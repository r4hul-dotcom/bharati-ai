import time
from pymongo import MongoClient
import re
import os
from datetime import datetime
from bson import ObjectId
# This import now works correctly!
from gmail_utils import gmail_authenticate, get_clean_email_body

from . import db, collection, leads_collection, campaigns_collection, sent_emails_collection, replies_collection, users_collection, email_replies_collection, paired_emails_collection, legacy_leads_collection, legacy_campaigns_collection

def clean_reply_body(body):
    """ Removes quoted text and signatures from an email body. """
    body = re.sub(r'On.*wrote:.*', '', body, flags=re.DOTALL)
    lines = [line for line in body.splitlines() if not line.strip().startswith('>')]
    cleaned_lines = []
    for line in lines:
        if line.strip() in ['--', '---', '–––']:
            break
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

def analyze_reply_intent(text):
    """ Categorizes a reply based on keywords. """
    text = text.lower()
    if any(kw in text for kw in ['interested', 'schedule a call', 'sounds good', 'let\'s talk', 'send more details', 'proposal']):
        return 'interested'
    if any(kw in text for kw in ['not interested', 'unsubscribe', 'remove me', 'not a good fit', 'stop sending']):
        return 'negative'
    return 'neutral'

def fetch_and_process_replies():
    """ Fetches replies, saves them to the replies collection, and updates lead status. """
    service = gmail_authenticate()
    results = service.users().messages().list(userId="me", q="is:unread").execute()
    messages = results.get("messages", [])

    if len(messages) == 0:
        return # No new messages, exit function early

    print(f"Found {len(messages)} unread messages. Checking for campaign replies...")

    for msg in messages:
        try:
            msg_data = service.users().messages().get(userId="me", id=msg["id"], format='full').execute()
            headers = msg_data['payload']['headers']
            in_reply_to = next((h['value'] for h in headers if h['name'] == 'In-Reply-To'), None)

            if not in_reply_to: continue

            original_sent_email = sent_emails_collection.find_one({'message_id': in_reply_to})
            if not original_sent_email: continue
            
            lead_id = original_sent_email['lead_id']
            lead = leads_collection.find_one({'_id': ObjectId(lead_id)})
            if not lead: continue

            # get_clean_email_body now returns 4 values, which we can use
            subject, body, from_name, from_email = get_clean_email_body(msg_data['payload'])
            cleaned_body = clean_reply_body(body)
            intent = analyze_reply_intent(cleaned_body)
            
            print(f"\n--- Found Reply for Lead: {lead.get('name', 'N/A')} ---")

            reply_doc = {
                "sent_email_id": original_sent_email['_id'],
                "lead_id": lead_id,
                "assigned_to_sales_person": lead.get('assigned_to'),
                "from_name": from_name, "from_email": from_email, "subject": subject,
                "body": cleaned_body, "category": intent, "timestamp": datetime.now()
            }
            replies_collection.insert_one(reply_doc)
            print(f"✅ Reply from '{from_email}' saved to database.")

            leads_collection.update_one(
                {'_id': ObjectId(lead_id)},
                {'$set': {'status': intent, 'last_reply_at': datetime.now()}}
            )
            print(f"✅ Updated lead status to '{intent}'.")
            
            service.users().messages().modify(userId='me', id=msg['id'], body={'removeLabelIds': ['UNREAD']}).execute()
        except Exception as e:
            print(f"!!-- Error processing message ID {msg.get('id')}: {e}")

if __name__ == "__main__":
    print("Starting Campaign Reply Fetching Service... (Checks every 60 seconds)")
    while True:
        fetch_and_process_replies()
        time.sleep(60)