# START OF FILE: fetch_gmail_replies.py (FINAL, CORRECTED VERSION)

import time
from pymongo import MongoClient
import re
import os
from datetime import datetime
from bson import ObjectId

# --- You will need to make sure you have a gmail_utils.py file with these functions ---
# --- or integrate their logic directly here.                                    ---
from gmail_utils import gmail_authenticate, get_clean_email_body


from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from email.message import EmailMessage
from email.mime.application import MIMEApplication
import base64

from . import db, collection, leads_collection, campaigns_collection, sent_emails_collection, replies_collection, users_collection, email_replies_collection, paired_emails_collection, legacy_leads_collection, legacy_campaigns_collection

def clean_reply_body(body):
    """ A more robust way to remove quoted text and signatures. """
    body = re.sub(r'On.*wrote:.*', '', body, flags=re.DOTALL)
    lines = [line for line in body.splitlines() if not line.strip().startswith('>')]
    cleaned_lines = []
    for line in lines:
        if line.strip() in ['--', '---', '–––']:
            break
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

def analyze_reply_intent(text):
    """ Simple keyword-based analysis to categorize a reply. """
    text = text.lower()
    positive_keywords = ['interested', 'schedule a call', 'sounds good', 'let\'s talk', 'send more details', 'proposal']
    negative_keywords = ['not interested', 'unsubscribe', 'remove me', 'not a good fit', 'stop sending']
    
    if any(kw in text for kw in positive_keywords):
        return 'interested'
    if any(kw in text for kw in negative_keywords):
        return 'negative'
    return 'neutral'

def fetch_and_process_replies():
    """ Fetches replies, saves them to the replies collection, and updates lead status. """
    service = gmail_authenticate()
    query = "is:unread" 
    results = service.users().messages().list(userId="me", q=query).execute()
    messages = results.get("messages", [])

    if len(messages) > 0:
        print(f"Found {len(messages)} unread messages. Checking for replies...")

    for msg in messages:
        try:
            msg_data = service.users().messages().get(userId="me", id=msg["id"], format='full').execute()
            headers = msg_data['payload']['headers']
            
            in_reply_to = next((h['value'] for h in headers if h['name'] == 'In-Reply-To'), None)
            if not in_reply_to:
                continue

            original_sent_email = sent_emails_collection.find_one({'message_id': in_reply_to})
            if not original_sent_email:
                continue

            # --- We found a valid reply to one of our campaigns ---
            lead_id = original_sent_email['lead_id']
            lead = leads_collection.find_one({'_id': ObjectId(lead_id)})
            if not lead:
                continue

            subject, body, from_name, from_email = get_clean_email_body(msg_data['payload'])
            cleaned_body = clean_reply_body(body)
            intent = analyze_reply_intent(cleaned_body)
            
            print(f"\n--- Found Reply for Lead: {lead.get('name', 'N/A')} ---")
            print(f"Intent Detected: {intent.upper()}")

            # --- THE CRITICAL FIX: Save the reply to the replies_collection ---
            reply_doc = {
                "sent_email_id": original_sent_email['_id'],
                "lead_id": lead_id,
                "assigned_to_sales_person": lead.get('assigned_to'),
                "from_name": from_name,
                "from_email": from_email,
                "subject": subject,
                "body": cleaned_body,
                "category": intent,
                "timestamp": datetime.now()
            }
            replies_collection.insert_one(reply_doc)
            print(f"✅ Reply from '{from_email}' saved to database.")
            # --- END OF CRITICAL FIX ---

            # Update the lead's status in the leads collection
            leads_collection.update_one(
                {'_id': ObjectId(lead_id)},
                {'$set': {'status': intent, 'last_reply_at': datetime.now()}}
            )
            print(f"✅ Updated lead status to '{intent}'.")
            
            # Mark the email as read
            service.users().messages().modify(userId='me', id=msg['id'], body={'removeLabelIds': ['UNREAD']}).execute()

        except Exception as e:
            print(f"!!-- Error processing message ID {msg.get('id')}: {e}")
            # Optionally, you could mark the message as unread again to retry later
            # service.users().messages().modify(userId='me', id=msg['id'], body={'addLabelIds': ['UNREAD']}).execute()


# --- THE CRUCIAL FIX: Make the script run forever ---
if __name__ == "__main__":
    print("Starting Reply Fetching Service... (Checks every 60 seconds)")
    while True:
        fetch_and_process_replies()
        time.sleep(60) # Wait for 60 seconds before checking for new emails again