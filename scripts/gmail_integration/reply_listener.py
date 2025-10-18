from celery_app import cel
from celery.schedules import crontab
from datetime import datetime, timedelta
from bson import ObjectId
from pymongo import MongoClient
import logging
import base64
import email

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from email.message import EmailMessage
from email.mime.application import MIMEApplication
import os

from . import db, collection, leads_collection, campaigns_collection, sent_emails_collection, replies_collection, users_collection, email_replies_collection, paired_emails_collection, legacy_leads_collection, legacy_campaigns_collection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_msg_id(headers):
    """Extract message-id from gmail headers"""
    refs = headers.get('References', '')
    in_reply_to = headers.get('In-Reply-To', '')
    return (refs + in_reply_to).split()

@cel.task(bind=True, max_retries=3)
def poll_inbox(self):
    """
    Poll Gmail inbox for new replies to sent emails
    """
    try:
        logger.info("Starting inbox polling for replies")
        
        # Import here to avoid circular imports
        from testapp import gmail_authenticate, classify_email
        
        service = gmail_authenticate()
        one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).isoformat() + 'Z'

        # Query emails received in last hour
        results = service.users().messages().list(
            userId='me',
            q=f'is:inbox after:{one_hour_ago}'
        ).execute()

        messages = results.get('messages', [])
        logger.info(f"Found {len(messages)} messages in the last hour")
        
        processed_count = 0
        for msg in messages:
            msg_id = msg['id']
            
            # Skip if already processed
            if replies.find_one({'msg_id': msg_id}):
                continue

            try:
                # Get message metadata
                full_msg = service.users().messages().get(
                    userId='me', 
                    id=msg_id, 
                    format='metadata',
                    metadataHeaders=['References', 'In-Reply-To', 'From', 'Subject']
                ).execute()

                headers = {h['name']: h['value'] for h in full_msg['payload']['headers']}
                refs = get_msg_id(headers)

                # Find original sent email
                sent = sent_emails.find_one({'msg_id': {'$in': refs}})
                if not sent:
                    continue

                # Get message body
                body_msg = service.users().messages().get(
                    userId='me', 
                    id=msg_id, 
                    format='raw'
                ).execute()
                
                mime_msg = email.message_from_bytes(
                    base64.urlsafe_b64decode(body_msg['raw'])
                )
                
                reply_text = ''
                if mime_msg.is_multipart():
                    for part in mime_msg.walk():
                        if part.get_content_type() == 'text/plain':
                            payload = part.get_payload(decode=True)
                            if payload:
                                reply_text += payload.decode('utf-8', 'ignore')
                else:
                    payload = mime_msg.get_payload(decode=True)
                    if payload:
                        reply_text = payload.decode('utf-8', 'ignore')

                # Classify reply
                category, meta = classify_email(reply_text)
                
                # Store reply
                replies.insert_one({
                    'sent_email_id': sent['_id'],
                    'msg_id': msg_id,
                    'reply_text': reply_text,
                    'category': category,
                    'confidence': meta.get('ml_confidence', 0),
                    'from': headers.get('From', ''),
                    'subject': headers.get('Subject', ''),
                    'timestamp': datetime.utcnow()
                })
                
                processed_count += 1
                logger.info(f"Processed reply from {headers.get('From', 'unknown')} - Category: {category}")
                
            except Exception as e:
                logger.error(f"Error processing message {msg_id}: {e}")
                continue

        logger.info(f"Processed {processed_count} new replies")
        return {"processed": processed_count, "total_messages": len(messages)}

    except Exception as exc:
        logger.error(f"Error in poll_inbox: {exc}")
        raise self.retry(countdown=300, exc=exc)  # Retry after 5 minutes

# Set up periodic task
@cel.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Configure periodic tasks"""
    sender.add_periodic_task(
        crontab(minute='*/5'),  # every 5 minutes (changed from 1 minute to reduce load)
        poll_inbox.s(),
        name='poll-for-replies'
    )