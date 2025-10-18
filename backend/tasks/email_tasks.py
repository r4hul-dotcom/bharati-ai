# === Standard Library Imports ===
import logging
import time
import base64
import os
from datetime import datetime
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()
from backend.app.models import CampaignReply # <-- ADD THIS

# === Third-Party Imports ===
from bson import ObjectId
from textblob import TextBlob
from pymongo import MongoClient

# === Local Application Imports ===
# Import the Celery app instance itself
from .celery_worker import celery_app

# Import the application factory to create a context for database access
from backend.app import create_app

# Import specific utilities from your project structure
# NOTE: Ensure 'send_email_with_retry' exists in your email_utils.py or rename it here.
# --- CORRECTED IMPORT ---
from backend.app.gmail_service import gmail_authenticate, get_clean_email_body, clean_reply_body

# Setup a logger for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===================================================================
# === MONGODB CONNECTION FOR CELERY ===
# ===================================================================
# Optionally set environment variables in code (override if not set in shell)
os.environ.setdefault('MONGO_URI', 'mongodb+srv://bharatiadmin:Exec%4002@cluster0.nt2rwkw.mongodb.net/bharati_ai?retryWrites=true&w=majority')
os.environ.setdefault('DB_NAME', 'bharati_ai')

MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('DB_NAME')

# Connect to MongoDB
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]

# MongoDB Collections
leads_collection = db['sales_leads']
campaigns_collection = db['sales_campaigns']
sent_emails_collection = db['sent_emails']
users_collection = db['users']
campaign_replies_collection = db['campaign_replies']  # ✅ Now using MongoDB!

logger.info(f"✅ MongoDB initialized in Celery: {DB_NAME}")


# ===================================================================
# === CONFIGURATION (Can be moved to a settings file later) ===
# ===================================================================
SALES_TEAM_CC_LIST = []
EXECUTIVE_CC_LIST = []

# ===================================================================
# === HELPER FUNCTIONS (Not Celery Tasks) ===
# ===================================================================

def classify_reply_category(text):
    """ Simple classifier for sales replies. """
    text = text.lower()
    if any(keyword in text for keyword in ["unsubscribe", "remove me", "stop sending"]):
        return "unsubscribed"
    if any(keyword in text for keyword in ["interested", "quote", "pricing", "like to know more", "send details", "schedule a call"]):
        return "interested"
    if any(keyword in text for keyword in ["not interested", "not for us", "no thanks"]):
        return "negative"
    
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0.3:
        return "positive"
    if sentiment < -0.2:
        return "negative"
    return "neutral"

# ===================================================================
# === CELERY TASKS ===
# ===================================================================
@celery_app.task(bind=True, max_retries=3)
def check_for_replies(self):
    """
    Checks for and processes replies to campaign emails for ALL users with Gmail connected.
    Saves replies to the MongoDB database.
    """
    logger.info("Starting campaign reply check for all connected users...")
    
    try:
        # Get all users who have Gmail connected
        users_with_gmail = users_collection.find({'gmail_connected': True})
        
        total_users_checked = 0
        total_replies_processed = 0
        
        for user in users_with_gmail:
            user_email = user.get('email', 'unknown')
            user_id = str(user.get('_id'))
            
            logger.info(f"Checking inbox for user: {user_email}")
            
            try:
                # Get user's Gmail credentials from database
                gmail_creds_encrypted = user.get('google_credentials_encrypted')
                
                if not gmail_creds_encrypted:
                    logger.warning(f"⚠️ User {user_email} has gmail_connected=True but no credentials found")
                    continue
                
                # Prepare credentials dictionary
                gmail_creds = {
                    'token': gmail_creds_encrypted.get('token'),
                    'refresh_token': gmail_creds_encrypted.get('refresh_token'),
                    'token_uri': gmail_creds_encrypted.get('token_uri'),
                    'client_id': gmail_creds_encrypted.get('client_id'),
                    'client_secret': gmail_creds_encrypted.get('client_secret'),
                    'scopes': gmail_creds_encrypted.get('scopes')
                }
                
                # Authenticate with this user's credentials
                service = gmail_authenticate(user_credentials=gmail_creds)
                
                if not service:
                    logger.error(f"❌ Gmail authentication failed for user {user_email}")
                    continue
                
                total_users_checked += 1
                
                # Check for unread messages
                results = service.users().messages().list(userId='me', q='is:unread').execute()
                messages = results.get('messages', [])
                
                if not messages:
                    logger.info(f"No unread messages for {user_email}")
                    continue
                
                # Process each message
                for message in messages:
                    msg_id = message['id']
                    campaign_id = is_campaign_reply(service, msg_id)
                    
                    if campaign_id:
                        logger.info(f"✅ Found campaign reply in {user_email}'s inbox: {msg_id}")
                        
                        # Get full message details
                        msg = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
                        
                        # Extract reply details
                        headers = msg['payload']['headers']
                        from_email = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
                        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                        date_str = next((h['value'] for h in headers if h['name'] == 'Date'), None)
                        
                        # Get email body
                        body = get_clean_email_body(msg)
                        cleaned_body = clean_reply_body(body)
                        
                        # Save to database
                        reply_data = {
                            'campaign_id': campaign_id,
                            'message_id': msg_id,
                            'from_email': from_email,
                            'subject': subject,
                            'body': cleaned_body,
                            'received_at': datetime.utcnow(),
                            'user_email': user_email,  # Track which user's inbox this came from
                            'user_id': user_id
                        }
                        
                        # Check if already exists
                        existing = replies_collection.find_one({'message_id': msg_id})
                        if not existing:
                            replies_collection.insert_one(reply_data)
                            total_replies_processed += 1
                            logger.info(f"💾 Saved reply from {from_email} to database")
                        else:
                            logger.info(f"⚠️ Reply {msg_id} already exists in database, skipping")
                        
                        # Mark as read
                        service.users().messages().modify(
                            userId='me',
                            id=msg_id,
                            body={'removeLabelIds': ['UNREAD']}
                        ).execute()
                        logger.info(f"📧 Marked message {msg_id} as read")
                
            except Exception as user_error:
                logger.error(f"❌ Error processing inbox for {user_email}: {str(user_error)}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        summary = f"Reply check complete: {total_users_checked} users checked, {total_replies_processed} new replies found"
        logger.info(summary)
        return summary
        
    except Exception as e:
        logger.error(f"❌ Critical error in check_for_replies: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise self.retry(exc=e, countdown=60)


def is_campaign_reply(service, msg_id):
    """
    Checks if a message is a reply to one of our campaigns by inspecting its thread.
    Returns the Campaign ID if it is, otherwise returns False.
    """
    try:
        msg_meta = service.users().messages().get(userId='me', id=msg_id, format='metadata', metadataHeaders=['threadId']).execute()
        thread_id = msg_meta.get('threadId')
        if not thread_id: return False

        thread = service.users().threads().get(userId='me', id=thread_id).execute()
        for thread_message in thread['messages']:
            headers = thread_message['payload'].get('headers', [])
            for header in headers:
                if header['name'] == 'X-Campaign-ID':
                    return header['value'] # Return the Campaign ID
        return False
    except Exception as e:
        logger.error(f"Error in is_campaign_reply for msg {msg_id}: {e}")
        return False

def process_campaign_reply(service, msg_id, campaign_id):
    """
    Fetches full details of a reply, saves it to MongoDB, and marks it as read.
    """
    try:
        # ✅ Check if reply already processed (MongoDB)
        if campaign_replies_collection.find_one({"message_id": msg_id}):
            logger.warning(f"Reply {msg_id} has already been processed. Skipping.")
            service.users().messages().modify(userId='me', id=msg_id, body={'removeLabelIds': ['UNREAD']}).execute()
            return

        # Get the full message
        msg = service.users().messages().get(userId='me', id=msg_id).execute()

        # Get sender from the main message headers
        headers = {h['name']: h['value'] for h in msg['payload'].get('headers', [])}
        sender = headers.get('From', 'Unknown Sender')

        # Get subject and body
        subject, raw_body = get_clean_email_body(msg['payload'])

        # Clean the raw body
        cleaned_body = clean_reply_body(raw_body)
        
        # Classify the reply
        category = classify_reply_category(cleaned_body)

        # ✅ Save to MongoDB
        reply_doc = {
            "message_id": msg_id,
            "thread_id": msg.get('threadId'),
            "campaign_id": campaign_id,  # Keep as string for now
            "sender_email": sender,
            "subject": subject,
            "body": cleaned_body,
            "category": category,
            "received_at": datetime.utcnow(),
            "processed": True
        }
        
        campaign_replies_collection.insert_one(reply_doc)
        logger.info(f"✅ Successfully saved reply from {sender} to MongoDB.")
        
        # Mark as read
        service.users().messages().modify(userId='me', id=msg_id, body={'removeLabelIds': ['UNREAD']}).execute()
        logger.info(f"✅ Marked message {msg_id} as read.")

    except Exception as e:
        logger.error(f"❌ Error processing and saving campaign reply: {e}", exc_info=True)


@celery_app.task(bind=True)
def send_campaign_batch(self, campaign_id, lead_ids, sender_user_id, signature_html):
    """ Celery task that queues up a batch of individual campaign emails. """
    total_leads = len(lead_ids)
    logger.info(f"Starting campaign batch for {total_leads} leads.")
    
    for i, lead_id in enumerate(lead_ids):
        # Call the more detailed, correct email sending task
        send_campaign_email.delay(lead_id, campaign_id, sender_user_id, signature_html)
        
        self.update_state(state='PROGRESS',
                          meta={'current': i + 1, 'total': total_leads,
                                'status': f'Queued email {i+1}/{total_leads}...'})
    
    return {'current': total_leads, 'total': total_leads, 'status': 'All emails have been queued.'}


@celery_app.task(name='backend.tasks.email_tasks.send_campaign_email', bind=True, max_retries=3, default_retry_delay=15)
def send_campaign_email(self, lead_id, campaign_id, sender_user_id, signature_html):
    """
    (Primary Version) Celery task to send a single personalized email with signature
    AND a custom X-Campaign-ID header for reply tracking.
    """
    logger.info(f"Task started for lead: {lead_id}, campaign: {campaign_id}")
    
    try:
        # ✅ Use collections initialized at module level (not from app context)
        lead = leads_collection.find_one({"_id": ObjectId(lead_id)})
        campaign = campaigns_collection.find_one({"_id": ObjectId(campaign_id)})
        sender = users_collection.find_one({'_id': ObjectId(sender_user_id)})

        if not all([lead, campaign, sender]):
            logger.warning(f"Skipping: Missing data for lead {lead_id}, campaign {campaign_id}, or sender {sender_user_id}.")
            return {'status': 'Skipped'}

        # ✅ Get Gmail credentials from database
        gmail_creds = sender.get('google_credentials_encrypted')
        if not gmail_creds:
            logger.error(f"❌ No Gmail credentials found for user {sender_user_id}. User must connect Gmail account.")
            return {'status': 'Error', 'message': 'Gmail not connected'}
        
        logger.info(f"🔐 Authenticating Gmail for user {sender_user_id}")
        
        # ✅ Authenticate with user's credentials from database
        service = gmail_authenticate(user_credentials=gmail_creds)
        
        if not service:
            logger.error(f"❌ Gmail authentication failed for user {sender_user_id}")
            return {'status': 'Error', 'message': 'Gmail authentication failed'}

        # --- EMAIL CONSTRUCTION LOGIC ---
        personalized_body = campaign['body'].replace("{{lead_name}}", lead.get('name', 'there'))
        final_email_html = personalized_body + signature_html

        # Create a proper MIME message
        message = MIMEText(final_email_html, 'html')
        message['to'] = lead['email']
        message['subject'] = campaign['subject']

        # CRITICAL: Add the custom header for tracking replies
        message['X-Campaign-ID'] = str(campaign['_id'])

        # Add CC if applicable
        cc_list = list(set(EXECUTIVE_CC_LIST))
        if cc_list:
            message['cc'] = ", ".join(cc_list)

        # Encode the message for the API
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {'raw': encoded_message}

        # Send the message
        logger.info(f"📧 Sending email to {lead['email']}")
        sent_message = service.users().messages().send(userId="me", body=create_message).execute()
        message_id = sent_message.get('id')

        if message_id:
            sent_emails_collection.insert_one({
                "lead_id": lead['_id'],
                "campaign_id": campaign['_id'],
                "message_id": message_id,
                "sent_by_id": sender['_id'],
                "sent_by_name": sender.get('name'),
                "timestamp": datetime.now()
            })
            leads_collection.update_one(
                {'_id': lead['_id']},
                {'$set': {'status': 'contacted', 'last_contacted_at': datetime.now()}}
            )
            logger.info(f"✅ Success: Sent email to {lead['email']} with message ID: {message_id}")
            return {'status': 'Sent', 'message_id': message_id}
        else:
            raise Exception("Failed to get a message ID from the Gmail API.")

    except Exception as exc:
        logger.exception(f"❌ Email to lead {lead_id} failed on attempt {self.request.retries + 1}. Error: {exc}")
        raise self.retry(exc=exc)

@celery_app.task
def send_assignment_notification_email(sales_person_email, sales_person_name, campaign_name, assigner_name):
    """
    Sends a simple email notification about a new campaign assignment.
    Uses token file authentication (system account).
    """
    logger.info(f"Sending assignment notification to {sales_person_email}")
    try:
        # ✅ For system notifications, use token file method
        service = gmail_authenticate()
        
        if not service:
            logger.error("Gmail authentication failed for notification email.")
            return {"status": "Failed", "error": "Authentication failed"}
        
        subject = f"New Campaign Assignment: {campaign_name}"
        body = f"""
        Hi {sales_person_name},<br><br>
        You have been assigned a new email campaign, '{campaign_name}', by {assigner_name}.<br><br>
        Thank you,<br>
        The Bharati AI Team
        """
        send_gmail_with_retry(service, to=sales_person_email, subject=subject, body=body)
        logger.info(f"✅ Sent campaign assignment notification to {sales_person_email}")
        return {"status": "Notification sent"}
    except Exception as e:
        logger.error(f"❌ Failed to send assignment notification: {e}", exc_info=True)
        return {"status": "Failed", "error": str(e)}

# Note: The other, simpler 'send_campaign_email' function has been removed to avoid conflict.
# The more detailed version above is now the single source of truth for sending campaign emails.
