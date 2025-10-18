import requests
import time
import os
from gmail_utils import gmail_authenticate, get_clean_email_body
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from email.message import EmailMessage
from email.mime.application import MIMEApplication
import base64


# IMPROVEMENT: Use environment variables for configuration
FLASK_APP_URL = os.environ.get("FLASK_APP_URL", "http://127.0.0.1:5000/")
CHECK_INTERVAL_SECONDS = int(os.environ.get("CHECK_INTERVAL_SECONDS", 60))

def fetch_and_process_emails():
    """ Connects to Gmail, fetches unread emails, and sends them to the Flask app. """
    print("Connecting to Gmail...")
    service = gmail_authenticate()
    print("Connection successful. Starting live email processing...")

    while True:
        try:
            results = service.users().messages().list(userId='me', labelIds=['INBOX'], q='is:unread').execute()
            messages = results.get('messages', [])

            if not messages:
                print(f"No unread messages found. Waiting for {CHECK_INTERVAL_SECONDS} seconds...")
            else:
                print(f"Found {len(messages)} unread message(s).")
                for message_info in messages:
                    msg_id = message_info['id']
                    msg = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
                    
                    # IMPROVEMENT: Use the robust, centralized parser
                    subject, email_body = get_clean_email_body(msg['payload'])

                    if email_body:
                        print(f"\n--- Processing: '{subject}' ---")
                        
                        try:
                            # IMPROVEMENT: Send both subject and body to the classifier
                            payload = {'email': email_body, 'subject': subject}
                            response = requests.post(FLASK_APP_URL, data=payload)
                            
                            if response.status_code == 200:
                                print("✅ Successfully sent to Flask app for classification.")
                                # Mark as read ONLY on success
                                service.users().messages().modify(
                                    userId='me', id=msg_id, body={'removeLabelIds': ['UNREAD']}
                                ).execute()
                                print("✅ Marked email as read.")
                            else:
                                print(f"⚠️ Error from Flask app (Status: {response.status_code}). Will retry later.")
                        
                        except requests.exceptions.ConnectionError:
                            # IMPROVEMENT: Doesn't crash the script, just warns the user.
                            print("❌ ERROR: Could not connect to the Flask app. Is it running? Will retry later.")
                        
                        print("---------------------------------")
                    else:
                        print(f"ℹ️ No text body found for email ID: {msg_id}. Marking as read to skip.")
                        service.users().messages().modify(
                            userId='me', id=msg_id, body={'removeLabelIds': ['UNREAD']}
                        ).execute()

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == '__main__':
    fetch_and_process_emails()