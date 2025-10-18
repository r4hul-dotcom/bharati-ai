from google.auth.transport.requests import Request as GoogleAuthRequest
from google.auth.transport.requests import AuthorizedSession
from googleapiclient.discovery import build

# START OF FILE: email_utils.py (DEFINITIVE, FINAL, CORRECTED VERSION 5)

import os
import base64
import bs4
import re

from googleapiclient.http import HttpRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
# We need the standard Request object specifically for token refreshes
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.auth.transport.requests import AuthorizedSession
from googleapiclient.discovery import build
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
# Get the absolute path of the project's root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

TOKEN_FILE = os.path.join(BASE_DIR, "token.json")
CREDS_FILE = os.path.join(BASE_DIR, "credentials.json")




def CustomHttpRequestBuilder(http):
    """
    Request builder factory that returns CustomHttpRequest instances.
    """
    def builder(http_instance, postproc, uri, method='GET', body=None, headers=None,
                methodId=None, resumable=None):
        return CustomHttpRequest(http, postproc, uri, method, body, headers,
                                methodId, resumable)
    return builder

class RequestsHttp(AuthorizedSession):
    """
    A custom adapter that makes the 'requests' library behave like the older
    'httplib2' library by returning a (response, content) tuple.
    This is required by the googleapiclient.discovery.build() service object.
    """
    def request(self, *args, **kwargs):
        """
        Handle the request with proper parameter parsing.
        The Google API client calls this method in different ways:
        - request(uri, method, body=None, headers=None, ...)
        - request(method, uri, body=None, headers=None, ...)
        """
        # Parse arguments
        if len(args) >= 2:
            if args[0].startswith("http"):
                url, method = args[0], args[1]
                remaining_args = args[2:]
            elif args[1].startswith("http"):
                method, url = args[0], args[1]
                remaining_args = args[2:]
            else:
                url, method = args[0], args[1]
                remaining_args = args[2:]
        elif len(args) == 1:
            url = args[0]
            method = kwargs.pop("method", "GET")
            remaining_args = []
        else:
            url = kwargs.pop("url", kwargs.pop("uri", ""))
            method = kwargs.pop("method", "GET")
            remaining_args = []

        # Extract common parameters
        body = kwargs.pop("body", kwargs.pop("data", None))
        headers = kwargs.pop("headers", {})

        # Clean up conflicting kwargs
        clean_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ["method", "url", "uri", "data"]
        }

        self.credentials.before_request(
            self,   # session
            method,
            url,
            headers
        )

        # Perform actual request using AuthorizedSession (requests underneath)
        response = super(AuthorizedSession, self).request(
            method=method,
            url=url,
            data=body,
            headers=headers,
            **clean_kwargs
        )

        # Wrap response to mimic httplib2.Response
        shim = HttpResponseShim(response)
        return shim, response.content

def gmail_authenticate(user_credentials=None, interactive=False):
    """
    Handles Gmail authentication, using credentials from database or token file.

    Args:
        user_credentials: Dictionary containing user's Gmail OAuth credentials from database
        interactive: Whether to start interactive flow if no credentials available

    Returns:
        Gmail API service object
    """
    creds = None

    # PRIORITY 1: Use credentials passed from database (for Celery tasks)
    if user_credentials:
        try:
            creds = Credentials(
                token=user_credentials.get('token'),
                refresh_token=user_credentials.get('refresh_token'),
                token_uri=user_credentials.get('token_uri'),
                client_id=user_credentials.get('client_id'),
                client_secret=user_credentials.get('client_secret'),
                scopes=user_credentials.get('scopes')
            )
            print(f"✅ Using credentials from database")
        except Exception as e:
            print(f"❌ Failed to create credentials from database: {e}")
            creds = None

    # PRIORITY 2: Fall back to token file (for backward compatibility)
    if not creds and os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        print(f"✅ Using credentials from token file")

    # Refresh token if expired
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Token has expired. Attempting to refresh silently...")
            try:
                creds.refresh(GoogleAuthRequest())

                # If credentials came from database, we should update them
                # (This will be handled by the calling code)
                if user_credentials and os.path.exists(TOKEN_FILE):
                    with open(TOKEN_FILE, "w") as token:
                        token.write(creds.to_json())

                print("Token refreshed successfully.")
            except Exception as e:
                print(f"CRITICAL: Could not refresh token. Error: {e}")
                raise Exception("Token refresh failed.")

        elif interactive:
            print("No valid token found. Starting interactive login flow...")
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
            creds = flow.run_local_server(port=0, open_browser=False)
            with open(TOKEN_FILE, "w") as token:
                token.write(creds.to_json())
            print("Credentials saved to token file.")
        else:
            raise Exception("No valid credentials available and interactive mode is disabled.")

    # Build the Gmail service with custom adapter
    try:
        service = build('gmail', 'v1', credentials=creds)
        print("✅ Gmail service built successfully")
        return service
    except Exception as e:
        print(f"❌ Failed to build Gmail service: {e}")
        raise


def parse_email_parts(payload):
    headers = payload.get("headers", [])
    subject = ""
    for h in headers:
        if h["name"].lower() == "subject":
            subject = h["value"]
    parts = payload.get('parts', [])
    body_plain, body_html = "", ""
    if parts:
        for part in parts:
            mime_type, body, data = part.get('mimeType'), part.get('body'), part.get('body', {}).get('data')
            if mime_type == 'text/plain' and data: body_plain += base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            elif mime_type == 'text/html' and data: body_html += base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            if 'parts' in part:
                nested_subject, nested_plain, nested_html = parse_email_parts(part)
                if not subject: subject = nested_subject
                body_plain += nested_plain
                body_html += nested_html
    elif 'data' in payload.get('body', {}):
        data = payload['body']['data']
        if payload['mimeType'] == 'text/plain': body_plain = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        elif payload['mimeType'] == 'text/html': body_html = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
    return subject, body_plain, body_html

# --- ADD/REPLACE THESE TWO FUNCTIONS AT THE END OF gmail_service.py ---

def get_clean_email_body(payload):
    """
    This version correctly returns ONLY subject and body.
    """
    subject, body_plain, body_html = parse_email_parts(payload)
    if body_plain and len(body_plain) > 20:
        return subject, body_plain
    if body_html:
        soup = bs4.BeautifulSoup(body_html, 'html.parser')
        return subject, soup.get_text(separator='\n', strip=True)
    return subject, ""

def clean_reply_body(body):
    """
    This version correctly cleans quoted text from replies.
    """
    body = re.sub(r'On.*wrote:.*', '', body, flags=re.DOTALL | re.IGNORECASE)
    lines = [line for line in body.splitlines() if not line.strip().startswith('>')]
    cleaned_lines = []
    for line in lines:
        if line.strip() in ['--', '---', '–––']: break
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def create_draft_with_cc(user_credentials, original_email_id, reply_text, cc_recipients, to_email, subject):
    """
    Creates a reply draft with CC recipients using Gmail API.
    Args:
        user_credentials: Dict with user's OAuth credentials from database
                         (has keys: token, refresh_token, token_uri, client_id, client_secret, scopes)
        original_email_id: Gmail message ID being replied to (for threading)
        reply_text: The reply body text (HTML or plain text)
        cc_recipients: List or comma-separated string of CC emails
        to_email: Who to reply to
        subject: Reply subject line
    Returns:
        Dict with 'success' and 'draft_id' or 'error' key
    """
    try:
        print(f"📧 Creating draft with CC recipients...")
        
        # Authenticate with user's credentials
        service = gmail_authenticate(user_credentials=user_credentials)
        if not service:
            print("❌ Gmail authentication failed")
            return {'success': False, 'error': 'Authentication failed'}
        
        # 🆕 STEP 1: Get the thread ID from the original message
        print(f"🔍 Fetching thread ID for message: {original_email_id}")
        try:
            original_message = service.users().messages().get(
                userId='me', 
                id=original_email_id,
                format='minimal'
            ).execute()
            thread_id = original_message.get('threadId')
            print(f"✓ Got thread ID: {thread_id}")
        except Exception as e:
            print(f"⚠️ Could not fetch thread ID: {str(e)}")
            print(f"   Will create draft without threading")
            thread_id = None
        
        # Parse CC recipients if it's a string
        if isinstance(cc_recipients, str):
            cc_list = [e.strip() for e in cc_recipients.split(',') if e.strip()]
        else:
            cc_list = cc_recipients if isinstance(cc_recipients, list) else []
        print(f"✓ Parsed {len(cc_list)} CC recipients")
        
        # Create MIME message
        from email.mime.text import MIMEText
        message = MIMEText(reply_text, 'html')
        message['to'] = to_email
        message['subject'] = subject
        
        # 🆕 Add In-Reply-To and References headers for proper threading
        message['In-Reply-To'] = f'<{original_email_id}@mail.gmail.com>'
        message['References'] = f'<{original_email_id}@mail.gmail.com>'
        
        # Add CC if there are any recipients
        if cc_list:
            message['cc'] = ', '.join(cc_list)
            print(f"✓ Added CC: {', '.join(cc_list)}")
        
        # Encode the message for the API
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        print(f"✓ Message encoded ({len(encoded_message)} bytes)")
        
        # Create draft body
        draft_body = {
            'message': {
                'raw': encoded_message
            }
        }
        
        # 🆕 Only add threadId if we successfully got it
        if thread_id:
            draft_body['message']['threadId'] = thread_id
            print(f"✓ Using thread ID: {thread_id}")
        else:
            print(f"⚠️ Creating draft without thread ID (will start new conversation)")
        
        print(f"📤 Calling Gmail API to create draft...")
        draft = service.users().drafts().create(userId='me', body=draft_body).execute()
        
        draft_id = draft.get('id')
        if draft_id:
            print(f"✅ Draft created successfully!")
            print(f"   Draft ID: {draft_id}")
            print(f"   To: {to_email}")
            print(f"   CC: {len(cc_list)} recipients")
            print(f"   Subject: {subject}")
            return {
                'success': True,
                'draft_id': draft_id,
                'message': f'Draft created with {len(cc_list)} CC recipients'
            }
        else:
            print("❌ Gmail API didn't return a draft ID")
            return {'success': False, 'error': 'No draft ID returned from Gmail API'}
            
    except Exception as e:
        print(f"❌ Error creating draft: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
