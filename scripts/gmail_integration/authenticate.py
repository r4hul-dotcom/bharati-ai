import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
TOKEN_FILE = "token.json"
CREDS_FILE = "credentials.json"

def gmail_authenticate():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Refreshing expired token...")
            creds.refresh(Request())
        else:
            print("🚀 Starting Gmail Authentication Process...")
            print("   Copy and paste the authorization URL into your browser:")
            print("-" * 65)
            
            if not os.path.exists(CREDS_FILE):
                print(f"❌ Error: {CREDS_FILE} not found!")
                print("Please download your OAuth credentials from Google Cloud Console")
                return None
                
            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
                
                # Set redirect_uri explicitly for headless environments
                flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
                
                # Get authorization URL with explicit redirect_uri
                auth_url, _ = flow.authorization_url(prompt='consent')
                print(f"\nAuthorization URL:\n{auth_url}\n")
                
                # Get authorization code from user
                auth_code = input("Enter the authorization code: ").strip()
                
                # Exchange code for token
                flow.fetch_token(code=auth_code)
                creds = flow.credentials
                
                print("-" * 65)
                print("✅ Authorization successful!")
            except Exception as e:
                print(f"❌ Authentication error: {e}")
                print("\nTroubleshooting steps:")
                print("1. Make sure your credentials.json is for a 'Desktop application'")
                print("2. Add 'urn:ietf:wg:oauth:2.0:oob' as an authorized redirect URI")
                print("3. Ensure the OAuth consent screen is properly configured")
                return None
        
        # Save the credentials for the next run
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
        print(f"✅ Token saved to {TOKEN_FILE}")
    
    return build("gmail", "v1", credentials=creds)

def test_gmail_connection():
    """Test the Gmail connection by getting user profile"""
    try:
        service = gmail_authenticate()
        
        # Test the connection
        profile = service.users().getProfile(userId='me').execute()
        print(f"\n🎉 Gmail connection successful!")
        print(f"   Email: {profile['emailAddress']}")
        print(f"   Total messages: {profile['messagesTotal']}")
        print(f"   Threads total: {profile['threadsTotal']}")
        
        # Test getting unread messages
        unread_results = service.users().messages().list(userId='me', q='is:unread').execute()
        unread_count = len(unread_results.get('messages', []))
        print(f"   Unread messages: {unread_count}")
        
        return True
    except Exception as e:
        print(f"\n❌ Gmail connection failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("   Gmail Authentication & Test")
    print("=" * 50)
    
    try:
        success = test_gmail_connection()
        if success:
            print("\n🎉 Setup complete! You can now use the Gmail integration.")
            print("   Your Celery workers should be able to access Gmail now.")
        else:
            print("\n💥 Setup failed. Please check your credentials.json file.")
    except KeyboardInterrupt:
        print("\n\n⚠️  Authentication cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure credentials.json exists and is valid")
        print("2. Check your internet connection")  
        print("3. Verify the OAuth consent screen is configured")
        print("4. Make sure you're using the correct OAuth client type (Desktop application)")
