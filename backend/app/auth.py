# backend/app/auth.py

from datetime import datetime
from bson import ObjectId
import logging

from flask import Blueprint, render_template, redirect, url_for, request, flash, session
from werkzeug.security import check_password_hash, generate_password_hash
from flask_login import login_user, logout_user, login_required, current_user
from functools import wraps
from google_auth_oauthlib.flow import Flow
from . import users_collection  # Import your users collection
from .utils import encrypt_token # Import the encryption utility
from bson.objectid import ObjectId # Ensure ObjectId is imported if not already
import os
from werkzeug.security import check_password_hash

# Import your User model and the database instance
from .models import User
#from . import db

# This should be your project's root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CLIENT_SECRETS_FILE = os.path.join(BASE_DIR, 'credentials.json')

# Define the scopes: what you want to access
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

auth = Blueprint('auth', __name__)

@auth.route('/authorize')
@login_required
def authorize():
    """Route to start the Google OAuth 2.0 flow."""
    redirect_uri = 'https://sales-dashboard.bharatifire.com/auth/oauth2callback'
    
    # DEBUG: Log the redirect URI
    print(f"=== DEBUG: Generated redirect URI: {redirect_uri} ===")
    import logging
    logging.warning(f"REDIRECT_URI: {redirect_uri}")
    
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=redirect_uri
    )
    # The prompt='consent' is important to ensure you get a refresh_token every time
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        prompt='consent'
    )
    session['state'] = state
    return redirect(authorization_url)


@auth.route('/oauth2callback')
def oauth2callback():
    try:
        from backend.app import users_collection
        from flask_login import current_user
        from datetime import datetime
        
        state = session.get('state')
        redirect_uri = 'https://sales-dashboard.bharatifire.com/auth/oauth2callback'

        logging.warning(f"CALLBACK REDIRECT_URI: {redirect_uri}")

        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            state=state,
            redirect_uri=redirect_uri
        )

        # Get the authorization response and force HTTPS
        authorization_response = request.url
        if authorization_response.startswith('http://'):
            authorization_response = authorization_response.replace('http://', 'https://', 1)

        # Fetch the token
        flow.fetch_token(authorization_response=authorization_response)

        # Store credentials in session (temporary)
        credentials = flow.credentials
        session['credentials'] = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }

        # CRITICAL FIX: Save credentials to database permanently
        if current_user.is_authenticated:
            result = users_collection.update_one(
                {'_id': ObjectId(current_user.id)},
                {
                    '$set': {
                        'gmail_connected': True,
			'google_credentials_encrypted': {
                            'token': credentials.token,
                            'refresh_token': credentials.refresh_token,
                            'token_uri': credentials.token_uri,
                            'client_id': credentials.client_id,
                            'client_secret': credentials.client_secret,
                            'scopes': credentials.scopes
                        },
                        'gmail_connected_at': datetime.utcnow()
                    }
                }
            )
            logging.warning(f"✅ Gmail credentials saved to database for user {current_user.id}, modified: {result.modified_count}")
            print(f"=== DEBUG: Credentials saved to DB, modified: {result.modified_count} ===")
        else:
            logging.error("❌ User not authenticated during OAuth callback")
        
        flash('Gmail connected successfully! You can now send campaigns.', 'success')
        # Redirect based on user role
        if current_user.is_authenticated and current_user.role == 'Sales Employee':
            return redirect('/sales_dashboard_personal')
        else:
            return redirect('/dashboard')

    except Exception as e:
        print(f"OAuth callback error: {str(e)}")
        logging.error(f"OAuth callback error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        flash(f'Authorization failed: {str(e)}', 'danger')
        # Redirect based on user role
        if current_user.is_authenticated and current_user.role == 'Sales Employee':
            return redirect('/sales_dashboard_personal')
        else:
            return redirect('/dashboard')


# 2. Attach your routes to the 'auth' blueprint instead of '@app'

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        if current_user.role == 'exec':
            return redirect(url_for('main.dashboard')) # Note: 'main.' prefix might be needed
        else:
            return redirect(url_for('main.sales_dashboard_personal'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_doc = users_collection.find_one({'email': email})

        if user_doc and check_password_hash(user_doc.get('password', ''), password):
            user_obj = User(user_doc)
            login_user(user_obj, remember=True)
            
            flash(f'Welcome, {user_obj.name}!', 'success')

            # Redirect based on role
            if user_obj.role == 'exec':
                return redirect(url_for('main.dashboard'))
            elif user_obj.role == 'sales':
                return redirect(url_for('main.sales_dashboard_personal'))
            else:
                return redirect(url_for('main.home'))
        else:
            flash('Invalid email or password. Please try again.', 'danger')
            return redirect(url_for('auth.login'))

    return render_template('login.html')


@auth.route('/logout')
@login_required
def logout():
    session.pop('_flashes', None)
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))

# 3. Your other auth-related functions can remain in this file as helpers
# They are not routes, so they don't need a decorator.

def create_user(email, password, role='sales', name=None):
    """Utility to create a new user with a hashed password."""
    if users_collection.find_one({'email': email}):
        print(f"User with email {email} already exists.")
        return

    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    
    user_data = {
        'email': email,
        'password': hashed_password,
        'role': role
    }
    if name:
        user_data['name'] = name
    
    users_collection.insert_one(user_data)
    print(f"User {email} created successfully.")


def role_required(role: str):
    def wrapper(fn):
        @wraps(fn)
        @login_required
        def decorated_view(*args, **kwargs):
            if current_user.role != role:
                from flask import abort
                abort(403)
            return fn(*args, **kwargs)
        return decorated_view
    return wrapper
