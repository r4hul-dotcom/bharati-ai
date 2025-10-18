# --- Flask and Third-Party Imports ---
from flask import (
    Blueprint, render_template, request, redirect, url_for, flash, jsonify,
    session, abort, send_file, Response, make_response, send_from_directory,
    render_template_string
)
from flask_login import login_required, current_user
from bson import ObjectId
import os
import pandas as pd
import traceback
import io
from datetime import datetime, timedelta # Added timedelta for notifications
from celery import Celery
import random
from werkzeug.security import generate_password_hash # Added for add_user route

from weasyprint import HTML

from werkzeug.security import check_password_hash
import jwt


# --- Blueprint Setup ---
main = Blueprint('main', __name__)

# Import db and collections from __init__.py
from . import (
    collection,
    leads_collection,
    campaigns_collection,
    sent_emails_collection,
    replies_collection,
    users_collection
)


# --- Local Application Imports ---
from .models import User
from .auth import role_required
from .utils import (
    nocache, extract_details, convert_numpy_types,
    generate_catalogue_reply, generate_complaint_reply, generate_followup_reply, generate_other_reply,
    get_my_reports_data, get_dashboard_data, prepare_executive_report_data, get_product_list_for_dropdown,
    create_reply_category_chart,
    generate_pdf_from_template # <-- ADD THIS LINE
)
from backend.ml_model.classifier import classify_email
from .models import CampaignReply # <-- ADD THIS

from flask import current_app

# --- Celery Client Setup ---
# This client is used ONLY to send tasks to the broker. It does not run worker code.
celery_client = Celery(
    'flask_client',
    broker=os.environ.get('CELERY_BROKER_URL', 'redis://127.0.0.1:6379/0'),
    # Point to the same backend to be able to retrieve task status
    backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://127.0.0.1:6379/0')
)


@main.route('/download-extension')
@login_required
def download_extension():
    """Page to download the Chrome extension"""
    return render_template('download_extension.html')

# Add this route for extension login
@main.route('/api/extension_login', methods=['POST'])
def extension_login():
    """
    Special login endpoint for Chrome extension authentication.
    Returns a JWT token that the extension can use.
    """
    from backend.app import users_collection
    
    data = request.json
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({'error': 'Missing email or password'}), 400
    
    email = data['email']
    password = data['password']
    
    # Find user in database
    user = users_collection.find_one({'email': email})
    
    if not user or not check_password_hash(user.get('password', ''), password):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Generate JWT token
    secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    token_payload = {
        'user_id': str(user['_id']),
        'email': email,
        'exp': datetime.utcnow() + timedelta(days=30)  # Token valid for 30 days
    }
    
    token = jwt.encode(token_payload, secret_key, algorithm='HS256')
    
    return jsonify({
        'token': token,
        'user': {
            'email': email,
            'name': user.get('name', ''),
            'role': user.get('role', '')
        }
    }), 200


@main.route("/")
@nocache
def home():
    """The main entry point, redirects authenticated users or sends them to login."""
    if current_user.is_authenticated:
        if current_user.role == 'exec':
            return redirect(url_for('main.dashboard'))
        else:
            return redirect(url_for('main.sales_dashboard_personal'))
    return redirect(url_for('auth.login'))

@main.route('/admin/users')
@login_required
@role_required('exec') # Only executives can access this
def manage_users():
    """Renders the user management page."""
    all_users = list(users_collection.find())
    return render_template('manage_users.html', users=all_users)


@main.route('/admin/add_user', methods=['POST'])
@login_required
@role_required('exec')
def add_user():
    """Handles the form submission for adding a new user."""
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    role = request.form.get('role')

    # Basic validation
    if not name or not email or not password or not role:
        flash("All fields are required.", "danger")
        return redirect(url_for('main.manage_users'))

    # Check if user already exists
    if users_collection.find_one({'email': email}):
        flash(f"A user with the email {email} already exists.", "warning")
        return redirect(url_for('main.manage_users'))

    # Hash the password for security
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

    users_collection.insert_one({
        'name': name,
        'email': email,
        'password': hashed_password,
        'role': role
    })

    flash(f"User '{name}' was created successfully.", "success")
    return redirect(url_for('main.manage_users'))

@main.route('/admin/delete_user/<user_id>', methods=['POST'])
@login_required
@role_required('exec')
def delete_user(user_id):
    """Handles the deletion of a user."""
    # Prevent an admin from deleting their own account
    if str(current_user.id) == user_id:
        flash("You cannot delete your own account.", "danger")
        return redirect(url_for('main.manage_users'))
    
    users_collection.delete_one({'_id': ObjectId(user_id)})
    flash("User has been deleted.", "success")
    return redirect(url_for('main.manage_users'))

# NEW UPDATED CODE - REPLACE THE OLD classify_and_draft_api FUNCTION WITH THIS
@main.route('/api/classify_and_draft', methods=['POST'])
def classify_and_draft_api():
    """
    API endpoint for Chrome Extension - accepts JWT token authentication
    """
    from backend.app import users_collection

    # Check for JWT token in Authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Missing or invalid authorization token'}), 401

    token = auth_header.split(' ')[1]

    try:
        # Verify JWT token
        secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        user_id = payload['user_id']

        # Get user from database
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 401
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Token expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401

    # Get email data
    data = request.json
    if not data or 'email_body' not in data:
        return jsonify({'error': 'Missing email_body in request'}), 400

    email_text = data.get('email_body')
    subject_text = data.get('subject', '')
    sender_email = data.get('from', '')
    cc_emails = data.get('cc', '')
    to_emails = data.get('to', '')

    # DEBUG: Log incoming data
    print("=" * 80)
    print("🔍 DEBUG - Incoming Email Data:")
    print(f"   FROM: '{sender_email}'")
    print(f"   TO: '{to_emails}'")
    print(f"   CC: '{cc_emails}'")
    print(f"   Subject: '{subject_text}'")
    print("=" * 80)

    # Classify the email
    category, metadata = classify_email(email_text, subject_text)

    # Generate reply draft
    details = extract_details(email_text)
    reply_drafts = {
        "quotation_request": generate_catalogue_reply(),
        "general_enquiry": generate_catalogue_reply(),
        "complaint": generate_complaint_reply(),
        "follow_up": generate_followup_reply(details, email_text),
        "other": generate_other_reply()
    }

    draft_reply = reply_drafts.get(category, generate_other_reply())

    # Parse and combine all recipients for CC
    reply_to = sender_email
    reply_cc = parse_and_combine_recipients(to_emails, cc_emails, exclude=sender_email)

    # DEBUG: Log processed data
    print("=" * 80)
    print("📤 DEBUG - Reply Recipients:")
    print(f"   Reply TO: '{reply_to}'")
    print(f"   Reply CC: '{reply_cc}'")
    print(f"   Reply CC Length: {len(reply_cc)}")
    print(f"   Reply CC Type: {type(reply_cc)}")
    print(f"   Reply CC is empty?: {reply_cc == ''}")
    print("=" * 80)

    response = {
        'category': category,
        'metadata': metadata,
        'draft_reply': draft_reply,
        'reply_to': reply_to,
        'reply_cc': reply_cc,
        'subject': f"Re: {subject_text}"
    }

    # DEBUG: Log final response
    print("=" * 80)
    print("📦 DEBUG - API Response:")
    print(f"   Category: {response['category']}")
    print(f"   Draft Reply Length: {len(response['draft_reply'])}")
    print(f"   Reply TO: '{response['reply_to']}'")
    print(f"   Reply CC: '{response['reply_cc']}'")
    print("=" * 80)

    return jsonify(response)



@main.route('/api/create_reply_draft', methods=['POST'])
def create_reply_draft_api():
    """
    Creates a reply draft with CC recipients via Gmail API.
    Called by Chrome extension after user approves the draft.
    
    Expected JSON payload:
    {
        "original_email_id": "message_id_from_gmail",
        "draft_reply": "the reply text (HTML)",
        "reply_cc": "email1@example.com, email2@example.com",
        "reply_to": "sender@example.com",
        "subject": "Re: Original Subject"
    }
    """
    from backend.app import users_collection
    from backend.app.gmail_service import create_draft_with_cc
    
    print("=" * 80)
    print("📨 POST /api/create_reply_draft called")
    print("=" * 80)
    
    # Step 1: Check for JWT token
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        print("❌ Missing authorization header")
        return jsonify({'error': 'Missing authorization token'}), 401
    
    token = auth_header.split(' ')[1]
    
    try:
        # Step 2: Verify JWT token
        secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        user_id = payload['user_id']
        print(f"✓ JWT token verified for user: {user_id}")
        
        # Step 3: Get user from database
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            print(f"❌ User not found in database: {user_id}")
            return jsonify({'error': 'User not found'}), 401
        
        print(f"✓ User found: {user.get('email')}")
    
    except jwt.ExpiredSignatureError:
        print("❌ JWT token expired")
        return jsonify({'error': 'Token expired'}), 401
    except jwt.InvalidTokenError as e:
        print(f"❌ Invalid JWT token: {str(e)}")
        return jsonify({'error': 'Invalid token'}), 401
    
    # Step 4: Get request data
    data = request.json
    if not data:
        print("❌ No JSON data in request")
        return jsonify({'error': 'No data provided'}), 400
    
    # Step 5: Extract and validate required fields
    original_email_id = data.get('original_email_id')
    reply_text = data.get('draft_reply')
    cc_recipients = data.get('reply_cc', '')
    to_email = data.get('reply_to')
    subject = data.get('subject', 'Re: Email')
    
    print(f"\n📋 Request Data:")
    print(f"   Original Email ID: {original_email_id}")
    print(f"   To: {to_email}")
    print(f"   CC: {cc_recipients}")
    print(f"   Subject: {subject}")
    print(f"   Reply Text Length: {len(reply_text) if reply_text else 0} chars")
    
    if not all([original_email_id, reply_text, to_email]):
        print("❌ Missing required fields")
        return jsonify({'error': 'Missing required fields: original_email_id, draft_reply, reply_to'}), 400
    
    # Step 6: Get user's Gmail credentials
    gmail_creds = user.get('google_credentials_encrypted')
    if not gmail_creds:
        print(f"❌ Gmail not connected for user {user.get('email')}")
        return jsonify({'error': 'Gmail not connected for this user. Please connect Gmail first.'}), 400
    
    print(f"✓ Gmail credentials found for user")
    
    # Step 7: Prepare credentials dictionary (same structure as in email_tasks.py)
    user_credentials = {
        'token': gmail_creds.get('token'),
        'refresh_token': gmail_creds.get('refresh_token'),
        'token_uri': gmail_creds.get('token_uri'),
        'client_id': gmail_creds.get('client_id'),
        'client_secret': gmail_creds.get('client_secret'),
        'scopes': gmail_creds.get('scopes')
    }
    
    print(f"✓ Credentials dictionary prepared")
    
    # Step 8: Call the Gmail API function to create draft
    print(f"\n📤 Calling create_draft_with_cc()...")
    result = create_draft_with_cc(
        user_credentials=user_credentials,
        original_email_id=original_email_id,
        reply_text=reply_text,
        cc_recipients=cc_recipients,
        to_email=to_email,
        subject=subject
    )
    
    # Step 9: Return result
    print(f"\n📦 Result from create_draft_with_cc:")
    print(f"   Success: {result.get('success')}")
    print(f"   Message: {result.get('message') or result.get('error')}")
    
    if result['success']:
        print(f"\n✅ Draft creation successful!")
        print("=" * 80)
        return jsonify({
            'success': True,
            'message': 'Draft created successfully via Gmail API',
            'draft_id': result.get('draft_id')
        }), 200
    else:
        print(f"\n❌ Draft creation failed!")
        print("=" * 80)
        return jsonify({
            'success': False,
            'error': result.get('error', 'Unknown error occurred')
        }), 500


def parse_and_combine_recipients(to_emails, cc_emails, exclude=None):
    """
    Parse and combine TO and CC recipients, excluding specified email
    """
    import re
    
    print(f"🔧 parse_and_combine_recipients called:")
    print(f"   to_emails: '{to_emails}'")
    print(f"   cc_emails: '{cc_emails}'")
    print(f"   exclude: '{exclude}'")
    
    # Email regex pattern
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    all_recipients = []
    
    # Parse TO emails
    if to_emails:
        to_list = re.findall(email_pattern, to_emails)
        print(f"   Parsed TO emails: {to_list}")
        all_recipients.extend(to_list)
    
    # Parse CC emails
    if cc_emails:
        cc_list = re.findall(email_pattern, cc_emails)
        print(f"   Parsed CC emails: {cc_list}")
        all_recipients.extend(cc_list)
    
    print(f"   All recipients before dedup: {all_recipients}")
    
    # Remove duplicates
    all_recipients = list(set(all_recipients))
    print(f"   All recipients after dedup: {all_recipients}")
    
    # Exclude the specified email (usually sender)
    if exclude:
        exclude_clean = re.findall(email_pattern, exclude)
        print(f"   Parsed exclude email: {exclude_clean}")
        if exclude_clean:
            all_recipients = [email for email in all_recipients if email not in exclude_clean]
    
    print(f"   Final recipients: {all_recipients}")
    
    # Return as comma-separated string
    result = ', '.join(all_recipients)
    print(f"   Returning: '{result}'")
    
    return result

@main.route('/my_reports')
@nocache 
@login_required
def my_reports():
    """Renders the personal reports page for a salesperson."""
    if current_user.role != 'sales':
        abort(403)
        
    context = get_my_reports_data()
    return render_template('my_reports.html', **context)


# FILE: backend/app/routes.py

# Replace your old dashboard function with this one
@main.route("/dashboard", methods=['GET', 'POST'])
@login_required
@nocache
@role_required("exec")
def dashboard():
    if request.method == 'POST':
        # For form submissions, get data from request.form
        category = request.form.get("category", "all")
        sla = request.form.get("sla", "all")
        product = request.form.get("product", "all")
        date_from = request.form.get("date_from", "")
        date_to = request.form.get("date_to", "")
    else:
        # For direct visits or links, get data from request.args (URL params)
        category = request.args.get("category", "all")
        sla = request.args.get("sla", "all")
        product = request.args.get("product", "all")
        date_from = request.args.get("date_from", "")
        date_to = request.args.get("date_to", "")

    # Call the data function with the extracted parameters
    context = get_dashboard_data(
        selected_category=category,
        sla_filter=sla,
        product_filter=product,
        date_from_str=date_from,
        date_to_str=date_to
    )

    # Note: 'all_products' is now returned by get_dashboard_data, so this line is no longer needed here.
    # context['all_products'] = get_product_list_for_dropdown()
    
    return render_template("dashboard.html", **context)

@main.route("/statistics")
@login_required
@nocache
@role_required("exec")
def statistics():
    """Renders the statistics page."""
    # For now, we'll pass a variable to tell the sidebar which page is active.
    # Later, you will add your data-fetching logic here.
    return render_template("statistics.html", active_page="statistics")

@main.route("/reports")
@login_required
@nocache
@role_required("exec")
def reports():
    """Renders the main reports page."""
    # We'll create the reports.html template in the next step.
    return render_template("reports.html", active_page="reports")

@main.route("/export/csv")
def export_csv():
    logs = list(collection.find())
    df = pd.DataFrame(logs)
    csv_path = "export/email_data.csv"
    df.to_csv(csv_path, index=False)
    return send_file(csv_path, as_attachment=True)

@main.route("/export/excel")
def export_excel():
    logs = list(collection.find())
    df = pd.DataFrame(logs)
    excel_path = "export/email_data.xlsx"
    df.to_excel(excel_path, index=False)
    return send_file(excel_path, as_attachment=True)

@main.route("/export/dashboard-pdf")
@login_required
@role_required("exec")
def export_dashboard_pdf():
    """Exports the main dashboard view as a PDF using WeasyPrint."""
    try:
        # Fetch the same data that your dashboard uses
        context = get_dashboard_data()

        # Use our new utility function to generate the PDF in memory
        # We are using 'print_dashboard.html' as the template
        pdf_buffer, filename = generate_pdf_from_template(
            template_name="print_dashboard.html",
            data=context
        )

        # Send the generated PDF to the user for download
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"Dashboard_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
        )

    except Exception as e:
        traceback.print_exc()
        flash(f"PDF generation failed: {str(e)}", "danger")
        return redirect(url_for('main.dashboard'))
    
@main.route("/print")
def print_preview():
    context = get_dashboard_data() # THE FIX
    return render_template("print_dashboard.html", **context)

@main.route("/pdfs/<path:filename>")
def serve_pdf(filename):
    return send_from_directory("pdfs", filename)

@main.route("/history")
def history():
    logs = list(collection.find().sort("timestamp", -1).limit(50))
    html = "<h2>📜 Recent Email History (Latest 50)</h2><ul>"
    for log in logs:
        html += f"<li><strong>{log.get('category', 'N/A').title()}</strong> — {log.get('timestamp', 'N/A')}</li>"
    html += "</ul><p><a href='/'>Back to Home</a></p>"
    return html

@main.route("/notifications")
def get_notifications():
    try:
        # --- 1. EFFICIENT QUERY: Use a single query with $or ---
        # This is faster as it's only one trip to the database.
        seven_days_ago = datetime.now() - timedelta(days=7)
        
        query = {
            "timestamp": {"$gte": seven_days_ago},
            "$or": [
                { "SLA_Met": False },
                { "category": "complaint", "urgency_label": "High" }
            ]
        }

        # Fetch, sort, and limit in one go.
        alerts = list(collection.find(query).sort("timestamp", -1).limit(10))

        # --- 2. SMARTER LOGIC & STRUCTURED RESPONSE ---
        notifications = []
        for alert in alerts:
            tags = []
            # Check for each condition independently
            if not alert.get('SLA_Met', True):
                tags.append("SLA Breach")
            
            if alert.get('category') == 'complaint' and alert.get('urgency_label') == 'High':
                # Avoid adding this tag if it's already an SLA breach of a complaint, to prevent redundancy
                if "SLA Breach" not in tags or alert.get('category') != 'complaint':
                     tags.append("High Urgency")

            # Format the message components
            category_title = alert.get('category', 'N/A').replace('_', ' ').title()
            timestamp_str = alert['timestamp'].strftime('%b %d, %H:%M')
            
            # Build the message with prefixes for clarity
            prefix = "".join([f"[{tag}] " for tag in tags])
            message = f"{prefix}'{category_title}' at {timestamp_str}"
            
            # Create a structured dictionary for each notification
            notifications.append({
                "id": str(alert['_id']),
                "message": message,
                "tags": tags # Pass tags for potential future UI styling
            })

        return jsonify(notifications)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed to load notifications"}), 500


@main.route("/upload_leads", methods=["GET", "POST"])
@role_required("exec")
def upload_leads():
    # This part handles the form submission
    if request.method == "POST":
        # Check if a file was actually included in the request
        if 'file' not in request.files or not request.files['file'].filename:
            flash("No file selected. Please choose an Excel file to upload.", "warning")
            return redirect(request.url) # Redirect back to the upload page

        file = request.files["file"]

        try:
            df = pd.read_excel(file)
            # Standardize column names to lowercase for consistency
            df.columns = [col.lower().strip() for col in df.columns]
            
            records = df.to_dict("records")

            # Handle case where Excel file is empty
            if not records:
                flash("The uploaded file appears to be empty.", "info")
                return redirect(url_for('main.manage_leads'))

            for r in records:
                r["status"] = "new"  # Set a default status for new leads
                r["assigned_to"] = None
            
            # Insert the records into the database
            result = leads_collection.insert_many(records)
            
            # Use flash to create a success message for the user
            flash(f"Success! {len(result.inserted_ids)} new leads have been uploaded.", "success")
        
        except Exception as e:
            # If anything goes wrong (e.g., wrong file format), show an error
            flash(f"An error occurred during upload: {e}", "danger")
            # Redirect back to the upload page so they can try again
            return redirect(request.url)

        # ======================= THE MAIN CHANGE =======================
        # After successfully processing, redirect to the new leads list page
        return redirect(url_for('main.manage_leads'))
        # ================================================================

    # This part handles the initial visit to the page (a GET request)
    # It just shows the upload form.
    return render_template("upload_leads.html")


@main.route('/campaigns')
@nocache
@role_required('exec')
def manage_campaigns():
    """Redirects to the sales dashboard and opens the 'campaigns' tab."""
    return redirect(url_for('main.sales_dashboard') + '#campaigns-pane')


@main.route('/create_campaign', methods=['GET', 'POST'])
@role_required('exec')
def create_campaign():
    if request.method == 'POST':
        campaign_name = request.form.get('campaign_name')
        subject = request.form.get('subject')
        body = request.form.get('body')
        
        campaigns_collection.insert_one({
            "name": campaign_name,
            "subject": subject,
            "body": body,
            "created_at": datetime.now(),
            "status": "active",
            "sent_count": 0,
            "reply_count": 0
        })
        flash('Campaign created successfully!', 'success')
        return redirect(url_for('main.manage_campaigns'))
        
    return render_template('create_campaign.html')



# REPLACE the old /start_campaign route with this one
@main.route('/start_campaign', methods=['GET', 'POST'])
@login_required
def start_campaign():
    # This route is for SALESPERSONS to start sending their assigned leads
    if current_user.role != 'sales':
        abort(403)

    if request.method == 'POST':
        from backend.tasks.email_tasks import send_campaign_email
        campaign_id = request.form.get('campaign_id')
        number_to_send = int(request.form.get('number_to_send', 0))

        if not campaign_id or number_to_send <= 0:
            flash("Please select a campaign and enter a valid number of leads.", "warning")
            return redirect(url_for('main.start_campaign'))

        # Find leads that are assigned to the current user and have a 'new' status
        leads_to_contact = list(leads_collection.find({
            'assigned_to': current_user.name,
            'status': 'new'
        }).limit(number_to_send))

        if not leads_to_contact:
            flash("No new leads available to contact for this campaign.", "info")
            return redirect(url_for('main.my_leads'))

        # Get the user's signature from the database
        user_doc = users_collection.find_one({'_id': ObjectId(current_user.id)})
        user_signature = user_doc.get('signature_html', f"<br><br><p><strong>{current_user.name}</strong></p>")
        
        # --- THIS IS THE CORRECTED LOGIC ---
        # Loop through each lead and dispatch a separate Celery task
        for lead in leads_to_contact:
            lead_id = str(lead['_id'])
            # Call the imported task directly with .delay()
            send_campaign_email.delay(lead_id, campaign_id, current_user.id, user_signature)
        # --- END OF CORRECTION ---
        
        flash(f"Success! Your campaign has been queued for {len(leads_to_contact)} leads.", "success")
        return redirect(url_for('main.my_leads'))

    # This part for the GET request remains unchanged.
    active_campaigns = list(campaigns_collection.find({'status': 'active'}))
    available_leads_count = leads_collection.count_documents({'assigned_to': current_user.name, 'status': 'new'})
    
    return render_template('start_campaign.html', campaigns=active_campaigns, lead_count=available_leads_count)

@main.route('/campaigns/edit/<campaign_id>', methods=['GET', 'POST'])
@login_required
@role_required('exec')
def edit_campaign(campaign_id):
    """ Edits an existing campaign template. """
    campaign = campaigns_collection.find_one({'_id': ObjectId(campaign_id)})
    if not campaign:
        abort(404)

    if request.method == 'POST':
        campaigns_collection.update_one(
            {'_id': ObjectId(campaign_id)},
            {'$set': {
                'name': request.form.get('campaign_name'),
                'subject': request.form.get('subject'),
                'body': request.form.get('body'),
                'status': request.form.get('status')
            }}
        )
        flash('Campaign updated successfully!', 'success')
        return redirect(url_for('main.manage_campaigns'))

    return render_template('edit_campaign.html', campaign=campaign)

@main.route('/campaigns/delete/<campaign_id>', methods=['POST'])
@login_required
@role_required('exec')
def delete_campaign(campaign_id):
    """ Deletes a campaign template. """
    result = campaigns_collection.delete_one({'_id': ObjectId(campaign_id)})
    if result.deleted_count > 0:
        flash('Campaign has been deleted.', 'success')
    else:
        flash('Campaign not found.', 'warning')
    return redirect(url_for('main.manage_campaigns'))

@main.route('/campaigns/assign/<campaign_id>', methods=['GET', 'POST'])
@login_required
@role_required('exec')
def assign_campaign(campaign_id):
    """ Page for an executive to assign a campaign to a salesperson. """
    campaign = campaigns_collection.find_one({'_id': ObjectId(campaign_id)})
    if not campaign:
        abort(404)

    sales_people = list(users_collection.find({'role': 'sales'}))

    if request.method == 'POST':
        sales_person_id = request.form.get('sales_person_id')
        sales_person = users_collection.find_one({'_id': ObjectId(sales_person_id)})

        if not sales_person:
            flash("Invalid salesperson selected.", "danger")
            return redirect(url_for('main.assign_campaign', campaign_id=campaign_id))

        campaigns_collection.update_one(
            {'_id': ObjectId(campaign_id)},
            {'$set': {'assigned_to': sales_person['name']}}
        )

        # ===============================================================
        # ======================= THE CORRECTED CALL ====================
        # ===============================================================
        # Define the arguments for the task in a list
        task_args = [
            sales_person['email'],
            sales_person['name'],
            campaign['name'],
            current_user.name
        ]

        # Call the task by its full string name using the celery_client
        celery_client.send_task('backend.tasks.email_tasks.send_assignment_notification_email', args=task_args)
        # ===============================================================
        
        flash(f"Campaign '{campaign['name']}' has been assigned to {sales_person['name']}.", "success")
        return redirect(url_for('main.manage_campaigns'))

    return render_template('assign_campaign.html', campaign=campaign, sales_people=sales_people)


# --- THIS IS THE CORRECTED ROUTE AND FUNCTION ---
@main.route('/campaigns/unassign/<campaign_id>', methods=['POST'])
@login_required
@role_required('exec')
def unassign_campaign(campaign_id): # <-- Now it accepts campaign_id
    """ Un-assigns a campaign from a salesperson. """
    try:
        # The campaign_id now comes directly from the URL, which is cleaner and safer.
        result = campaigns_collection.update_one(
            {'_id': ObjectId(campaign_id)},
            {'$set': {'assigned_to': None}} 
        )
        
        if result.modified_count > 0:
            flash('Campaign has been successfully unassigned.', 'success')
        else:
            # This can happen if the campaign was already unassigned.
            flash('Campaign was already unassigned or not found.', 'info')

    except Exception as e:
        flash(f"Error un-assigning campaign: {e}", "danger")
        
    return redirect(url_for('main.manage_campaigns'))

@main.route('/task_status/<task_id>')
def task_status(task_id):
    """AJAX endpoint to check the status of a background task."""
    
    # --- THIS IS THE FINAL FIX ---
    # Use the celery_client instance defined at the top of this file
    task = celery_client.AsyncResult(task_id)
    
    response = {'state': task.state, 'status': 'Pending...'}
    if task.state != 'PENDING' and task.state != 'FAILURE':
        response['status'] = task.info.get('status', '') if isinstance(task.info, dict) else ''
        response['result'] = task.info
    elif task.state == 'FAILURE':
        response['status'] = str(task.info) # The error message
    
    return jsonify(response)

@main.route('/replies')
@nocache
@role_required('exec') # It's good practice to add role protection here too
def view_replies():
    """Redirects to the sales dashboard and opens the 'replies' tab."""
    return redirect(url_for('main.sales_dashboard') + '#replies-pane')


@main.route("/sales") # <--- THE FIX: Changed from "/sales_dashboard" to "/sales"
@login_required
@nocache 
@role_required("exec")
def sales_dashboard():
    try:
        # This is now the ONE function for the executive sales view.
        # It handles the tabbed page with Overview, Leads, Campaigns, etc.
        
        # --- You will need to consolidate ALL data loading here ---
        # Data for Overview tab
        total_campaigns = campaigns_collection.count_documents({})
        total_leads = leads_collection.count_documents({})
        total_sent = sent_emails_collection.count_documents({})
        total_replies = replies_collection.count_documents({})
        response_rate = (total_replies / total_sent * 100) if total_sent > 0 else 0
        reply_category_pipeline = [{"$group": {"_id": "$category", "count": {"$sum": 1}}}]
        reply_stats = list(replies_collection.aggregate(reply_category_pipeline))
        hot_leads = list(replies_collection.find({"category": "interested"}).sort("timestamp", -1).limit(5))
        top_campaigns = list(campaigns_collection.find().sort("sent_count", -1).limit(5))
        reply_category_chart = create_reply_category_chart(reply_stats) if reply_stats else ""
        
        # Data for Leads and Campaigns tabs
        all_leads_list = list(leads_collection.find().sort("created_at", -1))
        all_campaigns_list = list(campaigns_collection.find().sort("created_at", -1))
        sales_people_list = list(users_collection.find({'role': 'sales'}))
        
        # Data for Replies tab
        all_replies_list = list(replies_collection.find({}).sort('timestamp', -1).limit(100))

        context = {
            # Overview data
            "total_campaigns": total_campaigns, "total_leads": total_leads, "total_sent": total_sent,
            "total_replies": total_replies, "response_rate": round(response_rate, 2),
            "reply_stats": reply_stats, "top_campaigns": top_campaigns,
            "hot_leads": hot_leads, "reply_category_chart": reply_category_chart,
            
            # Data for other tabs
            "leads": all_leads_list,
            "campaigns": all_campaigns_list,
            "sales_people": sales_people_list,
            "replies": all_replies_list
        }
        
        return render_template("sales_dashboard.html", **context)

    except Exception as e:
        traceback.print_exc()
        flash(f"Could not load sales dashboard. Error: {e}", "danger")
        return redirect(url_for('main.dashboard'))


# ======================= ADD THIS NEW ROUTE =======================
@main.route('/sales_dashboard_personal')
@nocache
@login_required
def sales_dashboard_personal():
    if current_user.role != 'sales':
        flash("Access denied. This page is for sales employees.", "danger")
        return redirect(url_for('main.dashboard'))
    
    # Refresh user data
    from backend.app import users_collection
    user_data = users_collection.find_one({'_id': ObjectId(current_user.id)})
    gmail_connected = user_data.get('gmail_connected', False) if user_data else False
    
    # Sales metrics
    sales_person_name = current_user.get('name')
    my_leads_count = leads_collection.count_documents({'assigned_to': sales_person_name})
    my_replies_count = replies_collection.count_documents({'assigned_to_sales_person': sales_person_name})
    
    my_hot_leads = list(leads_collection.find({
        'assigned_to': sales_person_name,
        'status': 'interested'
    }))
    
    context = {
        "my_leads_count": my_leads_count,
        "my_replies_count": my_replies_count,
        "my_hot_leads": my_hot_leads,
        "gmail_connected": gmail_connected,
    }
    
    return render_template('sales_person_dashboard.html', **context)

# ===================================================================

@main.route('/leads')
@nocache
@role_required('exec')
def manage_leads():
    """
    This route no longer renders a page. It redirects to the main
    sales dashboard and tells it to open the 'leads' tab.
    """
    # The '#' tells the browser which tab to open on the destination page.
    return redirect(url_for('main.sales_dashboard') + '#leads-pane')


@main.route('/add_lead', methods=['GET', 'POST'])
@login_required
def add_lead():
    """Handles the form to add a new lead manually."""
    if current_user.role != 'sales':
        abort(403)

    if request.method == 'POST':
        try:
            lead_name = request.form.get('name')
            lead_email = request.form.get('email')

            if not lead_name or not lead_email:
                flash("Lead Name and Email are required fields.", "danger")
                return redirect(url_for('main.add_lead'))

            if leads_collection.find_one({'email': lead_email}):
                flash(f"A lead with the email '{lead_email}' already exists.", "warning")
                return redirect(url_for('main.add_lead'))

            # The current user's name is automatically used for assignment
            assigned_to_name = current_user.name 

            new_lead = {
                "name": lead_name,
                "email": lead_email,
                "company": request.form.get('company'),
                "phone": request.form.get('phone'),
                "status": "new",
                "assigned_to": assigned_to_name, # Auto-assigned
                "created_at": datetime.now(),
                "last_contacted_at": None # Initialize as null
            }
            
            leads_collection.insert_one(new_lead)
            
            flash(f"Lead '{lead_name}' was successfully added and assigned to you.", "success")
            return redirect(url_for('main.my_leads'))

        except Exception as e:
            traceback.print_exc()
            flash(f"An error occurred: {e}", "danger")
            return redirect(url_for('main.add_lead'))

    # For a GET request, just render the form page
    return render_template('add_lead.html')

@main.route('/assign_lead', methods=['POST'])
@login_required
@role_required('exec')
def assign_lead():
    """Assigns a lead to a salesperson."""
    try:
        lead_id = request.form.get('lead_id')
        sales_person_name = request.form.get('sales_person_name')

        if not lead_id or not sales_person_name:
            flash("Missing lead ID or salesperson.", "danger")
            return redirect(url_for('main.manage_leads'))

        # Update the 'assigned_to' field for the specific lead in the database
        leads_collection.update_one(
            {'_id': ObjectId(lead_id)},
            {'$set': {'assigned_to': sales_person_name, 'status': 'new'}} # Also reset status to 'new'
        )

        flash(f"Lead successfully assigned to {sales_person_name}.", "success")
    except Exception as e:
        flash(f"Error assigning lead: {e}", "danger")
        
    return redirect(url_for('main.manage_leads'))

@main.route('/my_leads')
@nocache 
@login_required
def my_leads():
    """Shows a list of leads assigned specifically to the logged-in salesperson."""
    if current_user.role != 'sales':
        abort(403)
        
    # THE FIX: Ensure we query using the same robust .name property
    sales_person_name = current_user.name
    
    # Query the database for leads assigned to the current user's name
    my_assigned_leads = list(leads_collection.find({'assigned_to': sales_person_name}).sort("created_at", -1))
    
    return render_template('my_leads.html', leads=my_assigned_leads)


@main.route('/exec/edit_lead_status', methods=['POST'])
@login_required
@role_required('exec')
def exec_edit_lead_status():
    """ Allows an executive to update the status of any lead. """
    try:
        lead_id = request.form.get('lead_id')
        new_status = request.form.get('new_status')

        if not lead_id or not new_status:
            flash("Missing information to update lead.", "danger")
            return redirect(url_for('main.manage_leads'))

        # Executive can update any lead, so no 'assigned_to' check is needed here.
        result = leads_collection.update_one(
            {'_id': ObjectId(lead_id)},
            {'$set': {'status': new_status}}
        )

        if result.modified_count > 0:
            flash(f"Lead status updated to '{new_status}'.", "success")
        else:
            flash("Could not update lead status.", "warning")

    except Exception as e:
        flash(f"An error occurred: {e}", "danger")

    return redirect(url_for('main.manage_leads'))


@main.route('/exec/delete_lead/<lead_id>', methods=['POST'])
@login_required
@role_required('exec')
def exec_delete_lead(lead_id):
    """ Allows an executive to delete any lead. """
    try:
        # Executive can delete any lead.
        result = leads_collection.delete_one(
            {'_id': ObjectId(lead_id)}
        )

        if result.deleted_count > 0:
            flash("Lead has been deleted successfully.", "success")
        else:
            flash("Could not delete lead. It may have already been deleted.", "warning")

    except Exception as e:
        flash(f"An error occurred while deleting the lead: {e}", "danger")

    return redirect(url_for('main.manage_leads'))


@main.route('/edit_lead_status', methods=['POST'])
@login_required
def edit_lead_status():
    """ Updates the status of a specific lead and manages the 'last_contacted_at' timestamp. """
    if current_user.role != 'sales':
        abort(403)
    
    try:
        lead_id = request.form.get('lead_id')
        new_status = request.form.get('new_status')
        sales_person_name = current_user.name

        if not lead_id or not new_status:
            flash("Missing information to update lead.", "danger")
            return redirect(url_for('main.my_leads'))

        # --- THE FIX: Create a dynamic update query ---
        update_query = {'$set': {'status': new_status}}
        
        # If the lead is being marked as contacted, add the timestamp.
        if new_status == 'contacted':
            update_query['$set']['last_contacted_at'] = datetime.now()
        
        # If the lead is being reset to 'new', remove the timestamp for clarity.
        elif new_status == 'new':
            update_query['$unset'] = {'last_contacted_at': ""}

        # Security Check: Ensure the lead being updated belongs to the logged-in user
        result = leads_collection.update_one(
            {'_id': ObjectId(lead_id), 'assigned_to': sales_person_name},
            update_query
        )

        if result.modified_count > 0:
            flash(f"Lead status updated to '{new_status}'.", "success")
        else:
            flash("Could not update lead. It might not be assigned to you.", "warning")

    except Exception as e:
        flash(f"An error occurred: {e}", "danger")

    return redirect(url_for('main.my_leads'))

@main.route('/delete_lead/<lead_id>', methods=['POST'])
@login_required
def delete_lead(lead_id):
    """ Deletes a lead from the database. """
    if current_user.role != 'sales':
        abort(403)

    try:
        sales_person_name = current_user.name

        # Security Check: Ensure the lead being deleted belongs to the logged-in user
        result = leads_collection.delete_one(
            {'_id': ObjectId(lead_id), 'assigned_to': sales_person_name}
        )

        if result.deleted_count > 0:
            flash("Lead has been deleted successfully.", "success")
        else:
            flash("Could not delete lead. It might not be assigned to you or already deleted.", "warning")

    except Exception as e:
        flash(f"An error occurred while deleting the lead: {e}", "danger")

    return redirect(url_for('main.my_leads'))


@main.route('/export/my_leads/<export_type>')
@login_required
def export_my_leads(export_type):
    """
    Exports the leads assigned to the currently logged-in salesperson
    in the specified format (CSV, Excel, or PDF).
    """
    if current_user.role != 'sales':
        abort(403)

    try:
        sales_person_name = current_user.name
        
        # 1. Fetch this user's leads from the database
        my_leads = list(leads_collection.find({'assigned_to': sales_person_name}))
        
        if not my_leads:
            flash("You have no leads to export.", "info")
            return redirect(url_for('main.sales_dashboard_personal'))

        # 2. Convert to a pandas DataFrame for easy exporting
        df = pd.DataFrame(my_leads)
        
        # Clean up the DataFrame for a professional-looking export
        if '_id' in df.columns:
            df = df.drop(columns=['_id'])
        
        export_columns = ['name', 'email', 'company', 'status', 'assigned_to']
        df = df[[col for col in export_columns if col in df.columns]]

        # 3. Generate and send the file based on the requested type
        timestamp = datetime.now().strftime("%Y%m%d")
        filename_base = f"my_leads_{sales_person_name.replace(' ', '_')}_{timestamp}"

        if export_type == 'csv':
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            return Response(
                csv_buffer.getvalue(),
                mimetype="text/csv",
                headers={"Content-disposition": f"attachment; filename={filename_base}.csv"}
            )
            
        elif export_type == 'excel':
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='My Leads')
            return Response(
                excel_buffer.getvalue(),
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-disposition": f"attachment; filename={filename_base}.xlsx"}
            )

        # --- THIS BLOCK IS UPDATED FOR WEASYPRINT ---
        elif export_type == 'pdf':
            # Prepare the data dictionary to pass to the template
            context = {
                'leads': df.to_dict('records'),
                'user_name': sales_person_name,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Use our new WeasyPrint utility function to generate the PDF in memory
            # This assumes you have a template named 'export_leads_pdf.html'
            pdf_buffer, _ = generate_pdf_from_template(
                template_name='export_leads_pdf.html',
                data=context
            )
            
            # Create the Flask response object from the in-memory PDF buffer
            response = make_response(pdf_buffer.getvalue())
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = f'attachment; filename={filename_base}.pdf'
            return response
        # --- END OF UPDATED BLOCK ---

        else:
            flash("Invalid export format requested.", "danger")
            return redirect(url_for('main.sales_dashboard_personal'))

    except Exception as e:
        traceback.print_exc()
        flash(f"An error occurred during export: {str(e)}", "danger")
        return redirect(url_for('main.sales_dashboard_personal'))

@main.route('/mass_assign_leads', methods=['POST'])
@login_required
@role_required('exec')
def mass_assign_leads():
    """Assigns multiple selected leads to a single salesperson."""
    try:
        # .getlist() is crucial for fetching all values from checkboxes with the same name
        lead_ids = request.form.getlist('lead_ids')
        sales_person_name = request.form.get('sales_person_name')

        # --- Validation ---
        if not lead_ids:
            flash("You must select at least one lead to assign.", "warning")
            return redirect(url_for('main.manage_leads'))
        
        if not sales_person_name:
            flash("You must select a salesperson to assign the leads to.", "warning")
            return redirect(url_for('main.manage_leads'))

        # Convert the list of string IDs from the form to a list of MongoDB ObjectIds
        object_ids = [ObjectId(lead_id) for lead_id in lead_ids]

        # --- Database Operation ---
        # Use update_many with the $in operator to update all selected documents at once
        result = leads_collection.update_many(
            {'_id': {'$in': object_ids}},
            {'$set': {
                'assigned_to': sales_person_name,
                'status': 'new'  # It's good practice to reset the status to 'new' on assignment
            }}
        )

        flash(f"Successfully assigned {result.modified_count} leads to {sales_person_name}.", "success")

    except Exception as e:
        flash(f"An error occurred during mass assignment: {e}", "danger")
        traceback.print_exc()

    return redirect(url_for('main.manage_leads'))


@main.route('/mass_delete_leads', methods=['POST'])
@login_required
@role_required('exec')
def mass_delete_leads():
    """Deletes multiple selected leads from the database."""
    try:
        # .getlist() is crucial for fetching all values from checkboxes with the same name
        lead_ids = request.form.getlist('lead_ids')

        # --- Validation ---
        if not lead_ids:
            flash("You must select at least one lead to delete.", "warning")
            return redirect(url_for('main.manage_leads'))

        # Convert the list of string IDs from the form to a list of MongoDB ObjectIds
        object_ids = [ObjectId(lead_id) for lead_id in lead_ids]

        # --- Database Operation ---
        # Use delete_many with the $in operator to remove all selected documents at once
        result = leads_collection.delete_many(
            {'_id': {'$in': object_ids}}
        )

        flash(f"Successfully deleted {result.deleted_count} leads.", "success")

    except Exception as e:
        flash(f"An error occurred during mass deletion: {e}", "danger")
        traceback.print_exc()

    return redirect(url_for('main.manage_leads'))

# REPLACE your old my_replies function with this one
@main.route('/my_replies')
@nocache
@login_required
def my_replies():
    """
    (NEW VERSION) Shows a list of replies received by querying the new SQL database.
    """
    if current_user.role != 'sales':
        abort(403)

    sales_person_name = current_user.name
    print(f"--- LOADING REPLIES FOR: {sales_person_name} ---")

    # --- THIS IS THE NEW CORE LOGIC ---
    # 1. Query the new SQL 'campaign_replies' table.
    #    For now, we get all replies. Filtering by user is a future step.
    all_replies = list(replies_collection.find({}).sort('timestamp', -1))
    
    # 2. Format the data for the template.
    #    We create a list of dictionaries that looks similar to your old 'enriched_replies'.
    #    Your template likely uses keys like 'lead_name', 'reply_body', etc.
    formatted_replies = []
    for reply in all_replies:
    	formatted_replies.append({
            'lead_name': reply.get('sender_email', ''),
            'lead_email': reply.get('sender_email', ''),
            'subject': reply.get('subject', ''),
            'body': reply.get('body', ''),
            'received_at': reply.get('timestamp', '')
    	})

    # --- END OF NEW LOGIC ---

    print(f"Found {len(formatted_replies)} replies in the SQL database.")

    # Pass the newly formatted list to the template.
    return render_template('my_replies.html', replies=formatted_replies)

@main.route('/api/pdf-status')
@login_required
def pdf_generation_status():
    """An endpoint to quickly check if the WeasyPrint installation is working."""
    try:
        # Step 1: Check WeasyPrint import
        print("=== PDF DEBUG START ===")
        
        try:
            from weasyprint import HTML, __version__
            print(f"WeasyPrint version: {__version__}")
            print(f"HTML class: {HTML}")
        except ImportError as e:
            raise Exception(f"WeasyPrint import failed: {str(e)}")
        
        # Step 2: Check HTML class signature
        import inspect
        signature = inspect.signature(HTML.__init__)
        print(f"HTML.__init__ signature: {signature}")
        
        # Step 3: Try the most basic HTML creation
        print("Attempting basic HTML creation...")
        test_html = "<html><body><h1>Test</h1></body></html>"
        
        # Try different ways to create HTML object
        try:
            # Method 1: keyword argument
            html_doc = HTML(string=test_html)
            print("Method 1 (keyword) successful")
        except Exception as e1:
            print(f"Method 1 failed: {e1}")
            try:
                # Method 2: positional argument (older versions)
                html_doc = HTML(test_html)
                print("Method 2 (positional) successful")
            except Exception as e2:
                print(f"Method 2 failed: {e2}")
                raise Exception(f"Both HTML creation methods failed: {e1}, {e2}")
        
        # Step 4: Try PDF generation
        pdf_buffer = io.BytesIO()
        html_doc.write_pdf(pdf_buffer)
        
        # Check if the generated PDF has content
        if pdf_buffer.getbuffer().nbytes > 0:
            print("=== PDF DEBUG SUCCESS ===")
            return jsonify({
                'status': 'success',
                'message': 'PDF generation service is working correctly.',
                'weasyprint_available': True
            })
        else:
            raise Exception("Generated PDF buffer is empty.")

    except Exception as e:
        print(f"=== PDF DEBUG ERROR: {str(e)} ===")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error', 
            'message': f'PDF generation failed: {str(e)}',
            'weasyprint_available': False
        }), 500
    

@main.route('/export/executive-report-pdf')
@login_required
@role_required("exec")
def export_executive_report_pdf():
    """Generates and downloads a summarized PDF report for executives."""
    try:
        # 1. Gather report data
        context = prepare_executive_report_data()

        # 2. Generate PDF
        pdf_buffer, _ = generate_pdf_from_template(
            template_name="executive_report.html",
            data=context
        )

        # Ensure buffer is file-like
        if isinstance(pdf_buffer, bytes):
            pdf_buffer = io.BytesIO(pdf_buffer)
        pdf_buffer.seek(0)

        # 3. Return as downloadable file
        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"Executive_Summary_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
        )

    except Exception as e:
        traceback.print_exc()
        flash("Executive report generation failed. Please try again later.", "danger")
        return redirect(url_for("main.dashboard"))
