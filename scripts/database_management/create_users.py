# START OF FILE: create_users.py (CORRECTED)

from pymongo import MongoClient
from werkzeug.security import generate_password_hash
from . import db, collection, leads_collection, campaigns_collection, sent_emails_collection, replies_collection, users_collection, email_replies_collection, paired_emails_collection, legacy_leads_collection, legacy_campaigns_collection

# --- IMPORTANT: CONFIGURE YOUR USERS HERE ---
EXECUTIVE_EMAIL = "ee1.mybharati@gmail.com"
EXECUTIVE_NAME = "Executive User"

SALES_EMAIL = "bd3.bharatifire@gmail.com"
SALES_NAME = "Sales Employee 1"

EXECUTIVE_PASSWORD = "Exec@ee1"
SALES_PASSWORD = "Sales@bd3"
# ----------------------------------------------





def setup_users():
    """
    Creates an executive and a sales user in the database if they don't already exist.
    """
    print("--- Setting up user accounts ---")
    
    # --- Create Executive User ---
    if users_collection.find_one({'email': EXECUTIVE_EMAIL}):
        print(f"ℹ️ Executive user '{EXECUTIVE_EMAIL}' already exists. Skipping.")
    else:
        # THE FIX: Changed 'sha269' to the correct 'sha256'
        hashed_password = generate_password_hash(EXECUTIVE_PASSWORD, method='pbkdf2:sha256')
        users_collection.insert_one({
            'email': EXECUTIVE_EMAIL,
            'password': hashed_password,
            'name': EXECUTIVE_NAME,
            'role': 'exec'
        })
        print(f"✅ Executive user '{EXECUTIVE_NAME}' created successfully.")
        print(f"   - Email: {EXECUTIVE_EMAIL}")
        print(f"   - Password: {EXECUTIVE_PASSWORD}")

    # --- Create Sales User ---
    if users_collection.find_one({'email': SALES_EMAIL}):
        print(f"ℹ️ Sales user '{SALES_EMAIL}' already exists. Skipping.")
    else:
        # THE FIX: Changed 'sha269' to the correct 'sha256'
        hashed_password = generate_password_hash(SALES_PASSWORD, method='pbkdf2:sha256')
        users_collection.insert_one({
            'email': SALES_EMAIL,
            'password': hashed_password,
            'name': SALES_NAME,
            'role': 'sales',
            'manager_email': EXECUTIVE_EMAIL
        })
        print(f"✅ Sales user '{SALES_NAME}' created successfully.")
        print(f"   - Email: {SALES_EMAIL}")
        print(f"   - Password: {SALES_PASSWORD}")
        
    print("\n--- User setup complete. ---")


if __name__ == "__main__":
    setup_users()