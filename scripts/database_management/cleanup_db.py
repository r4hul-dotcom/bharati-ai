from pymongo import MongoClient

from . import db, collection, leads_collection, campaigns_collection, sent_emails_collection, replies_collection, users_collection, email_replies_collection, paired_emails_collection, legacy_leads_collection, legacy_campaigns_collection

# --- IMPORTANT: MAKE SURE YOUR testapp.py is NOT RUNNING when you run this script ---



# Define the categories you want to remove
unwanted_categories = ["dispatch_update", "sales_approval", "feedback"]

# Create a query to find documents with these categories
query = { "category": { "$in": unwanted_categories } }

# Delete the matching documents
result = collection.delete_many(query)

print(f"Cleanup complete.")
print(f"Total documents removed: {result.deleted_count}")