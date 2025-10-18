from pymongo import MongoClient
from bson import ObjectId

from . import db, collection, leads_collection, campaigns_collection, sent_emails_collection, replies_collection, users_collection, email_replies_collection, paired_emails_collection, legacy_leads_collection, legacy_campaigns_collection

KISHAN_RAJ_SIGNATURE = """
<br><br>
<div style="font-family: Arial, sans-serif; font-size: 11pt; color: #000080;">
    <p style="margin: 0;">Thanks & Regards</p>
    <p style="margin: 0; font-weight: bold; font-size: 12pt;">Kishan Raj</p>
    <p style="margin: 0;">Sales Executive</p>
    <p style="margin: 0; font-weight: bold;">Bharati Fire Engineers</p>
    <hr style="border: none; border-top: 1px solid #FFA500; margin: 10px 0;">
    <p style="margin: 0;">
        <img src="https://i.imgur.com/8aZ3E2G.png" alt="Phone" style="vertical-align: middle; height: 14px; width: 14px;">
        <span style="vertical-align: middle;"> +91 6202503433 / +91 8591998712</span>
    </p>
    <p style="margin: 5px 0;">
        <img src="https://i.imgur.com/k2tC8h6.png" alt="Location" style="vertical-align: middle; height: 14px; width: 14px;">
        <span style="vertical-align: middle;"> Plant: Plot No. A-427 & A-428, TTC Ind Estate, MIDC Mahape, Navi Mumbai - 400 709</span>
    </p>
    <p style="margin: 5px 0;">
        <span style="vertical-align: middle;"> Corp Office: 206-B, Flying Colors, PDU Marg, Mulund (W), Mumbai - 400 080 INDIA.</span>
    </p>
    <p style="margin: 5px 0;">
        <img src="https://i.imgur.com/o2xYp3A.png" alt="Website" style="vertical-align: middle; height: 14px; width: 14px;">
        <a href="http://bharatifire.com" style="vertical-align: middle; color: #000080; text-decoration: none;"> bharatifire.com</a>
    </p>
    <hr style="border: none; border-top: 1px solid #FFA500; margin: 10px 0;">
</div>
"""

SHYAM_YADAV_SIGNATURE = """
<br><br>
<div style="font-family: Arial, sans-serif; font-size: 11pt; color: #333333;">
    <p style="margin: 0;">Shyam Yadav</p>
    <p style="margin: 0;">Sales Executive</p>
    <p style="margin: 0;">Bharati Fire Engineers</p>
    <hr style="border: none; border-top: 1px solid #FFA500; margin: 10px 0;">
    <p style="margin: 0;">
        <img src="https://i.imgur.com/Qh6S3hD.png" alt="Phone" style="vertical-align: middle; height: 14px; width: 14px;">
        <span style="vertical-align: middle;"> +91 22 25681269, 25684298</span>
    </p>
    <p style="margin: 5px 0;">
        <img src="https://i.imgur.com/8aZ3E2G.png" alt="Mobile" style="vertical-align: middle; height: 14px; width: 14px;">
        <span style="vertical-align: middle;"> +91 8545881539</span>
    </p>
    <p style="margin: 5px 0;">
        <img src="https://i.imgur.com/k2tC8h6.png" alt="Location" style="vertical-align: middle; height: 14px; width: 14px;">
        <span style="vertical-align: middle;"> Plant: Plot No. A-427 & A-428, TTC Ind Estate, MIDC Mahape, Navi Mumbai - 400 709</span>
    </p>
    <p style="margin: 5px 0;">
        <span style="vertical-align: middle;"> Corp Office: 206-B, Flying Colors, PDU Marg, Mulund (W), Mumbai - 400 080 INDIA.</span>
    </p>
    <p style="margin: 5px 0;">
        <img src="https://i.imgur.com/o2xYp3A.png" alt="Website" style="vertical-align: middle; height: 14px; width: 14px;">
        <a href="http://bharatifire.com" style="vertical-align: middle; color: #E87722; text-decoration: none;"> bharatifire.com</a>
    </p>
    <hr style="border: none; border-top: 1px solid #FFA500; margin: 10px 0;">
</div>
"""

SIGNATURE_MAP = {
    "Kishan Raj": KISHAN_RAJ_SIGNATURE,
    "Shyam Yadav": SHYAM_YADAV_SIGNATURE
}

def update_user_signatures():
    updated_count = 0
    for name, signature_html in SIGNATURE_MAP.items():
        user = users_collection.find_one({'name': name, 'role': 'sales'})
        
        if user:
            result = users_collection.update_one(
                {'_id': user['_id']},
                {'$set': {'signature_html': signature_html}}
            )
            if result.modified_count > 0:
                print(f"✅ Successfully updated signature for: {name}")
                updated_count += 1
            else:
                print(f"ℹ️ Signature for {name} is already up to date.")
        else:
            print(f"⚠️ WARNING: Salesperson named '{name}' not found in the database.")
            
    print(f"\nFinished. Updated {updated_count} user signature(s).")

if __name__ == "__main__":
    update_user_signatures()