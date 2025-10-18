from pymongo import MongoClient
from difflib import SequenceMatcher
from datetime import datetime

from . import db, collection, leads_collection, campaigns_collection, sent_emails_collection, replies_collection, users_collection, email_replies_collection, paired_emails_collection, legacy_leads_collection, legacy_campaigns_collection

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def normalize_subject(subject):
    subject = subject.lower().strip()
    for prefix in ["re:", "fw:", "fwd:", "re :", "fwd :"]:
        if subject.startswith(prefix):
            subject = subject[len(prefix):].strip()
    return subject

def match_and_pair():
    all_incoming = list(incoming.find())
    all_replies = list(replies.find())
    count = 0

    for reply in all_replies:
        reply_subj = normalize_subject(reply.get("subject", ""))
        reply_time = reply.get("timestamp", datetime.now())

        best_match = None
        best_score = 0.0

        for email in all_incoming:
            email_subj = normalize_subject(email.get("subject", ""))
            score = similar(reply_subj, email_subj)

            # Optional: filter by timestamp proximity (within 30 days)
            if score > best_score and abs((reply_time - email.get("timestamp", reply_time)).days) <= 30:
                best_score = score
                best_match = email

        if best_score > 0.85 and best_match:
            paired.insert_one({
                "email_text": best_match["email_text"],
                "category": best_match.get("category", ""),
                "intent": best_match.get("intent", ""),
                "reply_text": reply["reply_body"],
                "subject": best_match.get("subject", ""),
                "timestamp_original": best_match["timestamp"],
                "timestamp_reply": reply["timestamp"]
            })
            count += 1
            print(f"[✓] Paired: {reply['subject'][:60]}...")

    print(f"\n✅ Total Pairs Created: {count}")

if __name__ == "__main__":
    match_and_pair()
