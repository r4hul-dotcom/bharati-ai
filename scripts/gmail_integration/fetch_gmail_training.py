# START OF FILE: fetch_gmail_training.py (IMPROVED)

import argparse
import csv
from gmail_utils import gmail_authenticate, get_clean_email_body
# IMPORTANT: This script should NOT import from testapp.py. It's a data-gathering tool.

def fetch_emails_for_dataset(query, output_file, limit=100):
    """
    Fetches emails based on a query and saves them to a CSV file for training.
    This does NOT classify them; it prepares them for human review and labeling.
    """
    service = gmail_authenticate()
    results = service.users().messages().list(userId="me", q=query, maxResults=limit).execute()
    messages = results.get("messages", [])

    print(f"Found {len(messages)} emails matching query: '{query}'")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'category']) # Write header

        for msg_info in messages:
            msg_data = service.users().messages().get(userId="me", id=msg_info["id"]).execute()
            subject, body = get_clean_email_body(msg_data["payload"])
            
            if body:
                full_text = f"{subject}\n\n{body}"
                # Write the text and an EMPTY category column for a human to fill in.
                writer.writerow([full_text, '']) 
                print(f"✅ Extracted: '{subject[:60]}...'")

    print(f"\n🎉 Success! Data exported to '{output_file}'.")
    print("Next step: Open the CSV file and manually fill in the 'category' for each email.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Gmail emails to create a training dataset.")
    parser.add_argument(
        "-q", "--query", 
        default="after:2024/01/01", 
        help="Gmail search query (e.g., 'after:2024/01/01')."
    )
    parser.add_argument(
        "-o", "--output", 
        default="new_training_data.csv", 
        help="Output CSV file name."
    )
    parser.add_argument(
        "-l", "--limit", 
        type=int,
        default=100,
        help="Maximum number of emails to fetch."
    )
    args = parser.parse_args()
    
    fetch_emails_for_dataset(args.query, args.output, args.limit)