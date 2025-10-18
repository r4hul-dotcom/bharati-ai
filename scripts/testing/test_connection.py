# test_connection.py (Version 2 - Simplified)

import requests

print("--- Running SIMPLIFIED Connection Test ---")
print("This version lets the 'requests' library find the certificate bundle automatically.")

# The URL your Celery task is trying to reach
url = "https://gmail.googleapis.com/gmail/v1/users/me/profile"

try:
    print(f"Attempting to connect to: {url}")
    # We are NOT setting any environment variables.
    response = requests.get(url, headers={'Authorization': 'Bearer dummy_token'}, timeout=15)
    
    print("\n--- CONNECTION SUCCEEDED ---")
    print("This means 'requests' successfully found the certificate bundle on its own.")
    print(f"Status Code: {response.status_code}")
    print(f"Response Text: {response.text[:100]}...") # Print first 100 chars

except Exception as e:
    print("\n--- CONNECTION FAILED AGAIN ---")
    print("This indicates a very deep problem with the Python SSL or requests/certifi installation.")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error: {e}")

print("\n--- Test Complete ---")