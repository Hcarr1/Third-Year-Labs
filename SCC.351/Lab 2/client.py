# This builds upon the skeleton code file for lab 1. Any instructions that are new have NEW at the beginning of the line.

# Step 1: Import the requests library to send HTTP requests to the server.
#   NEW Import libraries for cryptographic operations.

import requests
import json
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

# Step 2: Define the base URL of the server
#   This should match the host and port used in server.py

BASE_URL = "http://127.0.0.1:5000"

# NEW Step 3: Write a function that takes as input a voter ID and loads the voter's private key from a PEM file.
#   NEW This key is used to sign the vote before submission.
#   NEW The function should return an error if the voter ID is not an eligible voter.
def load_private_key(voter_id):
    try:
        with open(f"PEM files/{voter_id}_private_key.pem", "rb") as f:
            private_pem = f.read()
        private_key = serialization.load_pem_private_key(
            private_pem,
            password=None,
            backend=default_backend()
        )
        return private_key
    except FileNotFoundError:
        print(f"Error: No private key found for voter ID '{voter_id}'.")
        return None

# NEW Step 4: Write a function to sign the vote.
#   NEW Your function should take as input a private signing key and a candidate, and should return a signature.
#   NEW Convert signature to hex string for transmission
def sign_vote(private_key, candidate):
    signature = private_key.sign(
        candidate.encode('utf-8'),
        ec.ECDSA(hashes.SHA256())
    )
    return signature.hex()

# Step 5: Write a function to send a vote
#   NEW Generate a signature using the signing function in step 4.
#   Prepare the headers and JSON payload
#   NEW Include voter ID, candidate name, and signature in the payload.
#   NEW Print the payload for testing purposes.
#   Send a POST request to the /vote endpoint and raise an error if the server responds with a failure code (e.g. 400 or 500)
#   Print the server's response (should be a confirmation message)
def send_vote(voter_id, candidate):
    private_key = load_private_key(voter_id)
    if private_key is None:
        return

    signature = sign_vote(private_key, candidate)

    url = f"{BASE_URL}/vote"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "voter_id": voter_id,
        "candidate": candidate,
        "signature": signature
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    print(response.json())

# Step 6: Write a function to fetch current vote results
#   Send a GET request to the /results endpoint
#   Print the vote tally in a readable format
def fetch_results():
    url = f"{BASE_URL}/results"
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    results = response.json().get("results", {})

    print("Current Vote Tally:")
    for candidate, votes in results.items():
        print(f"{candidate}: {votes} votes")

def extract_error_message(response):
    if response is None:
        return "No response from server"
    try:
        data = response.json()
        if isinstance(data, dict):
            return data.get("error") or data.get("message") or str(data)
        return str(data)
    except Exception:
        text = (response.text or "").strip()
        if text:
            return text
        return f"HTTP {response.status_code} {response.reason}"

def is_registered_voter(voter_id):
    try:
        with open("public_keys.json", "r") as f:
            public_key_db = json.load(f)
        return voter_id in public_key_db
    except FileNotFoundError:
        return False

# Step 7: Create a simple command-line interface
#   This lets users type commands to vote or view results
#   NEW The interface should prompt the user for their voter ID before alloing them to vote or view results.
#   NEW The voter ID must match a registered key.
#   If the user types a vote command, extract the candidate name and send the vote
#   If the user types 'results', fetch and display the current tally
def main():
    while True:
        voter_id = input("Enter your voter ID: ").strip()

        if not is_registered_voter(voter_id):
            print(f"Error: Voter ID '{voter_id}' is not registered.")
            continue

        command = input("Enter command (`vote <candidate>` / `results` / `exit`): ").strip()
        if command.startswith("vote "):
            candidate = command.split(" ", 1)[1]
            try:
                send_vote(voter_id, candidate)
            except requests.HTTPError as e:
                msg = extract_error_message(getattr(e, "response", None))
                print(f"Error: {msg}")
        elif command == "results":
            try:
                fetch_results()
            except requests.HTTPError as e:
                msg = extract_error_message(getattr(e, "response", None))
                print(f"Error: {msg}")
        elif command == "exit":
            break
        else:
            print("Invalid command. Please try again.")

if __name__ == "__main__":
    main()

