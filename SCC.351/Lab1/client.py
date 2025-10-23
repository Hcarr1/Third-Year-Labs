# Step 1: Import the requests library to send HTTP requests to the server
import requests

# Step 2: Define the base URL of the server
# This should match the host and port used in server.py

BASE_URL = "http://127.0.0.1:5000"

# Step 3: Write a function to send a vote
#   Prepare the headers and JSON payload
#   Send a POST request to the /vote endpoint and raise an error if the server responds with a failure code (e.g. 400 or 500)
#   Print the server's response (should be a confirmation message)
def send_vote(candidate):
    url = f"{BASE_URL}/vote"
    headers = {'Content-Type': 'application/json'}
    payload = {"candidate": candidate}

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise an error for bad responses
    print(response.json())

# Step 4: Write a function to fetch current vote results
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

# Step 5: Create a simple command-line interface
#   This lets users type commands to vote or view results
#   If the user types a vote command, extract the candidate name and send the vote
#   If the user types 'results', fetch and display the current tally

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


def main():
    while True:
        command = input("Enter command (`vote <candidate>` / `results` / `exit`): ").strip()
        if command.startswith("vote "):
            candidate = command.split(" ", 1)[1]
            try:
                send_vote(candidate)
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
            print("Exiting...")
            break
        else:
            print("Invalid command. Please use `vote <candidate>`, `results`, or `exit`.")

if __name__ == "__main__":
    main()