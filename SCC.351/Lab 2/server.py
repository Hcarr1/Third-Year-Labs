# Step 1: Import helper functions for handling web requests and JSON responses
#   NEW Import functions from cryptography.io for cryptographic operations.
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes

# Step 2: Use a dictionary to store vote counts for each candidate. It should be held in memory and updated as votes are received.
# Your voting server should accept votes for three candidates: Alice, Bob and Charlie.
vote_counts = {
    "Alice": 0,
    "Bob": 0,
    "Charlie": 0
}


# NEW Step 3: Define a function to load the public key for a given voter ID from the registrar's database.
#   NEW This key is used to verify the signature on the submitted vote.

def load_public_key(voter_id):
    try:
        with open("public_keys.json", "r") as f:
            public_key_db = json.load(f)
        public_pem = public_key_db.get(voter_id)
        if not public_pem:
            return None
        public_key = serialization.load_pem_public_key(
            public_pem.encode('utf-8'),
            backend=default_backend()
        )
        return public_key
    except FileNotFoundError:
        return None


# Step 4: Define a custom request handler class to manage incoming HTTP requests.
class RequestHandler(BaseHTTPRequestHandler):

    def __send_json(self, payload, status_code=200):
        body = json.dumps(payload).encode()
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def __debug_request(self, body):
        print(f"{self.command} {self.path}")
        for header, value in self.headers.items():
            print(f"{header}: {value}")
        if body is not None:
            try:
                print("Body:", body.decode(errors='replace'))
            except Exception:
                print("Body: (binary)")

    # It should handle POST requests.
    # Write a function do_POST(self) that:
    #   Checks that the incoming request contains JSON data.
    #   Extract the candidate name and validate that the candidate is in the allowed list defined in step 3.
    #   Return an error response if the candidate is not in the allowed list.
    #   NEW Check for any missing fields in the JSON data. Return an error response if any fields are missing.
    #   NEW Verify the signature.
    #   NEW If the candidate is in the allowed list, there are no missing fields, and the signature verifies, increment the vote count for that candidate and return a success message.
    def do_POST(self):
        if self.path != '/vote':
            self.__send_json({"error": "Not Found"}, status_code=404)
            return

        try:
            content_length = int(self.headers['Content-Length'])
        except ValueError:
            content_length = None

        body = self.rfile.read(content_length) if content_length else None
        self.__debug_request(body)

        if not body:
            self.__send_json({"error": "Missing request body"}, status_code=400)
            return

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.__send_json({"error": "Invalid JSON"}, status_code=400)
            return

        required_fields = {"voter_id", "candidate", "signature"}
        if not all(field in data for field in required_fields):
            self.__send_json({"error": "Missing fields in request"}, status_code=400)
            return

        voter_id = data.get("voter_id")
        candidate = data.get("candidate")
        signature_hex = data.get("signature")
        public_key = load_public_key(voter_id)
        if public_key is None:
            self.__send_json({"error": "Unregistered voter"}, status_code=400)
            return
        signature = bytes.fromhex(signature_hex)
        try:
            public_key.verify(
                signature,
                candidate.encode('utf-8'),
                ec.ECDSA(hashes.SHA256())
            )
        except Exception:
            self.__send_json({"error": "Invalid signature"}, status_code=400)
            return

        if candidate not in vote_counts:
            self.__send_json({"error": "Invalid candidate"}, status_code=400)
            return

        vote_counts[candidate] += 1
        self.__send_json({"message": f"Vote recorded for {candidate}"})

    # Write a function do_GET(self) that:
    #   Handles GET requests used to retrieve the current vote results.
    #   It should check that the request is targeting the /results endpoint and respond with the current vote tally in JSON format.

    def do_GET(self):
        if self.path != '/results':
            self.__send_json({"error": "Not Found"}, status_code=404)
        else:
            self.__send_json({"results": vote_counts})


# Step 5: Write a function that sets up and runs the HTTP server.
# It should bind to localhost on port 5000.
# It should run until manually stopped.
def Server():
    server_address = ('127.0.0.1', 5000)
    httpd = HTTPServer(server_address, RequestHandler)
    print('Starting voting server on port 5000...')
    httpd.serve_forever()


if __name__ == '__main__':
    Server()

