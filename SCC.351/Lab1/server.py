# Step 1: Import helper functions for handling web requests and JSON responses
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

# Step 2: Use a dictionary to store vote counts for each candidate. It should be held in memory and updated as votes are received.
# Your voting server should accept votes for three candidates: Alice, Bob and Charlie.
vote_counts = {
    "Alice": 0,
    "Bob": 0,
    "Charlie": 0
}

# Step 3: Define a custom request handler class to manage incoming HTTP requests.
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
#   Return an error response if the canddiate is not in the allowed list.
#   If the candidate is in the allowed list, increment the vote count for that candidate and return a success message.

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

        candidate = data.get("candidate") if isinstance(data, dict) else None
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

# Step 4: Write a function that sets up and runs the HTTP server.
# It should bind to localhost on port 5000.
# It should run until manually stopped.
def Server():
    server_address = ('127.0.0.1', 5000)
    httpd = HTTPServer(server_address, RequestHandler)
    print('Starting voting server on port 5000...')
    httpd.serve_forever()

if __name__ == '__main__':
    Server()