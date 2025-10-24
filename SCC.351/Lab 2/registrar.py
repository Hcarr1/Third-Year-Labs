# Step 1: Import helper functions and functions from cryptography.io for cryptographic operations.
import os
import json
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

# Step 2: Generate an ECDSA key pair for a given voter ID.
#   The private key is saved locally; the public key is returned for inclusion in the public database.
#   Generate a new ECDSA private key using the SECP256R1 curve
#   Serialize and save the private key to a PEM file (no encryption)
#   Serialize the public key to a PEM string (used by the server for signature verification)
def generate_key_pair(voter_id):
    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())

    # Save the private key to a PEM file
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    )
    with open(f"PEM files/{voter_id}_private_key.pem", "wb") as f:
        f.write(private_pem)

    # Serialize the public key to PEM format
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    return public_pem.decode('utf-8')

# Step 3: Define a function main that prompts the user for number of voters and generates key pairs for each.        
#   Ask the user how many voters to register    
#   Load existing public key database if it exists, or start a new one
#   Generate key pairs for each voter and update the public key database
#   Save the updated public key database to disk
def main():
    num_voters = int(input("Number of voters: "))

    public_key_db = {}
    if os.path.exists("public_keys.json"):
        with open("public_keys.json", "r") as f:
            public_key_db = json.load(f)

    for i in range(num_voters):
        voter_id = input(f"Enter voter ID for voter {i+1}: ").strip()
        public_key_pem = generate_key_pair(voter_id)
        public_key_db[voter_id] = public_key_pem
        print(f"Generated keys for voter ID: {voter_id}")

    with open("public_keys.json", "w") as f:
        json.dump(public_key_db, f, indent=4)

main()