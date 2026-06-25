
import hashlib
import json
import time
import uuid
import base64

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

class KeyManager:

    def __init__(self):
        self.private_key = None
        self.public_key = None

    def generate_keypair(self):

        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )

        self.private_key = private_key
        self.public_key = private_key.public_key()

class OfferSigner:

    def __init__(self, key_manager):
        self.key_manager = key_manager

    def sign(self, offer):

        payload = {
            **offer,
            "offer_id": str(uuid.uuid4()),
            "signed_at": int(time.time())
        }

        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(',', ':')
        ).encode()

        sha256_hash = hashlib.sha256(canonical).hexdigest()

        signature = self.key_manager.private_key.sign(
            canonical,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        return {
            "payload": payload,
            "sha256_hash": sha256_hash,
            "signature": base64.b64encode(signature).decode()
        }
