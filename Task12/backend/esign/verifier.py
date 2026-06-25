
import hashlib
import json
import base64

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

class TamperVerifier:

    def __init__(self, key_manager):
        self.key_manager = key_manager

    def verify(self, signed_offer):

        payload = signed_offer["payload"]

        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(',', ':')
        ).encode()

        recomputed_hash = hashlib.sha256(canonical).hexdigest()

        hash_ok = recomputed_hash == signed_offer["sha256_hash"]

        try:
            self.key_manager.public_key.verify(
                base64.b64decode(signed_offer["signature"]),
                canonical,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            sig_ok = True

        except Exception:
            sig_ok = False

        return {
            "valid": hash_ok and sig_ok,
            "hash_match": hash_ok,
            "signature_valid": sig_ok
        }
