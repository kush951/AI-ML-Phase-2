
from backend.esign.signer import KeyManager, OfferSigner
from backend.esign.verifier import TamperVerifier

class ESignPipeline:

    def __init__(self):
        self.key_manager = KeyManager()
        self.key_manager.generate_keypair()

        self.signer = OfferSigner(self.key_manager)
        self.verifier = TamperVerifier(self.key_manager)

    def sign_offer(self, offer_data):
        return self.signer.sign(offer_data)

    def verify_offer(self, signed_offer):
        return self.verifier.verify(signed_offer)
