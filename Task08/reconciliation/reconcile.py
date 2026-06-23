
gateway_record = {
    "txn_id": "TXN1001",
    "amount": 499,
    "status": "SUCCESS"
}

internal_record = {
    "txn_id": "TXN1001",
    "amount": 499,
    "status": "SUCCESS"
}

if gateway_record == internal_record:
    print("Reconciliation SUCCESS")
else:
    print("Mismatch detected")
