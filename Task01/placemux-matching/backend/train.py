"""
Standalone training run, separate from the API so you can re-run
experiments and inspect numbers without booting the server.

Usage:
    python train.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.matching import train_and_evaluate  # noqa: E402

if __name__ == "__main__":
    result = train_and_evaluate(pairs_per_job=25)
    print("\n=== Training run complete ===")
    print(json.dumps(result, indent=2))
    print(f"\nArtifacts written to: backend/artifacts/")
    print(f"  - match_model.joblib   (trained classifier)")
    print(f"  - metrics.json         (latest held-out metrics)")
    print(f"  - experiment_log.jsonl (append-only run history)")
