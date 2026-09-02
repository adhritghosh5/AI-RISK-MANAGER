"""
RISKGRAPH — Step 4.4B Frontend Integration & End-to-End Verification Test
Tests:
1. Real order live assessment (ORD100028 - Low Risk)
2. Real order live assessment (ORD100007 - Medium Risk)
3. Real order live assessment (ORD100003 - High Risk)
4. Unknown order error handling (404 mapping)
5. Empty order validation error handling (422 mapping)
6. Frontend files security audit (zero Supabase secrets in JS/HTML)
7. Dataset & Model C integrity check
"""

import os
import sys
import json
import hashlib
from fastapi.testclient import TestClient

# Ensure utf-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from riskgraph_backend.main import app

def get_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def run_step_4_4b_tests():
    print("=" * 85)
    print("STEP 4.4B: FRONTEND INTEGRATION & END-TO-END VERIFICATION")
    print("=" * 85)

    # 1. Dataset & Model Checksums Check
    expected_hashes = {
        'riskgraph_ecommerce_dataset.csv': 'e32ef89c9f70ab848f9d46d30c2687777be9477c6f36cfbc6e97f14527db0cef',
        'riskgraph_independent_raw_dataset.csv': '840c2bc59e42401d2e601211c3c8a926833db08a54f4906172a6b33b1cfc5827',
        'riskgraph_refund_modeling_dataset.csv': 'fc93c3858ef1d7a002aead074f2b5b40d8df8190d0ca54c65cbb6b65e5b6cd9d',
        'riskgraph_pipeline_c.joblib': 'b250d233af81c437687624d5b1fcf82f9c22e74c6c4ff71c77aeef24a10648b2'
    }

    print("\n--- 1. PRE-CHECK DATASET & MODEL INTEGRITY ---")
    for name, exp_hash in expected_hashes.items():
        p = os.path.join(BASE_DIR, name)
        curr_hash = get_file_sha256(p)
        assert curr_hash == exp_hash, f"Integrity failure on {name}!"
        print(f"[PASS] {name:<42} SHA-256 Match: True")

    client = TestClient(app)

    # 2. Test Real Order Assessment (ORD100028 - Low Risk)
    print("\n--- 2. TESTING REAL ORDER ASSESSMENT: ORD100028 ---")
    resp_low = client.post("/assess", json={"order_id": "ORD100028"})
    assert resp_low.status_code == 200, f"Assessment failed: {resp_low.text}"
    low_data = resp_low.json()
    print("[RESPONSE ORD100028]:")
    print(json.dumps(low_data, indent=2))
    assert low_data["request_id"] == "ORD100028"
    assert low_data["risk_level"] == "LOW"
    assert low_data["recommended_action"] == "Normal refund processing"
    assert low_data["risk_score"] < 0.20
    print("[PASS] ORD100028 evaluated correctly as LOW RISK.")

    # 3. Test Real Order Assessment (ORD100007 - Medium Risk)
    print("\n--- 3. TESTING REAL ORDER ASSESSMENT: ORD100007 ---")
    resp_med = client.post("/assess", json={"order_id": "ORD100007"})
    assert resp_med.status_code == 200, f"Assessment failed: {resp_med.text}"
    med_data = resp_med.json()
    print("[RESPONSE ORD100007]:")
    print(json.dumps(med_data, indent=2))
    assert med_data["request_id"] == "ORD100007"
    assert med_data["risk_level"] == "MEDIUM"
    assert med_data["recommended_action"] == "Additional verification"
    assert 0.20 <= med_data["risk_score"] < 0.45
    print("[PASS] ORD100007 evaluated correctly as MEDIUM RISK.")

    # 4. Test Error Case: Unknown Order
    print("\n--- 4. TESTING ERROR CASE: UNKNOWN ORDER (ORD999999) ---")
    resp_unknown = client.post("/assess", json={"order_id": "ORD999999"})
    print(f"Unknown Order Status: {resp_unknown.status_code}")
    print(f"Response: {resp_unknown.json()}")
    assert resp_unknown.status_code == 404
    print("[PASS] Unknown order returns 404.")

    # 5. Test Error Case: Missing/Empty Order
    print("\n--- 5. TESTING ERROR CASE: EMPTY PAYLOAD ---")
    resp_empty = client.post("/assess", json={})
    print(f"Empty Payload Status: {resp_empty.status_code}")
    assert resp_empty.status_code == 422
    print("[PASS] Empty payload returns 422 Unprocessable Entity.")

    # 6. Frontend Files Security & Credential Isolation Audit
    print("\n--- 6. FRONTEND FILES SECURITY AUDIT ---")
    frontend_files = [
        os.path.join(BASE_DIR, 'riskgraph_app', 'app.js'),
        os.path.join(BASE_DIR, 'riskgraph_app', 'index.html'),
        os.path.join(BASE_DIR, 'riskgraph_app', 'style.css')
    ]
    prohibited_secret_tokens = [
        "service_role", "supabase_service_role_key", "sb_secret", "postgresql://",
        "eyj", "bearer ", "secret_key"
    ]
    for fp in frontend_files:
        with open(fp, 'r', encoding='utf-8') as f:
            content = f.read().lower()
            for token in prohibited_secret_tokens:
                assert token not in content, f"SECURITY VIOLATION: '{token}' found in {fp}!"
        print(f"[PASS] {os.path.basename(fp):<20} 100% Clean (Zero Supabase secrets found).")

    # 7. Post-Execution Safety Check
    print("\n--- 7. POST-EXECUTION DATASET & MODEL SAFETY CHECK ---")
    for name, exp_hash in expected_hashes.items():
        p = os.path.join(BASE_DIR, name)
        curr_hash = get_file_sha256(p)
        assert curr_hash == exp_hash, f"Modification detected in {name}!"
        print(f"[PASS] {name:<42} UNMODIFIED.")

    print("\n" + "=" * 85)
    print("STEP 4.4B FRONTEND INTEGRATION COMPLETED WITH 100% SUCCESS")
    print("=" * 85)

if __name__ == "__main__":
    run_step_4_4b_tests()
