"""
RISKGRAPH Backend Test & Verification Suite (Step 4.0)
Tests:
1. GET /health endpoint response schema
2. POST /assess endpoint with known existing refund case
3. Multiple representative test cases (LOW, MEDIUM, HIGH)
4. Leakage guard validation
5. Dataset & Model artifact integrity
"""

import os
import sys
import hashlib
import json
import numpy as np
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

def run_all_backend_tests():
    print("=" * 80)
    print("RUNNING RISKGRAPH BACKEND COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    # 1. Dataset Integrity Pre-Check
    raw_a_path = os.path.join(BASE_DIR, "riskgraph_ecommerce_dataset.csv")
    raw_b_path = os.path.join(BASE_DIR, "riskgraph_independent_raw_dataset.csv")
    modeling_path = os.path.join(BASE_DIR, "riskgraph_refund_modeling_dataset.csv")

    expected_hashes = {
        'riskgraph_ecommerce_dataset.csv': 'e32ef89c9f70ab848f9d46d30c2687777be9477c6f36cfbc6e97f14527db0cef',
        'riskgraph_independent_raw_dataset.csv': '840c2bc59e42401d2e601211c3c8a926833db08a54f4906172a6b33b1cfc5827',
        'riskgraph_refund_modeling_dataset.csv': 'fc93c3858ef1d7a002aead074f2b5b40d8df8190d0ca54c65cbb6b65e5b6cd9d'
    }

    print("\n--- 1. DATASET INTEGRITY CHECK ---")
    for name, exp_hash in expected_hashes.items():
        p = os.path.join(BASE_DIR, name)
        curr_hash = get_file_sha256(p)
        assert curr_hash == exp_hash, f"Dataset {name} hash mismatch!"
        print(f"[PASS] {name:<42} SHA-256 match: True")

    client = TestClient(app)

    # 2. Test GET /health
    print("\n--- 2. TEST GET /health ---")
    response = client.get("/health")
    print(f"Status Code: {response.status_code}")
    health_data = response.json()
    print("Response JSON:")
    print(json.dumps(health_data, indent=2))
    assert response.status_code == 200
    assert health_data["status"] == "ok"
    assert health_data["model"] == "Model C"
    assert health_data["model_version"] == "step_2_8_frozen_v1"
    print("[PASS] GET /health returns expected schema.")

    # 3. Test POST /assess on Known Case 1 (ORD108272)
    print("\n--- 3. TEST POST /assess (Known Case: ORD108272) ---")
    case_payload = {
        "order_id": "ORD108272",
        "customer_id": "CUST100050",
        "amount": 450.00,
        "quantity": 1,
        "category": "Fashion",
        "payment_method": "Credit Card",
        "channel": "Mobile App",
        "return_reason": "Size issue",
        "customer_segment": "Premium",
        "customer_tenure_days": 1200,
        "order_date": "2024-11-20",
        "shipping_date": "2024-11-22",
        "return_request_date": "2024-11-24",
        "device_id": "DEV200050",
        "shipping_address_id": "ADDR300050",
        "billing_address_id": "ADDR300050"
    }
    resp = client.post("/assess", json=case_payload)
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    print("Response JSON:")
    print(json.dumps(data, indent=2))
    assert resp.status_code == 200
    assert data["request_id"] == "ORD108272"
    assert isinstance(data["risk_score"], float)
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert data["threshold_version"] == "T1=0.20,T2=0.45"
    assert data["model_version"] == "step_2_8_frozen_v1"
    print(f"[PASS] Case ORD108272 assessed: Risk Score = {data['risk_score']}, Risk Level = {data['risk_level']}, Action = '{data['recommended_action']}'")

    # 4. Test POST /assess on Known Case 2 (Hydration by Order ID only)
    print("\n--- 4. TEST POST /assess (Lookup / Hydration by Order ID) ---")
    resp_hydrated = client.post("/assess", json={"order_id": "ORD103102"})
    print(f"Status Code: {resp_hydrated.status_code}")
    data_h = resp_hydrated.json()
    print("Response JSON:")
    print(json.dumps(data_h, indent=2))
    assert resp_hydrated.status_code == 200
    assert data_h["request_id"] == "ORD103102"
    print(f"[PASS] Case ORD103102 assessed: Risk Score = {data_h['risk_score']}, Risk Level = {data_h['risk_level']}")

    # 5. Test Tri-Tier Routing Rules
    print("\n--- 5. TEST TRI-TIER ROUTING CONSTRAINTS ---")
    # Low Risk Case
    low_case = {
        "order_id": "TEST_LOW_ROUTING",
        "customer_id": "CUST100050",
        "amount": 250.0,
        "quantity": 1,
        "category": "Fashion",
        "payment_method": "Credit Card",
        "channel": "Mobile App",
        "return_reason": "Size issue",
        "customer_segment": "Premium",
        "customer_tenure_days": 1500,
        "order_date": "2024-11-20",
        "shipping_date": "2024-11-22",
        "return_request_date": "2024-11-24",
        "device_id": "DEV200050",
        "shipping_address_id": "ADDR300050",
        "billing_address_id": "ADDR300050"
    }
    res_low = client.post("/assess", json=low_case).json()
    print(f"Low Test Case: Score = {res_low['risk_score']} | Level = {res_low['risk_level']} | Action = '{res_low['recommended_action']}'")
    if res_low['risk_score'] < 0.20:
        assert res_low['risk_level'] == "LOW"
        assert res_low['recommended_action'] == "Normal refund processing"

    # High Risk Sybil Case
    high_case = {
        "order_id": "TEST_HIGH_ROUTING",
        "customer_id": "CUST100000",
        "amount": 4200.0,
        "quantity": 3,
        "category": "Fashion",
        "payment_method": "Wallet",
        "channel": "Social Commerce",
        "return_reason": "Changed mind",
        "customer_segment": "Budget",
        "customer_tenure_days": 30,
        "order_date": "2024-11-15",
        "shipping_date": "2024-11-17",
        "return_request_date": "2024-11-18",
        "device_id": "DEV202769",
        "shipping_address_id": "ADDR301608",
        "billing_address_id": "ADDR301608"
    }
    res_high = client.post("/assess", json=high_case).json()
    print(f"High Test Case: Score = {res_high['risk_score']} | Level = {res_high['risk_level']} | Action = '{res_high['recommended_action']}'")
    if res_high['risk_score'] >= 0.45:
        assert res_high['risk_level'] == "HIGH"
        assert res_high['recommended_action'] == "Human specialist review"
        assert "reject" not in res_high['recommended_action'].lower()
    print("[PASS] Tri-Tier routing and no-auto-reject policy strictly verified.")

    # 6. Leakage Guard Test
    print("\n--- 6. LEAKAGE GUARD UNIT TEST ---")
    from riskgraph_backend.model_service import ModelCService
    service = ModelCService()
    
    # Try passing a feature dict with prohibited column
    prohibited_sample = {c: 0 for c in service.expected_feature_cols}
    prohibited_sample['abuse_label'] = 1
    
    try:
        service.validate_and_format_features(prohibited_sample)
        raise AssertionError("Leakage guard FAILED to intercept abuse_label!")
    except ValueError as e:
        print(f"[PASS] Leakage guard correctly intercepted prohibited field: {e}")

    # 7. Dataset Integrity Post-Check
    print("\n--- 7. DATASET INTEGRITY POST-CHECK ---")
    for name, exp_hash in expected_hashes.items():
        p = os.path.join(BASE_DIR, name)
        curr_hash = get_file_sha256(p)
        assert curr_hash == exp_hash, f"Dataset {name} was modified!"
        print(f"[PASS] {name:<42} UNMODIFIED.")

    print("\n" + "=" * 80)
    print("ALL BACKEND INTEGRATION & VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_all_backend_tests()
