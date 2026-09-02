"""
RISKGRAPH — Step 4.4A Backend & Frontend API Contract Verification
Tests:
1. GET /health contract & response
2. CORS headers and preflight handling (OPTIONS)
3. POST /assess with frontend-friendly request
4. Exact clean response structure (zero PII, zero credentials)
5. Structured HTTP error handling (404 for unknown orders, 422 for invalid payloads)
6. Model C and dataset cryptographic checksum preservation
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

def run_step_4_4a_tests():
    print("=" * 85)
    print("STEP 4.4A: BACKEND API CONTRACT & CORS PREPARATION VERIFICATION")
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

    # 2. Test GET /health
    print("\n--- 2. TESTING GET /health ---")
    resp_health = client.get("/health")
    assert resp_health.status_code == 200, f"Health check failed: {resp_health.status_code}"
    health_json = resp_health.json()
    print("[HEALTH RESPONSE]:", json.dumps(health_json, indent=2))
    assert health_json["status"] == "ok"
    assert health_json["model"] == "Model C"
    assert health_json["model_version"] == "step_2_8_frozen_v1"
    print("[PASS] GET /health matches contract exactly.")

    # 3. Test CORS Headers & Preflight
    print("\n--- 3. TESTING CORS & PREFLIGHT OPTIONS ---")
    headers = {
        "Origin": "http://localhost:8080",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type"
    }
    resp_cors = client.options("/assess", headers=headers)
    print(f"CORS OPTIONS /assess Status: {resp_cors.status_code}")
    print("CORS Headers:", {k: v for k, v in resp_cors.headers.items() if 'access-control' in k.lower()})
    assert resp_cors.status_code == 200
    assert "access-control-allow-origin" in resp_cors.headers
    print("[PASS] CORS preflight OPTIONS returns 200 with allowed origin headers.")

    # 4. Test POST /assess with Clean Frontend Request
    print("\n--- 4. TESTING POST /assess (FRONTEND CONTRACT) ---")
    req_payload = {"order_id": "ORD100028"}
    resp_assess = client.post("/assess", json=req_payload, headers={"Origin": "http://localhost:8080"})
    assert resp_assess.status_code == 200, f"Assessment failed: {resp_assess.text}"
    assess_json = resp_assess.json()
    print("[ASSESSMENT RESPONSE]:")
    print(json.dumps(assess_json, indent=2))

    # Verify exact required fields
    expected_fields = ["request_id", "risk_score", "risk_level", "recommended_action", "model_version", "threshold_version", "top_signals"]
    for f in expected_fields:
        assert f in assess_json, f"Missing field {f} in response!"
    
    assert assess_json["request_id"] == "ORD100028"
    assert isinstance(assess_json["risk_score"], float)
    assert assess_json["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert assess_json["model_version"] == "step_2_8_frozen_v1"
    assert assess_json["threshold_version"] == "T1=0.20,T2=0.45"
    assert isinstance(assess_json["top_signals"], list)
    print("[PASS] Clean frontend response contract verified.")

    # 5. Test PII & Credential Isolation
    print("\n--- 5. TESTING CREDENTIAL & PII ISOLATION ---")
    sensitive_keys = ["customer_id", "device_id", "shipping_address_id", "billing_address_id", "supabase_url", "supabase_secret_key", "service_role_key"]
    response_keys_lower = [k.lower() for k in assess_json.keys()]
    for sk in sensitive_keys:
        assert sk not in response_keys_lower, f"PII/Credential leak in response key: {sk}"
    
    response_str = json.dumps(assess_json).lower()
    for sk in ["sb_secret", "supabase_url", "postgres", "service_role"]:
        assert sk not in response_str, f"Credential leaked in response body: {sk}"
    print("[PASS] Zero PII or database credentials exposed in API response.")

    # 6. Test Error Handling
    print("\n--- 6. TESTING ERROR HANDLING ---")
    
    # 6a. Unknown order -> 404
    resp_404 = client.post("/assess", json={"order_id": "ORD999999"})
    print(f"Unknown order (ORD999999) status code: {resp_404.status_code}")
    print(f"Error detail: {resp_404.json()}")
    assert resp_404.status_code == 404
    assert "not found" in resp_404.json()["detail"].lower()
    print("[PASS] Unknown order correctly returns HTTP 404.")

    # 6b. Missing required order_id -> 422 Unprocessable Entity
    resp_422 = client.post("/assess", json={})
    print(f"Empty payload status code: {resp_422.status_code}")
    assert resp_422.status_code == 422
    print("[PASS] Invalid payload correctly returns HTTP 422 validation error.")

    # 7. Post-Execution Safety Check
    print("\n--- 7. POST-EXECUTION DATASET & MODEL SAFETY CHECK ---")
    for name, exp_hash in expected_hashes.items():
        p = os.path.join(BASE_DIR, name)
        curr_hash = get_file_sha256(p)
        assert curr_hash == exp_hash, f"Modification detected in {name}!"
        print(f"[PASS] {name:<42} UNMODIFIED.")

    print("\n" + "=" * 85)
    print("STEP 4.4A BACKEND & API CONTRACT VERIFICATION COMPLETED WITH 100% SUCCESS")
    print("=" * 85)

if __name__ == "__main__":
    run_step_4_4a_tests()
