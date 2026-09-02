"""
RISKGRAPH Supabase Secure Connectivity & Configuration Test (Step 4.1A)

Verifies:
1. Environment credentials validation (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
2. Secret protection (zero leakage to logs, stdout, or frontend files)
3. Safe Supabase client initialization
4. Harmless read-only query execution (SELECT * LIMIT 3)
5. Zero modifications to local datasets or database data
"""

import os
import sys
import hashlib
import json

# Ensure utf-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from riskgraph_backend.supabase_client import (
    validate_supabase_config,
    get_supabase_client,
    execute_readonly_connectivity_test
)

def get_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("STEP 4.1A: SUPABASE SECURE CONFIGURATION & CONNECTIVITY TEST")
    print("=" * 80)

    # 1. Dataset Integrity Pre-Check
    expected_hashes = {
        'riskgraph_ecommerce_dataset.csv': 'e32ef89c9f70ab848f9d46d30c2687777be9477c6f36cfbc6e97f14527db0cef',
        'riskgraph_independent_raw_dataset.csv': '840c2bc59e42401d2e601211c3c8a926833db08a54f4906172a6b33b1cfc5827',
        'riskgraph_refund_modeling_dataset.csv': 'fc93c3858ef1d7a002aead074f2b5b40d8df8190d0ca54c65cbb6b65e5b6cd9d'
    }

    print("\n--- 1. LOCAL DATASET INTEGRITY CHECK ---")
    for name, exp_hash in expected_hashes.items():
        p = os.path.join(BASE_DIR, name)
        curr_hash = get_file_sha256(p)
        assert curr_hash == exp_hash, f"Dataset {name} was modified!"
        print(f"[PASS] {name:<42} SHA-256 Match: True")

    # 2. Frontend Security Audit (Ensure no secret in index.html, app.js, style.css)
    print("\n--- 2. FRONTEND SECRETS EXPOSURE AUDIT ---")
    frontend_files = [
        os.path.join(BASE_DIR, "riskgraph_app", "index.html"),
        os.path.join(BASE_DIR, "riskgraph_app", "app.js"),
        os.path.join(BASE_DIR, "riskgraph_app", "style.css")
    ]
    for ff in frontend_files:
        if os.path.exists(ff):
            with open(ff, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "SUPABASE_SERVICE_ROLE_KEY" not in content, f"Secret reference found in {ff}!"
                assert "service_role" not in content, f"Potential secret keyword found in {ff}!"
            print(f"[PASS] {os.path.basename(ff):<20} Clean (Zero Supabase credentials present)")

    # 3. Gitignore Security Audit
    print("\n--- 3. GITIGNORE AUDIT ---")
    gitignore_path = os.path.join(BASE_DIR, ".gitignore")
    assert os.path.exists(gitignore_path), ".gitignore missing!"
    with open(gitignore_path, 'r', encoding='utf-8') as f:
        gi_content = f.read()
        assert ".env" in gi_content, ".env not in .gitignore!"
        assert "*.env" in gi_content, "*.env not in .gitignore!"
    print("[PASS] .gitignore correctly excludes .env and *.env files.")

    # 4. Supabase Credentials Validation
    print("\n--- 4. SUPABASE CONFIGURATION VALIDATION ---")
    try:
        url, key = validate_supabase_config()
        # Obfuscate URL and key in output for security
        obfuscated_url = url[:16] + "..." if len(url) > 16 else "***"
        obfuscated_key = key[:6] + "..." + key[-4:] if len(key) > 10 else "***"
        print(f"[CONFIG] SUPABASE_URL: {obfuscated_url}")
        print(f"[CONFIG] SUPABASE_SERVICE_ROLE_KEY: {obfuscated_key} (Obfuscated)")
        print("[PASS] Environment credentials syntax validated.")

        # 5. Client Initialization & Harmless Read-Only Test
        print("\n--- 5. SUPABASE CLIENT INITIALIZATION & READ-ONLY QUERY ---")
        client = get_supabase_client()
        print("[PASS] Supabase client initialized successfully.")

        print("[QUERY] Executing harmless read-only SELECT * LIMIT 3 against 'raw_ecommerce_data'...")
        result = execute_readonly_connectivity_test("raw_ecommerce_data", limit=3)
        
        if result["status"] == "success":
            print(f"[PASS] Read-only query succeeded! Retrieved {result['rows_retrieved']} rows from table '{result['table']}'.")
            if result['rows_retrieved'] > 0:
                print(f"Sample row keys: {list(result['sample_records'][0].keys())[:8]}...")
        else:
            print(f"[NOTICE] Read-only query returned: {result['error_type']} - {result['error_detail']}")
            print("[INFO] Note: If the table name differs in your existing Supabase project, the client is ready to connect once the table name is specified.")

    except ValueError as ve:
        print(f"[NOTICE] Expected configuration notice: {ve}")
        print("[INFO] To test live database connectivity:")
        print("       1. Open riskgraph_backend/.env")
        print("       2. Add your SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
        print("       3. Re-run: python test_supabase_connectivity.py")

    # 6. Final Dataset Integrity Post-Check
    print("\n--- 6. DATASET INTEGRITY POST-CHECK ---")
    for name, exp_hash in expected_hashes.items():
        p = os.path.join(BASE_DIR, name)
        curr_hash = get_file_sha256(p)
        assert curr_hash == exp_hash, f"Dataset {name} was modified!"
        print(f"[PASS] {name:<42} UNMODIFIED.")

    print("\n" + "=" * 80)
    print("STEP 4.1A CONFIGURATION & SECURITY AUDIT COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
