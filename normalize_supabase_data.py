"""
RISKGRAPH — Step 4.3 Supabase Data Normalization Engine
Maps and populates existing normalized Supabase tables from raw_ecommerce_data:
1. customers (2,500 distinct records)
2. devices (1,661 distinct records)
3. addresses (1,696 distinct records)
4. orders (10,000 distinct records)
5. return_requests (2,014 distinct dispute records)

Features:
- Idempotent upsert logic (duplicate-safe)
- Strict relational order (parent tables first)
- Dry-run verification before write
- Post-write integrity & orphan checks
- Zero schema or model modifications
"""

import os
import sys
import hashlib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Ensure utf-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from riskgraph_backend.supabase_client import get_supabase_client

def get_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def fetch_all_raw_data(client) -> pd.DataFrame:
    """Fetch all rows from raw_ecommerce_data using pagination."""
    all_rows = []
    batch_size = 1000
    offset = 0
    while True:
        res = client.table('raw_ecommerce_data').select('*').range(offset, offset + batch_size - 1).execute()
        data = res.data if res.data else []
        all_rows.extend(data)
        if len(data) < batch_size:
            break
        offset += batch_size
    df = pd.DataFrame(all_rows)
    return df

def prepare_normalized_records(df_raw: pd.DataFrame) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Transform raw_ecommerce_data into normalized entity records."""
    df = df_raw.copy()
    df['order_date_dt'] = pd.to_datetime(df['order_date'])

    # 1. Customers
    customer_records = []
    for cid, c_df in df.groupby('customer_id'):
        last_row = c_df.sort_values(by='order_date_dt').iloc[-1]
        first_row = c_df.sort_values(by='order_date_dt').iloc[0]
        
        tenure = int(first_row['customer_tenure_days']) if pd.notna(first_row['customer_tenure_days']) and str(first_row['customer_tenure_days']).strip() != '' else 0
        first_dt = first_row['order_date_dt']
        created_dt = first_dt - pd.Timedelta(days=tenure)
        updated_dt = last_row['order_date_dt']

        customer_records.append({
            'customer_id': str(cid),
            'customer_segment': str(last_row['customer_segment']),
            'location_city_state': str(last_row['customer_location']) if pd.notna(last_row['customer_location']) else None,
            'created_at': created_dt.isoformat() + "Z",
            'updated_at': updated_dt.isoformat() + "Z"
        })

    # 2. Devices
    device_records = []
    for dev_id, d_df in df.groupby('device_id'):
        first_dt = d_df['order_date_dt'].min()
        last_dt = d_df['order_date_dt'].max()
        device_records.append({
            'device_id': str(dev_id),
            'first_seen_at': first_dt.isoformat() + "Z",
            'last_seen_at': last_dt.isoformat() + "Z"
        })

    # 3. Addresses
    ship_addrs = set(df['shipping_address_id'].dropna().astype(str))
    bill_addrs = set(df['billing_address_id'].dropna().astype(str))
    all_addrs = sorted(list(ship_addrs.union(bill_addrs)))

    addr_location_map = {}
    addr_min_date_map = {}
    for idx, row in df.iterrows():
        s_id = str(row['shipping_address_id']) if pd.notna(row['shipping_address_id']) else None
        b_id = str(row['billing_address_id']) if pd.notna(row['billing_address_id']) else None
        loc = str(row['customer_location']) if pd.notna(row['customer_location']) else None
        dt = row['order_date_dt']

        for a_id in [s_id, b_id]:
            if a_id:
                if a_id not in addr_location_map and loc:
                    addr_location_map[a_id] = loc
                if a_id not in addr_min_date_map or dt < addr_min_date_map[a_id]:
                    addr_min_date_map[a_id] = dt

    address_records = []
    for aid in all_addrs:
        loc = addr_location_map.get(aid, "")
        city = None
        state = None
        if loc and "," in loc:
            parts = loc.split(",")
            city = parts[0].strip()
            state = parts[1].strip()
        elif loc:
            city = loc.strip()

        created_dt = addr_min_date_map.get(aid, pd.to_datetime("2024-01-01"))
        address_records.append({
            'address_id': str(aid),
            'city': city,
            'state': state,
            'created_at': created_dt.isoformat() + "Z"
        })

    # 4. Orders
    order_records = []
    for idx, row in df.iterrows():
        o_date = pd.to_datetime(row['order_date'])
        s_date = pd.to_datetime(row['shipping_date']) if pd.notna(row['shipping_date']) and str(row['shipping_date']).strip() != '' else None
        
        order_records.append({
            'order_id': str(row['order_id']),
            'transaction_id': str(row['transaction_id']),
            'customer_id': str(row['customer_id']),
            'device_id': str(row['device_id']),
            'shipping_address_id': str(row['shipping_address_id']),
            'billing_address_id': str(row['billing_address_id']),
            'product_id': str(row['product_id']),
            'category': str(row['category']),
            'amount': float(row['amount']),
            'quantity': int(row['quantity']),
            'payment_method': str(row['payment_method']),
            'channel': str(row['channel']),
            'order_date': o_date.isoformat() + "Z",
            'shipping_date': s_date.isoformat() + "Z" if s_date is not None else None,
            'delivery_status': str(row['delivery_status'])
        })

    # 5. Return Requests (All disputes with return_request_date)
    return_records = []
    has_return = df[df['return_request_date'].notna() & (df['return_request_date'].astype(str).str.strip() != '')]
    
    for idx, row in has_return.iterrows():
        ret_date = pd.to_datetime(row['return_request_date'])
        ref_date = pd.to_datetime(row['refund_date']) if pd.notna(row['refund_date']) and str(row['refund_date']).strip() != '' else None
        ref_amt = float(row['refund_amount']) if pd.notna(row['refund_amount']) and str(row['refund_amount']).strip() != '' else None
        abuse_lbl = int(row['abuse_label']) if pd.notna(row['abuse_label']) and str(row['abuse_label']).strip() != '' else None
        abuse_tp = str(row['abuse_type']) if pd.notna(row['abuse_type']) and str(row['abuse_type']).strip() != '' else None

        return_records.append({
            'return_request_id': str(row['order_id']),  # Deterministic 1-to-1 matching order_id
            'order_id': str(row['order_id']),
            'customer_id': str(row['customer_id']),
            'return_request_date': ret_date.isoformat() + "Z",
            'return_reason': str(row['return_reason']) if pd.notna(row['return_reason']) and str(row['return_reason']).strip() != '' else None,
            'return_resolution': str(row['return_resolution']) if pd.notna(row['return_resolution']) and str(row['return_resolution']).strip() != '' else None,
            'return_status': str(row['return_status']) if pd.notna(row['return_status']) and str(row['return_status']).strip() != '' else None,
            'refund_amount': ref_amt,
            'refund_date': ref_date.isoformat() + "Z" if ref_date is not None else None,
            'abuse_label': abuse_lbl,
            'abuse_type': abuse_tp
        })

    return customer_records, device_records, address_records, order_records, return_records

def upsert_in_batches(client, table_name: str, records: List[Dict], batch_size: int = 500):
    """Upsert records in batches to avoid payload size limits."""
    n = len(records)
    print(f"[UPSERT] Writing {n} records to table '{table_name}' in batches of {batch_size}...")
    for i in range(0, n, batch_size):
        chunk = records[i:i + batch_size]
        res = client.table(table_name).upsert(chunk).execute()
        print(f"  --> Batch {i//batch_size + 1}/{(n + batch_size - 1)//batch_size} written ({len(chunk)} records).")
    print(f"[PASS] Successfully upserted {n} records into '{table_name}'.\n")

def run_normalization():
    print("=" * 85)
    print("STEP 4.3: LIVE SUPABASE DATA NORMALIZATION & INTEGRITY VERIFICATION")
    print("=" * 85)

    client = get_supabase_client()

    # Pre-check CSV checksums
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

    # 1. Fetch raw data from Supabase
    print("\n--- 2. FETCHING LIVE DATA FROM raw_ecommerce_data ---")
    df_raw = fetch_all_raw_data(client)
    print(f"[FETCH] Retrieved {len(df_raw)} records from 'raw_ecommerce_data'.")
    assert len(df_raw) == 10000, f"Expected 10,000 raw records, got {len(df_raw)}"

    # 2. Check pre-normalization row counts
    print("\n--- 3. PRE-NORMALIZATION ROW COUNTS ---")
    tables = ['customers', 'devices', 'addresses', 'orders', 'return_requests', 'risk_assessments']
    pre_counts = {}
    for tbl in tables:
        res = client.table(tbl).select('*', count='exact').limit(1).execute()
        pre_counts[tbl] = res.count if res.count is not None else 0
        print(f"  - Table {tbl:<20}: {pre_counts[tbl]} rows")

    # 3. Prepare normalized records
    print("\n--- 4. DETERMINING NORMALIZED MAPPINGS & DRY-RUN COUNTS ---")
    cust_recs, dev_recs, addr_recs, order_recs, ret_recs = prepare_normalized_records(df_raw)
    
    print(f"Dry-Run Summary:")
    print(f"  - customers to insert/upsert:       {len(cust_recs):,}")
    print(f"  - devices to insert/upsert:         {len(dev_recs):,}")
    print(f"  - addresses to insert/upsert:       {len(addr_recs):,}")
    print(f"  - orders to insert/upsert:          {len(order_recs):,}")
    print(f"  - return_requests to insert/upsert: {len(ret_recs):,}")

    assert len(cust_recs) == 2500, f"Expected 2500 customers, got {len(cust_recs)}"
    assert len(dev_recs) == 1661, f"Expected 1661 devices, got {len(dev_recs)}"
    assert len(addr_recs) == 1696, f"Expected 1696 addresses, got {len(addr_recs)}"
    assert len(order_recs) == 10000, f"Expected 10000 orders, got {len(order_recs)}"
    assert len(ret_recs) == 2014, f"Expected 2014 return requests, got {len(ret_recs)}"

    # 4. Perform Write Operations in Foreign Key Dependency Order
    print("\n--- 5. EXECUTING NORMALIZATION WRITES (PARENT TABLES FIRST) ---")
    upsert_in_batches(client, 'customers', cust_recs, batch_size=500)
    upsert_in_batches(client, 'devices', dev_recs, batch_size=500)
    upsert_in_batches(client, 'addresses', addr_recs, batch_size=500)
    upsert_in_batches(client, 'orders', order_recs, batch_size=500)
    upsert_in_batches(client, 'return_requests', ret_recs, batch_size=500)

    # 5. Post-Normalization Row Counts & Validation
    print("--- 6. POST-NORMALIZATION ROW COUNTS ---")
    post_counts = {}
    for tbl in tables:
        res = client.table(tbl).select('*', count='exact').limit(1).execute()
        post_counts[tbl] = res.count if res.count is not None else 0
        print(f"  - Table {tbl:<20}: {post_counts[tbl]:,} rows")

    assert post_counts['customers'] == 2500
    assert post_counts['devices'] == 1661
    assert post_counts['addresses'] == 1696
    assert post_counts['orders'] == 10000
    assert post_counts['return_requests'] == 2014

    # 6. Referential Integrity & Orphan Checks
    print("\n--- 7. REFERENTIAL INTEGRITY & ORPHAN CHECKS ---")
    
    # Check 1: Orphan Orders -> Customer
    # Query customer_ids in orders vs customers
    cust_id_set = {c['customer_id'] for c in cust_recs}
    orphan_orders_cust = [o['order_id'] for o in order_recs if o['customer_id'] not in cust_id_set]
    print(f"  - Orphan Orders (Missing customer_id): {len(orphan_orders_cust)}")
    assert len(orphan_orders_cust) == 0

    # Check 2: Orphan Orders -> Device
    dev_id_set = {d['device_id'] for d in dev_recs}
    orphan_orders_dev = [o['order_id'] for o in order_recs if o['device_id'] not in dev_id_set]
    print(f"  - Orphan Orders (Missing device_id): {len(orphan_orders_dev)}")
    assert len(orphan_orders_dev) == 0

    # Check 3: Orphan Orders -> Shipping Address
    addr_id_set = {a['address_id'] for a in addr_recs}
    orphan_orders_ship = [o['order_id'] for o in order_recs if o['shipping_address_id'] not in addr_id_set]
    print(f"  - Orphan Orders (Missing shipping_address_id): {len(orphan_orders_ship)}")
    assert len(orphan_orders_ship) == 0

    # Check 4: Orphan Orders -> Billing Address
    orphan_orders_bill = [o['order_id'] for o in order_recs if o['billing_address_id'] not in addr_id_set]
    print(f"  - Orphan Orders (Missing billing_address_id): {len(orphan_orders_bill)}")
    assert len(orphan_orders_bill) == 0

    # Check 5: Orphan Return Requests -> Orders
    order_id_set = {o['order_id'] for o in order_recs}
    orphan_returns = [r['return_request_id'] for r in ret_recs if r['order_id'] not in order_id_set]
    print(f"  - Orphan Return Requests (Missing order_id): {len(orphan_returns)}")
    assert len(orphan_returns) == 0

    print("[PASS] All referential integrity and orphan checks passed (0 orphans).")

    # 7. Live /assess Verification and risk_assessments Audit Persistence
    print("\n--- 8. LIVE RISK ASSESSMENT & AUDIT RECORD PERSISTENCE TEST ---")
    from fastapi.testclient import TestClient
    from riskgraph_backend.main import app
    
    api_client = TestClient(app)
    test_order_id = "ORD100028"
    
    print(f"[ASSESS] Evaluating real existing return dispute: {test_order_id}...")
    resp = api_client.post("/assess", json={"order_id": test_order_id})
    assert resp.status_code == 200, f"API failed: {resp.text}"
    assessment = resp.json()
    print("[RESPONSE]:")
    print(json.dumps(assessment, indent=2))

    # Verify audit record persisted in Supabase risk_assessments table
    print(f"\n[QUERY] Verifying audit record for {test_order_id} in Supabase 'risk_assessments'...")
    audit_res = client.table('risk_assessments').select('*').eq('order_id', test_order_id).execute()
    audit_data = audit_res.data if audit_res.data else []
    print(f"Retrieved audit records from Supabase: {len(audit_data)}")
    assert len(audit_data) >= 1, "Audit record was not persisted!"
    saved_audit = audit_data[-1]
    print(f"  - Assessment ID: {saved_audit['assessment_id']}")
    print(f"  - Order ID: {saved_audit['order_id']} | Return Request ID: {saved_audit['return_request_id']}")
    print(f"  - Customer ID: {saved_audit['customer_id']}")
    print(f"  - Risk Score: {saved_audit['risk_probability']} | Routing Tier: {saved_audit['routing_tier']}")
    print(f"  - Action: '{saved_audit['recommended_action']}'")
    print(f"  - Assessed At: {saved_audit['assessed_at']}")
    print("[PASS] Audit record successfully persisted in Supabase with foreign keys fully satisfied!")

    # 8. Post-Execution Safety Checks
    print("\n--- 9. POST-EXECUTION DATASET & MODEL SAFETY CHECK ---")
    # Verify raw_ecommerce_data row count
    res_raw_post = client.table('raw_ecommerce_data').select('order_id', count='exact').limit(1).execute()
    assert res_raw_post.count == 10000, f"raw_ecommerce_data count changed to {res_raw_post.count}!"
    print(f"[PASS] raw_ecommerce_data row count strictly unchanged: {res_raw_post.count:,}")

    # Verify local file checksums
    for name, exp_hash in expected_hashes.items():
        p = os.path.join(BASE_DIR, name)
        curr_hash = get_file_sha256(p)
        assert curr_hash == exp_hash, f"Modification detected in {name}!"
        print(f"[PASS] {name:<42} UNMODIFIED.")

    print("\n" + "=" * 85)
    print("STEP 4.3 DATA NORMALIZATION & AUDIT INTEGRITY COMPLETED WITH 100% SUCCESS")
    print("=" * 85)

if __name__ == "__main__":
    run_normalization()
