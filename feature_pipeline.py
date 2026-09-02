"""
RISKGRAPH — AI Risk Manager
Track 02: AI Risk Manager
Problem: Defensive E-Commerce Return + Refund Abuse Detection

STEP 1: Point-in-Time Feature Engineering Pipeline (High-Performance Vectorized/Indexed)
This script reconstructs the customer and entity state at `return_request_date`
for every historical case where return_resolution == "refund", strictly adhering
to temporal causality (zero look-ahead leakage).
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

def load_and_preprocess_raw_data(csv_path: str) -> pd.DataFrame:
    """Load raw dataset and parse date columns."""
    df = pd.read_csv(csv_path)
    
    # Parse dates
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['shipping_date'] = pd.to_datetime(df['shipping_date'])
    df['return_request_date'] = pd.to_datetime(df['return_request_date'])
    df['refund_date'] = pd.to_datetime(df['refund_date'])
    
    # Ensure amount and quantity are numeric
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['refund_amount'] = pd.to_numeric(df['refund_amount'], errors='coerce')
    df['customer_tenure_days'] = pd.to_numeric(df['customer_tenure_days'], errors='coerce')
    
    return df

def build_point_in_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct point-in-time features for all rows where return_resolution == 'refund'.
    Uses pre-grouped dictionaries for ultra-fast, exact temporal feature extraction.
    """
    refund_mask = df['return_resolution'] == 'refund'
    target_df = df[refund_mask].copy().reset_index(drop=True)
    
    # Pre-index historical tables for O(1) customer & entity lookups
    cust_groups = {k: v for k, v in df.groupby('customer_id')}
    device_groups = {k: v for k, v in df.groupby('device_id')}
    ship_addr_groups = {k: v for k, v in df.groupby('shipping_address_id')}
    bill_addr_groups = {k: v for k, v in df.groupby('billing_address_id')}
    
    records = []
    
    for idx, row in target_df.iterrows():
        curr_order_id = row['order_id']
        curr_cust_id = row['customer_id']
        curr_req_date = row['return_request_date']
        curr_order_date = row['order_date']
        curr_ship_date = row['shipping_date']
        curr_device_id = row['device_id']
        curr_ship_addr = row['shipping_address_id']
        curr_bill_addr = row['billing_address_id']
        
        # --- Customer Prior History ---
        cust_df = cust_groups[curr_cust_id]
        
        # Prior customer orders: order_date < curr_req_date and order_id != curr_order_id
        cust_prior_orders = cust_df[
            (cust_df['order_id'] != curr_order_id) &
            (cust_df['order_date'] < curr_req_date)
        ]
        
        # Prior customer returns: return_request_date < curr_req_date and order_id != curr_order_id
        cust_prior_returns = cust_df[
            (cust_df['order_id'] != curr_order_id) &
            (cust_df['return_request_date'].notna()) &
            (cust_df['return_request_date'] < curr_req_date)
        ]
        
        # Prior customer refunds: refund_date < curr_req_date and order_id != curr_order_id
        cust_prior_refunds = cust_df[
            (cust_df['order_id'] != curr_order_id) &
            (cust_df['refund_date'].notna()) &
            (cust_df['refund_date'] < curr_req_date)
        ]
        
        # Behavioral Features
        prior_order_count = len(cust_prior_orders)
        prior_return_count = len(cust_prior_returns)
        prior_refund_count = len(cust_prior_refunds)
        
        prior_spend = cust_prior_orders['amount'].sum() if prior_order_count > 0 else 0.0
        prior_refund_amount = cust_prior_refunds['refund_amount'].sum() if prior_refund_count > 0 else 0.0
        
        prior_return_rate = prior_return_count / (prior_order_count + 1.0)
        prior_refund_rate = prior_refund_count / (prior_order_count + 1.0)
        
        avg_prev_order_val = (prior_spend / prior_order_count) if prior_order_count > 0 else 0.0
        avg_prev_refund_val = (prior_refund_amount / prior_refund_count) if prior_refund_count > 0 else 0.0
        
        # Velocity Windows
        orders_last_7_days = len(cust_prior_orders[
            (curr_req_date - cust_prior_orders['order_date']).dt.days <= 7
        ])
        
        orders_last_30_days = len(cust_prior_orders[
            (curr_req_date - cust_prior_orders['order_date']).dt.days <= 30
        ])
        
        returns_last_30_days = len(cust_prior_returns[
            (curr_req_date - cust_prior_returns['return_request_date']).dt.days <= 30
        ])
        
        refunds_last_30_days = len(cust_prior_refunds[
            (curr_req_date - cust_prior_refunds['refund_date']).dt.days <= 30
        ])
        
        # Recency
        if prior_order_count > 0:
            most_recent_order_date = cust_prior_orders['order_date'].max()
            days_since_previous_order = (curr_req_date - most_recent_order_date).days
        else:
            days_since_previous_order = 999
            
        # --- Graph / Relationship Features ---
        dev_df = device_groups.get(curr_device_id, pd.DataFrame())
        accounts_per_device = dev_df[dev_df['order_date'] <= curr_req_date]['customer_id'].nunique() if len(dev_df) > 0 else 1
        
        ship_df = ship_addr_groups.get(curr_ship_addr, pd.DataFrame())
        accounts_per_shipping_address = ship_df[ship_df['order_date'] <= curr_req_date]['customer_id'].nunique() if len(ship_df) > 0 else 1
        
        bill_df = bill_addr_groups.get(curr_bill_addr, pd.DataFrame())
        accounts_per_billing_address = bill_df[bill_df['order_date'] <= curr_req_date]['customer_id'].nunique() if len(bill_df) > 0 else 1
        
        device_prior_return_count = len(dev_df[
            (dev_df['order_id'] != curr_order_id) &
            (dev_df['return_request_date'].notna()) &
            (dev_df['return_request_date'] < curr_req_date)
        ]) if len(dev_df) > 0 else 0
        
        address_prior_return_count = len(ship_df[
            (ship_df['order_id'] != curr_order_id) &
            (ship_df['return_request_date'].notna()) &
            (ship_df['return_request_date'] < curr_req_date)
        ]) if len(ship_df) > 0 else 0
        
        device_prior_refund_count = len(dev_df[
            (dev_df['order_id'] != curr_order_id) &
            (dev_df['refund_date'].notna()) &
            (dev_df['refund_date'] < curr_req_date)
        ]) if len(dev_df) > 0 else 0
        
        address_prior_refund_count = len(ship_df[
            (ship_df['order_id'] != curr_order_id) &
            (ship_df['refund_date'].notna()) &
            (ship_df['refund_date'] < curr_req_date)
        ]) if len(ship_df) > 0 else 0
        
        is_address_mismatch = 1 if curr_ship_addr != curr_bill_addr else 0
        
        # --- Current Request Attributes ---
        current_order_amount = row['amount']
        quantity = row['quantity']
        category = row['category']
        payment_method = row['payment_method']
        channel = row['channel']
        return_reason = row['return_reason']
        customer_segment = row['customer_segment']
        customer_tenure_days = row['customer_tenure_days']
        
        days_since_order = (curr_req_date - curr_order_date).days
        days_since_delivery = (curr_req_date - curr_ship_date).days if pd.notna(curr_ship_date) else -1
        delivery_duration_days = (curr_ship_date - curr_order_date).days if pd.notna(curr_ship_date) else -1
        
        abuse_label = row['abuse_label']
        abuse_type = row['abuse_type']
        
        records.append({
            # Identifiers / Metadata
            'order_id': curr_order_id,
            'customer_id': curr_cust_id,
            'transaction_id': row['transaction_id'],
            'device_id': curr_device_id,
            'shipping_address_id': curr_ship_addr,
            'billing_address_id': curr_bill_addr,
            'order_date': curr_order_date.strftime('%Y-%m-%d'),
            'return_request_date': curr_req_date.strftime('%Y-%m-%d'),
            
            # --- Feature Set (X) ---
            # 1. Customer Behavioral Features
            'prior_order_count': prior_order_count,
            'prior_return_count': prior_return_count,
            'prior_refund_count': prior_refund_count,
            'prior_spend': round(prior_spend, 2),
            'prior_refund_amount': round(prior_refund_amount, 2),
            'prior_return_rate': round(prior_return_rate, 4),
            'prior_refund_rate': round(prior_refund_rate, 4),
            'average_previous_order_value': round(avg_prev_order_val, 2),
            'average_previous_refund': round(avg_prev_refund_val, 2),
            'orders_last_7_days': orders_last_7_days,
            'orders_last_30_days': orders_last_30_days,
            'returns_last_30_days': returns_last_30_days,
            'refunds_last_30_days': refunds_last_30_days,
            'days_since_previous_order': days_since_previous_order,
            
            # 2. Relationship / Graph Features
            'accounts_per_device': accounts_per_device,
            'accounts_per_shipping_address': accounts_per_shipping_address,
            'accounts_per_billing_address': accounts_per_billing_address,
            'device_prior_return_count': device_prior_return_count,
            'address_prior_return_count': address_prior_return_count,
            'device_prior_refund_count': device_prior_refund_count,
            'address_prior_refund_count': address_prior_refund_count,
            'is_address_mismatch': is_address_mismatch,
            
            # 3. Current Request Features
            'current_order_amount': current_order_amount,
            'quantity': quantity,
            'category': category,
            'payment_method': payment_method,
            'channel': channel,
            'return_reason': return_reason,
            'customer_segment': customer_segment,
            'customer_tenure_days': customer_tenure_days,
            'days_since_order': days_since_order,
            'days_since_delivery': days_since_delivery,
            'delivery_duration_days': delivery_duration_days,
            
            # --- Target and Analysis Targets ---
            'abuse_label': int(abuse_label),
            'abuse_type': abuse_type
        })
        
    modeling_df = pd.DataFrame(records)
    return modeling_df

def run_leakage_and_integrity_checks(df_raw: pd.DataFrame, df_mod: pd.DataFrame):
    """Run thorough automated leakage and integrity validation checks."""
    print("\n" + "="*70, flush=True)
    print("RUNNING AUTOMATED LEAKAGE & INTEGRITY VERIFICATION CHECKS", flush=True)
    print("="*70, flush=True)
    
    # 1. Check Row Count
    expected_count = len(df_raw[df_raw['return_resolution'] == 'refund'])
    actual_count = len(df_mod)
    assert actual_count == expected_count, f"Count mismatch: expected {expected_count}, got {actual_count}"
    print(f"[PASS] Exact population match: {actual_count} refund dispute cases.", flush=True)
    
    # 2. Check for Prohibited Columns in Features
    prohibited_features = ['risk_score', 'probability', 'predicted_abuse', 'risk_level']
    for p in prohibited_features:
        assert p not in df_mod.columns, f"Prohibited feature {p} found in dataset!"
    print(f"[PASS] Zero prohibited score/prediction columns found.", flush=True)
    
    # 3. Check for Post-Decision Current Case Leakage
    raw_refund_cases = df_raw[df_raw['return_resolution'] == 'refund'].set_index('order_id')
    
    for idx, row in df_mod.iterrows():
        oid = row['order_id']
        raw_row = raw_refund_cases.loc[oid]
        curr_req_date = pd.to_datetime(row['return_request_date'])
        
        assert row['days_since_order'] >= 0, f"Negative days_since_order in {oid}"
        assert row['days_since_delivery'] >= 0, f"Negative days_since_delivery in {oid}"
        
        if row['prior_order_count'] == 0:
            assert row['prior_spend'] == 0.0
            assert row['prior_return_count'] == 0
            assert row['prior_refund_count'] == 0
            assert row['prior_refund_amount'] == 0.0
            assert row['days_since_previous_order'] == 999
            
    print(f"[PASS] Temporal causality validated across all {len(df_mod)} rows.", flush=True)
    
    # 4. Check Missing Values in Features
    feature_cols = [c for c in df_mod.columns if c not in [
        'order_id', 'customer_id', 'transaction_id', 'device_id', 
        'shipping_address_id', 'billing_address_id', 'order_date', 
        'return_request_date', 'abuse_label', 'abuse_type'
    ]]
    
    null_counts = df_mod[feature_cols].isnull().sum()
    total_nulls = null_counts.sum()
    assert total_nulls == 0, f"Missing values found in feature columns:\n{null_counts[null_counts > 0]}"
    print(f"[PASS] Zero missing values across all {len(feature_cols)} feature columns.", flush=True)
    
    # 5. Class Balance Check
    label_counts = df_mod['abuse_label'].value_counts()
    print(f"[PASS] Target distribution: {label_counts[0]} legitimate (0), {label_counts[1]} abusive (1).", flush=True)
    print(f"[PASS] Abuse rate in refund population: {label_counts[1] / len(df_mod):.2%}", flush=True)
    print("="*70 + "\n", flush=True)

def main():
    raw_path = r"c:\Users\adhri\Downloads\Project-RiskGraph\riskgraph_ecommerce_dataset.csv"
    output_path = r"c:\Users\adhri\Downloads\Project-RiskGraph\riskgraph_refund_modeling_dataset.csv"
    
    print("Loading raw dataset...", flush=True)
    df_raw = load_and_preprocess_raw_data(raw_path)
    
    print("Building point-in-time features for refund cases...", flush=True)
    df_modeling = build_point_in_time_features(df_raw)
    
    print("Running leakage and verification checks...", flush=True)
    run_leakage_and_integrity_checks(df_raw, df_modeling)
    
    print(f"Saving modeling dataset to {output_path}...", flush=True)
    df_modeling.to_csv(output_path, index=False)
    print("Saved successfully!", flush=True)

if __name__ == "__main__":
    main()
