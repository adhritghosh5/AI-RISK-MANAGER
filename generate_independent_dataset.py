"""
RISKGRAPH — AI Risk Manager
Track 02: AI Risk Manager
Problem: Defensive E-Commerce Return + Refund Abuse Detection

STEP 2.7: Independent Synthetic Raw Dataset Generator
Generates a completely independent 10,000-row e-commerce transaction dataset
(riskgraph_independent_raw_dataset.csv) with zero ID overlap, ~26% abuse prevalence
in refund cases, and realistic behavioral distributions.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set fixed seed for genuine reproducibility
np.random.seed(42)
random.seed(42)

def generate_independent_raw_dataset(output_path: str, n_rows=10000):
    cities = [
        "Mumbai, Maharashtra", "Delhi, Delhi", "Bengaluru, Karnataka",
        "Hyderabad, Telangana", "Ahmedabad, Gujarat", "Chennai, Tamil Nadu",
        "Kolkata, West Bengal", "Surat, Gujarat", "Pune, Maharashtra",
        "Jaipur, Rajasthan", "Lucknow, Uttar Pradesh", "Nagpur, Maharashtra",
        "Indore, Madhya Pradesh", "Patna, Bihar", "Bhopal, Madhya Pradesh",
        "Vadodara, Gujarat", "Ludhiana, Punjab", "Agra, Uttar Pradesh",
        "Nashik, Maharashtra", "Ranchi, Jharkhand", "Varanasi, Uttar Pradesh",
        "Chandigarh, Chandigarh", "Guwahati, Assam", "Mysuru, Karnataka",
        "Kochi, Kerala", "Vijayawada, Andhra Pradesh", "Amritsar, Punjab",
        "Thiruvananthapuram, Kerala", "Raipur, Chhattisgarh", "Coimbatore, Tamil Nadu"
    ]
    
    categories = ["Fashion", "Electronics", "Home", "Beauty", "Footwear", "Sports", "Grocery", "Accessories"]
    category_price_ranges = {
        "Fashion": (300, 3000),
        "Electronics": (400, 6500),
        "Home": (150, 2500),
        "Beauty": (100, 1500),
        "Footwear": (300, 4000),
        "Sports": (250, 4500),
        "Grocery": (50, 1000),
        "Accessories": (100, 2000)
    }
    
    payment_methods = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Wallet", "EMI"]
    payment_weights = [0.38, 0.22, 0.18, 0.10, 0.08, 0.04]
    
    channels = ["Mobile App", "Website", "Marketplace", "Social Commerce"]
    channel_weights = [0.48, 0.28, 0.16, 0.08]
    
    customer_segments = ["Budget", "Mass Market", "High Value", "Premium"]
    segment_weights = [0.40, 0.35, 0.15, 0.10]
    
    return_reasons = [
        "Size issue", "Changed mind", "Quality issue", "Product not as described",
        "Damaged product", "Wrong product", "Late delivery", "Other", "Duplicate order"
    ]
    reason_weights = [0.24, 0.19, 0.14, 0.13, 0.12, 0.10, 0.05, 0.02, 0.01]
    
    # 1. Generate Customers (~2,500 distinct customers with disjoint ID space)
    n_cust = 2500
    cust_ids = [f"CUST5{i:05d}" for i in range(n_cust)]
    
    cust_profiles = {}
    for cid in cust_ids:
        seg = random.choices(customer_segments, weights=segment_weights)[0]
        loc = random.choice(cities)
        tenure = random.randint(30, 1800)
        dev_id = f"DEV8{random.randint(1000, 9999):05d}"
        addr_id = f"ADDR9{random.randint(1000, 9999):05d}"
        cust_profiles[cid] = {
            'segment': seg, 'location': loc, 'tenure': tenure,
            'device_id': dev_id, 'address_id': addr_id,
            'is_abusive_profile': False,
            'abuse_pattern': 'none'
        }
        
    # Inject calibrated abuse syndicates targeting ~25%-27% positive prevalence in refund disputes
    # a. Serial Returners (~55 customers)
    serial_custs = random.sample(cust_ids, 55)
    for cid in serial_custs:
        cust_profiles[cid]['is_abusive_profile'] = True
        cust_profiles[cid]['abuse_pattern'] = 'serial_return_abuse'
        
    # b. Policy Exploiters (~25 customers)
    remaining = [c for c in cust_ids if not cust_profiles[c]['is_abusive_profile']]
    policy_custs = random.sample(remaining, 25)
    for cid in policy_custs:
        cust_profiles[cid]['is_abusive_profile'] = True
        cust_profiles[cid]['abuse_pattern'] = 'policy_exploitation'
        
    # c. Refund INR Abusers (~22 customers)
    remaining = [c for c in cust_ids if not cust_profiles[c]['is_abusive_profile']]
    refund_abuse_custs = random.sample(remaining, 22)
    for cid in refund_abuse_custs:
        cust_profiles[cid]['is_abusive_profile'] = True
        cust_profiles[cid]['abuse_pattern'] = 'refund_abuse'
        
    # d. Coordinated Address Rings (~8 rings)
    remaining = [c for c in cust_ids if not cust_profiles[c]['is_abusive_profile']]
    for ring_i in range(8):
        ring_custs = random.sample(remaining, 3)
        shared_addr = f"ADDR9{8000+ring_i:05d}"
        for cid in ring_custs:
            cust_profiles[cid]['address_id'] = shared_addr
            cust_profiles[cid]['is_abusive_profile'] = True
            cust_profiles[cid]['abuse_pattern'] = 'coordinated_address_abuse'
            remaining.remove(cid)
            
    # e. Coordinated Device Rings (~6 rings)
    for ring_i in range(6):
        ring_custs = random.sample(remaining, 3)
        shared_dev = f"DEV8{8000+ring_i:05d}"
        for cid in ring_custs:
            cust_profiles[cid]['device_id'] = shared_dev
            cust_profiles[cid]['is_abusive_profile'] = True
            cust_profiles[cid]['abuse_pattern'] = 'coordinated_device_abuse'
            remaining.remove(cid)
            
    # f. Account Multiplication (~8 rings)
    for ring_i in range(8):
        ring_custs = random.sample(remaining, 3)
        shared_dev = f"DEV8{9000+ring_i:05d}"
        shared_addr = f"ADDR9{9000+ring_i:05d}"
        for cid in ring_custs:
            cust_profiles[cid]['device_id'] = shared_dev
            cust_profiles[cid]['address_id'] = shared_addr
            cust_profiles[cid]['is_abusive_profile'] = True
            cust_profiles[cid]['abuse_pattern'] = 'account_multiplication'
            remaining.remove(cid)

    # 2. Generate Orders across 2024
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_delta_days = (end_date - start_date).days
    
    rows = []
    order_counter = 500001
    
    for cid, prof in cust_profiles.items():
        if prof['abuse_pattern'] == 'serial_return_abuse':
            n_orders = random.randint(4, 8)
        elif prof['abuse_pattern'] == 'account_multiplication':
            n_orders = random.randint(1, 3)
        else:
            n_orders = random.choices([1, 2, 3, 4, 5, 6, 7], weights=[0.22, 0.26, 0.22, 0.15, 0.08, 0.05, 0.02])[0]
            
        cust_order_dates = sorted([
            start_date + timedelta(days=random.randint(0, date_delta_days))
            for _ in range(n_orders)
        ])
        
        for o_date in cust_order_dates:
            cat = random.choice(categories)
            price_min, price_max = category_price_ranges[cat]
            amount = round(random.uniform(price_min, price_max), 2)
            quantity = random.choices([1, 2, 3], weights=[0.75, 0.20, 0.05])[0]
            pm = random.choices(payment_methods, weights=payment_weights)[0]
            ch = random.choices(channels, weights=channel_weights)[0]
            prod_id = f"{cat[:3].upper()}-{random.randint(1001, 1099):04d}"
            
            ship_lag = random.randint(1, 4)
            shipping_date = o_date + timedelta(days=ship_lag)
            delivery_status = "Delivered"
            
            bill_addr_id = prof['address_id'] if random.random() > 0.08 else f"ADDR9{random.randint(1000, 9999):05d}"
            
            is_return = False
            return_status = "Not Returned"
            return_req_date = None
            return_reason = None
            return_resolution = None
            refund_amount = None
            refund_date = None
            abuse_label = 0
            abuse_type = "none"
            
            if prof['abuse_pattern'] == 'serial_return_abuse':
                is_return = random.random() < 0.62
            elif prof['abuse_pattern'] in ['policy_exploitation', 'refund_abuse', 'account_multiplication', 'coordinated_address_abuse', 'coordinated_device_abuse']:
                is_return = random.random() < 0.48
            else:
                # Legitimate dispute rate ~18%
                is_return = random.random() < 0.18
                
            if is_return:
                req_lag = random.randint(1, 12)
                return_req_date = shipping_date + timedelta(days=req_lag)
                return_reason = random.choices(return_reasons, weights=reason_weights)[0]
                
                res_type = "refund" if random.random() < 0.73 else "replacement"
                return_resolution = res_type
                
                if res_type == "refund":
                    if prof['is_abusive_profile']:
                        abuse_label = 1
                        abuse_type = prof['abuse_pattern']
                    else:
                        abuse_label = 0
                        abuse_type = "none"
                        
                    if random.random() < 0.96:
                        return_status = "Refunded"
                        ref_lag = random.randint(1, 8)
                        refund_date = return_req_date + timedelta(days=ref_lag)
                        refund_amount = round(amount * random.uniform(0.75, 1.0), 2)
                    else:
                        return_status = "Return Requested"
                        refund_date = None
                        refund_amount = None
                else:
                    if prof['is_abusive_profile']:
                        abuse_label = 1
                        abuse_type = prof['abuse_pattern']
                    else:
                        abuse_label = 0
                        abuse_type = "none"
                    return_status = random.choice(["Returned", "Return Requested"])
                    refund_date = None
                    refund_amount = None
                    
            rows.append({
                'order_id': f"ORD{order_counter:06d}",
                'transaction_id': f"TXN{order_counter:06d}",
                'customer_id': cid,
                'product_id': prod_id,
                'category': cat,
                'amount': amount,
                'quantity': quantity,
                'order_date': o_date.strftime('%Y-%m-%d'),
                'payment_method': pm,
                'customer_location': prof['location'],
                'device_id': prof['device_id'],
                'channel': ch,
                'shipping_date': shipping_date.strftime('%Y-%m-%d'),
                'delivery_status': delivery_status,
                'return_status': return_status,
                'return_request_date': return_req_date.strftime('%Y-%m-%d') if return_req_date else "",
                'return_reason': return_reason if return_reason else "",
                'return_resolution': return_resolution if return_resolution else "",
                'refund_amount': refund_amount if refund_amount else "",
                'refund_date': refund_date.strftime('%Y-%m-%d') if refund_date else "",
                'customer_segment': prof['segment'],
                'customer_tenure_days': prof['tenure'],
                'shipping_address_id': prof['address_id'],
                'billing_address_id': bill_addr_id,
                'abuse_label': abuse_label,
                'abuse_type': abuse_type
            })
            order_counter += 1
            
    # Trim or pad to exactly n_rows
    if len(rows) > n_rows:
        rows = rows[:n_rows]
    elif len(rows) < n_rows:
        while len(rows) < n_rows:
            cid = random.choice(cust_ids)
            prof = cust_profiles[cid]
            o_date = start_date + timedelta(days=random.randint(0, date_delta_days))
            cat = random.choice(categories)
            amount = round(random.uniform(category_price_ranges[cat][0], category_price_ranges[cat][1]), 2)
            rows.append({
                'order_id': f"ORD{order_counter:06d}",
                'transaction_id': f"TXN{order_counter:06d}",
                'customer_id': cid,
                'product_id': f"{cat[:3].upper()}-{random.randint(1001, 1099):04d}",
                'category': cat,
                'amount': amount,
                'quantity': 1,
                'order_date': o_date.strftime('%Y-%m-%d'),
                'payment_method': random.choice(payment_methods),
                'customer_location': prof['location'],
                'device_id': prof['device_id'],
                'channel': random.choice(channels),
                'shipping_date': (o_date + timedelta(days=2)).strftime('%Y-%m-%d'),
                'delivery_status': 'Delivered',
                'return_status': 'Not Returned',
                'return_request_date': "",
                'return_reason': "",
                'return_resolution': "",
                'refund_amount': "",
                'refund_date': "",
                'customer_segment': prof['segment'],
                'customer_tenure_days': prof['tenure'],
                'shipping_address_id': prof['address_id'],
                'billing_address_id': prof['address_id'],
                'abuse_label': 0,
                'abuse_type': "none"
            })
            order_counter += 1
            
    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(output_path, index=False)
    print(f"Generated {len(df_raw)} rows in independent raw dataset: {output_path}")
    
    refund_cases = df_raw[df_raw['return_resolution'] == 'refund']
    print(f"Total refund dispute cases (Dataset B population): {len(refund_cases)}")
    print(f"Abuse distribution in refund cases:\n{refund_cases['abuse_label'].value_counts(normalize=True)}")
    print(f"Abuse types in refund cases:\n{refund_cases['abuse_type'].value_counts()}")

if __name__ == "__main__":
    out_file = r"c:\Users\adhri\Downloads\Project-RiskGraph\riskgraph_independent_raw_dataset.csv"
    generate_independent_raw_dataset(out_file)
