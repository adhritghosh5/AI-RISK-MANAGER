"""
RISKGRAPH — AI Risk Manager
Track 02: AI Risk Manager
Problem: Defensive E-Commerce Return + Refund Abuse Detection

STEP 3.3: Hybrid ML + Entity Graph Risk Model Benchmark
Controlled Models:
- Model A: Baseline 45 Features (Step 2.8 Set B)
- Model B: 45 Features + Device Graph (48 features)
- Model C: 45 Features + Full Entity Graph (Device, Shipping Address, Billing Address, Cluster Size - 54 features)

Evaluates strictly across:
1. Dataset A Chronological Train (70%) - Fitting
2. Dataset A Validation (15%) - Threshold Selection
3. Dataset A Temporal Test (15%) - Held-Out Temporal Evaluation
4. Independent Dataset B (10,000 rows / 1,193 refund cases) - Out-of-Distribution Generalization Test
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# Ensure utf-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_auc_score, auc, confusion_matrix
)

from entity_graph_analysis import DefensiveEntityGraphEngine

def extract_hybrid_dataset_features(raw_csv_path: str) -> pd.DataFrame:
    """
    Extracts all 45 baseline features + all point-in-time entity graph signals
    strictly before return_request_date.
    """
    engine = DefensiveEntityGraphEngine(raw_csv_path)
    df = engine.df_raw
    
    # Pre-index events for fast point-in-time calculation
    cust_orders = defaultdict(list)
    cust_returns = defaultdict(list)
    cust_refunds = defaultdict(list)
    
    for idx, row in df.iterrows():
        cid = row['customer_id']
        oid = row['order_id']
        odate = row['order_date']
        amt = float(row['amount'])
        
        cust_orders[cid].append((odate, oid, amt))
        if pd.notna(row['return_request_date']):
            cust_returns[cid].append((row['return_request_date'], oid))
        if pd.notna(row['refund_date']) and row['return_status'] == 'Refunded':
            ref_amt = float(row['refund_amount']) if pd.notna(row['refund_amount']) and str(row['refund_amount']).strip() != '' else amt
            cust_refunds[cid].append((row['refund_date'], oid, ref_amt))
            
    refund_mask = (df['return_resolution'] == 'refund') & (df['return_request_date'].notna())
    cases = df[refund_mask].copy().sort_values(by=['return_request_date', 'order_date', 'order_id']).reset_index(drop=True)
    
    records = []
    for idx, row in cases.iterrows():
        t_cutoff = row['return_request_date']
        curr_cust = row['customer_id']
        curr_dev = row['device_id']
        curr_ship_addr = row['shipping_address_id']
        curr_bill_addr = row['billing_address_id']
        curr_order_id = row['order_id']
        
        # 1. Point-in-time Customer Behavioral features
        p_orders = [o for o in cust_orders[curr_cust] if o[0] < t_cutoff and o[1] != curr_order_id]
        p_returns = [r for r in cust_returns[curr_cust] if r[0] < t_cutoff and r[1] != curr_order_id]
        p_refunds = [rf for rf in cust_refunds[curr_cust] if rf[0] < t_cutoff and rf[1] != curr_order_id]
        
        prior_order_count = len(p_orders)
        prior_return_count = len(p_returns)
        prior_refund_count = len(p_refunds)
        prior_spend = sum(o[2] for o in p_orders) if prior_order_count > 0 else 0.0
        prior_refund_amount = sum(rf[2] for rf in p_refunds) if prior_refund_count > 0 else 0.0
        prior_return_rate = (prior_return_count / prior_order_count) if prior_order_count > 0 else 0.0
        prior_refund_rate = (prior_refund_count / prior_order_count) if prior_order_count > 0 else 0.0
        
        if prior_order_count > 0:
            last_order_date = max(o[0] for o in p_orders)
            days_since_prev_order = (row['order_date'] - last_order_date).days
            avg_prev_order_val = prior_spend / prior_order_count
        else:
            days_since_prev_order = 999
            avg_prev_order_val = 0.0
            
        t_30 = t_cutoff - pd.Timedelta(days=30)
        t_14 = t_cutoff - pd.Timedelta(days=14)
        t_7 = t_cutoff - pd.Timedelta(days=7)
        
        orders_30 = len([o for o in p_orders if o[0] >= t_30])
        returns_30 = len([r for r in p_returns if r[0] >= t_30])
        refunds_30 = len([rf for rf in p_refunds if rf[0] >= t_30])
        spend_30 = sum(o[2] for o in p_orders if o[0] >= t_30) if orders_30 > 0 else 0.0
        refund_amt_30 = sum(rf[2] for rf in p_refunds if rf[0] >= t_30) if refunds_30 > 0 else 0.0
        
        orders_14 = len([o for o in p_orders if o[0] >= t_14])
        returns_14 = len([r for r in p_returns if r[0] >= t_14])
        refunds_14 = len([rf for rf in p_refunds if rf[0] >= t_14])
        
        orders_7 = len([o for o in p_orders if o[0] >= t_7])
        returns_7 = len([r for r in p_returns if r[0] >= t_7])
        refunds_7 = len([rf for rf in p_refunds if rf[0] >= t_7])
        
        days_since_last_ret = (t_cutoff - max(r[0] for r in p_returns)).days if prior_return_count > 0 else 999
        days_since_last_ref = (t_cutoff - max(rf[0] for rf in p_refunds)).days if prior_refund_count > 0 else 999
        
        return_rate_30 = (returns_30 / orders_30) if orders_30 > 0 else 0.0
        refund_rate_30 = (refunds_30 / orders_30) if orders_30 > 0 else 0.0
        
        refund_to_spend_ratio = (prior_refund_amount / prior_spend) if prior_spend > 0 else 0.0
        amount_to_avg_ratio = (float(row['amount']) / (prior_spend / prior_order_count)) if prior_order_count > 0 else 1.0
        
        days_since_order = (t_cutoff - row['order_date']).days
        days_since_delivery = (t_cutoff - row['shipping_date']).days if pd.notna(row['shipping_date']) else 0
        is_addr_mismatch = int(curr_ship_addr != curr_bill_addr)
        
        # 2. Point-in-time Entity Graph Resolution via Step 3.2 engine
        graph_data = engine.resolve_entity_neighborhood(
            customer_id=curr_cust,
            device_id=curr_dev,
            shipping_address_id=curr_ship_addr,
            billing_address_id=curr_bill_addr,
            cutoff_timestamp=t_cutoff,
            current_order_id=curr_order_id
        )
        
        rec = {
            # Metadata
            'order_id': row['order_id'], 'customer_id': row['customer_id'],
            'transaction_id': row['transaction_id'], 'device_id': row['device_id'],
            'shipping_address_id': row['shipping_address_id'], 'billing_address_id': row['billing_address_id'],
            'order_date': row['order_date'].strftime('%Y-%m-%d'),
            'return_request_date': row['return_request_date'].strftime('%Y-%m-%d'),
            'abuse_label': int(row['abuse_label']), 'abuse_type': row['abuse_type'],
            
            # --- MODEL A (45 Baseline Features) ---
            'category': row['category'], 'payment_method': row['payment_method'],
            'channel': row['channel'], 'return_reason': row['return_reason'],
            'customer_segment': row['customer_segment'],
            'current_order_amount': float(row['amount']), 'current_order_quantity': int(row['quantity']),
            'customer_tenure_days': int(row['customer_tenure_days']),
            'days_since_order': int(days_since_order), 'days_since_delivery': int(days_since_delivery),
            'is_address_mismatch': int(is_addr_mismatch),
            'prior_order_count': int(prior_order_count), 'prior_return_count': int(prior_return_count),
            'prior_refund_count': int(prior_refund_count), 'prior_spend': float(prior_spend),
            'prior_refund_amount': float(prior_refund_amount), 'prior_return_rate': float(prior_return_rate),
            'prior_refund_rate': float(prior_refund_rate), 'days_since_previous_order': int(days_since_prev_order),
            'average_previous_order_value': float(avg_prev_order_val),
            'orders_last_30_days': int(orders_30), 'returns_last_30_days': int(returns_30),
            'refunds_last_30_days': int(refunds_30), 'spend_last_30_days': float(spend_30),
            'refund_amount_last_30_days': float(refund_amt_30),
            'device_prior_return_count': int(graph_data['device_prior_return_count']),
            'device_prior_refund_count': int(graph_data['device_prior_refund_count']),
            'accounts_per_device': int(graph_data['accounts_per_device']),
            'address_prior_return_count': int(graph_data['address_prior_return_count']),
            'address_prior_refund_count': int(graph_data['address_prior_refund_count']),
            'accounts_per_shipping_address': int(graph_data['accounts_per_shipping_address']),
            'is_weekend_order': int(row['order_date'].weekday() >= 5),
            'is_weekend_return_request': int(row['return_request_date'].weekday() >= 5),
            'orders_last_7_days': int(orders_7), 'returns_last_7_days': int(returns_7),
            'refunds_last_7_days': int(refunds_7), 'orders_last_14_days': int(orders_14),
            'returns_last_14_days': int(returns_14), 'refunds_last_14_days': int(refunds_14),
            'days_since_last_return': int(days_since_last_ret), 'days_since_last_refund': int(days_since_last_ref),
            'return_rate_last_30_days': float(return_rate_30), 'refund_rate_last_30_days': float(refund_rate_30),
            'refund_to_spend_ratio': float(refund_to_spend_ratio), 'amount_to_avg_ratio': float(amount_to_avg_ratio),
            
            # --- MODEL B: Additional Device Graph Features ---
            'device_prior_order_count': int(graph_data['device_prior_order_count']),
            'device_prior_refund_amount': float(graph_data['device_prior_refund_amount']),
            'device_distinct_accounts': len(graph_data['device_other_accounts']),
            
            # --- MODEL C: Additional Address & Cluster Graph Features ---
            'address_prior_order_count': int(graph_data['address_prior_order_count']),
            'address_prior_refund_amount': float(graph_data['address_prior_refund_amount']),
            'billing_address_prior_order_count': int(graph_data['billing_address_prior_order_count']),
            'billing_address_prior_return_count': int(graph_data['billing_address_prior_return_count']),
            'billing_address_prior_refund_count': int(graph_data['billing_address_prior_refund_count']),
            'accounts_per_billing_address': int(graph_data['accounts_per_billing_address']),
            'total_linked_external_accounts': int(graph_data['total_linked_external_accounts'])
        }
        records.append(rec)
        
    return pd.DataFrame(records)

def evaluate_predictions(y_true, y_prob, threshold=0.50, fp_cost_per_case=50.0):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    p_c, r_c, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(r_c, p_c)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    fp_cost = fp * fp_cost_per_case
    cost_1k = (fp / len(y_true)) * 1000 * fp_cost_per_case
    
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'pr_auc': pr_auc, 'roc_auc': roc_auc, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'fp_cost': fp_cost, 'cost_1k': cost_1k
    }

def find_best_val_threshold(y_val, val_probs):
    best_th = 0.50
    best_f1 = -1.0
    for th in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
        pred = (val_probs >= th).astype(int)
        f1 = f1_score(y_val, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th, best_f1

def main():
    raw_a_path = r"c:\Users\adhri\Downloads\Project-RiskGraph\riskgraph_ecommerce_dataset.csv"
    raw_b_path = r"c:\Users\adhri\Downloads\Project-RiskGraph\riskgraph_independent_raw_dataset.csv"
    
    print("="*95)
    print("STEP 3.3: HYBRID ML + ENTITY GRAPH MODEL BENCHMARK EXPERIMENT")
    print("="*95)
    
    df_feat_a = extract_hybrid_dataset_features(raw_a_path)
    df_feat_b = extract_hybrid_dataset_features(raw_b_path)
    
    print(f"Dataset A Processed: {len(df_feat_a)} cases")
    print(f"Dataset B Processed: {len(df_feat_b)} cases")
    
    cat_cols = ['category', 'payment_method', 'channel', 'return_reason', 'customer_segment']
    
    # Feature Set Definitions
    model_a_cols = [
        'category', 'payment_method', 'channel', 'return_reason', 'customer_segment',
        'current_order_amount', 'current_order_quantity', 'customer_tenure_days',
        'days_since_order', 'days_since_delivery', 'is_address_mismatch',
        'prior_order_count', 'prior_return_count', 'prior_refund_count',
        'prior_spend', 'prior_refund_amount', 'prior_return_rate', 'prior_refund_rate',
        'days_since_previous_order', 'average_previous_order_value',
        'orders_last_30_days', 'returns_last_30_days', 'refunds_last_30_days',
        'spend_last_30_days', 'refund_amount_last_30_days',
        'device_prior_return_count', 'device_prior_refund_count', 'accounts_per_device',
        'address_prior_return_count', 'address_prior_refund_count', 'accounts_per_shipping_address',
        'is_weekend_order', 'is_weekend_return_request',
        'orders_last_7_days', 'returns_last_7_days', 'refunds_last_7_days',
        'orders_last_14_days', 'returns_last_14_days', 'refunds_last_14_days',
        'days_since_last_return', 'days_since_last_refund',
        'return_rate_last_30_days', 'refund_rate_last_30_days',
        'refund_to_spend_ratio', 'amount_to_avg_ratio'
    ]
    
    model_b_cols = model_a_cols + [
        'device_prior_order_count', 'device_prior_refund_amount', 'device_distinct_accounts'
    ]
    
    model_c_cols = model_b_cols + [
        'address_prior_order_count', 'address_prior_refund_amount',
        'billing_address_prior_order_count', 'billing_address_prior_return_count',
        'billing_address_prior_refund_count', 'accounts_per_billing_address',
        'total_linked_external_accounts'
    ]
    
    print(f"Model A Feature Count: {len(model_a_cols)} features (Baseline 45)")
    print(f"Model B Feature Count: {len(model_b_cols)} features (45 + Device Graph)")
    print(f"Model C Feature Count: {len(model_c_cols)} features (45 + Full Entity Graph)")
    print("="*95 + "\n")
    
    # Chronological Split on Dataset A (70% Train, 15% Val, 15% Test)
    df_feat_a['return_request_date'] = pd.to_datetime(df_feat_a['return_request_date'])
    df_feat_a['order_date'] = pd.to_datetime(df_feat_a['order_date'])
    df_feat_a = df_feat_a.sort_values(by=['return_request_date', 'order_date', 'order_id']).reset_index(drop=True)
    
    n_a = len(df_feat_a)
    train_a = df_feat_a.iloc[:int(n_a*0.70)].copy().reset_index(drop=True)
    val_a = df_feat_a.iloc[int(n_a*0.70):int(n_a*0.85)].copy().reset_index(drop=True)
    test_a = df_feat_a.iloc[int(n_a*0.85):].copy().reset_index(drop=True)
    
    y_train = train_a['abuse_label'].values
    y_val = val_a['abuse_label'].values
    y_test = test_a['abuse_label'].values
    y_b = df_feat_b['abuse_label'].values
    
    models = {
        'Model A (Baseline 45)': model_a_cols,
        'Model B (45 + Device Graph)': model_b_cols,
        'Model C (45 + Full Graph)': model_c_cols
    }
    
    results = []
    fitted_models = {}
    
    print("="*105)
    print("1. VALIDATION SET BENCHMARK & THRESHOLD TUNING")
    print("="*105)
    print(f"{'Model Architecture':<30} | {'Features':>8} | {'Val Th':>6} | {'Val Prec':>8} | {'Val Rec':>8} | {'Val F1':>8} | {'Val PR-AUC':>10} | {'Val ROC':>8}")
    print("-" * 105)
    
    for m_name, fcols in models.items():
        num_cols = [c for c in fcols if c not in cat_cols]
        prep = ColumnTransformer([
            ('num', 'passthrough', num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])
        
        # Fit ONLY on TRAIN
        X_tr = prep.fit_transform(train_a[fcols])
        X_va = prep.transform(val_a[fcols])
        X_te = prep.transform(test_a[fcols])
        X_b = prep.transform(df_feat_b[fcols])
        
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        clf.fit(X_tr, y_train)
        
        val_prob = clf.predict_proba(X_va)[:, 1]
        test_prob = clf.predict_proba(X_te)[:, 1]
        b_prob = clf.predict_proba(X_b)[:, 1]
        
        best_th, best_f1 = find_best_val_threshold(y_val, val_prob)
        val_m = evaluate_predictions(y_val, val_prob, threshold=best_th)
        test_m = evaluate_predictions(y_test, test_prob, threshold=best_th)
        b_m = evaluate_predictions(y_b, b_prob, threshold=best_th)
        
        print(f"{m_name:<30} | {len(fcols):8d} | {best_th:6.2f} | {val_m['precision']:8.4f} | {val_m['recall']:8.4f} | {val_m['f1']:8.4f} | {val_m['pr_auc']:10.4f} | {val_m['roc_auc']:8.4f}")
        
        results.append({
            'model_name': m_name, 'n_features': len(fcols), 'val_th': best_th,
            'val_m': val_m, 'test_m': test_m, 'b_m': b_m,
            'test_prob': test_prob, 'b_prob': b_prob,
            'clf': clf, 'prep': prep, 'num_cols': num_cols, 'fcols': fcols
        })
        fitted_models[m_name] = clf
        
    print("="*105 + "\n")
    
    # -------------------------------------------------------------
    # 2. TEMPORAL TEST SET COMPARISON
    # -------------------------------------------------------------
    print("="*120)
    print("2. HELD-OUT TEMPORAL TEST SET BENCHMARK (N = 219)")
    print("="*120)
    print(f"{'Model Architecture':<30} | {'Thresh':>6} | {'Acc':>6} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'PR-AUC':>6} | {'ROC':>6} | {'FP':>3} | {'FN':>3} | {'FP Cost':>8}")
    print("-" * 120)
    
    base_test = results[0]['test_m']
    for r in results:
        tm = r['test_m']
        print(f"{r['model_name']:<30} | {r['val_th']:6.2f} | {tm['accuracy']:6.4f} | {tm['precision']:6.4f} | {tm['recall']:6.4f} | {tm['f1']:6.4f} | {tm['pr_auc']:6.4f} | {tm['roc_auc']:6.4f} | {tm['fp']:3d} | {tm['fn']:3d} | INR {tm['fp_cost']:4.0f}")
        if r['model_name'] != 'Model A (Baseline 45)':
            d_pr = tm['pr_auc'] - base_test['pr_auc']
            d_f1 = tm['f1'] - base_test['f1']
            d_prec = tm['precision'] - base_test['precision']
            d_rec = tm['recall'] - base_test['recall']
            d_cost = tm['fp_cost'] - base_test['fp_cost']
            print(f"  --> Delta vs Model A: PR-AUC: {d_pr:+6.4f} | F1: {d_f1:+6.4f} | Prec: {d_prec:+6.4f} | Rec: {d_rec:+6.4f} | FP Cost: INR {d_cost:+4.0f}")
            print("-" * 120)
            
    print("="*120 + "\n")
    
    # -------------------------------------------------------------
    # 3. INDEPENDENT DATASET B GENERALIZATION BENCHMARK
    # -------------------------------------------------------------
    print("="*120)
    print("3. INDEPENDENT DATASET B GENERALIZATION BENCHMARK (N = 1,193)")
    print("="*120)
    print(f"{'Model Architecture':<30} | {'Thresh':>6} | {'Acc':>6} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'PR-AUC':>6} | {'ROC':>6} | {'FP':>3} | {'FN':>3} | {'FP Cost':>8}")
    print("-" * 120)
    
    base_b = results[0]['b_m']
    for r in results:
        bm = r['b_m']
        print(f"{r['model_name']:<30} | {r['val_th']:6.2f} | {bm['accuracy']:6.4f} | {bm['precision']:6.4f} | {bm['recall']:6.4f} | {bm['f1']:6.4f} | {bm['pr_auc']:6.4f} | {bm['roc_auc']:6.4f} | {bm['fp']:3d} | {bm['fn']:3d} | INR {bm['fp_cost']:4.0f}")
        if r['model_name'] != 'Model A (Baseline 45)':
            d_pr = bm['pr_auc'] - base_b['pr_auc']
            d_f1 = bm['f1'] - base_b['f1']
            d_prec = bm['precision'] - base_b['precision']
            d_rec = bm['recall'] - base_b['recall']
            d_cost = bm['fp_cost'] - base_b['fp_cost']
            print(f"  --> Delta vs Model A on Indep Dataset B: PR-AUC: {d_pr:+6.4f} | F1: {d_f1:+6.4f} | Prec: {d_prec:+6.4f} | Rec: {d_rec:+6.4f} | FP Cost: INR {d_cost:+4.0f}")
            print("-" * 120)
            
    print("="*120 + "\n")
    
    # -------------------------------------------------------------
    # 4. FEATURE IMPORTANCE (MODEL B & MODEL C)
    # -------------------------------------------------------------
    for r in results[1:]:
        clf = r['clf']
        fcols = r['fcols']
        num_cols = r['num_cols']
        encoded_names = num_cols + list(r['prep'].named_transformers_['cat'].get_feature_names_out(cat_cols))
        
        imp_df = pd.DataFrame({
            'Feature': encoded_names,
            'Importance': clf.feature_importances_
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        print(f"TOP 10 FEATURES IN {r['model_name'].upper()}:")
        for i, row in imp_df.head(10).iterrows():
            print(f"  {i+1:2d}. {row['Feature']:<35} | Importance: {row['Importance']:.5f} ({row['Importance']*100:.2f}%)")
        print("\n")
        
    # -------------------------------------------------------------
    # 5. ERROR ANALYSIS: MODEL A VS MODEL B
    # -------------------------------------------------------------
    test_eval = test_a.copy()
    test_eval['prob_a'] = results[0]['test_prob']
    test_eval['pred_a'] = (test_eval['prob_a'] >= results[0]['val_th']).astype(int)
    
    test_eval['prob_b'] = results[1]['test_prob']
    test_eval['pred_b'] = (test_eval['prob_b'] >= results[1]['val_th']).astype(int)
    
    # Cases fixed by Model B (where Model A missed abuse, but Model B caught it)
    fixed_fn = test_eval[(test_eval['abuse_label'] == 1) & (test_eval['pred_a'] == 0) & (test_eval['pred_b'] == 1)]
    
    # Cases where Model B avoided False Positive (Model A flagged good customer, Model B cleared them)
    fixed_fp = test_eval[(test_eval['abuse_label'] == 0) & (test_eval['pred_a'] == 1) & (test_eval['pred_b'] == 0)]
    
    print("="*85)
    print("5. ERROR ANALYSIS: IMPACT OF GRAPH ENHANCEMENT")
    print("="*85)
    print(f"False Negatives Fixed by Model B (Abuse Caught via Device Graph): {len(fixed_fn)}")
    print(f"False Positives Fixed by Model B (Legitimate Customers Cleared):  {len(fixed_fp)}")
    
    if len(fixed_fn) > 0:
        ex = fixed_fn.iloc[0]
        print(f"\nExample Fixed FN [Order {ex['order_id']}]:")
        print(f"  Cust: {ex['customer_id']} | Amount: ₹{ex['current_order_amount']} | Reason: {ex['return_reason']}")
        print(f"  Model A Score: {ex['prob_a']:.4f} (Missed) --> Model B Score: {ex['prob_b']:.4f} (Caught!)")
        print(f"  Device Prior Refunds: {ex['device_prior_refund_count']} | Accounts on Device: {ex['accounts_per_device']}")
        
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
