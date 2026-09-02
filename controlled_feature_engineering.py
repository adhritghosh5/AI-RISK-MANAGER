"""
RISKGRAPH — AI Risk Manager
Track 02: AI Risk Manager
Problem: Defensive E-Commerce Return + Refund Abuse Detection

STEP 2.8: Controlled Feature Engineering, Ablation, & Generalization Pipeline (High Performance)
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_auc_score, auc, confusion_matrix
)

def extract_extended_features_fast(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['shipping_date'] = pd.to_datetime(df['shipping_date'])
    df['return_request_date'] = pd.to_datetime(df['return_request_date'])
    df['refund_date'] = pd.to_datetime(df['refund_date'])
    
    # Pre-index events by customer, device, shipping_address, billing_address
    cust_orders = defaultdict(list)
    cust_returns = defaultdict(list)
    cust_refunds = defaultdict(list)
    
    dev_returns = defaultdict(list)
    dev_refunds = defaultdict(list)
    dev_custs = defaultdict(list) # (order_date, cust_id)
    
    ship_returns = defaultdict(list)
    ship_refunds = defaultdict(list)
    ship_custs = defaultdict(list)
    
    bill_returns = defaultdict(list)
    bill_refunds = defaultdict(list)
    bill_custs = defaultdict(list)
    
    for idx, row in df.iterrows():
        cid = row['customer_id']
        oid = row['order_id']
        odate = row['order_date']
        amt = float(row['amount'])
        dev = row['device_id']
        ship_addr = row['shipping_address_id']
        bill_addr = row['billing_address_id']
        
        cust_orders[cid].append((odate, oid, amt))
        dev_custs[dev].append((odate, cid))
        ship_custs[ship_addr].append((odate, cid))
        bill_custs[bill_addr].append((odate, cid))
        
        if pd.notna(row['return_request_date']):
            ret_date = row['return_request_date']
            cust_returns[cid].append((ret_date, oid))
            dev_returns[dev].append((ret_date, oid))
            ship_returns[ship_addr].append((ret_date, oid))
            bill_returns[bill_addr].append((ret_date, oid))
            
        if pd.notna(row['refund_date']) and row['return_status'] == 'Refunded':
            ref_date = row['refund_date']
            ref_amt = float(row['refund_amount']) if pd.notna(row['refund_amount']) and str(row['refund_amount']).strip() != '' else amt
            cust_refunds[cid].append((ref_date, oid, ref_amt))
            dev_refunds[dev].append((ref_date, oid, ref_amt))
            ship_refunds[ship_addr].append((ref_date, oid, ref_amt))
            bill_refunds[bill_addr].append((ref_date, oid, ref_amt))
            
    # Filter to prediction population: return_resolution == 'refund'
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
        
        # Prior orders
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
        p_orders_30 = [o for o in p_orders if o[0] >= t_30]
        p_returns_30 = [r for r in p_returns if r[0] >= t_30]
        p_refunds_30 = [rf for rf in p_refunds if rf[0] >= t_30]
        orders_30 = len(p_orders_30)
        returns_30 = len(p_returns_30)
        refunds_30 = len(p_refunds_30)
        spend_30 = sum(o[2] for o in p_orders_30) if orders_30 > 0 else 0.0
        refund_amt_30 = sum(rf[2] for rf in p_refunds_30) if refunds_30 > 0 else 0.0
        
        days_since_order = (t_cutoff - row['order_date']).days
        days_since_delivery = (t_cutoff - row['shipping_date']).days if pd.notna(row['shipping_date']) else 0
        is_addr_mismatch = int(curr_ship_addr != curr_bill_addr)
        
        # Device & Address Counts
        dev_prior_returns = len([r for r in dev_returns[curr_dev] if r[0] < t_cutoff and r[1] != curr_order_id])
        dev_prior_refunds = len([rf for rf in dev_refunds[curr_dev] if rf[0] < t_cutoff and rf[1] != curr_order_id])
        dev_prior_accounts = len(set(c[1] for c in dev_custs[curr_dev] if c[0] < t_cutoff))
        
        addr_prior_returns = len([r for r in ship_returns[curr_ship_addr] if r[0] < t_cutoff and r[1] != curr_order_id])
        addr_prior_refunds = len([rf for rf in ship_refunds[curr_ship_addr] if rf[0] < t_cutoff and rf[1] != curr_order_id])
        addr_prior_accounts = len(set(c[1] for c in ship_custs[curr_ship_addr] if c[0] < t_cutoff))
        
        # New Recency & Velocity (7d, 14d)
        t_7 = t_cutoff - pd.Timedelta(days=7)
        orders_7 = len([o for o in p_orders if o[0] >= t_7])
        returns_7 = len([r for r in p_returns if r[0] >= t_7])
        refunds_7 = len([rf for rf in p_refunds if rf[0] >= t_7])
        
        t_14 = t_cutoff - pd.Timedelta(days=14)
        orders_14 = len([o for o in p_orders if o[0] >= t_14])
        returns_14 = len([r for r in p_returns if r[0] >= t_14])
        refunds_14 = len([rf for rf in p_refunds if rf[0] >= t_14])
        
        days_since_last_ret = (t_cutoff - max(r[0] for r in p_returns)).days if prior_return_count > 0 else 999
        days_since_last_ref = (t_cutoff - max(rf[0] for rf in p_refunds)).days if prior_refund_count > 0 else 999
        
        return_rate_30 = (returns_30 / orders_30) if orders_30 > 0 else 0.0
        refund_rate_30 = (refunds_30 / orders_30) if orders_30 > 0 else 0.0
        
        # Monetary Intensity
        avg_prior_order_amt = (prior_spend / prior_order_count) if prior_order_count > 0 else 0.0
        avg_prior_refund_amt = (prior_refund_amount / prior_refund_count) if prior_refund_count > 0 else 0.0
        refund_to_spend_ratio = (prior_refund_amount / prior_spend) if prior_spend > 0 else 0.0
        amount_to_avg_ratio = (float(row['amount']) / avg_prior_order_amt) if avg_prior_order_amt > 0 else 1.0
        
        # Billing Address Graph Counts
        bill_prior_returns = len([r for r in bill_returns[curr_bill_addr] if r[0] < t_cutoff and r[1] != curr_order_id])
        bill_prior_refunds = len([rf for rf in bill_refunds[curr_bill_addr] if rf[0] < t_cutoff and rf[1] != curr_order_id])
        bill_prior_accounts = len(set(c[1] for c in bill_custs[curr_bill_addr] if c[0] < t_cutoff))
        
        rec = {
            'order_id': row['order_id'], 'customer_id': row['customer_id'],
            'transaction_id': row['transaction_id'], 'device_id': row['device_id'],
            'shipping_address_id': row['shipping_address_id'], 'billing_address_id': row['billing_address_id'],
            'order_date': row['order_date'].strftime('%Y-%m-%d'),
            'return_request_date': row['return_request_date'].strftime('%Y-%m-%d'),
            'abuse_label': int(row['abuse_label']), 'abuse_type': row['abuse_type'],
            
            # --- SET A: Baseline 33 ---
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
            'device_prior_return_count': int(dev_prior_returns),
            'device_prior_refund_count': int(dev_prior_refunds),
            'accounts_per_device': int(dev_prior_accounts),
            'address_prior_return_count': int(addr_prior_returns),
            'address_prior_refund_count': int(addr_prior_refunds),
            'accounts_per_shipping_address': int(addr_prior_accounts),
            'is_weekend_order': int(row['order_date'].weekday() >= 5),
            'is_weekend_return_request': int(row['return_request_date'].weekday() >= 5),
            
            # --- SET B Additions (12) ---
            'orders_last_7_days': int(orders_7), 'returns_last_7_days': int(returns_7),
            'refunds_last_7_days': int(refunds_7), 'orders_last_14_days': int(orders_14),
            'returns_last_14_days': int(returns_14), 'refunds_last_14_days': int(refunds_14),
            'days_since_last_return': int(days_since_last_ret), 'days_since_last_refund': int(days_since_last_ref),
            'return_rate_last_30_days': float(return_rate_30), 'refund_rate_last_30_days': float(refund_rate_30),
            'refund_to_spend_ratio': float(refund_to_spend_ratio), 'amount_to_avg_ratio': float(amount_to_avg_ratio),
            
            # --- SET C Additions (5) ---
            'billing_address_prior_return_count': int(bill_prior_returns),
            'billing_address_prior_refund_count': int(bill_prior_refunds),
            'accounts_per_billing_address': int(bill_prior_accounts),
            'average_prior_order_amount': float(avg_prior_order_amt),
            'average_prior_refund_amount': float(avg_prior_refund_amt)
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
    
    print("="*85)
    print("STEP 2.8: EXTRACTING CONTROLLED CANDIDATE FEATURES (FAST DICTIONARY INDEXING)")
    print("="*85)
    df_raw_a = pd.read_csv(raw_a_path)
    df_feat_a = extract_extended_features_fast(df_raw_a)
    
    df_raw_b = pd.read_csv(raw_b_path)
    df_feat_b = extract_extended_features_fast(df_raw_b)
    
    print(f"Dataset A Processed: {len(df_feat_a)} refund cases | {len(df_feat_a.columns)-10} features")
    print(f"Dataset B Processed: {len(df_feat_b)} refund cases | {len(df_feat_b.columns)-10} features")
    
    set_a_cols = [
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
        'is_weekend_order', 'is_weekend_return_request'
    ]
    
    set_b_additions = [
        'orders_last_7_days', 'returns_last_7_days', 'refunds_last_7_days',
        'orders_last_14_days', 'returns_last_14_days', 'refunds_last_14_days',
        'days_since_last_return', 'days_since_last_refund',
        'return_rate_last_30_days', 'refund_rate_last_30_days',
        'refund_to_spend_ratio', 'amount_to_avg_ratio'
    ]
    set_b_cols = set_a_cols + set_b_additions
    
    set_c_additions = [
        'billing_address_prior_return_count', 'billing_address_prior_refund_count',
        'accounts_per_billing_address', 'average_prior_order_amount',
        'average_prior_refund_amount'
    ]
    set_c_cols = set_b_cols + set_c_additions
    
    print(f"Set A Count: {len(set_a_cols)} features")
    print(f"Set B Count: {len(set_b_cols)} features (+12 Behavioural / Recency / Monetary)")
    print(f"Set C Count: {len(set_c_cols)} features (+17 Full Suite with Entity Graph)")
    print("="*85 + "\n")
    
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
    
    cat_cols = ['category', 'payment_method', 'channel', 'return_reason', 'customer_segment']
    
    feature_sets = {
        'SET A (33)': set_a_cols,
        'SET B (45)': set_b_cols,
        'SET C (50)': set_c_cols
    }
    
    model_factories = {
        'Gradient Boosting': lambda: GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42),
        'Random Forest': lambda: RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=8, min_samples_leaf=5, random_state=42),
        'HistGradientBoosting': lambda: HistGradientBoostingClassifier(class_weight='balanced', max_iter=150, min_samples_leaf=15, learning_rate=0.08, random_state=42),
        'Elastic-Net LR': lambda: LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', class_weight='balanced', max_iter=1000, random_state=42, C=0.5)
    }
    
    ablation_results = []
    
    print("="*105)
    print("3. VALIDATION SET ABLATION EXPERIMENT ACROSS FEATURE SETS")
    print("="*105)
    print(f"{'Model':<24} | {'Feature Set':<10} | {'Val Th':>6} | {'Val Prec':>8} | {'Val Rec':>8} | {'Val F1':>8} | {'Val PR-AUC':>10} | {'Val ROC':>8}")
    print("-" * 105)
    
    for fset_name, fcols in feature_sets.items():
        num_cols = [c for c in fcols if c not in cat_cols]
        
        prep_lin = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])
        
        prep_tree = ColumnTransformer([
            ('num', 'passthrough', num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])
        
        X_tr_lin = prep_lin.fit_transform(train_a[fcols])
        X_va_lin = prep_lin.transform(val_a[fcols])
        X_te_lin = prep_lin.transform(test_a[fcols])
        X_b_lin = prep_lin.transform(df_feat_b[fcols])
        
        X_tr_tree = prep_tree.fit_transform(train_a[fcols])
        X_va_tree = prep_tree.transform(val_a[fcols])
        X_te_tree = prep_tree.transform(test_a[fcols])
        X_b_tree = prep_tree.transform(df_feat_b[fcols])
        
        for m_name, m_factory in model_factories.items():
            model = m_factory()
            if m_name == 'Elastic-Net LR':
                model.fit(X_tr_lin, y_train)
                val_prob = model.predict_proba(X_va_lin)[:, 1]
                test_prob = model.predict_proba(X_te_lin)[:, 1]
                b_prob = model.predict_proba(X_b_lin)[:, 1]
            else:
                model.fit(X_tr_tree, y_train)
                val_prob = model.predict_proba(X_va_tree)[:, 1]
                test_prob = model.predict_proba(X_te_tree)[:, 1]
                b_prob = model.predict_proba(X_b_tree)[:, 1]
                
            best_th, best_f1 = find_best_val_threshold(y_val, val_prob)
            val_metrics = evaluate_predictions(y_val, val_prob, threshold=best_th)
            test_metrics = evaluate_predictions(y_test, test_prob, threshold=best_th)
            b_metrics = evaluate_predictions(y_b, b_prob, threshold=best_th)
            
            print(f"{m_name:<24} | {fset_name:<10} | {best_th:6.2f} | {val_metrics['precision']:8.4f} | {val_metrics['recall']:8.4f} | {val_metrics['f1']:8.4f} | {val_metrics['pr_auc']:10.4f} | {val_metrics['roc_auc']:8.4f}")
            
            ablation_results.append({
                'model_name': m_name, 'feature_set': fset_name, 'val_threshold': best_th,
                'val_metrics': val_metrics, 'test_metrics': test_metrics, 'b_metrics': b_metrics,
                'model_obj': model, 'encoded_features': num_cols
            })
            
    print("="*105 + "\n")
    
    # -------------------------------------------------------------
    # 4. HELD-OUT TEMPORAL TEST EVALUATION (COMPARING SET A VS B VS C)
    # -------------------------------------------------------------
    print("="*115)
    print("4. HELD-OUT TEMPORAL TEST SET PERFORMANCE (ORIGINAL TEST SET N = 219)")
    print("="*115)
    print(f"{'Model':<24} | {'Feature Set':<10} | {'Thresh':>6} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'PR-AUC':>6} | {'ROC':>6} | {'FP':>3} | {'FN':>3} | {'FP Cost':>8}")
    print("-" * 115)
    for res in ablation_results:
        tm = res['test_metrics']
        print(f"{res['model_name']:<24} | {res['feature_set']:<10} | {res['val_threshold']:6.2f} | {tm['precision']:6.4f} | {tm['recall']:6.4f} | {tm['f1']:6.4f} | {tm['pr_auc']:6.4f} | {tm['roc_auc']:6.4f} | {tm['fp']:3d} | {tm['fn']:3d} | INR {tm['fp_cost']:4.0f}")
    print("="*115 + "\n")
    
    # -------------------------------------------------------------
    # 5. INDEPENDENT DATASET B GENERALIZATION COMPARISON
    # -------------------------------------------------------------
    print("="*115)
    print("5. INDEPENDENT DATASET B GENERALIZATION PERFORMANCE (DATASET B N = 1,193)")
    print("="*115)
    print(f"{'Model':<24} | {'Feature Set':<10} | {'Thresh':>6} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'PR-AUC':>6} | {'ROC':>6} | {'FP':>3} | {'FN':>3} | {'FP Cost':>8}")
    print("-" * 115)
    for res in ablation_results:
        bm = res['b_metrics']
        print(f"{res['model_name']:<24} | {res['feature_set']:<10} | {res['val_threshold']:6.2f} | {bm['precision']:6.4f} | {bm['recall']:6.4f} | {bm['f1']:6.4f} | {bm['pr_auc']:6.4f} | {bm['roc_auc']:6.4f} | {bm['fp']:3d} | {bm['fn']:3d} | INR {bm['fp_cost']:4.0f}")
    print("="*115 + "\n")
    
    # -------------------------------------------------------------
    # 6. FEATURE IMPORTANCE & ERROR ANALYSIS (BEST IMPROVED MODEL)
    # -------------------------------------------------------------
    rf_res = [r for r in ablation_results if r['model_name'] == 'Random Forest' and r['feature_set'] == 'SET C (50)'][0]
    rf_model = rf_res['model_obj']
    
    prep_tree = ColumnTransformer([
        ('num', 'passthrough', [c for c in set_c_cols if c not in cat_cols]),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    prep_tree.fit(train_a[set_c_cols])
    encoded_c_names = (
        [c for c in set_c_cols if c not in cat_cols] + 
        list(prep_tree.named_transformers_['cat'].get_feature_names_out(cat_cols))
    )
    
    imp_df = pd.DataFrame({
        'Feature': encoded_c_names,
        'Gini_Importance': rf_model.feature_importances_
    }).sort_values(by='Gini_Importance', ascending=False).reset_index(drop=True)
    
    print("="*85)
    print("6. TOP 15 PREDICTIVE FEATURES IN IMPROVED SET C (RANDOM FOREST)")
    print("="*85)
    for i, r in imp_df.head(15).iterrows():
        print(f"  {i+1:2d}. {r['Feature']:<35} | Importance: {r['Gini_Importance']:.5f} ({r['Gini_Importance']*100:.2f}%)")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
