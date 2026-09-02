"""
RISKGRAPH — AI Risk Manager
Track 02: AI Risk Manager
Problem: Defensive E-Commerce Return + Refund Abuse Detection

STEP 2.7: Independent Generalization Test Execution
- Generates point-in-time features for Dataset B using feature_pipeline logic
- Enforces strict zero-leakage, zero-retraining freeze on models and preprocessors
- Evaluates frozen models with original validation thresholds on Dataset B
- Compares Dataset A (Test) vs Dataset B performance
- Produces 10-bin calibration/reliability analysis
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure utf-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_auc_score, auc, confusion_matrix
)
from scipy.special import expit

from feature_pipeline import load_and_preprocess_raw_data, build_point_in_time_features

def run_independent_generalization_experiment():
    raw_a_path = r"c:\Users\adhri\Downloads\Project-RiskGraph\riskgraph_ecommerce_dataset.csv"
    mod_a_path = r"c:\Users\adhri\Downloads\Project-RiskGraph\riskgraph_refund_modeling_dataset.csv"
    
    raw_b_path = r"c:\Users\adhri\Downloads\Project-RiskGraph\riskgraph_independent_raw_dataset.csv"
    mod_b_path = r"c:\Users\adhri\Downloads\Project-RiskGraph\riskgraph_independent_modeling_dataset.csv"
    
    # -------------------------------------------------------------
    # 1. PROCESS DATASET B VIA EXISTING POINT-IN-TIME FEATURE ENGINE
    # -------------------------------------------------------------
    print("="*85)
    print("1. RUNNING EXISTING POINT-IN-TIME FEATURE PIPELINE ON DATASET B")
    print("="*85)
    df_raw_b = load_and_preprocess_raw_data(raw_b_path)
    df_mod_b = build_point_in_time_features(df_raw_b)
    df_mod_b.to_csv(mod_b_path, index=False)
    print(f"[PASS] Successfully generated Dataset B modeling features: {mod_b_path}")
    print(f"       Dataset B Refund Population (N): {len(df_mod_b)} cases")
    print(f"       Abuse Label Counts: Legitimate (0) = {(df_mod_b['abuse_label']==0).sum()}, Abusive (1) = {(df_mod_b['abuse_label']==1).sum()} ({df_mod_b['abuse_label'].mean():.2%})\n")
    
    # -------------------------------------------------------------
    # 2. INDEPENDENT DATA QUALITY & NON-OVERLAP CHECKS (PART L)
    # -------------------------------------------------------------
    print("="*85)
    print("2. RUNNING INDEPENDENT DATA INTEGRITY & NON-OVERLAP CHECKS (PART L)")
    print("="*85)
    df_raw_a = pd.read_csv(raw_a_path)
    
    # Check ID disjointness
    cust_overlap = set(df_raw_a['customer_id']).intersection(set(df_raw_b['customer_id']))
    order_overlap = set(df_raw_a['order_id']).intersection(set(df_raw_b['order_id']))
    txn_overlap = set(df_raw_a['transaction_id']).intersection(set(df_raw_b['transaction_id']))
    dev_overlap = set(df_raw_a['device_id']).intersection(set(df_raw_b['device_id']))
    addr_overlap = set(df_raw_a['shipping_address_id']).intersection(set(df_raw_b['shipping_address_id']))
    
    assert len(cust_overlap) == 0, f"Customer ID overlap detected: {len(cust_overlap)}"
    assert len(order_overlap) == 0, f"Order ID overlap detected: {len(order_overlap)}"
    assert len(txn_overlap) == 0, f"Transaction ID overlap detected: {len(txn_overlap)}"
    assert len(dev_overlap) == 0, f"Device ID overlap detected: {len(dev_overlap)}"
    assert len(addr_overlap) == 0, f"Address ID overlap detected: {len(addr_overlap)}"
    
    print("[PASS] Zero ID overlap: 100% disjoint customer, order, transaction, device, and address IDs.")
    
    df_mod_a = pd.read_csv(mod_a_path)
    assert list(df_mod_a.columns) == list(df_mod_b.columns), "Feature schema mismatch between Dataset A and Dataset B!"
    print("[PASS] Exact schema parity: All 33 input features, target, and metadata match 1-to-1.")
    print("="*85 + "\n")
    
    # -------------------------------------------------------------
    # 3. TRAIN & FREEZE ORIGINAL MODELS ON DATASET A
    # -------------------------------------------------------------
    print("="*85)
    print("3. FREEZING ORIGINAL MODELS & PREPROCESSORS ON DATASET A TRAIN (70%)")
    print("="*85)
    df_mod_a['return_request_date'] = pd.to_datetime(df_mod_a['return_request_date'])
    df_mod_a['order_date'] = pd.to_datetime(df_mod_a['order_date'])
    df_mod_a = df_mod_a.sort_values(by=['return_request_date', 'order_date', 'order_id']).reset_index(drop=True)
    
    n_a = len(df_mod_a)
    train_a = df_mod_a.iloc[:int(n_a*0.70)].copy().reset_index(drop=True)
    val_a = df_mod_a.iloc[int(n_a*0.70):int(n_a*0.85)].copy().reset_index(drop=True)
    test_a = df_mod_a.iloc[int(n_a*0.85):].copy().reset_index(drop=True)
    
    metadata_cols = [
        'order_id', 'customer_id', 'transaction_id', 'device_id',
        'shipping_address_id', 'billing_address_id', 'order_date',
        'return_request_date', 'abuse_label', 'abuse_type'
    ]
    feature_cols = [c for c in train_a.columns if c not in metadata_cols]
    cat_cols = ['category', 'payment_method', 'channel', 'return_reason', 'customer_segment']
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    X_train_a = train_a[feature_cols].copy()
    y_train_a = train_a['abuse_label'].values
    X_val_a = val_a[feature_cols].copy()
    y_val_a = val_a['abuse_label'].values
    X_test_a = test_a[feature_cols].copy()
    y_test_a = test_a['abuse_label'].values
    
    # Dataset B Features & Target (NEVER FITTED ON)
    X_b = df_mod_b[feature_cols].copy()
    y_b = df_mod_b['abuse_label'].values
    
    # Preprocessor 1: Linear (Scaled + One-Hot)
    prep_linear = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    # Preprocessor 2: Tree (Passthrough Num + One-Hot)
    prep_tree = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    # FIT PREPROCESSORS EXCLUSIVELY ON DATASET A TRAIN
    X_tr_lin_a = prep_linear.fit_transform(X_train_a)
    X_va_lin_a = prep_linear.transform(X_val_a)
    X_te_lin_a = prep_linear.transform(X_test_a)
    X_b_lin = prep_linear.transform(X_b)  # STRICTLY TRANSFORM ONLY
    
    X_tr_tree_a = prep_tree.fit_transform(X_train_a)
    X_va_tree_a = prep_tree.transform(X_val_a)
    X_te_tree_a = prep_tree.transform(X_test_a)
    X_b_tree = prep_tree.transform(X_b)    # STRICTLY TRANSFORM ONLY
    
    # Train Models on Dataset A Train
    models = {}
    
    # 1. Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    gbc.fit(X_tr_tree_a, y_train_a)
    models['Gradient Boosting Classifier'] = {'model': gbc, 'type': 'tree', 'orig_val_th': 0.20}
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=8, min_samples_leaf=5, random_state=42)
    rf.fit(X_tr_tree_a, y_train_a)
    models['Random Forest'] = {'model': rf, 'type': 'tree', 'orig_val_th': 0.35}
    
    # 3. Linear Discriminant Analysis (LDA)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_tr_lin_a, y_train_a)
    models['Linear Discriminant Analysis'] = {'model': lda, 'type': 'linear', 'orig_val_th': 0.20}
    
    # 4. Elastic-Net Logistic Regression
    enet = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', class_weight='balanced', max_iter=1000, random_state=42, C=0.5)
    enet.fit(X_tr_lin_a, y_train_a)
    models['Elastic-Net Logistic Regression'] = {'model': enet, 'type': 'linear', 'orig_val_th': 0.40}
    
    # 5. HistGradientBoosting
    hgb = HistGradientBoostingClassifier(class_weight='balanced', max_iter=150, min_samples_leaf=15, learning_rate=0.08, random_state=42)
    hgb.fit(X_tr_tree_a, y_train_a)
    models['HistGradientBoosting'] = {'model': hgb, 'type': 'tree', 'orig_val_th': 0.35}
    
    # 6. Logistic Regression (L2)
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, C=0.5)
    lr.fit(X_tr_lin_a, y_train_a)
    models['Logistic Regression (L2)'] = {'model': lr, 'type': 'linear', 'orig_val_th': 0.45}
    
    print("[PASS] All 6 models and 2 preprocessors successfully frozen with original validation thresholds.")
    print("="*85 + "\n")
    
    # -------------------------------------------------------------
    # 4. EVALUATE ON DATASET A (TEST) AND DATASET B (INDEPENDENT)
    # -------------------------------------------------------------
    def compute_metrics(y_true, y_prob, threshold, model_name, dataset_name):
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
        
        fp_cost = fp * 50.0
        cost_1k = (fp / len(y_true)) * 1000 * 50.0
        
        return {
            'model_name': model_name, 'dataset': dataset_name, 'threshold': threshold,
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'pr_auc': pr_auc, 'roc_auc': roc_auc, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'fp_cost': fp_cost, 'cost_1k': cost_1k, 'n_total': len(y_true), 'n_pos': int(y_true.sum())
        }
        
    comparison_rows = []
    
    # Evaluate Majority Baseline on both
    maj_p_a = np.zeros_like(y_test_a)
    maj_p_b = np.zeros_like(y_b)
    m_a_maj = compute_metrics(y_test_a, maj_p_a, 0.5, "Majority Class Baseline", "Dataset A (Test)")
    m_b_maj = compute_metrics(y_b, maj_p_b, 0.5, "Majority Class Baseline", "Dataset B (Indep)")
    comparison_rows.append((m_a_maj, m_b_maj))
    
    for name, m_info in models.items():
        m = m_info['model']
        th = m_info['orig_val_th']
        m_type = m_info['type']
        
        if m_type == 'linear':
            p_a = m.predict_proba(X_te_lin_a)[:, 1]
            p_b = m.predict_proba(X_b_lin)[:, 1]
        else:
            p_a = m.predict_proba(X_te_tree_a)[:, 1]
            p_b = m.predict_proba(X_b_tree)[:, 1]
            
        m_a = compute_metrics(y_test_a, p_a, th, name, "Dataset A (Test)")
        m_b = compute_metrics(y_b, p_b, th, name, "Dataset B (Indep)")
        comparison_rows.append((m_a, m_b))
        
    # Print Detailed Side-by-Side Comparison Table
    print("="*125)
    print("4. DATASET A (HELD-OUT TEST) VS DATASET B (INDEPENDENT) GENERALIZATION BENCHMARK")
    print("="*125)
    print(f"{'Model Name':<32} | {'Set':<14} | {'Thresh':>6} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'PR-AUC':>6} | {'ROC':>6} | {'FP':>4} | {'FN':>4} | {'FP Cost':>8}")
    print("-" * 125)
    
    for m_a, m_b in comparison_rows:
        print(f"{m_a['model_name']:<32} | {'Dataset A (Test)':<14} | {m_a['threshold']:6.2f} | {m_a['accuracy']:6.4f} | {m_a['precision']:6.4f} | {m_a['recall']:6.4f} | {m_a['f1']:6.4f} | {m_a['pr_auc']:6.4f} | {m_a['roc_auc']:6.4f} | {m_a['fp']:4d} | {m_a['fn']:4d} | INR {m_a['fp_cost']:4.0f}")
        print(f"{'':<32} | {'Dataset B (Ind)':<14} | {m_b['threshold']:6.2f} | {m_b['accuracy']:6.4f} | {m_b['precision']:6.4f} | {m_b['recall']:6.4f} | {m_b['f1']:6.4f} | {m_b['pr_auc']:6.4f} | {m_b['roc_auc']:6.4f} | {m_b['fp']:4d} | {m_b['fn']:4d} | INR {m_b['fp_cost']:4.0f}")
        # Compute Delta
        d_f1 = m_b['f1'] - m_a['f1']
        d_pr = m_b['pr_auc'] - m_a['pr_auc']
        d_roc = m_b['roc_auc'] - m_a['roc_auc']
        print(f"{'':<32} | {'Delta (B - A)':<14} | {'':>6} | {m_b['accuracy']-m_a['accuracy']:+6.4f} | {m_b['precision']-m_a['precision']:+6.4f} | {m_b['recall']-m_a['recall']:+6.4f} | {d_f1:+6.4f} | {d_pr:+6.4f} | {d_roc:+6.4f} | {m_b['fp']-m_a['fp']:+4d} | {m_b['fn']-m_a['fn']:+4d} | INR {m_b['fp_cost']-m_a['fp_cost']:+4.0f}")
        print("-" * 125)
    print("="*125 + "\n")
    
    # -------------------------------------------------------------
    # 5. PART M: PROBABILITY BAND CALIBRATION ANALYSIS (DATASET B)
    # -------------------------------------------------------------
    print("="*85)
    print("5. PART M: 10-BIN PROBABILITY CALIBRATION ANALYSIS ON DATASET B (RANDOM FOREST)")
    print("="*85)
    rf_probs_b = rf.predict_proba(X_b_tree)[:, 1]
    
    bins = np.linspace(0.0, 1.0, 11)
    bin_labels = [f"{bins[i]:.1f} - {bins[i+1]:.1f}" for i in range(10)]
    
    bin_indices = np.digitize(rf_probs_b, bins, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, 9)
    
    print(f"{'Probability Band':<18} | {'Case Count':>10} | {'Observed Abuse Count':>20} | {'Observed Abuse Rate':>20} | {'Mean Pred Prob':>15}")
    print("-" * 85)
    
    for b_idx in range(10):
        mask = (bin_indices == b_idx)
        count = mask.sum()
        if count > 0:
            obs_abuse = int(y_b[mask].sum())
            obs_rate = obs_abuse / count
            mean_prob = float(rf_probs_b[mask].mean())
            print(f"{bin_labels[b_idx]:<18} | {count:10d} | {obs_abuse:20d} | {obs_rate:19.2%} | {mean_prob:15.4f}")
        else:
            print(f"{bin_labels[b_idx]:<18} | {0:10d} | {0:20d} | {'0.00%':>20} | {'N/A':>15}")
            
    print("="*85 + "\n")

if __name__ == "__main__":
    run_independent_generalization_experiment()
