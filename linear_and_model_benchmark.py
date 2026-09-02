"""
RISKGRAPH — AI Risk Manager
Track 02: AI Risk Manager
Problem: Defensive E-Commerce Return + Refund Abuse Detection

STEP 2.6: Comprehensive Linear & Tree Model Benchmark
Benchmark Suite:
1. Majority Class Baseline (Always Legitimate)
2. Naive Rule-Based Baseline (Return velocity / rate heuristic)
3. Standard Logistic Regression (L2 / Balanced)
4. Ridge Classifier (L2 Linear)
5. LinearSVC (Support Vector Classification)
6. SGDClassifier (Logistic Loss / Elastic-Net)
7. SGDClassifier (Modified Huber Loss)
8. Linear Discriminant Analysis (LDA)
9. Elastic-Net Logistic Regression (SAGA Solver)
10. Random Forest Classifier
11. HistGradientBoosting Classifier
12. Gradient Boosting Classifier
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure utf-8 output encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_auc_score, auc, confusion_matrix
)
from scipy.special import expit

def load_and_split():
    df = pd.read_csv(r"c:\Users\adhri\Downloads\Project-RiskGraph\riskgraph_refund_modeling_dataset.csv")
    df['return_request_date'] = pd.to_datetime(df['return_request_date'])
    df['order_date'] = pd.to_datetime(df['order_date'])
    df = df.sort_values(by=['return_request_date', 'order_date', 'order_id']).reset_index(drop=True)
    
    n = len(df)
    train_df = df.iloc[:int(n*0.70)].copy().reset_index(drop=True)
    val_df = df.iloc[int(n*0.70):int(n*0.85)].copy().reset_index(drop=True)
    test_df = df.iloc[int(n*0.85):].copy().reset_index(drop=True)
    
    metadata_cols = [
        'order_id', 'customer_id', 'transaction_id', 'device_id',
        'shipping_address_id', 'billing_address_id', 'order_date',
        'return_request_date', 'abuse_label', 'abuse_type'
    ]
    feature_cols = [c for c in train_df.columns if c not in metadata_cols]
    cat_cols = ['category', 'payment_method', 'channel', 'return_reason', 'customer_segment']
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df['abuse_label'].values
    X_val = val_df[feature_cols].copy()
    y_val = val_df['abuse_label'].values
    X_test = test_df[feature_cols].copy()
    y_test = test_df['abuse_label'].values
    
    # Preprocessor for Linear Models (Scaled + One-Hot)
    prep_linear = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    # Preprocessor for Tree Models (Passthrough Num + One-Hot)
    prep_tree = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    # Fit ONLY on TRAIN
    X_tr_lin = prep_linear.fit_transform(X_train)
    X_va_lin = prep_linear.transform(X_val)
    X_te_lin = prep_linear.transform(X_test)
    
    X_tr_tree = prep_tree.fit_transform(X_train)
    X_va_tree = prep_tree.transform(X_val)
    X_te_tree = prep_tree.transform(X_test)
    
    return {
        'train_df': train_df, 'val_df': val_df, 'test_df': test_df,
        'X_train_raw': X_train, 'X_val_raw': X_val, 'X_test_raw': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'X_tr_lin': X_tr_lin, 'X_va_lin': X_va_lin, 'X_te_lin': X_te_lin,
        'X_tr_tree': X_tr_tree, 'X_va_tree': X_va_tree, 'X_te_tree': X_te_tree,
        'feature_cols': feature_cols
    }

def evaluate_model_at_threshold(y_true, y_pred, y_prob=None, model_name="Model", fp_cost_unit=50.0):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    if y_prob is not None:
        p_cur, r_cur, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(r_cur, p_cur)
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except:
            roc_auc = np.nan
    else:
        pr_auc = np.nan
        roc_auc = np.nan
        
    fp_cost = fp * fp_cost_unit
    cost_per_1k = (fp / len(y_true)) * 1000 * fp_cost_unit
    
    return {
        'model_name': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'fp_cost': fp_cost,
        'cost_per_1k': cost_per_1k
    }

def find_best_val_threshold(y_val, val_scores, thresholds=[0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]):
    best_th = 0.50
    best_f1 = -1.0
    val_records = []
    
    for th in thresholds:
        pred = (val_scores >= th).astype(int)
        f1 = f1_score(y_val, pred, zero_division=0)
        p = precision_score(y_val, pred, zero_division=0)
        r = recall_score(y_val, pred, zero_division=0)
        val_records.append({'threshold': th, 'precision': p, 'recall': r, 'f1': f1})
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            
    return best_th, best_f1, val_records

def main():
    data = load_and_split()
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    X_tr_lin, X_va_lin, X_te_lin = data['X_tr_lin'], data['X_va_lin'], data['X_te_lin']
    X_tr_tree, X_va_tree, X_te_tree = data['X_tr_tree'], data['X_va_tree'], data['X_te_tree']
    X_val_raw, X_test_raw = data['X_val_raw'], data['X_test_raw']
    
    results = []
    
    # -------------------------------------------------------------
    # 1. Majority Class Baseline (Always Predict 0)
    # -------------------------------------------------------------
    maj_pred = np.zeros_like(y_test)
    maj_res = evaluate_model_at_threshold(y_test, maj_pred, y_prob=maj_pred, model_name="Majority Class Baseline")
    results.append({**maj_res, 'category': 'Baseline', 'val_th': 'N/A'})
    
    # -------------------------------------------------------------
    # 2. Naive Rule-Based Baseline
    # -------------------------------------------------------------
    naive_test_pred = (
        (X_test_raw['prior_return_rate'] >= 0.20) | 
        (X_test_raw['returns_last_30_days'] >= 1)
    ).astype(int).values
    naive_res = evaluate_model_at_threshold(y_test, naive_test_pred, model_name="Naive Rule Baseline")
    results.append({**naive_res, 'category': 'Baseline', 'val_th': 'Rule'})
    
    # -------------------------------------------------------------
    # 3. Logistic Regression (L2 / Balanced)
    # -------------------------------------------------------------
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, C=0.5)
    lr.fit(X_tr_lin, y_train)
    val_p_lr = lr.predict_proba(X_va_lin)[:, 1]
    test_p_lr = lr.predict_proba(X_te_lin)[:, 1]
    th_lr, f1_lr, _ = find_best_val_threshold(y_val, val_p_lr)
    test_pred_lr = (test_p_lr >= th_lr).astype(int)
    results.append({
        **evaluate_model_at_threshold(y_test, test_pred_lr, y_prob=test_p_lr, model_name="Logistic Regression (L2)"),
        'category': 'Linear', 'val_th': f"{th_lr:.2f}"
    })
    
    # -------------------------------------------------------------
    # 4. Ridge Classifier
    # -------------------------------------------------------------
    ridge = RidgeClassifier(class_weight='balanced', random_state=42)
    ridge.fit(X_tr_lin, y_train)
    # Convert decision function to pseudo-probability via sigmoid
    val_score_ridge = expit(ridge.decision_function(X_va_lin))
    test_score_ridge = expit(ridge.decision_function(X_te_lin))
    th_ridge, _, _ = find_best_val_threshold(y_val, val_score_ridge)
    test_pred_ridge = (test_score_ridge >= th_ridge).astype(int)
    results.append({
        **evaluate_model_at_threshold(y_test, test_pred_ridge, y_prob=test_score_ridge, model_name="Ridge Classifier"),
        'category': 'Linear', 'val_th': f"{th_ridge:.2f}"
    })
    
    # -------------------------------------------------------------
    # 5. LinearSVC
    # -------------------------------------------------------------
    svc = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000, dual=False)
    svc.fit(X_tr_lin, y_train)
    val_score_svc = expit(svc.decision_function(X_va_lin))
    test_score_svc = expit(svc.decision_function(X_te_lin))
    th_svc, _, _ = find_best_val_threshold(y_val, val_score_svc)
    test_pred_svc = (test_score_svc >= th_svc).astype(int)
    results.append({
        **evaluate_model_at_threshold(y_test, test_pred_svc, y_prob=test_score_svc, model_name="LinearSVC"),
        'category': 'Linear', 'val_th': f"{th_svc:.2f}"
    })
    
    # -------------------------------------------------------------
    # 6. SGDClassifier (Log-Loss)
    # -------------------------------------------------------------
    sgd_log = SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42, max_iter=1000)
    sgd_log.fit(X_tr_lin, y_train)
    val_p_sgd_log = sgd_log.predict_proba(X_va_lin)[:, 1]
    test_p_sgd_log = sgd_log.predict_proba(X_te_lin)[:, 1]
    th_sgd_log, _, _ = find_best_val_threshold(y_val, val_p_sgd_log)
    test_pred_sgd_log = (test_p_sgd_log >= th_sgd_log).astype(int)
    results.append({
        **evaluate_model_at_threshold(y_test, test_pred_sgd_log, y_prob=test_p_sgd_log, model_name="SGDClassifier (Log Loss)"),
        'category': 'Linear', 'val_th': f"{th_sgd_log:.2f}"
    })
    
    # -------------------------------------------------------------
    # 7. SGDClassifier (Modified Huber Loss)
    # -------------------------------------------------------------
    sgd_huber = SGDClassifier(loss='modified_huber', class_weight='balanced', random_state=42, max_iter=1000)
    sgd_huber.fit(X_tr_lin, y_train)
    val_p_sgd_huber = sgd_huber.predict_proba(X_va_lin)[:, 1]
    test_p_sgd_huber = sgd_huber.predict_proba(X_te_lin)[:, 1]
    th_sgd_huber, _, _ = find_best_val_threshold(y_val, val_p_sgd_huber)
    test_pred_sgd_huber = (test_p_sgd_huber >= th_sgd_huber).astype(int)
    results.append({
        **evaluate_model_at_threshold(y_test, test_pred_sgd_huber, y_prob=test_p_sgd_huber, model_name="SGDClassifier (Modified Huber)"),
        'category': 'Linear', 'val_th': f"{th_sgd_huber:.2f}"
    })
    
    # -------------------------------------------------------------
    # 8. Linear Discriminant Analysis (LDA)
    # -------------------------------------------------------------
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_tr_lin, y_train)
    val_p_lda = lda.predict_proba(X_va_lin)[:, 1]
    test_p_lda = lda.predict_proba(X_te_lin)[:, 1]
    th_lda, _, _ = find_best_val_threshold(y_val, val_p_lda)
    test_pred_lda = (test_p_lda >= th_lda).astype(int)
    results.append({
        **evaluate_model_at_threshold(y_test, test_pred_lda, y_prob=test_p_lda, model_name="Linear Discriminant Analysis"),
        'category': 'Linear', 'val_th': f"{th_lda:.2f}"
    })
    
    # -------------------------------------------------------------
    # 9. Elastic-Net Logistic Regression (SAGA Solver)
    # -------------------------------------------------------------
    enet = LogisticRegression(
        penalty='elasticnet', l1_ratio=0.5, solver='saga',
        class_weight='balanced', max_iter=1000, random_state=42, C=0.5
    )
    enet.fit(X_tr_lin, y_train)
    val_p_enet = enet.predict_proba(X_va_lin)[:, 1]
    test_p_enet = enet.predict_proba(X_te_lin)[:, 1]
    th_enet, _, _ = find_best_val_threshold(y_val, val_p_enet)
    test_pred_enet = (test_p_enet >= th_enet).astype(int)
    results.append({
        **evaluate_model_at_threshold(y_test, test_pred_enet, y_prob=test_p_enet, model_name="Elastic-Net Logistic Regression"),
        'category': 'Linear', 'val_th': f"{th_enet:.2f}"
    })
    
    # -------------------------------------------------------------
    # 10. Random Forest
    # -------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=200, class_weight='balanced', max_depth=8,
        min_samples_leaf=5, random_state=42
    )
    rf.fit(X_tr_tree, y_train)
    val_p_rf = rf.predict_proba(X_va_tree)[:, 1]
    test_p_rf = rf.predict_proba(X_te_tree)[:, 1]
    th_rf, _, _ = find_best_val_threshold(y_val, val_p_rf)
    test_pred_rf = (test_p_rf >= th_rf).astype(int)
    results.append({
        **evaluate_model_at_threshold(y_test, test_pred_rf, y_prob=test_p_rf, model_name="Random Forest"),
        'category': 'Tree / Ensemble', 'val_th': f"{th_rf:.2f}"
    })
    
    # -------------------------------------------------------------
    # 11. HistGradientBoosting
    # -------------------------------------------------------------
    hgb = HistGradientBoostingClassifier(
        class_weight='balanced', max_iter=150, min_samples_leaf=15,
        learning_rate=0.08, random_state=42
    )
    hgb.fit(X_tr_tree, y_train)
    val_p_hgb = hgb.predict_proba(X_va_tree)[:, 1]
    test_p_hgb = hgb.predict_proba(X_te_tree)[:, 1]
    th_hgb, _, _ = find_best_val_threshold(y_val, val_p_hgb)
    test_pred_hgb = (test_p_hgb >= th_hgb).astype(int)
    results.append({
        **evaluate_model_at_threshold(y_test, test_pred_hgb, y_prob=test_p_hgb, model_name="HistGradientBoosting"),
        'category': 'Tree / Ensemble', 'val_th': f"{th_hgb:.2f}"
    })
    
    # -------------------------------------------------------------
    # 12. Gradient Boosting Classifier (Standard Sklearn)
    # -------------------------------------------------------------
    gbc = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42
    )
    gbc.fit(X_tr_tree, y_train)
    val_p_gbc = gbc.predict_proba(X_va_tree)[:, 1]
    test_p_gbc = gbc.predict_proba(X_te_tree)[:, 1]
    th_gbc, _, _ = find_best_val_threshold(y_val, val_p_gbc)
    test_pred_gbc = (test_p_gbc >= th_gbc).astype(int)
    results.append({
        **evaluate_model_at_threshold(y_test, test_pred_gbc, y_prob=test_p_gbc, model_name="Gradient Boosting Classifier"),
        'category': 'Tree / Ensemble', 'val_th': f"{th_gbc:.2f}"
    })
    
    # Build Summary DataFrame
    res_df = pd.DataFrame(results)
    
    print("\n" + "="*115)
    print("STEP 2.6: COMPLETE LINEAR & NON-LINEAR MODEL BENCHMARK (HELD-OUT TEST SET)")
    print("="*115)
    print(f"{'Model Name':<34} | {'Val Th':>6} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'PR-AUC':>6} | {'ROC':>6} | {'FP':>3} | {'FN':>3} | {'FP Cost':>8}")
    print("-" * 115)
    
    for _, r in res_df.iterrows():
        pr_s = f"{r['pr_auc']:.4f}" if not np.isnan(r['pr_auc']) else " N/A  "
        roc_s = f"{r['roc_auc']:.4f}" if not np.isnan(r['roc_auc']) else " N/A  "
        print(f"{r['model_name']:<34} | {r['val_th']:>6} | {r['accuracy']:6.4f} | {r['precision']:6.4f} | {r['recall']:6.4f} | {r['f1']:6.4f} | {pr_s:>6} | {roc_s:>6} | {r['fp']:3d} | {r['fn']:3d} | INR {r['fp_cost']:4.0f}")
        
    print("="*115 + "\n")
    
if __name__ == "__main__":
    main()
