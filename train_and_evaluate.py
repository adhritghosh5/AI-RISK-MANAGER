"""
RISKGRAPH — AI Risk Manager
Track 02: AI Risk Manager
Problem: Defensive E-Commerce Return + Refund Abuse Detection

STEP 2: Model Training & Evaluation Pipeline
- Temporal Train / Validation / Test chronological split (70% / 15% / 15%)
- Preprocessing fitted exclusively on Training data
- Models: Naive Baseline, Logistic Regression, Tree-Based Model (HistGradientBoosting / RandomForest)
- Validation Threshold Tuning & Held-out Test Set Evaluation
- False Positive Business Cost Analysis (Assumption: INR 50 / review)
- Defensive Risk Routing Policy Prototype (LOW / MEDIUM / HIGH)
- Interpretability & Error Analysis
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Ensure utf-8 stdout encoding on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, precision_recall_curve,
    auc, confusion_matrix, accuracy_score, roc_auc_score
)
from sklearn.inspection import permutation_importance

def load_data(csv_path: str):
    """Load modeling dataset and ensure correct chronological ordering."""
    df = pd.read_csv(csv_path)
    df['return_request_date'] = pd.to_datetime(df['return_request_date'])
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    # Sort chronologically by return_request_date
    df = df.sort_values(by=['return_request_date', 'order_date', 'order_id']).reset_index(drop=True)
    return df

def temporal_split(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    """
    Split dataset chronologically into Train (70%), Validation (15%), and Test (15%).
    Ensures zero temporal overlap across splits.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    # Verify strict non-overlapping temporal order
    max_train_date = train_df['return_request_date'].max()
    min_val_date = val_df['return_request_date'].min()
    max_val_date = val_df['return_request_date'].max()
    min_test_date = test_df['return_request_date'].min()
    
    print("\n" + "="*75)
    print("1. TEMPORAL CHRONOLOGICAL SPLIT SUMMARY")
    print("="*75)
    print(f"TRAIN SET:      {len(train_df):4d} rows | Date Range: {train_df['return_request_date'].min().strftime('%Y-%m-%d')} to {max_train_date.strftime('%Y-%m-%d')}")
    print(f"  - Abuse (1):  {train_df['abuse_label'].sum():4d} ({train_df['abuse_label'].mean():.2%}) | Legitimate (0): {(train_df['abuse_label']==0).sum():4d}")
    print(f"VALIDATION SET: {len(val_df):4d} rows | Date Range: {min_val_date.strftime('%Y-%m-%d')} to {max_val_date.strftime('%Y-%m-%d')}")
    print(f"  - Abuse (1):  {val_df['abuse_label'].sum():4d} ({val_df['abuse_label'].mean():.2%}) | Legitimate (0): {(val_df['abuse_label']==0).sum():4d}")
    print(f"TEST SET:       {len(test_df):4d} rows | Date Range: {min_test_date.strftime('%Y-%m-%d')} to {test_df['return_request_date'].max().strftime('%Y-%m-%d')}")
    print(f"  - Abuse (1):  {test_df['abuse_label'].sum():4d} ({test_df['abuse_label'].mean():.2%}) | Legitimate (0): {(test_df['abuse_label']==0).sum():4d}")
    print(f"TOTAL ROWS:     {len(df):4d}")
    print("="*75 + "\n")
    
    assert max_train_date <= min_val_date, "Temporal overlap between Train and Validation!"
    assert max_val_date <= min_test_date, "Temporal overlap between Validation and Test!"
    
    return train_df, val_df, test_df

def prepare_feature_matrices(train_df, val_df, test_df):
    """Separate features X, target y, and metadata for all splits."""
    metadata_cols = [
        'order_id', 'customer_id', 'transaction_id', 'device_id',
        'shipping_address_id', 'billing_address_id', 'order_date',
        'return_request_date', 'abuse_label', 'abuse_type'
    ]
    
    feature_cols = [c for c in train_df.columns if c not in metadata_cols]
    assert len(feature_cols) == 33, f"Expected 33 features, found {len(feature_cols)}!"
    
    categorical_cols = ['category', 'payment_method', 'channel', 'return_reason', 'customer_segment']
    numerical_cols = [c for c in feature_cols if c not in categorical_cols]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df['abuse_label'].values
    
    X_val = val_df[feature_cols].copy()
    y_val = val_df['abuse_label'].values
    
    X_test = test_df[feature_cols].copy()
    y_test = test_df['abuse_label'].values
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, numerical_cols, categorical_cols

def evaluate_predictions(y_true, y_pred, y_prob=None, model_name="Model", fp_cost_per_case=50.0):
    """Compute comprehensive defensive risk metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    if y_prob is not None:
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(r_curve, p_curve)
        roc_auc = roc_auc_score(y_true, y_prob)
    else:
        pr_auc = np.nan
        roc_auc = np.nan
        
    total_fp_cost = fp * fp_cost_per_case
    cost_per_1000 = (fp / len(y_true)) * 1000 * fp_cost_per_case
    
    return {
        'model_name': model_name,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'accuracy': acc,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'fp_cost': total_fp_cost,
        'cost_per_1000': cost_per_1000
    }

def run_pipeline():
    csv_path = r"c:\Users\adhri\Downloads\Project-RiskGraph\riskgraph_refund_modeling_dataset.csv"
    df = load_data(csv_path)
    
    # 1. Temporal Split
    train_df, val_df, test_df = temporal_split(df)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, num_cols, cat_cols = prepare_feature_matrices(
        train_df, val_df, test_df
    )
    
    # 2. Build Preprocessors (Fitted ONLY on Train)
    preprocessor_lr = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    preprocessor_tree = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    # Fit preprocessors on TRAIN ONLY
    X_train_lr = preprocessor_lr.fit_transform(X_train)
    X_val_lr = preprocessor_lr.transform(X_val)
    X_test_lr = preprocessor_lr.transform(X_test)
    
    X_train_tree = preprocessor_tree.fit_transform(X_train)
    X_val_tree = preprocessor_tree.transform(X_val)
    X_test_tree = preprocessor_tree.transform(X_test)
    
    encoded_feature_names = (
        num_cols + 
        list(preprocessor_tree.named_transformers_['cat'].get_feature_names_out(cat_cols))
    )
    
    # -------------------------------------------------------------------------
    # MODEL 0: Naive Rule-Based Baseline
    # Rule: Flag as potential abuse if prior_return_rate >= 0.20 OR returns_last_30_days >= 1
    # -------------------------------------------------------------------------
    val_baseline_pred = (
        (X_val['prior_return_rate'] >= 0.20) | 
        (X_val['returns_last_30_days'] >= 1)
    ).astype(int).values
    
    test_baseline_pred = (
        (X_test['prior_return_rate'] >= 0.20) | 
        (X_test['returns_last_30_days'] >= 1)
    ).astype(int).values
    
    baseline_val_metrics = evaluate_predictions(y_val, val_baseline_pred, model_name="Naive Baseline")
    baseline_test_metrics = evaluate_predictions(y_test, test_baseline_pred, model_name="Naive Baseline")
    
    # -------------------------------------------------------------------------
    # MODEL 1: Logistic Regression
    # -------------------------------------------------------------------------
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, C=0.5)
    lr_model.fit(X_train_lr, y_train)
    
    val_prob_lr = lr_model.predict_proba(X_val_lr)[:, 1]
    test_prob_lr = lr_model.predict_proba(X_test_lr)[:, 1]
    
    # -------------------------------------------------------------------------
    # MODEL 2: Tree-Based Model (HistGradientBoosting & RandomForest)
    # -------------------------------------------------------------------------
    tree_model = HistGradientBoostingClassifier(
        class_weight='balanced',
        max_iter=150,
        min_samples_leaf=15,
        learning_rate=0.08,
        random_state=42
    )
    tree_model.fit(X_train_tree, y_train)
    
    val_prob_tree = tree_model.predict_proba(X_val_tree)[:, 1]
    test_prob_tree = tree_model.predict_proba(X_test_tree)[:, 1]
    
    # Also evaluate Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        max_depth=8,
        min_samples_leaf=5,
        random_state=42
    )
    rf_model.fit(X_train_tree, y_train)
    val_prob_rf = rf_model.predict_proba(X_val_tree)[:, 1]
    test_prob_rf = rf_model.predict_proba(X_test_tree)[:, 1]
    
    # -------------------------------------------------------------------------
    # THRESHOLD ANALYSIS ON VALIDATION SET (DO NOT TOUCH TEST SET)
    # -------------------------------------------------------------------------
    threshold_grid = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]
    
    print("\n" + "="*75)
    print("2. VALIDATION SET THRESHOLD TUNING (Tree Model - HistGradientBoosting)")
    print("="*75)
    print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'FP':>5} | {'FN':>5} | {'FP Cost (INR 50)':>16}")
    print("-" * 75)
    
    val_thresh_records = []
    for th in threshold_grid:
        v_pred = (val_prob_tree >= th).astype(int)
        m = evaluate_predictions(y_val, v_pred, y_prob=val_prob_tree, model_name=f"Tree (th={th:.2f})")
        val_thresh_records.append(m)
        print(f"{th:10.2f} | {m['precision']:10.4f} | {m['recall']:10.4f} | {m['f1']:10.4f} | {m['fp']:5d} | {m['fn']:5d} | INR {m['fp_cost']:10.2f}")
    print("="*75 + "\n")
    
    # Optimal defensive threshold chosen on validation set: 0.50
    chosen_threshold = 0.50
    print(f"[DECISION] Selected Optimal Validation Threshold: {chosen_threshold:.2f}")
    
    # -------------------------------------------------------------------------
    # FINAL HELD-OUT TEST EVALUATION (APPLIED ONCE)
    # -------------------------------------------------------------------------
    test_pred_lr = (test_prob_lr >= chosen_threshold).astype(int)
    test_pred_tree = (test_prob_tree >= chosen_threshold).astype(int)
    test_pred_rf = (test_prob_rf >= chosen_threshold).astype(int)
    
    lr_test_metrics = evaluate_predictions(y_test, test_pred_lr, y_prob=test_prob_lr, model_name="Logistic Regression")
    tree_test_metrics = evaluate_predictions(y_test, test_pred_tree, y_prob=test_prob_tree, model_name="HistGradientBoosting (Tree)")
    rf_test_metrics = evaluate_predictions(y_test, test_pred_rf, y_prob=test_prob_rf, model_name="Random Forest (Tree)")
    
    # -------------------------------------------------------------------------
    # PRINT MODEL COMPARISON TABLE ON HELD-OUT TEST SET
    # -------------------------------------------------------------------------
    print("\n" + "="*85)
    print("3. HELD-OUT TEST SET EVALUATION COMPARISON (Threshold = 0.50)")
    print("="*85)
    print(f"{'Model':<28} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'PR-AUC':>6} | {'FP':>4} | {'FN':>4} | {'FP Cost':>10} | {'Cost/1k':>11}")
    print("-" * 85)
    for res in [baseline_test_metrics, lr_test_metrics, tree_test_metrics, rf_test_metrics]:
        pr_str = f"{res['pr_auc']:.4f}" if not np.isnan(res['pr_auc']) else " N/A  "
        print(f"{res['model_name']:<28} | {res['precision']:6.4f} | {res['recall']:6.4f} | {res['f1']:6.4f} | {pr_str:>6} | {res['fp']:4d} | {res['fn']:4d} | INR {res['fp_cost']:6.0f} | INR {res['cost_per_1000']:7.2f}")
    print("="*85 + "\n")
    
    # -------------------------------------------------------------------------
    # FEATURE IMPORTANCE (Permutation Importance on Test Set & Coefficients)
    # -------------------------------------------------------------------------
    perm_importance = permutation_importance(tree_model, X_test_tree, y_test, n_repeats=10, random_state=42)
    importance_df = pd.DataFrame({
        'Feature': encoded_feature_names,
        'Importance_Mean': perm_importance.importances_mean,
        'Importance_Std': perm_importance.importances_std
    }).sort_values(by='Importance_Mean', ascending=False).reset_index(drop=True)
    
    print("TOP 12 PREDICTIVE FEATURES (Tree Model Permutation Importance):")
    for i, r in importance_df.head(12).iterrows():
        print(f"  {i+1:2d}. {r['Feature']:<32} | Importance: {r['Importance_Mean']:.5f} (±{r['Importance_Std']:.5f})")
    print("\n")
    
    # Logistic Regression Top Coefficients
    lr_coefs = pd.DataFrame({
        'Feature': encoded_feature_names,
        'Coefficient': lr_model.coef_[0]
    }).sort_values(by='Coefficient', ascending=False).reset_index(drop=True)
    
    print("TOP 6 RISK-ELEVATING SIGNALS (Logistic Regression):")
    for i, r in lr_coefs.head(6).iterrows():
        print(f"  + {r['Feature']:<32} | Coef: {r['Coefficient']:+.4f}")
    print("\nTOP 6 RISK-REDUCING SIGNALS (Logistic Regression):")
    for i, r in lr_coefs.tail(6).iloc[::-1].iterrows():
        print(f"  - {r['Feature']:<32} | Coef: {r['Coefficient']:+.4f}")
    print("\n")
    
    # -------------------------------------------------------------------------
    # ERROR ANALYSIS & PROTOTYPE DEFENSIVE RISK ROUTING (HELD-OUT TEST)
    # -------------------------------------------------------------------------
    test_eval_df = test_df.copy()
    test_eval_df['prob_tree'] = test_prob_tree
    test_eval_df['pred_tree'] = test_pred_tree
    
    # Prototype Routing: LOW (<0.30), MEDIUM (0.30 - 0.60), HIGH (>=0.60)
    def assign_routing(p):
        if p < 0.30:
            return 'LOW (Auto-Approve)'
        elif p < 0.60:
            return 'MEDIUM (Verify Customer)'
        else:
            return 'HIGH (Human Review Queue)'
            
    test_eval_df['risk_routing'] = test_eval_df['prob_tree'].apply(assign_routing)
    
    print("PROTOTYPE RISK ROUTING DISTRIBUTION ON TEST SET:")
    print(test_eval_df['risk_routing'].value_counts())
    print("\n")
    
    # Inspect TP, TN, FP, FN Examples
    tp_cases = test_eval_df[(test_eval_df['abuse_label'] == 1) & (test_eval_df['pred_tree'] == 1)]
    tn_cases = test_eval_df[(test_eval_df['abuse_label'] == 0) & (test_eval_df['pred_tree'] == 0)]
    fp_cases = test_eval_df[(test_eval_df['abuse_label'] == 0) & (test_eval_df['pred_tree'] == 1)]
    fn_cases = test_eval_df[(test_eval_df['abuse_label'] == 1) & (test_eval_df['pred_tree'] == 0)]
    
    print("ERROR ANALYSIS SAMPLE SIZES ON TEST SET:")
    print(f"  - True Positives (Abuse Caught):            {len(tp_cases):3d}")
    print(f"  - True Negatives (Legitimate Cleared):      {len(tn_cases):3d}")
    print(f"  - False Positives (Friction on Legitimate): {len(fp_cases):3d}")
    print(f"  - False Negatives (Abuse Missed):           {len(fn_cases):3d}")
    print("\n")
    
    return {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'baseline_test': baseline_test_metrics,
        'lr_test': lr_test_metrics,
        'tree_test': tree_test_metrics,
        'rf_test': rf_test_metrics,
        'val_threshold_table': val_thresh_records,
        'importance_df': importance_df,
        'lr_coefs': lr_coefs,
        'test_eval_df': test_eval_df,
        'tp_cases': tp_cases,
        'tn_cases': tn_cases,
        'fp_cases': fp_cases,
        'fn_cases': fn_cases
    }

if __name__ == "__main__":
    run_pipeline()
