"""
RISKGRAPH — AI Risk Manager
Track 02: AI Risk Manager
Problem: Defensive E-Commerce Return + Refund Abuse Detection

STEP 2.5: Diagnostic Audit Script
Calculates exact probability distributions, calibration metrics (Brier score, ECE),
class-wise feature summaries, and error analysis for Random Forest and HistGradientBoosting.
"""

import sys
import os
import pandas as pd
import numpy as np

# Ensure utf-8 stdout encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, precision_recall_curve,
    auc, confusion_matrix, accuracy_score, roc_auc_score, brier_score_loss
)
from sklearn.calibration import calibration_curve

def run_diagnostics():
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
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc = preprocessor.transform(X_val)
    X_test_enc = preprocessor.transform(X_test)
    
    encoded_feature_names = (
        num_cols + 
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    )
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        max_depth=8,
        min_samples_leaf=5,
        random_state=42
    )
    rf.fit(X_train_enc, y_train)
    
    val_prob_rf = rf.predict_proba(X_val_enc)[:, 1]
    test_prob_rf = rf.predict_proba(X_test_enc)[:, 1]
    
    # Train HistGradientBoosting
    hgb = HistGradientBoostingClassifier(
        class_weight='balanced',
        max_iter=150,
        min_samples_leaf=15,
        learning_rate=0.08,
        random_state=42
    )
    hgb.fit(X_train_enc, y_train)
    val_prob_hgb = hgb.predict_proba(X_val_enc)[:, 1]
    test_prob_hgb = hgb.predict_proba(X_test_enc)[:, 1]
    
    # -------------------------------------------------------------
    # 1. THRESHOLD COMPARISON ON VALIDATION SET FOR RANDOM FOREST
    # -------------------------------------------------------------
    print("="*75)
    print("1. RANDOM FOREST VALIDATION SET THRESHOLD GRID")
    print("="*75)
    for th in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        pred = (val_prob_rf >= th).astype(int)
        cm = confusion_matrix(y_val, pred)
        tn, fp, fn, tp = cm.ravel()
        p = precision_score(y_val, pred, zero_division=0)
        r = recall_score(y_val, pred, zero_division=0)
        f1 = f1_score(y_val, pred, zero_division=0)
        cost = fp * 50.0
        print(f"Th: {th:.2f} | Prec: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | TP: {tp:2d} | TN: {tn:3d} | FP: {fp:2d} | FN: {fn:2d} | FP Cost: INR {cost:7.2f}")
    print("="*75 + "\n")
    
    # -------------------------------------------------------------
    # 2. PROBABILITY DISTRIBUTION STATS (RANDOM FOREST)
    # -------------------------------------------------------------
    print("="*75)
    print("2. RANDOM FOREST PREDICTED PROBABILITY SUMMARY STATISTICS")
    print("="*75)
    def print_prob_stats(name, probs, y_true=None):
        print(f"\n--- {name} (N = {len(probs)}) ---")
        print(f"  Min:    {np.min(probs):.4f}")
        print(f"  25th %: {np.percentile(probs, 25):.4f}")
        print(f"  Median: {np.median(probs):.4f}")
        print(f"  Mean:   {np.mean(probs):.4f}")
        print(f"  75th %: {np.percentile(probs, 75):.4f}")
        print(f"  Max:    {np.max(probs):.4f}")
        if y_true is not None:
            legit_probs = probs[y_true == 0]
            abuse_probs = probs[y_true == 1]
            print(f"  Actual Legitimate (y=0, N={len(legit_probs)}): Median={np.median(legit_probs):.4f}, Mean={np.mean(legit_probs):.4f}, IQR=[{np.percentile(legit_probs, 25):.4f}, {np.percentile(legit_probs, 75):.4f}]")
            print(f"  Actual Abusive    (y=1, N={len(abuse_probs)}): Median={np.median(abuse_probs):.4f}, Mean={np.mean(abuse_probs):.4f}, IQR=[{np.percentile(abuse_probs, 25):.4f}, {np.percentile(abuse_probs, 75):.4f}]")
            
    print_prob_stats("Validation Set Overall", val_prob_rf, y_val)
    print_prob_stats("Test Set Overall", test_prob_rf, y_test)
    print("="*75 + "\n")
    
    # -------------------------------------------------------------
    # 3. CALIBRATION & BRIER SCORE
    # -------------------------------------------------------------
    brier_val = brier_score_loss(y_val, val_prob_rf)
    brier_test = brier_score_loss(y_test, test_prob_rf)
    print("="*75)
    print("3. CALIBRATION DIAGNOSTICS")
    print("="*75)
    print(f"Validation Brier Score Loss: {brier_val:.4f}")
    print(f"Test Brier Score Loss:       {brier_test:.4f}")
    
    # Calibration bins on test set
    prob_true, prob_pred = calibration_curve(y_test, test_prob_rf, n_bins=5)
    print("\nTest Set Reliability Bins (5 quantile/uniform bins):")
    for pt, pp in zip(prob_true, prob_pred):
        print(f"  Predicted Mean Probability: {pp:.4f} | Observed True Abuse Fraction: {pt:.4f}")
    print("="*75 + "\n")
    
    # -------------------------------------------------------------
    # 4. RANDOM FOREST GINI FEATURE IMPORTANCES (TOP 10)
    # -------------------------------------------------------------
    rf_feat_imp = pd.DataFrame({
        'Feature': encoded_feature_names,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    print("="*75)
    print("4. TOP 10 RANDOM FOREST FEATURE IMPORTANCES (GINI)")
    print("="*75)
    for i, r in rf_feat_imp.head(10).iterrows():
        print(f"  {i+1:2d}. {r['Feature']:<32} | Gini Importance: {r['Importance']:.5f} ({r['Importance']*100:.2f}%)")
    print("="*75 + "\n")
    
    # -------------------------------------------------------------
    # 5. ERROR ANALYSIS PATTERNS ON TEST SET (RF at Thresh 0.50 & 0.40)
    # -------------------------------------------------------------
    for test_th in [0.40, 0.50]:
        test_pred = (test_prob_rf >= test_th).astype(int)
        cm = confusion_matrix(y_test, test_pred)
        tn, fp, fn, tp = cm.ravel()
        p = precision_score(y_test, test_pred, zero_division=0)
        r = recall_score(y_test, test_pred, zero_division=0)
        f1 = f1_score(y_test, test_pred, zero_division=0)
        print(f"TEST SET PERFORMANCE at Threshold = {test_th:.2f}:")
        print(f"  Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
        print(f"  TP: {tp:2d} | TN: {tn:3d} | FP: {fp:2d} | FN: {fn:2d} | FP Cost: INR {fp*50:.2f} | Cost/1k: INR {(fp/len(y_test))*1000*50:.2f}\n")

if __name__ == "__main__":
    run_diagnostics()
