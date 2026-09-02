"""
RISKGRAPH — Model C Serialization & Verification Script (Step 3.5.1)
Reproduces and serializes the exact frozen Model C from Step 3.3 / Step 3.4.

Configuration:
- Model: GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
- Features: Exact 55 Features (5 categorical + 50 numerical)
- Preprocessing: ColumnTransformer with passthrough numerical & OneHotEncoder categorical
- Split: Chronological 70% Train, 15% Val, 15% Test
- Artifacts Saved:
    - riskgraph_model_c.joblib
    - riskgraph_preprocessor_c.joblib
    - riskgraph_pipeline_c.joblib
    - model_c_metadata.json
"""

import os
import sys
import json
import hashlib
import joblib
import pandas as pd
import numpy as np

# Ensure utf-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

from hybrid_riskgraph_model import extract_hybrid_dataset_features

def get_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_a_path = os.path.join(base_dir, "riskgraph_ecommerce_dataset.csv")
    raw_b_path = os.path.join(base_dir, "riskgraph_independent_raw_dataset.csv")
    modeling_path = os.path.join(base_dir, "riskgraph_refund_modeling_dataset.csv")

    print("=" * 80)
    print("STEP 3.5.1: SERIALIZING EXACT FROZEN MODEL C & PREPROCESSOR")
    print("=" * 80)

    # 1. Capture initial dataset hashes
    datasets = {
        'riskgraph_ecommerce_dataset.csv': raw_a_path,
        'riskgraph_independent_raw_dataset.csv': raw_b_path,
        'riskgraph_refund_modeling_dataset.csv': modeling_path
    }
    initial_hashes = {}
    for name, path in datasets.items():
        if os.path.exists(path):
            initial_hashes[name] = get_file_sha256(path)
            print(f"[DATASET] {name:<42} SHA-256: {initial_hashes[name]}")

    # 2. Extract features using exact Step 3.3 feature extraction
    print("\n[PROCESS] Extracting Step 3.3 hybrid features for Dataset A...")
    df_feat_a = extract_hybrid_dataset_features(raw_a_path)
    df_feat_a['return_request_date'] = pd.to_datetime(df_feat_a['return_request_date'])
    df_feat_a['order_date'] = pd.to_datetime(df_feat_a['order_date'])
    df_feat_a = df_feat_a.sort_values(by=['return_request_date', 'order_date', 'order_id']).reset_index(drop=True)

    n_a = len(df_feat_a)
    train_a = df_feat_a.iloc[:int(n_a * 0.70)].copy().reset_index(drop=True)
    val_a = df_feat_a.iloc[int(n_a * 0.70):int(n_a * 0.85)].copy().reset_index(drop=True)
    test_a = df_feat_a.iloc[int(n_a * 0.85):].copy().reset_index(drop=True)

    print(f"[SPLIT] Chronological 70/15/15 split: Train={len(train_a)}, Val={len(val_a)}, Test={len(test_a)}")

    # 3. Exact 55 Features definition
    cat_cols = ['category', 'payment_method', 'channel', 'return_reason', 'customer_segment']
    
    metadata_cols = [
        'order_id', 'customer_id', 'transaction_id', 'device_id',
        'shipping_address_id', 'billing_address_id', 'order_date',
        'return_request_date', 'abuse_label', 'abuse_type'
    ]
    
    feature_cols = [c for c in df_feat_a.columns if c not in metadata_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    assert len(feature_cols) == 55, f"Expected 55 features, got {len(feature_cols)}"
    assert len(cat_cols) == 5, f"Expected 5 categorical features, got {len(cat_cols)}"
    assert len(num_cols) == 50, f"Expected 50 numerical features, got {len(num_cols)}"

    print(f"[FEATURES] Total Features: {len(feature_cols)} (5 Categorical, 50 Numerical)")

    # 4. Create and fit preprocessor & model
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

    X_train_enc = preprocessor.fit_transform(train_a[feature_cols])
    y_train = train_a['abuse_label'].values

    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    clf.fit(X_train_enc, y_train)
    print("[FIT] Fitted Preprocessor and GradientBoostingClassifier on Train set successfully.")

    # Create combined pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    # 5. Reference predictions before serialization
    X_val = val_a[feature_cols]
    y_val = val_a['abuse_label'].values
    ref_val_prob = clf.predict_proba(preprocessor.transform(X_val))[:, 1]

    X_test = test_a[feature_cols]
    y_test = test_a['abuse_label'].values
    ref_test_prob = clf.predict_proba(preprocessor.transform(X_test))[:, 1]

    # Extract Independent Dataset B for out-of-distribution check
    print("\n[PROCESS] Extracting Step 3.3 hybrid features for Dataset B...")
    df_feat_b = extract_hybrid_dataset_features(raw_b_path)
    X_b = df_feat_b[feature_cols]
    ref_b_prob = clf.predict_proba(preprocessor.transform(X_b))[:, 1]

    # 6. Save artifacts
    model_path = os.path.join(base_dir, "riskgraph_model_c.joblib")
    prep_path = os.path.join(base_dir, "riskgraph_preprocessor_c.joblib")
    pipe_path = os.path.join(base_dir, "riskgraph_pipeline_c.joblib")
    meta_path = os.path.join(base_dir, "model_c_metadata.json")

    print(f"\n[SAVE] Saving artifacts to {base_dir}...")
    joblib.dump(clf, model_path, compress=3)
    joblib.dump(preprocessor, prep_path, compress=3)
    joblib.dump(pipeline, pipe_path, compress=3)

    metadata = {
        'model_name': 'GradientBoosting_Hybrid_Model_C',
        'model_version': 'v3.4_frozen_model_c',
        'feature_set_version': 'SET_C_55_HYBRID_FEATURES',
        'n_features': 55,
        'categorical_features': cat_cols,
        'numerical_features': num_cols,
        'feature_columns_order': feature_cols,
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'random_state': 42
        },
        'thresholds': {
            't1_low_med': 0.20,
            't2_med_high': 0.45,
            'policy': 'T1_0.20_T2_0.45_val_optimized'
        },
        'train_samples': len(train_a),
        'val_samples': len(val_a),
        'test_samples': len(test_a),
        'dataset_a_sha256': initial_hashes.get('riskgraph_ecommerce_dataset.csv', ''),
        'dataset_b_sha256': initial_hashes.get('riskgraph_independent_raw_dataset.csv', '')
    }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"  --> Saved: riskgraph_model_c.joblib ({os.path.getsize(model_path):,} bytes)")
    print(f"  --> Saved: riskgraph_preprocessor_c.joblib ({os.path.getsize(prep_path):,} bytes)")
    print(f"  --> Saved: riskgraph_pipeline_c.joblib ({os.path.getsize(pipe_path):,} bytes)")
    print(f"  --> Saved: model_c_metadata.json ({os.path.getsize(meta_path):,} bytes)")

    # 7. Verification: Load saved artifacts back from disk
    print("\n" + "=" * 80)
    print("VERIFYING LOADED ARTIFACTS AGAINST IN-MEMORY REFERENCE IMPLEMENTATION")
    print("=" * 80)

    loaded_model = joblib.load(model_path)
    loaded_prep = joblib.load(prep_path)
    loaded_pipeline = joblib.load(pipe_path)

    # Verification on Validation Set (N=219)
    val_enc_loaded = loaded_prep.transform(X_val)
    loaded_val_prob = loaded_model.predict_proba(val_enc_loaded)[:, 1]
    pipe_val_prob = loaded_pipeline.predict_proba(X_val)[:, 1]

    val_max_diff_model = np.max(np.abs(ref_val_prob - loaded_val_prob))
    val_max_diff_pipe = np.max(np.abs(ref_val_prob - pipe_val_prob))

    print(f"[CHECK] Validation Set (N={len(val_a)}):")
    print(f"  - Model + Prep Max Prob Difference: {val_max_diff_model:.2e} (Pass: {val_max_diff_model < 1e-7})")
    print(f"  - Combined Pipeline Max Difference: {val_max_diff_pipe:.2e} (Pass: {val_max_diff_pipe < 1e-7})")
    assert val_max_diff_model < 1e-7, "Validation prediction mismatch!"
    assert val_max_diff_pipe < 1e-7, "Pipeline prediction mismatch!"

    # Verification on Held-Out Temporal Test Set (N=219)
    test_enc_loaded = loaded_prep.transform(X_test)
    loaded_test_prob = loaded_model.predict_proba(test_enc_loaded)[:, 1]
    pipe_test_prob = loaded_pipeline.predict_proba(X_test)[:, 1]

    test_max_diff_model = np.max(np.abs(ref_test_prob - loaded_test_prob))
    test_max_diff_pipe = np.max(np.abs(ref_test_prob - pipe_test_prob))

    print(f"[CHECK] Temporal Test Set (N={len(test_a)}):")
    print(f"  - Model + Prep Max Prob Difference: {test_max_diff_model:.2e} (Pass: {test_max_diff_model < 1e-7})")
    print(f"  - Combined Pipeline Max Difference: {test_max_diff_pipe:.2e} (Pass: {test_max_diff_pipe < 1e-7})")
    assert test_max_diff_model < 1e-7, "Test prediction mismatch!"
    assert test_max_diff_pipe < 1e-7, "Pipeline test prediction mismatch!"

    # Verification on Independent Dataset B (N=1,193)
    b_enc_loaded = loaded_prep.transform(X_b)
    loaded_b_prob = loaded_model.predict_proba(b_enc_loaded)[:, 1]
    pipe_b_prob = loaded_pipeline.predict_proba(X_b)[:, 1]

    b_max_diff_model = np.max(np.abs(ref_b_prob - loaded_b_prob))
    b_max_diff_pipe = np.max(np.abs(ref_b_prob - pipe_b_prob))

    print(f"[CHECK] Independent Dataset B (N={len(df_feat_b)}):")
    print(f"  - Model + Prep Max Prob Difference: {b_max_diff_model:.2e} (Pass: {b_max_diff_model < 1e-7})")
    print(f"  - Combined Pipeline Max Difference: {b_max_diff_pipe:.2e} (Pass: {b_max_diff_pipe < 1e-7})")
    assert b_max_diff_model < 1e-7, "Dataset B prediction mismatch!"
    assert b_max_diff_pipe < 1e-7, "Pipeline Dataset B prediction mismatch!"

    # 8. Check specific known sample cases
    print("\n[CHECK] Known Individual Test Cases:")
    sample_indices = [0, 50, 100, 150, 200]
    for idx in sample_indices:
        case_row = test_a.iloc[idx]
        p_ref = ref_test_prob[idx]
        p_load = loaded_test_prob[idx]
        p_pipe = pipe_test_prob[idx]
        print(f"  - Case {case_row['order_id']} ({case_row['customer_id']}): Reference = {p_ref:.6f} | Loaded = {p_load:.6f} | Pipeline = {p_pipe:.6f} | Delta = {abs(p_ref - p_load):.2e}")

    # 9. Verify dataset integrity (Ensure no CSV was touched)
    print("\n" + "=" * 80)
    print("VERIFYING DATASET INTEGRITY (ZERO MODIFICATIONS)")
    print("=" * 80)
    for name, path in datasets.items():
        if os.path.exists(path):
            current_hash = get_file_sha256(path)
            match = (current_hash == initial_hashes[name])
            print(f"[INTEGRITY] {name:<42} Match: {match} (Initial: {initial_hashes[name][:12]}... == Current: {current_hash[:12]}...)")
            assert match, f"CRITICAL: Dataset {name} was modified!"

    print("\n" + "=" * 80)
    print("ALL SERIALIZATION AND EXACT VERIFICATION CHECKS PASSED (DIFF < 1e-7)")
    print("=" * 80)

if __name__ == "__main__":
    main()
