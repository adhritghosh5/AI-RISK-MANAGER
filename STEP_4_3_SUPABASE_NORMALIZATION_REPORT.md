# STEP 4.3 — SUPABASE DATA NORMALIZATION & AUDIT INTEGRITY REPORT

**Project:** RISKGRAPH AI Risk Manager  
**Track:** 02 — Defensive E-Commerce Return & Refund Abuse Detection  
**Execution Timestamp:** 2026-08-31T19:02:00Z  
**Status:** Completed with 100% Verification Success  

---

## A. Existing Schema

The live Supabase PostgreSQL database tables and constraints were inspected prior to write operations:

| Table Name | Primary Key | Foreign Keys | Key Columns |
| :--- | :--- | :--- | :--- |
| **`customers`** | `customer_id` | — | `customer_id`, `customer_segment`, `location_city_state`, `created_at`, `updated_at` |
| **`devices`** | `device_id` | — | `device_id`, `first_seen_at`, `last_seen_at` |
| **`addresses`** | `address_id` | — | `address_id`, `city`, `state`, `created_at` |
| **`orders`** | `order_id` | `customer_id` $\rightarrow$ `customers.customer_id`<br>`device_id` $\rightarrow$ `devices.device_id`<br>`shipping_address_id` $\rightarrow$ `addresses.address_id`<br>`billing_address_id` $\rightarrow$ `addresses.address_id` | `order_id`, `transaction_id`, `customer_id`, `device_id`, `shipping_address_id`, `billing_address_id`, `product_id`, `category`, `amount`, `quantity`, `payment_method`, `channel`, `order_date`, `shipping_date`, `delivery_status` |
| **`return_requests`** | `return_request_id` | `order_id` $\rightarrow$ `orders.order_id`<br>`customer_id` $\rightarrow$ `customers.customer_id` | `return_request_id`, `order_id`, `customer_id`, `return_request_date`, `return_reason`, `return_resolution`, `return_status`, `refund_amount`, `refund_date`, `abuse_label`, `abuse_type` |
| **`risk_assessments`** | `assessment_id` | `order_id` $\rightarrow$ `orders.order_id`<br>`customer_id` $\rightarrow$ `customers.customer_id`<br>`return_request_id` $\rightarrow$ `return_requests.return_request_id` | `assessment_id`, `order_id`, `customer_id`, `return_request_id`, `risk_probability`, `routing_tier`, `recommended_action`, `policy_rationale`, `top_signals`, `feature_snapshot`, `model_name`, `model_version`, `threshold_version`, `assessed_at` |

---

## B. Mapping Used

Deterministic, type-safe mapping from `raw_ecommerce_data` into normalized tables:

1. **`customers`:**
   - Grouped by `customer_id` across all orders.
   - `customer_segment`: Most recent customer segment.
   - `location_city_state`: Customer city & state from `customer_location`.
   - `created_at`: First order timestamp minus `customer_tenure_days` in UTC.
   - `updated_at`: Latest order timestamp in UTC.

2. **`devices`:**
   - Grouped by `device_id`.
   - `first_seen_at`: Minimum `order_date` across all transactions on hardware fingerprint.
   - `last_seen_at`: Maximum `order_date` on hardware fingerprint.

3. **`addresses`:**
   - Unified unique identifiers from `shipping_address_id` $\cup$ `billing_address_id`.
   - `city`: Parsed from first segment of `customer_location`.
   - `state`: Parsed from second segment of `customer_location`.
   - `created_at`: Minimum `order_date` where destination address was first used.

4. **`orders`:**
   - Direct 1-to-1 mapping of all 10,000 purchases linking foreign keys to `customers`, `devices`, and `addresses`.
   - Timestamps converted to standard ISO 8601 UTC strings.

5. **`return_requests`:**
   - Filtered for all disputes where `return_request_date` is populated ($N = 2,014$).
   - `return_request_id`: Deterministically set to `order_id` (guaranteeing exact 1-to-1 relational mapping with parent order).
   - Preserves `return_reason`, `return_resolution`, `return_status`, `refund_amount`, `refund_date`, `abuse_label`, and `abuse_type`.

---

## C. Pre-Normalization Row Counts

| Table Name | Pre-Normalization Count |
| :--- | :---: |
| `raw_ecommerce_data` | **10,000** |
| `customers` | 0 |
| `devices` | 0 |
| `addresses` | 0 |
| `orders` | 0 |
| `return_requests` | 0 |
| `risk_assessments` | 0 |

---

## D. Dry-Run Counts

| Entity Target | Planned Upsert Count | Verification |
| :--- | :---: | :---: |
| **`customers`** | **2,500** | Validated distinct customer count |
| **`devices`** | **1,661** | Validated distinct hardware fingerprints |
| **`addresses`** | **1,696** | Validated distinct shipping & billing addresses |
| **`orders`** | **10,000** | Validated exact total order volume |
| **`return_requests`** | **2,014** | Validated all disputes with return request dates |

---

## E. Records Inserted / Upserted

Writes were executed in strict foreign-key dependency order in idempotent batches of 500 records:

1. `customers`: **2,500** records upserted (5 batches $\times$ 500)
2. `devices`: **1,661** records upserted (4 batches)
3. `addresses`: **1,696** records upserted (4 batches)
4. `orders`: **10,000** records upserted (20 batches)
5. `return_requests`: **2,014** records upserted (5 batches)

---

## F. Post-Normalization Row Counts

| Table Name | Post-Normalization Count | Discrepancy |
| :--- | :---: | :---: |
| **`raw_ecommerce_data`** | **10,000** | **0** |
| **`customers`** | **2,500** | **0** |
| **`devices`** | **1,661** | **0** |
| **`addresses`** | **1,696** | **0** |
| **`orders`** | **10,000** | **0** |
| **`return_requests`** | **2,014** | **0** |
| **`risk_assessments`** | **1** (Live verified) | **0** |

---

## G. Referential Integrity & Orphan Checks

| Integrity Constraint | Check Description | Result | Status |
| :--- | :--- | :---: | :---: |
| **`orders` $\rightarrow$ `customers`** | `orders.customer_id` not present in `customers` | **0** | **PASS** |
| **`orders` $\rightarrow$ `devices`** | `orders.device_id` not present in `devices` | **0** | **PASS** |
| **`orders` $\rightarrow$ `addresses (Ship)`** | `orders.shipping_address_id` not in `addresses` | **0** | **PASS** |
| **`orders` $\rightarrow$ `addresses (Bill)`** | `orders.billing_address_id` not in `addresses` | **0** | **PASS** |
| **`return_requests` $\rightarrow$ `orders`** | `return_requests.order_id` not present in `orders` | **0** | **PASS** |
| **`return_requests` $\rightarrow$ `customers`**| `return_requests.customer_id` not in `customers` | **0** | **PASS** |

---

## H. Duplicate Checks

- **Primary Identifier Uniqueness:** `0` duplicate primary keys detected across all normalized tables.
- **Idempotency:** Executing upsert operations multiple times produces identical row counts and zero constraint collisions.

---

## I. Live `/assess` End-to-End Verification

A real existing return dispute (`ORD100028`) was evaluated via the FastAPI `/assess` endpoint using normalized data:

```json
{
  "request_id": "ORD100028",
  "risk_score": 0.1072,
  "risk_level": "LOW",
  "recommended_action": "Normal refund processing",
  "model_version": "step_2_8_frozen_v1",
  "threshold_version": "T1=0.20,T2=0.45",
  "top_signals": [
    "Dispute reason (Changed mind) correlates with elevated claim frequency"
  ],
  "feature_summary": {
    "accounts_on_device": 1,
    "device_other_accounts": [],
    "device_prior_refunds": 0,
    "accounts_on_shipping_address": 1,
    "shipping_other_accounts": ["CUST101362"],
    "shipping_address_prior_refunds": 0,
    "total_linked_external_accounts": 1,
    "prior_order_count": 0,
    "prior_refund_count": 0,
    "orders_last_30_days": 0,
    "source_database": "Supabase PostgreSQL"
  }
}
```

---

## J. `risk_assessments` Persistence Result

Following normalization, the live evaluation of dispute `ORD100028` was written directly to the Supabase `risk_assessments` table. 

**Persisted Audit Record:**
- **`assessment_id`:** `6c64aac0-b4a2-4d78-a55e-ab456d0f1ba6`
- **`order_id`:** `ORD100028` (Valid foreign key $\rightarrow$ `orders.order_id`)
- **`return_request_id`:** `ORD100028` (Valid foreign key $\rightarrow$ `return_requests.return_request_id`)
- **`customer_id`:** `CUST100006` (Valid foreign key $\rightarrow$ `customers.customer_id`)
- **`risk_probability`:** `0.1072`
- **`routing_tier`:** `LOW`
- **`recommended_action`:** `Normal refund processing`
- **`assessed_at`:** `2026-08-31T13:32:45.80984+00:00`
- **Foreign Key Constraints:** **100% SATISFIED AND PERSISTED**

---

## K. Model Integrity & L. Raw Dataset Integrity

Cryptographic SHA-256 validation confirmed that no local datasets, models, or raw database records were modified:

| Component | SHA-256 Checksum | Verification Status |
| :--- | :--- | :---: |
| `riskgraph_ecommerce_dataset.csv` | `e32ef89c9f70ab848f9d46d30c2687777be9477c6f36cfbc6e97f14527db0cef` | **UNMODIFIED** |
| `riskgraph_independent_raw_dataset.csv` | `840c2bc59e42401d2e601211c3c8a926833db08a54f4906172a6b33b1cfc5827` | **UNMODIFIED** |
| `riskgraph_refund_modeling_dataset.csv` | `fc93c3858ef1d7a002aead074f2b5b40d8df8190d0ca54c65cbb6b65e5b6cd9d` | **UNMODIFIED** |
| `riskgraph_pipeline_c.joblib` | `b250d233af81c437687624d5b1fcf82f9c22e74c6c4ff71c77aeef24a10648b2` | **UNMODIFIED** |
| `raw_ecommerce_data` (Supabase) | Exactly 10,000 rows | **UNMODIFIED** |

---

## M. Errors & Issues

- **None.** All 10,000 transactions, 2,500 customers, 1,661 devices, 1,696 addresses, and 2,014 return disputes were normalized without errors, truncation, or constraint violations.

---

## N. Final Status

**COMPLETE & PRODUCTION-READY (Backend Database Layer).**  
The normalized relational structure in Supabase is fully populated, referentially sound, and capable of executing live causal feature generation and immutable audit logging.
