"""
RISKGRAPH — AI Risk Manager
Track 02: AI Risk Manager
Problem: Defensive E-Commerce Return + Refund Abuse Detection

STEP 3.5: AI Risk Manager Backend & Decision Support Server
- Lightweight HTTP API serving the RiskGraph decision support interface.
- Encapsulates frozen Model C (55 Point-in-Time Features) and T1=0.20 / T2=0.45 thresholds.
- Reconstructs point-in-time historical and graph signals on the fly.
- Zero future lookahead, zero target leakage, human-in-the-loop guarantee.
"""

import os
import sys
import json
import mimetypes
import pandas as pd
import numpy as np
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Ensure utf-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path to import defensive_risk_router and entity_graph_analysis
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defensive_risk_router import DefensiveRiskRouter
from entity_graph_analysis import DefensiveEntityGraphEngine
from hybrid_riskgraph_model import extract_hybrid_dataset_features

# Global Server State
RAW_DATASET_PATH = r"c:\Users\adhri\Downloads\Project-RiskGraph\riskgraph_ecommerce_dataset.csv"
ENGINE = None
ROUTER = None
SESSION_STATS = {
    'total_assessed': 0,
    'low_risk': 0,
    'medium_risk': 0,
    'high_risk': 0,
    'history': []
}

def initialize_ai_risk_manager():
    global ENGINE, ROUTER
    print("[INIT] Loading historical transaction database and initializing RiskGraph Engine...")
    ENGINE = DefensiveEntityGraphEngine(RAW_DATASET_PATH)
    
    df_feat_a = extract_hybrid_dataset_features(RAW_DATASET_PATH)
    df_feat_a['return_request_date'] = pd.to_datetime(df_feat_a['return_request_date'])
    df_feat_a['order_date'] = pd.to_datetime(df_feat_a['order_date'])
    df_feat_a = df_feat_a.sort_values(by=['return_request_date', 'order_date', 'order_id']).reset_index(drop=True)
    
    n_a = len(df_feat_a)
    train_a = df_feat_a.iloc[:int(n_a * 0.70)].copy().reset_index(drop=True)
    
    metadata_cols = [
        'order_id', 'customer_id', 'transaction_id', 'device_id',
        'shipping_address_id', 'billing_address_id', 'order_date',
        'return_request_date', 'abuse_label', 'abuse_type'
    ]
    model_c_features = [c for c in df_feat_a.columns if c not in metadata_cols]
    
    ROUTER = DefensiveRiskRouter(t1=0.20, t2=0.45)
    ROUTER.fit_and_freeze_model_c(train_a, model_c_features)
    print(f"[INIT] RiskGraph Frozen Model C (55 features) ready with thresholds T1=0.20, T2=0.45.")

def get_demo_cases():
    """Returns 3 preloaded demonstration cases representing Low, Medium, and High risk profiles."""
    return {
        'low_risk_demo': {
            'order_id': 'DEMO-LOW-001',
            'customer_id': 'CUST100050',
            'amount': 450.00,
            'quantity': 1,
            'category': 'Fashion',
            'payment_method': 'Credit Card',
            'channel': 'Mobile App',
            'return_reason': 'Size issue',
            'customer_segment': 'Premium',
            'customer_tenure_days': 1200,
            'order_date': '2024-11-20',
            'shipping_date': '2024-11-22',
            'return_request_date': '2024-11-24',
            'device_id': 'DEV200050',
            'shipping_address_id': 'ADDR300050',
            'billing_address_id': 'ADDR300050',
            'demo_label': 'Low-Risk Profile (Established Premium Buyer, Size Issue Return)'
        },
        'medium_risk_demo': {
            'order_id': 'DEMO-MED-002',
            'customer_id': 'CUST100120',
            'amount': 1850.00,
            'quantity': 1,
            'category': 'Electronics',
            'payment_method': 'UPI',
            'channel': 'Website',
            'return_reason': 'Changed mind',
            'customer_segment': 'Mass Market',
            'customer_tenure_days': 250,
            'order_date': '2024-11-25',
            'shipping_date': '2024-11-27',
            'return_request_date': '2024-11-28',
            'device_id': 'DEV200120',
            'shipping_address_id': 'ADDR300120',
            'billing_address_id': 'ADDR300120',
            'demo_label': 'Medium-Risk Profile (Changed Mind on Electronics, Shared Hardware)'
        },
        'high_risk_demo': {
            'order_id': 'DEMO-HIGH-003',
            'customer_id': 'CUST100000',
            'amount': 3800.00,
            'quantity': 2,
            'category': 'Fashion',
            'payment_method': 'Wallet',
            'channel': 'Social Commerce',
            'return_reason': 'Changed mind',
            'customer_segment': 'Budget',
            'customer_tenure_days': 45,
            'order_date': '2024-11-15',
            'shipping_date': '2024-11-17',
            'return_request_date': '2024-11-18',
            'device_id': 'DEV202769',
            'shipping_address_id': 'ADDR301608',
            'billing_address_id': 'ADDR301608',
            'demo_label': 'High-Risk Profile (Sybil Order Churn, Multiple Prior Device Refunds)'
        }
    }

class RiskGraphRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        
        if path == '/api/health':
            self._send_json({
                'status': 'healthy',
                'model_name': ROUTER.model_name,
                'model_version': ROUTER.model_version,
                'thresholds': {'t1': ROUTER.t1, 't2': ROUTER.t2},
                'session_stats': SESSION_STATS
            })
            return
            
        elif path == '/api/demo_cases':
            self._send_json(get_demo_cases())
            return
            
        # Serve static files
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if path in ['/', '/index.html']:
            file_path = os.path.join(base_dir, 'index.html')
            content_type = 'text/html; charset=utf-8'
        elif path == '/style.css':
            file_path = os.path.join(base_dir, 'style.css')
            content_type = 'text/css; charset=utf-8'
        elif path == '/app.js':
            file_path = os.path.join(base_dir, 'app.js')
            content_type = 'application/javascript; charset=utf-8'
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return
            
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"File not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        
        if path == '/api/assess':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                request_data = json.loads(body)
            except Exception as e:
                self._send_json({'error': f'Invalid JSON: {str(e)}'}, 400)
                return
                
            # Generate point-in-time features & graph neighborhood
            try:
                t_cutoff = pd.to_datetime(request_data['return_request_date'])
                curr_cust = request_data['customer_id']
                curr_dev = request_data['device_id']
                curr_ship = request_data['shipping_address_id']
                curr_bill = request_data.get('billing_address_id', curr_ship)
                curr_order = request_data.get('order_id', 'REQ_ASSESS')
                
                # Extract graph neighborhood strictly before t_cutoff
                graph_data = ENGINE.resolve_entity_neighborhood(
                    customer_id=curr_cust,
                    device_id=curr_dev,
                    shipping_address_id=curr_ship,
                    billing_address_id=curr_bill,
                    cutoff_timestamp=t_cutoff,
                    current_order_id=curr_order
                )
                
                # Extract customer historical behavioral stats
                df = ENGINE.df_raw
                p_orders = df[(df['customer_id'] == curr_cust) & (df['order_date'] < t_cutoff) & (df['order_id'] != curr_order)]
                p_returns = df[(df['customer_id'] == curr_cust) & (df['return_request_date'].notna()) & (df['return_request_date'] < t_cutoff) & (df['order_id'] != curr_order)]
                p_refunds = df[(df['customer_id'] == curr_cust) & (df['refund_date'].notna()) & (df['refund_date'] < t_cutoff) & (df['return_status'] == 'Refunded') & (df['order_id'] != curr_order)]
                
                prior_orders_count = len(p_orders)
                prior_returns_count = len(p_returns)
                prior_refunds_count = len(p_refunds)
                prior_spend = float(p_orders['amount'].sum()) if prior_orders_count > 0 else 0.0
                prior_refund_amt = float(p_refunds['refund_amount'].astype(float).sum()) if prior_refunds_count > 0 else 0.0
                prior_ret_rate = (prior_returns_count / prior_orders_count) if prior_orders_count > 0 else 0.0
                prior_ref_rate = (prior_refunds_count / prior_orders_count) if prior_orders_count > 0 else 0.0
                
                order_dt = pd.to_datetime(request_data['order_date'])
                if prior_orders_count > 0:
                    last_ord = p_orders['order_date'].max()
                    days_since_prev_ord = (order_dt - last_ord).days
                    avg_prev_ord_val = prior_spend / prior_orders_count
                else:
                    days_since_prev_ord = 999
                    avg_prev_ord_val = 0.0
                    
                t_30 = t_cutoff - pd.Timedelta(days=30)
                t_14 = t_cutoff - pd.Timedelta(days=14)
                t_7 = t_cutoff - pd.Timedelta(days=7)
                
                orders_30 = len(p_orders[p_orders['order_date'] >= t_30])
                returns_30 = len(p_returns[p_returns['return_request_date'] >= t_30])
                refunds_30 = len(p_refunds[p_refunds['refund_date'] >= t_30])
                spend_30 = float(p_orders[p_orders['order_date'] >= t_30]['amount'].sum()) if orders_30 > 0 else 0.0
                refund_amt_30 = float(p_refunds[p_refunds['refund_date'] >= t_30]['refund_amount'].astype(float).sum()) if refunds_30 > 0 else 0.0
                
                orders_14 = len(p_orders[p_orders['order_date'] >= t_14])
                returns_14 = len(p_returns[p_returns['return_request_date'] >= t_14])
                refunds_14 = len(p_refunds[p_refunds['refund_date'] >= t_14])
                
                orders_7 = len(p_orders[p_orders['order_date'] >= t_7])
                returns_7 = len(p_returns[p_returns['return_request_date'] >= t_7])
                refunds_7 = len(p_refunds[p_refunds['refund_date'] >= t_7])
                
                days_since_last_ret = (t_cutoff - p_returns['return_request_date'].max()).days if prior_returns_count > 0 else 999
                days_since_last_ref = (t_cutoff - p_refunds['refund_date'].max()).days if prior_refunds_count > 0 else 999
                
                return_rate_30 = (returns_30 / orders_30) if orders_30 > 0 else 0.0
                refund_rate_30 = (refunds_30 / orders_30) if orders_30 > 0 else 0.0
                refund_to_spend_ratio = (prior_refund_amt / prior_spend) if prior_spend > 0 else 0.0
                amount_to_avg_ratio = (float(request_data['amount']) / (prior_spend / prior_orders_count)) if prior_orders_count > 0 else 1.0
                
                ship_dt = pd.to_datetime(request_data.get('shipping_date', order_dt + pd.Timedelta(days=2)))
                days_since_order = (t_cutoff - order_dt).days
                days_since_delivery = (t_cutoff - ship_dt).days
                is_addr_mismatch = int(curr_ship != curr_bill)
                
                # Build complete 55-feature dictionary
                full_feature_dict = {
                    'order_id': curr_order, 'customer_id': curr_cust,
                    'category': request_data['category'],
                    'payment_method': request_data['payment_method'],
                    'channel': request_data['channel'],
                    'return_reason': request_data['return_reason'],
                    'customer_segment': request_data['customer_segment'],
                    'current_order_amount': float(request_data['amount']),
                    'current_order_quantity': int(request_data.get('quantity', 1)),
                    'customer_tenure_days': int(request_data.get('customer_tenure_days', 100)),
                    'days_since_order': int(days_since_order),
                    'days_since_delivery': int(days_since_delivery),
                    'is_address_mismatch': int(is_addr_mismatch),
                    'prior_order_count': int(prior_orders_count),
                    'prior_return_count': int(prior_returns_count),
                    'prior_refund_count': int(prior_refunds_count),
                    'prior_spend': float(prior_spend),
                    'prior_refund_amount': float(prior_refund_amt),
                    'prior_return_rate': float(prior_ret_rate),
                    'prior_refund_rate': float(prior_ref_rate),
                    'days_since_previous_order': int(days_since_prev_ord),
                    'average_previous_order_value': float(avg_prev_ord_val),
                    'orders_last_30_days': int(orders_30),
                    'returns_last_30_days': int(returns_30),
                    'refunds_last_30_days': int(refunds_30),
                    'spend_last_30_days': float(spend_30),
                    'refund_amount_last_30_days': float(refund_amt_30),
                    'device_prior_return_count': int(graph_data['device_prior_return_count']),
                    'device_prior_refund_count': int(graph_data['device_prior_refund_count']),
                    'accounts_per_device': int(graph_data['accounts_per_device']),
                    'address_prior_return_count': int(graph_data['address_prior_return_count']),
                    'address_prior_refund_count': int(graph_data['address_prior_refund_count']),
                    'accounts_per_shipping_address': int(graph_data['accounts_per_shipping_address']),
                    'is_weekend_order': int(order_dt.weekday() >= 5),
                    'is_weekend_return_request': int(t_cutoff.weekday() >= 5),
                    'orders_last_7_days': int(orders_7),
                    'returns_last_7_days': int(returns_7),
                    'refunds_last_7_days': int(refunds_7),
                    'orders_last_14_days': int(orders_14),
                    'returns_last_14_days': int(returns_14),
                    'refunds_last_14_days': int(refunds_14),
                    'days_since_last_return': int(days_since_last_ret),
                    'days_since_last_refund': int(days_since_last_ref),
                    'return_rate_last_30_days': float(return_rate_30),
                    'refund_rate_last_30_days': float(refund_rate_30),
                    'refund_to_spend_ratio': float(refund_to_spend_ratio),
                    'amount_to_avg_ratio': float(amount_to_avg_ratio),
                    'device_prior_order_count': int(graph_data['device_prior_order_count']),
                    'device_prior_refund_amount': float(graph_data['device_prior_refund_amount']),
                    'device_distinct_accounts': len(graph_data['device_other_accounts']),
                    'address_prior_order_count': int(graph_data['address_prior_order_count']),
                    'address_prior_refund_amount': float(graph_data['address_prior_refund_amount']),
                    'billing_address_prior_order_count': int(graph_data['billing_address_prior_order_count']),
                    'billing_address_prior_return_count': int(graph_data['billing_address_prior_return_count']),
                    'billing_address_prior_refund_count': int(graph_data['billing_address_prior_refund_count']),
                    'accounts_per_billing_address': int(graph_data['accounts_per_billing_address']),
                    'total_linked_external_accounts': int(graph_data['total_linked_external_accounts'])
                }
                
                # Execute Risk Routing
                assessment = ROUTER.route_request(full_feature_dict)
                
                # Separate Behavioral vs Entity/Graph signals for clean explainability
                behavioral_signals = {
                    'orders_last_30_days': orders_30,
                    'orders_last_14_days': orders_14,
                    'prior_order_count': prior_orders_count,
                    'prior_return_count': prior_returns_count,
                    'prior_refund_count': prior_refunds_count,
                    'prior_return_rate': round(prior_ret_rate, 2),
                    'prior_spend': f"INR {prior_spend:.2f}",
                    'customer_tenure_days': request_data.get('customer_tenure_days', 100),
                    'days_since_delivery': days_since_delivery
                }
                
                entity_signals = {
                    'device_id': curr_dev,
                    'accounts_on_device': graph_data['accounts_per_device'],
                    'device_prior_refunds': graph_data['device_prior_refund_count'],
                    'device_other_accounts': graph_data['device_other_accounts'],
                    'shipping_address_id': curr_ship,
                    'accounts_on_shipping_address': graph_data['accounts_per_shipping_address'],
                    'shipping_address_prior_refunds': graph_data['address_prior_refund_count'],
                    'shipping_other_accounts': graph_data['shipping_other_accounts'],
                    'billing_address_id': curr_bill,
                    'accounts_on_billing_address': graph_data['accounts_per_billing_address']
                }
                
                # Format simple relationship tree structure
                graph_tree = {
                    'customer': curr_cust,
                    'device': {
                        'device_id': curr_dev,
                        'connected_accounts': graph_data['device_other_accounts']
                    },
                    'shipping_address': {
                        'address_id': curr_ship,
                        'connected_accounts': graph_data['shipping_other_accounts']
                    },
                    'billing_address': {
                        'address_id': curr_bill,
                        'connected_accounts': graph_data['billing_other_accounts']
                    }
                }
                
                # Update Session Stats
                SESSION_STATS['total_assessed'] += 1
                if assessment['routing_tier'] == 'LOW':
                    SESSION_STATS['low_risk'] += 1
                elif assessment['routing_tier'] == 'MEDIUM':
                    SESSION_STATS['medium_risk'] += 1
                else:
                    SESSION_STATS['high_risk'] += 1
                    
                response_payload = {
                    'order_id': curr_order,
                    'customer_id': curr_cust,
                    'risk_probability': assessment['risk_probability'],
                    'risk_percentage': f"{int(assessment['risk_probability'] * 100)}%",
                    'routing_tier': assessment['routing_tier'],
                    'recommended_action': assessment['recommended_action'],
                    'policy_rationale': assessment['policy_rationale'],
                    'top_signals': assessment['top_signals'],
                    'behavioral_signals': behavioral_signals,
                    'entity_signals': entity_signals,
                    'graph_tree': graph_tree,
                    'audit': assessment,
                    'session_stats': SESSION_STATS
                }
                
                self._send_json(response_payload, 200)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({'error': f'Assessment error: {str(e)}'}, 500)
                
        elif path == '/api/review_action':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            review_payload = json.loads(body)
            # Record analyst investigation notes
            self._send_json({
                'status': 'recorded',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'review_log': review_payload
            }, 200)
        else:
            self.send_response(404)
            self.end_headers()

def start_server(port=8080):
    initialize_ai_risk_manager()
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, RiskGraphRequestHandler)
    print(f"\n=======================================================")
    print(f"RISKGRAPH AI Risk Manager running at: http://127.0.0.1:{port}/")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down gracefully.")
        httpd.server_close()

if __name__ == "__main__":
    start_server(port=8080)
