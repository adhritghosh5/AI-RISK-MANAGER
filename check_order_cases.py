import os
import sys
import json
import traceback

try:
    from riskgraph_backend.main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)

    cases = ['ORD100028', 'ORD100007', 'ORD100003', 'ORD100001', 'ORD100002', 'ORD100004', 'ORD100005', 'ORD100006', 'ORD100010', 'ORD100015', 'ORD100020']
    results = {}
    for oid in cases:
        resp = client.post('/assess', json={'order_id': oid})
        if resp.status_code == 200:
            d = resp.json()
            results[oid] = {
                'risk_score': d.get('risk_score'),
                'risk_level': d.get('risk_level'),
                'recommended_action': d.get('recommended_action'),
                'top_signals': d.get('top_signals')
            }
        else:
            results[oid] = {'error': resp.status_code, 'text': resp.text}

    out_path = os.path.join(r"c:\Users\adhri\Downloads\Project-RiskGraph", "order_check_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print("SUCCESS: Wrote", out_path)
except Exception as e:
    err_path = os.path.join(r"c:\Users\adhri\Downloads\Project-RiskGraph", "error.txt")
    with open(err_path, 'w', encoding='utf-8') as f:
        f.write(traceback.format_exc())
    print("ERROR:", e)


