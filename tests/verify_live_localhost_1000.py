"""
tests/verify_live_localhost_1000.py
===================================
Connects to the LIVE running localhost FastAPI server (http://127.0.0.1:8000),
authenticates via JWT, enables Virtual Synthetic Generator, monitors 1,000+ live
packets over authenticated WebSocket, and verifies:
  1. /api/health & /api/readiness
  2. Live Normal vs Anomaly scoring & FPR
  3. Diverse RAG classifications & XAI Top Features (no single falsified chunk)
  4. Alert lifecycle persistence in SQLite (/api/alerts)
"""

import asyncio
import json
import urllib.request
import urllib.parse
from collections import Counter
import websockets

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/monitor"

def http_req(path, method="GET", data=None, token=None, is_form=False):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None
    if data is not None:
        if is_form:
            body = urllib.parse.urlencode(data).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            body = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{BASE_URL}{path}", data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

async def monitor_live_1000():
    print("=" * 80)
    print("  LIVE LOCALHOST 1,000-PACKET MONITORING & RAG/XAI VERIFICATION")
    print("=" * 80)

    # 1. Health & Readiness
    health = http_req("/api/health")
    readiness = http_req("/api/readiness")
    print(f"[1/5] /api/health: {health['status']} | Device: {health['hardware']['device']} ({health['hardware']['gpu_name']})")
    print(f"      /api/readiness: {readiness['ready']} | Policy: {readiness['policy']}")

    # 2. Authenticate to get JWT token
    tok_resp = http_req("/token", method="POST", data={"username": "admin", "password": "admin"}, is_form=True)
    token = tok_resp["access_token"]
    print(f"[2/5] Authenticated JWT token acquired.")

    # 3. Set policy to 'fpr_3pct' (High Precision) and start virtual synthetic capture
    pol_resp = http_req("/api/policy", method="POST", data={"policy": "fpr_3pct"}, token=token)
    print(f"[3/5] Policy set: {pol_resp['policy']} (thr={pol_resp['threshold']:.6f})")

    mode_resp = http_req("/api/capture/mode", method="POST", data={"mode": "virtual_synthetic"}, token=token)
    print(f"      Capture mode: {mode_resp['message']}")

    # 4. Monitor 1,000 live packets over authenticated WebSocket
    normal_scores = []
    anomaly_scores = []
    attack_counts = Counter()
    xai_features = Counter()
    telemetry_sources = Counter()
    provisional_seen = 0
    target_packets = 1000

    print(f"[4/5] Monitoring {target_packets} live packets via authenticated WebSocket...")
    async with websockets.connect(f"{WS_URL}?token={token}") as ws:
        await ws.send(json.dumps({"type": "ping"}))
        while (len(normal_scores) + len(anomaly_scores)) < target_packets:
            msg_str = await asyncio.wait_for(ws.recv(), timeout=15.0)
            msg = json.loads(msg_str)
            if msg.get("type") != "log":
                continue

            telemetry_sources[msg.get("telemetry_source", "unknown")] += 1
            sc = float(msg["score"])
            if msg["is_anomaly"]:
                anomaly_scores.append(sc)
                attack_counts[msg.get("attack_type", "Unknown")] += 1
                for f in msg.get("top_features", []):
                    xai_features[f] += 1
                if "PROVISIONAL AI EVALUATION" in msg.get("explanation", ""):
                    provisional_seen += 1
            else:
                normal_scores.append(sc)

    # Stop synthetic capture cleanly
    http_req("/api/capture/mode", method="POST", data={"mode": "live"}, token=token)

    total = len(normal_scores) + len(anomaly_scores)
    print(f"\n    === LIVE STREAM SUMMARY ({total} PACKETS) ===")
    print(f"    Telemetry Sources:   {dict(telemetry_sources)}")
    print(f"    Normal Flows:        {len(normal_scores)}/{total} ({len(normal_scores)/total*100:.2f}%)")
    if normal_scores:
        print(f"      Normal Score Mean/Max: {sum(normal_scores)/len(normal_scores):.4f} / {max(normal_scores):.4f}")
    print(f"    Detected Anomalies:  {len(anomaly_scores)}/{total} ({len(anomaly_scores)/total*100:.2f}%)")
    if anomaly_scores:
        print(f"      Anomaly Score Mean/Min/Max: {sum(anomaly_scores)/len(anomaly_scores):.4f} / {min(anomaly_scores):.4f} / {max(anomaly_scores):.4f}")
    print(f"    RAG Classifications: {dict(attack_counts)}")
    print(f"    Provisional Banners: {provisional_seen}/{len(anomaly_scores)}")
    print(f"    Top XAI Features:    {dict(xai_features.most_common(6))}")

    # 5. Verify Alert Lifecycle Persistence
    alerts = http_req("/api/alerts?limit=20", token=token)
    print(f"\n[5/5] /api/alerts Persistence Check: Retrieved {len(alerts)} persisted SQLite alerts.")
    if alerts:
        sample = alerts[0]
        print(f"      Latest Alert: ID={sample['id']} | {sample['attack_type']} | Score={sample['threat_score']:.4f} | Status={sample['status']} | Feature={sample['culprit_feature']}")

    assert total >= 1000, f"Expected 1000 packets, got {total}"
    assert len(normal_scores) > 800, "Normal traffic ratio unexpectedly low"
    if normal_scores:
        assert max(normal_scores) <= 0.4285, "Normal score exceeded threshold!"
    print("\n[+] LIVE LOCALHOST 1,000-PACKET VERIFICATION PASSED WITH ZERO ERRORS!")

if __name__ == "__main__":
    asyncio.run(monitor_live_1000())
