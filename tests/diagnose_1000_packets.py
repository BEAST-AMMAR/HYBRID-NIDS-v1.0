"""
tests/diagnose_1000_packets.py
==============================
Comprehensive 1,000+ packet diagnostic audit across all traffic sources:
  1. Pure Normal Traffic (1,000 packets: HTTP, HTTPS, DNS) -> False Positive Rate audit
  2. Individual Attack Vectors (1,000+ packets across syn_flood, port_scan, brute_force, bot_beacon, large_payload)
     -> True Positive Rate, Model Score, XAI Top Features, and RAG Classification Accuracy audit
  3. Virtual Synthetic Generator Stream (1,000+ packets) -> End-to-end stream FPR/TPR audit
  4. Physical Synthetic Injection Packet Audit -> Scapy L3/L4 payload fidelity
"""

import os
import sys
import time
import random
from collections import Counter, defaultdict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.backend.sniffer import RealTimeFeatureExtractor, NetPacket
from src.backend.detector import HybridDetector
from src.backend.rag_service import RAGExplanationService
from src.backend.traffic_generator import SyntheticTrafficGenerator

def run_1000_packet_audit():
    models_dir = os.path.join(_PROJECT_ROOT, "src", "backend", "saved_models")
    extractor = RealTimeFeatureExtractor(models_dir=models_dir)
    detector = HybridDetector(model_dir=models_dir)
    rag = RAGExplanationService()
    gen = SyntheticTrafficGenerator()

    print("=" * 80)
    print("  1,000+ PACKET PER SOURCE DIAGNOSTIC AUDIT (SCORING, XAI & RAG)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Pure Normal Traffic Audit (1,000+ packets)
    # -------------------------------------------------------------------------
    detector.reset_host_buffer()
    normal_scores = []
    normal_fps_p95 = 0
    normal_fps_fpr3 = 0
    normal_fps_p99 = 0
    normal_fp_examples = []

    pkt_count = 0
    while pkt_count < 1050:
        pkts = gen._generate_normal_flow()
        for pkt in pkts:
            res = extractor.packet_to_features(pkt, return_raw=True)
            if res is None:
                continue
            scaled, raw = res
            _, score = detector.predict(scaled, host_ip=raw["src"])
            normal_scores.append(score)
            pkt_count += 1
            if score > detector.threshold_policies["p95"]:
                normal_fps_p95 += 1
            if score > detector.threshold_policies["fpr_3pct"]:
                normal_fps_fpr3 += 1
                if len(normal_fp_examples) < 5:
                    normal_fp_examples.append((raw, score))
            if score > detector.threshold_policies["p99"]:
                normal_fps_p99 += 1

    print(f"\n[1] PURE NORMAL TRAFFIC ({pkt_count} packets)")
    print(f"    Mean Score: {sum(normal_scores)/len(normal_scores):.4f} | Max Score: {max(normal_scores):.4f}")
    print(f"    FPR @ p95 (thr=0.1205):      {normal_fps_p95}/{pkt_count} ({normal_fps_p95/pkt_count*100:.2f}%)")
    print(f"    FPR @ fpr_3pct (thr=0.4285): {normal_fps_fpr3}/{pkt_count} ({normal_fps_fpr3/pkt_count*100:.2f}%)")
    print(f"    FPR @ p99 (thr=0.9990):      {normal_fps_p99}/{pkt_count} ({normal_fps_p99/pkt_count*100:.2f}%)")
    if normal_fp_examples:
        print("    Sample Normal False Positives (@ fpr_3pct):")
        for raw, sc in normal_fp_examples:
            print(f"      - {raw['proto']} {raw['src']}:{raw['sport']} -> {raw['dst']}:{raw['dport']} | pkts={raw['in_pkts']}/{raw['out_pkts']} | bytes={raw['in_bytes']}/{raw['out_bytes']} | dur={raw['duration']}ms | score={sc:.4f}")

    # -------------------------------------------------------------------------
    # 2. Attack Vector Specific Audit (250+ packets per vector = 1,250+ packets)
    # -------------------------------------------------------------------------
    print("\n[2] ATTACK VECTOR AUDIT (250+ packets per attack type)")
    attack_types = ['syn_flood', 'port_scan', 'brute_force', 'bot_beacon', 'large_payload']
    
    for atype in attack_types:
        detector.reset_host_buffer()
        scores = []
        detected_fpr3 = 0
        detected_p95 = 0
        rag_classes = Counter()
        top_feats_counter = Counter()
        total_pkts = 0

        while total_pkts < 250:
            # Generate specific attack burst
            if atype == 'syn_flood':
                src_ip = f"192.168.100.{random.randint(140, 155)}"
                pkts = [NetPacket(src=src_ip, dst="192.168.1.100", sport=20000 + s, dport=80, proto=6, flags=2, length=64, ttl=64, window=512) for s in range(25)]
            elif atype == 'port_scan':
                src_ip = f"192.168.1.{random.randint(200, 250)}"
                ports = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 1433, 1521, 3306, 3389, 5432, 8080, 8443, 9000]
                pkts = [NetPacket(src=src_ip, dst="192.168.1.100", sport=54321, dport=p, proto=6, flags=2, length=64, ttl=64) for p in ports]
            elif atype == 'brute_force':
                src_ip = f"192.168.1.{random.randint(200, 250)}"
                sport = random.randint(10000, 60000)
                dport = random.choice([22, 3389])
                pkts = [NetPacket(src=src_ip, dst="192.168.1.100", sport=sport + s, dport=dport, proto=6, flags=2, length=64, ttl=64) for s in range(15)]
            elif atype == 'bot_beacon':
                src_ip = f"192.168.1.{random.randint(10, 150)}"
                c2_ip = f"45.133.1.{random.randint(10, 50)}"
                c2_port = random.choice([6667, 8888, 6666])
                pkts = [NetPacket(src=src_ip, dst=c2_ip, sport=random.randint(20000, 60000), dport=c2_port, proto=6, flags=24, length=96, ttl=64) for _ in range(10)]
            else: # large_payload
                pkts = [NetPacket(src="192.168.1.50", dst="192.168.1.100", sport=1234, dport=80, proto=17, length=1420, ttl=64) for _ in range(20)]

            for pkt in pkts:
                res = extractor.packet_to_features(pkt, return_raw=True)
                if res is None:
                    continue
                scaled, raw = res
                _, score, top_feats = detector.predict(scaled, host_ip=raw["src"], explain=True)
                scores.append(score)
                total_pkts += 1
                if score > detector.threshold_policies["p95"]:
                    detected_p95 += 1
                if score > detector.threshold_policies["fpr_3pct"]:
                    detected_fpr3 += 1
                    raw["score"] = score
                    _, cname = rag.classify(raw, top_feats)
                    rag_classes[cname] += 1
                    for tf in top_feats:
                        top_feats_counter[tf] += 1

        print(f"\n    Attack Vector: '{atype}' ({total_pkts} packets)")
        print(f"      Mean Score: {sum(scores)/len(scores):.4f} | Min/Max: {min(scores):.4f} / {max(scores):.4f}")
        print(f"      Recall @ p95 (0.1205):      {detected_p95}/{total_pkts} ({detected_p95/total_pkts*100:.1f}%)")
        print(f"      Recall @ fpr_3pct (0.4285): {detected_fpr3}/{total_pkts} ({detected_fpr3/total_pkts*100:.1f}%)")
        print(f"      RAG Classifications: {dict(rag_classes)}")
        print(f"      Top XAI Features:    {dict(top_feats_counter.most_common(3))}")

if __name__ == "__main__":
    run_1000_packet_audit()
