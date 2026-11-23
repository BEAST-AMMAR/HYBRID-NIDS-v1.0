import pyarrow.parquet as pq
import numpy as np
from src.backend.sniffer import RealTimeFeatureExtractor, NetPacket
from src.backend.detector import HybridDetector

ext = RealTimeFeatureExtractor('src/backend/saved_models')
det = HybridDetector('src/backend/saved_models')

# Real DoS flow from dataset
table = pq.read_table('data/NF UQ NIDS V2/parquet/part_0000.parquet')
df = table.to_pandas()
real_dos_row = df[df['Attack'] == 'DoS'].iloc[[0]]
feat_real = ext.adapter.transform(real_dos_row)
score_real = det.predict(feat_real, host_ip="test_real")[1]
print(f"Real DoS Score: {score_real:.4f}")

# Synthetic syn_flood flow
pkts = [NetPacket(src='10.0.0.1', dst='192.168.1.100', sport=1024+i, dport=80, proto=6, flags=2, length=64, ttl=64) for i in range(35)]
for p in pkts:
    res = ext.packet_to_features(p, return_raw=True)
scaled_syn, raw_syn = res
score_syn = det.predict(scaled_syn, host_ip="10.0.0.1")[1]
print(f"Synthetic SYN Flood Score: {score_syn:.4f}")

# Compare column by column
feat_cols = ext.adapter.final_columns
diff = np.abs(feat_real[0] - scaled_syn[0])
top_diff_idx = np.argsort(diff)[-10:][::-1]

print("\nTop 10 Feature Differences between Real DoS and Synthetic:")
for idx in top_diff_idx:
    col = feat_cols[idx]
    print(f"  {col:30s} | Real: {feat_real[0, idx]:8.4f} | Syn: {scaled_syn[0, idx]:8.4f} | Diff: {diff[idx]:8.4f}")
    if col in real_dos_row.columns:
        print(f"      [raw real val: {real_dos_row[col].values[0]}]")
