import pyarrow.parquet as pq
import glob
from src.backend.detector import HybridDetector
from src.backend.sequences import PureNfUqAdapter

det = HybridDetector('src/backend/saved_models')
adapter = PureNfUqAdapter(model_dir='src/backend/saved_models')

files = glob.glob('data/NF UQ NIDS V2/parquet/*.parquet')
table = pq.read_table(files[0])
df = table.to_pandas()

attacks = df[df['Label'] == 1].head(10)
benign = df[df['Label'] == 0].head(10)

print('--- TESTING REAL BENIGN FLOWS ---')
for idx, row in benign.iterrows():
    row_df = df.iloc[[idx]]
    feat = adapter.transform(row_df)
    anom, score = det.predict(feat, host_ip=str(row['IPV4_SRC_ADDR']))
    print(f"Benign {row['IPV4_SRC_ADDR']} -> {row['IPV4_DST_ADDR']} ({row['Attack']}) | Score: {score:.4f} | Anomaly: {anom}")

print('\n--- TESTING REAL ATTACK FLOWS ---')
for idx, row in attacks.iterrows():
    row_df = df.iloc[[idx]]
    feat = adapter.transform(row_df)
    anom, score = det.predict(feat, host_ip=str(row['IPV4_SRC_ADDR']))
    print(f"Attack {row['IPV4_SRC_ADDR']} -> {row['IPV4_DST_ADDR']} ({row['Attack']}) | Score: {score:.4f} | Anomaly: {anom}")
