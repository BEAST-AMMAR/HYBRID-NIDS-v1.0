import pyarrow.parquet as pq
import glob
import pandas as pd

files = glob.glob('data/NF UQ NIDS V2/parquet/*.parquet')
table = pq.read_table(files[0])
df = table.to_pandas()

benign = df[df['Label'] == 0].head(10)
high_fp = benign.iloc[:5]
low_fp = benign.iloc[5:]

cols_to_compare = [
    'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 
    'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
    'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
    'FLOW_DURATION_MILLISECONDS', 'MIN_TTL', 'MAX_TTL', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT'
]

print("=== FIRST 5 BENIGN (SCORED ~0.99) ===")
print(high_fp[cols_to_compare].to_string())

print("\n=== LAST 5 BENIGN (SCORED ~0.005) ===")
print(low_fp[cols_to_compare].to_string())
