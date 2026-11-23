import pyarrow.parquet as pq
import glob
import pandas as pd

files = glob.glob('data/NF UQ NIDS V2/parquet/*.parquet')
table = pq.read_table(files[0])
df = table.to_pandas()

dos_rows = df[df['Attack'] == 'DoS'].head(3)
ddos_rows = df[df['Attack'] == 'DDoS'].head(3)
scan_rows = df[df['Attack'] == 'scanning'].head(3)

cols = [
    'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 
    'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
    'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
    'FLOW_DURATION_MILLISECONDS', 'MIN_TTL', 'MAX_TTL', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT',
    'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT'
]

print("=== REAL DoS SAMPLES ===")
print(dos_rows[cols].to_string())

print("\n=== REAL DDoS SAMPLES ===")
print(ddos_rows[cols].to_string())

print("\n=== REAL SCANNING SAMPLES ===")
print(scan_rows[cols].to_string())
