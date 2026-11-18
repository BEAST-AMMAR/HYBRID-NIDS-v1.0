# NF-UQ-NIDS-v2 Data Notes

**Generated:** 2026-10-21 04:41:53

## Full Dataset Summary
- **Total Rows:** 75,987,976
- **Total Columns:** 46
- **Parquet Storage:** 52 parts (Snappy compressed, ~2.7 GB total)

### Full Dataset Label Distribution
| Label | Count | Percentage |
|---|---|---|
| 0 | 25,165,295 | 33.12% |
| 1 | 50,822,681 | 66.88% |

### Source Datasets
| Source Dataset | Flow Count | Percentage |
|---|---|---|
| NF-BoT-IoT-v2 | 37,763,497 | 49.70% |
| NF-CSE-CIC-IDS2018-v2 | 18,893,708 | 24.86% |
| NF-ToN-IoT-v2 | 16,940,496 | 22.29% |
| NF-UNSW-NB15-v2 | 2,390,275 | 3.15% |

### Attack Classes (Full Dataset)
| Attack Class | Count | Percentage |
|---|---|---|
| Benign | 25,165,295 | 33.1175% |
| DDoS | 21,748,351 | 28.6208% |
| DoS | 17,875,585 | 23.5242% |
| scanning | 3,781,419 | 4.9763% |
| Reconnaissance | 2,633,778 | 3.4660% |
| xss | 2,455,020 | 3.2308% |
| password | 1,153,323 | 1.5178% |
| injection | 684,897 | 0.9013% |
| Bot | 143,097 | 0.1883% |
| Brute Force | 123,982 | 0.1632% |
| Infilteration | 116,361 | 0.1531% |
| Exploits | 31,551 | 0.0415% |
| Fuzzers | 22,310 | 0.0294% |
| Backdoor | 18,978 | 0.0250% |
| Generic | 16,560 | 0.0218% |
| mitm | 7,723 | 0.0102% |
| ransomware | 3,425 | 0.0045% |
| Theft | 2,431 | 0.0032% |
| Analysis | 2,299 | 0.0030% |
| Shellcode | 1,427 | 0.0019% |
| Worms | 164 | 0.0002% |

## Subsampled Dataset (Option B Protocol)
- **Subsampled Rows:** 3,040,308
- Label=1: 1,540,308 (50.66%)
- Label=0: 1,500,000 (49.34%)

### Subsampled Attack Breakdown
| Attack Class | Subsampled Count |
|---|---|
| Benign | 1,500,000 |
| scanning | 150,000 |
| injection | 150,000 |
| Reconnaissance | 150,000 |
| xss | 150,000 |
| password | 150,000 |
| DoS | 150,000 |
| DDoS | 150,000 |
| Bot | 143,097 |
| Brute Force | 123,982 |
| Infilteration | 116,361 |
| Exploits | 31,551 |
| Fuzzers | 22,310 |
| Backdoor | 18,978 |
| Generic | 16,560 |
| mitm | 7,723 |
| ransomware | 3,425 |
| Theft | 2,431 |
| Analysis | 2,299 |
| Shellcode | 1,427 |
| Worms | 164 |
