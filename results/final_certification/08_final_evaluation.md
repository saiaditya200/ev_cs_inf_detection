# Final Evaluation Evidence

## Quarantined Outer-Test Evaluation (35 Stations across 8 Held-Out Districts)

| Model Identity | Role | Feature Contract | Threshold | Accuracy | Balanced Accuracy | AC Recall | DC Recall | Macro F1 | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Current Bakeoff Best RF** | benchmark | Contract A (21 feat) | 0.755 | 74.29% | 60.90% | 33.33% (3/9) | 88.46% (23/26) | 0.6182 | 0.8013 |
| **RF_0039** | secondary benchmark | Contract A (21 feat) | 0.720 | 80.00% | 64.74% | 33.33% (3/9) | 96.15% (25/26) | 0.6694 | 0.8759 |
| **RF_0002** | secondary benchmark | Contract A (21 feat) | 0.740 | 71.43% | 51.71% | 11.11% (1/9) | 92.31% (24/26) | 0.4971 | 0.7723 |
| **Active Production Model** | production | Contract A (21 feat) | 0.760 | 71.43% | 51.71% | 11.11% (1/9) | 92.31% (24/26) | 0.4971 | 0.7723 |
| **Fixed RF Baseline (Contract A)** | candidate | Contract A (21 feat) | 0.740 | 74.29% | 57.26% | 22.22% (2/9) | 92.31% (24/26) | 0.5749 | 0.7672 |
| **Fixed RF Baseline (Contract A + Block 1)** | candidate | Contract A + Block 1 (23 feat) | 0.740 | 74.29% | 57.26% | 22.22% (2/9) | 92.31% (24/26) | 0.5749 | 0.7460 |

*Note: Benchmark results represent comparative research findings and are not automatic production-selection instructions.*
