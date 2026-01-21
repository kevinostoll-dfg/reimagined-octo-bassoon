# 📊 Memgraph Data Ingestion Report
**Date:** Today

## 1. Pipeline Summary
| Stage | Metric | Value | Status |
|:------|:-------|:------|:-------|
| **Source (GCS)** | Total Transcripts | **390** | ✅ Active |
| | Total Size | 18.01 MB | |
| **Processor** | Checkpoint Count | **388** | ⏳ (vs GCS) |
| **Destination** | Total Nodes | **11,695** | ✅ Online |
| | Relationships | **489,437** | |

## 2. Graph Database Composition
### Node Distribution
| Node Label | Count | % Distribution |
|:-----------|------:|:---------------|
| `CONCEPT` | 8,514 |   72.8% ██████████████ |
| `MONEY` | 875 |    7.5% █ |
| `CARDINAL` | 730 |    6.2% █ |
| `RISK` | 296 |    2.5%  |
| `PERCENT` | 248 |    2.1%  |
| `METRIC` | 203 |    1.7%  |
| `DATE` | 202 |    1.7%  |
| `ORG` | 133 |    1.1%  |
| `PRODUCT` | 91 |    0.8%  |
| `STATEMENT` | 56 |    0.5%  |
| `LAW` | 51 |    0.4%  |
| `PERSON` | 48 |    0.4%  |
| `GPE` | 40 |    0.3%  |
| `GEOGRAPHY` | 33 |    0.3%  |
| `ACTION` | 32 |    0.3%  |
| `REGULATION` | 31 |    0.3%  |
| `SECTION` | 22 |    0.2%  |
| `SEGMENT` | 20 |    0.2%  |
| `METRIC_DEFINITION` | 16 |    0.1%  |
| `LOC` | 15 |    0.1%  |
| `QUANTITY` | 15 |    0.1%  |
| `NORP` | 8 |    0.1%  |
| `ORDINAL` | 6 |    0.1%  |
| `TIME` | 5 |    0.0%  |
| `ROLE` | 2 |    0.0%  |
| `WORK_OF_ART` | 2 |    0.0%  |
| `EVENT` | 1 |    0.0%  |

### Relationship Distribution
| Relationship Type | Count | % Distribution |
|:------------------|------:|:---------------|
| `CO_MENTIONED` | 457,286 |   93.4% ██████████████████ |
| `TEMPORAL_CONTEXT` | 26,771 |    5.5% █ |
| `QUANTITY_OF` | 4,119 |    0.8%  |
| `SVO_TRIPLE` | 588 |    0.1%  |
| `QUANTITY_IN_ACTION` | 261 |    0.1%  |
| `SAME_AS` | 203 |    0.0%  |
| `MITIGATES_RISK` | 119 |    0.0%  |
| `SAID` | 56 |    0.0%  |
| `EXPOSES_TO_RISK` | 20 |    0.0%  |
| `OPERATES_IN` | 12 |    0.0%  |
| `HAS_ROLE` | 2 |    0.0%  |

## 3. Health & Quality Check
> **Ingestion Ratios**
> - **Nodes per Transcript:** 30
> - **Edges per Transcript:** 1,261
> - **Statements per Transcript:** 0.1

### ⚠️ Critical Warning
**Extremely low statement count (0.1/file).**
This suggests that `STATEMENT` nodes (speaker attributions) are NOT being created for most transcripts.
- Check `v1.0-graph-ea-scripts.py` speaker extraction patterns.
- Verify transcript format in GCS.