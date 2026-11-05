# Milestone Checklist & Exit Criteria

This document defines the formal, objective exit criteria for each milestone in the Customer Purchase Propensity project. A milestone can only transition from `IN_PROGRESS` or `REVIEW_PENDING` to `COMPLETED` when all specified exit criteria are verified and signed off.

---

## M00 — Project Initialization & Foundation Setup

| Exit Criterion | Verification Method | Status |
| :--- | :--- | :--- |
| **Strict 6 Root-Level Items** | File system audit confirms exactly `docs/`, `notebooks/`, `src/`, `sample_data/`, `pyproject.toml`, and `README.md` at root. No additional files or folders exist. | [x] Verified |
| **Valid pyproject.toml Configuration** | Successfully parsed by Python 3.12 TOML parser (`py -3.12 -c "import tomllib; ..."`). Python `>=3.12` declared. Minimal data processing dependencies only; no premature ML packages. | [x] Verified |
| **Complete Documentation Suite** | All seven required files exist in `docs/`: `PROJECT_MASTER_PLAN.md`, `PROJECT_STATE.md`, `DECISION_LOG.md`, `MILESTONE_CHECKLIST.md`, `BUSINESS_UNDERSTANDING.md`, `DATA_CONTRACT.md`, `HANDOFF.md`. | [x] Verified |
| **No Unverified Data Claims** | No speculative dataset statistics or unverified claims treated as ground truth. Open assumptions cataloged in `DECISION_LOG.md`. | [x] Verified |
| **Strict Boundary Between M00 & M01** | Zero datasets downloaded, zero ML code written, zero package installations executed, zero EDA performed in M00. | [x] Verified |
| **Collaboration & Handoff Protocol Defined** | `docs/HANDOFF.md` populated with current state, implementation summary, verification results, and ChatGPT review prompt. | [x] Verified |
| **Review Gate Set** | `PROJECT_STATE.md` milestone status set to `REVIEW_PENDING`. Advancement to M01 blocked until review sign-off. | [x] Verified |

---

## M01 — Data Acquisition, Environment Provisioning & Ingestion Setup

| Exit Criterion | Verification Method | Status |
| :--- | :--- | :--- |
| **Virtual Environment Activation** | Python 3.12 virtual environment initialized and activated. Core packages installed and verified via `python -V` and `pip list`. | [ ] Pending |
| **Raw Dataset Download & Integrity** | T-ECD raw archive downloaded to local storage. Checksum (SHA-256 or MD5) verified against source metadata. | [ ] Pending |
| **Raw Schema & Column Audit** | Column names, inferred data types, row counts, and disk footprint documented in `docs/DATA_AUDIT_REPORT.md` without memory exhaustion. | [ ] Pending |
| **Micro-Sample Generation** | Representative deterministic micro-sample (e.g., 5,000–10,000 interaction rows) extracted and saved in `sample_data/` for rapid testing. | [ ] Pending |
| **Laptop Resource Safety Guard** | Script demonstrates streaming or chunked reads to avoid OOM on 16 GB RAM laptop. | [ ] Pending |
| **M01 Review Sign-Off** | `docs/HANDOFF.md` updated with M01 findings; ChatGPT sign-off recorded before proceeding to M02. | [ ] Pending |

---

## M02 — Exploratory Data Analysis (EDA) & Data Quality Assessment

| Exit Criterion | Verification Method | Status |
| :--- | :--- | :--- |
| **Interaction Distribution Profiling** | Quantitative breakdown of interaction types (click, view, cart, favorite, purchase) computed and visualized in `notebooks/01_eda.ipynb`. | [ ] Pending |
| **Temporal Span & Stationarity Check** | Timestamp range, daily transaction volume, and potential holiday spikes (e.g., 11.11) analyzed and documented. | [ ] Pending |
| **Data Quality Audit** | Missing value ratios, duplicate event rates, and timestamp anomalies cataloged in `docs/DATA_QUALITY_ASSESSMENT.md`. | [ ] Pending |
| **7-Day Purchase Horizon Validation** | Statistical verification of purchase rates over 7-day lookaheads. Confirmation that conversion volume supports gradient boosting. | [ ] Pending |
| **Customer Eligibility Rules Finalized** | Documented threshold for customer inclusion (e.g., minimum interactions during lookback window) to filter out dormant non-informative accounts. | [ ] Pending |
| **M02 Review Sign-Off** | `docs/HANDOFF.md` updated; ChatGPT review approved prior to M03. | [ ] Pending |

---

## M03 — Data Preparation, Sampling Strategy & Feature Engineering

| Exit Criterion | Verification Method | Status |
| :--- | :--- | :--- |
| **Leakage-Free Snapshot Architecture** | Automated test validates that for any snapshot $(u_i, t_s)$, no event with $t > t_s$ enters the feature computation pipeline. | [ ] Pending |
| **Strict Chronological Splits** | Train, validation, and test datasets partitioned using non-overlapping time boundaries ($T_{\text{train}} < T_{\text{val}} < T_{\text{test}}$). No random splits. | [ ] Pending |
| **Resource-Efficient Sampling Pipeline** | Sampling strategy produces training/validation datasets that comfortably fit within laptop memory (< 4 GB per partition). | [ ] Pending |
| **Modular Feature Pipeline** | Reusable, unit-tested feature transformations covering recency, frequency, engagement intensity, and category affinity implemented in `src/features/`. | [ ] Pending |
| **Feature Store Documentation** | Complete data dictionary and transformation formulas documented in `docs/FEATURE_STORE.md`. | [ ] Pending |
| **M03 Review Sign-Off** | `docs/HANDOFF.md` updated; ChatGPT review approved prior to M04. | [ ] Pending |

---

## M04 — Baseline Modeling & Evaluation Strategy

| Exit Criterion | Verification Method | Status |
| :--- | :--- | :--- |
| **Evaluation Suite Implementation** | Standalone evaluation module (`src/evaluation/metrics.py`) computing PR-AUC, ROC-AUC, Brier score, and Precision/Recall@Top-K. | [ ] Pending |
| **Rule-Based & Heuristic Baselines** | Rule-based baseline (e.g., recent cart-add heuristic) benchmarked on chronological test set. | [ ] Pending |
| **Statistical Model Baseline** | Logistic Regression or shallow Decision Tree trained and evaluated on identical chronological splits. | [ ] Pending |
| **Baseline Benchmark Report** | Documented performance numbers, confusion matrices, and calibration curves in `docs/BASELINE_REPORT.md`. | [ ] Pending |
| **M04 Review Sign-Off** | `docs/HANDOFF.md` updated; ChatGPT review approved prior to M05. | [ ] Pending |

---

## M05 — Advanced Modeling, Hyperparameter Optimization & Error Analysis

| Exit Criterion | Verification Method | Status |
| :--- | :--- | :--- |
| **Gradient Boosting Optimization** | High-performance GBDT model (LightGBM/CatBoost) trained using time-series cross-validation; achieves superior PR-AUC over baselines. | [ ] Pending |
| **Probability Calibration** | Calibrated probabilities validated via Brier score and reliability curves on the chronological test set. | [ ] Pending |
| **Feature Attribution & Explainability** | Global and local SHAP feature importances computed and interpreted in `docs/MODEL_INTERPRETABILITY.md`. | [ ] Pending |
| **Cohort & Slice Error Analysis** | Performance segmented across customer activity levels, tenure, and purchase history in `docs/ERROR_ANALYSIS.md`. | [ ] Pending |
| **Laptop Efficiency Compliance** | Complete model training and tuning completes without OOM within acceptable runtime (< 30 minutes on laptop CPU). | [ ] Pending |
| **M05 Review Sign-Off** | `docs/HANDOFF.md` updated; ChatGPT review approved prior to M06. | [ ] Pending |

---

## M06 — Deployment Readiness, Inference Pipeline & Project Packaging

| Exit Criterion | Verification Method | Status |
| :--- | :--- | :--- |
| **Batch Inference Engine** | CLI script (`src/inference/predict_batch.py`) scores new customer batches from snapshot Parquet files. | [ ] Pending |
| **Input Schema & Data Drift Validation** | Automated schema validation fails fast on missing columns or anomalous type mismatches. | [ ] Pending |
| **Propensity Decile Binning** | Output contains continuous propensity probabilities alongside calibrated decile ranking buckets (1–10). | [ ] Pending |
| **End-to-End Integration Testing** | Automated test executes complete pipeline from sample input to prediction output. | [ ] Pending |
| **Project Packaging & Final Delivery** | Complete documentation, reproducible execution instructions in `README.md`, and clean project archive. | [ ] Pending |
| **Final Stakeholder Sign-Off** | Formal project completion sign-off recorded in `docs/FINAL_DELIVERY_REPORT.md`. | [ ] Pending |

