# Milestone Checklist & Exit Criteria

This document defines the formal, objective exit criteria for each milestone in the Customer Purchase Propensity project. A milestone can only transition from `IN_PROGRESS` or `REVIEW_PENDING` to `COMPLETED` when all specified exit criteria are verified, reviewed by ChatGPT, and **formally approved by the Project Owner**.

---

## M00 — Project Initialization & Foundation Setup

| Exit Criterion | Verification Method | Status |
| :--- | :--- | :--- |
| **Strict 6 Root-Level Items** | File system audit confirms exactly `docs/`, `notebooks/`, `src/`, `sample_data/`, `pyproject.toml`, and `README.md` at root. No additional files or folders exist (`.git` permitted as metadata). | [x] Verified |
| **Valid pyproject.toml Configuration** | Successfully parsed by Python 3.12 TOML parser (`py -3.12 -c "import tomllib; ..."`). Python `>=3.12` declared. Minimal data processing dependencies only; no premature ML packages. | [x] Verified |
| **Complete Documentation Suite** | All seven required files exist in `docs/`: `PROJECT_MASTER_PLAN.md`, `PROJECT_STATE.md`, `DECISION_LOG.md`, `MILESTONE_CHECKLIST.md`, `BUSINESS_UNDERSTANDING.md`, `DATA_CONTRACT.md`, `HANDOFF.md`. | [x] Verified |
| **No Unverified Data Claims** | No speculative dataset statistics or unverified claims treated as ground truth. Open assumptions cataloged in `DECISION_LOG.md`. Speculative duration minimums removed. | [x] Verified |
| **Three-Party Governance Established** | Workflow documented: Antigravity implements $\to$ ChatGPT provides review recommendation $\to$ Project Owner has final approval authority. Milestones cannot be auto-completed without Project Owner decision. | [x] Verified |
| **Strict Boundary Between M01 & M02** | Responsibilities decoupled: M01 defines business definitions, target objectives, eligibility policies, feature hypotheses, and setup; M02 performs empirical schema inspection and feasibility testing. | [x] Verified |
| **Personal Machine Execution Policy** | Operational rules established under AD-007 (routine non-destructive operations authorized, external venv/cache, resource checks before download). | [x] Verified |
| **Review Gate Set** | `PROJECT_STATE.md` milestone status set to `REVIEW_PENDING`. Advancement to M01 blocked until explicit Project Owner approval. | [x] Verified |

---

## M01 — Data Acquisition, Environment Provisioning & Ingestion Setup

| Exit Criterion | Verification Method | Status |
| :--- | :--- | :--- |
| **Business Meaning of Valid Purchase** | Documented formal definition of what constitutes a valid purchase transaction from a business perspective (order vs. payment vs. returns). | [ ] Pending |
| **Target Objectives & Expectations** | Business target definition, intended 7-day prediction horizon, and stakeholder KPI thresholds formally defined. | [ ] Pending |
| **Preliminary Eligibility Policies** | Business rules governing which customer accounts qualify for propensity scoring documented. | [ ] Pending |
| **Feature Hypotheses Catalog** | Comprehensive catalog of domain feature hypotheses (recency, frequency, intensity, category affinity) recorded for subsequent validation. | [ ] Pending |
| **Virtual Environment Provisioning** | Python 3.12 virtual environment provisioned on personal computer (kept external to repo root) with core tooling verified. | [ ] Pending |
| **Safe Ingestion Mechanics** | Ingestion script implemented with pre-download size checks to prevent personal laptop disk/memory overflow. | [ ] Pending |
| **M01 Governance Sign-Off** | `docs/HANDOFF.md` updated; ChatGPT review recommendation obtained; explicit Project Owner approval recorded before proceeding to M02. | [ ] Pending |

---

## M02 — Exploratory Data Analysis (EDA) & Empirical Data Quality Assessment

| Exit Criterion | Verification Method | Status |
| :--- | :--- | :--- |
| **Actual Dataset Schema Inspected** | Direct empirical inspection of raw files; physical column names, inferred types, row counts, and null rates documented without speculation. | [ ] Pending |
| **Event Semantics & Timestamps Validated** | Empirical validation of interaction codes (e.g., `buy`, `cart`, `fav`, `pv`), timestamp spans, clock anomalies, and ordering. | [ ] Pending |
| **Feature Availability Assessed** | Verification of whether hypothesized features from M01 can be constructed from actual dataset fields. | [ ] Pending |
| **Target Constructibility Verified** | Statistical verification that 7-day forward purchase events exist with sufficient frequency to support gradient boosting. | [ ] Pending |
| **Sampling Feasibility Evaluated** | Subsampling strategies evaluated and verified to execute safely within personal laptop RAM limits (< 4 GB working memory). | [ ] Pending |
| **Exploratory Profiling Documented** | Reproducible EDA notebooks in `notebooks/` and comprehensive audit report in `docs/DATA_QUALITY_ASSESSMENT.md`. | [ ] Pending |
| **M02 Governance Sign-Off** | `docs/HANDOFF.md` updated; ChatGPT review recommendation obtained; explicit Project Owner approval recorded before proceeding to M03. | [ ] Pending |

---

## M03 — Data Preparation, Sampling Strategy & Feature Engineering

| Exit Criterion | Verification Method | Status |
| :--- | :--- | :--- |
| **Leakage-Free Snapshot Architecture** | Automated test validates that for any snapshot $(u_i, t_s)$, no event with $t > t_s$ enters the feature computation pipeline. | [ ] Pending |
| **Strict Chronological Splits** | Train, validation, and test datasets partitioned using non-overlapping time boundaries ($T_{\text{train}} < T_{\text{val}} < T_{\text{test}}$) with label maturity buffers. No random splits. | [ ] Pending |
| **Resource-Efficient Sampling Pipeline** | Sampling strategy produces training/validation datasets that comfortably fit within laptop memory (< 4 GB per partition). | [ ] Pending |
| **Modular Feature Pipeline** | Reusable, unit-tested feature transformations covering recency, frequency, engagement intensity, and category affinity implemented in `src/features/`. | [ ] Pending |
| **Feature Store Documentation** | Complete data dictionary and transformation formulas documented in `docs/FEATURE_STORE.md`. | [ ] Pending |
| **M03 Governance Sign-Off** | `docs/HANDOFF.md` updated; ChatGPT review recommendation obtained; explicit Project Owner approval recorded before proceeding to M04. | [ ] Pending |

---

## M04 — Baseline Modeling & Evaluation Strategy

| Exit Criterion | Verification Method | Status |
| :--- | :--- | :--- |
| **Evaluation Suite Implementation** | Standalone evaluation module (`src/evaluation/metrics.py`) computing PR-AUC, ROC-AUC, Brier score, and Precision/Recall@Top-K. | [ ] Pending |
| **Rule-Based & Heuristic Baselines** | Rule-based baseline (e.g., recent cart-add heuristic) benchmarked on chronological test set. | [ ] Pending |
| **Statistical Model Baseline** | Logistic Regression or shallow Decision Tree trained and evaluated on identical chronological splits. | [ ] Pending |
| **Baseline Benchmark Report** | Documented performance numbers, confusion matrices, and calibration curves in `docs/BASELINE_REPORT.md`. | [ ] Pending |
| **M04 Governance Sign-Off** | `docs/HANDOFF.md` updated; ChatGPT review recommendation obtained; explicit Project Owner approval recorded before proceeding to M05. | [ ] Pending |

---

## M05 — Advanced Modeling, Hyperparameter Optimization & Error Analysis

| Exit Criterion | Verification Method | Status |
| :--- | :--- | :--- |
| **Gradient Boosting Optimization** | High-performance GBDT model (LightGBM/CatBoost) trained using time-series cross-validation; achieves superior PR-AUC over baselines. | [ ] Pending |
| **Probability Calibration** | Calibrated probabilities validated via Brier score and reliability curves on the chronological test set. | [ ] Pending |
| **Feature Attribution & Explainability** | Global and local SHAP feature importances computed and interpreted in `docs/MODEL_INTERPRETABILITY.md`. | [ ] Pending |
| **Cohort & Slice Error Analysis** | Performance segmented across customer activity levels, tenure, and purchase history in `docs/ERROR_ANALYSIS.md`. | [ ] Pending |
| **Laptop Efficiency Compliance** | Complete model training and tuning completes without OOM within acceptable runtime (< 30 minutes on laptop CPU). | [ ] Pending |
| **M05 Governance Sign-Off** | `docs/HANDOFF.md` updated; ChatGPT review recommendation obtained; explicit Project Owner approval recorded before proceeding to M06. | [ ] Pending |

---

## M06 — Deployment Readiness, Inference Pipeline & Project Packaging

| Exit Criterion | Verification Method | Status |
| :--- | :--- | :--- |
| **Batch Inference Engine** | CLI script (`src/inference/predict_batch.py`) scores new customer batches from snapshot Parquet files. | [ ] Pending |
| **Input Schema & Data Drift Validation** | Automated schema validation fails fast on missing columns or anomalous type mismatches. | [ ] Pending |
| **Propensity Decile Binning** | Output contains continuous propensity probabilities alongside calibrated decile ranking buckets (1–10). | [ ] Pending |
| **End-to-End Integration Testing** | Automated test executes complete pipeline from sample input to prediction output. | [ ] Pending |
| **Project Packaging & Final Delivery** | Complete documentation, reproducible execution instructions in `README.md`, and clean project archive. | [ ] Pending |
| **Final Project Owner Sign-Off** | Formal project completion sign-off recorded by Project Owner in `docs/FINAL_DELIVERY_REPORT.md`. | [ ] Pending |
