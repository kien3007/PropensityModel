# Decision Log

This document records the architectural, methodological, and operational decisions for the Customer Purchase Propensity project. It strictly categorizes decisions into **Approved Decisions**, **Proposals Under Consideration**, and **Unverified Assumptions**.

---

## 1. Approved Decisions (AD)

| Decision ID | Date | Category | Title | Summary & Rationale | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AD-001** | 2026-11-05 | Architecture | Strict 6 Root-Level Components | The repository root strictly contains only `docs/`, `notebooks/`, `src/`, `sample_data/`, `pyproject.toml`, and `README.md` (`.git` permitted as repository store). Prevents root clutter and enforces standard ML project organization. | **APPROVED** |
| **AD-002** | 2026-11-05 | Methodology | CRISP-DM Milestone Roadmap | Adopted standard CRISP-DM mapped across milestones M00 (Foundation), M01 (Ingestion Setup), M02 (EDA & Feasibility), M03 (Preparation), M04 (Baselines), M05 (Advanced Modeling), and M06 (Deployment). | **APPROVED** |
| **AD-003** | 2026-11-05 | Environment | Python 3.12 Baseline with Deferred Installation | Configured `pyproject.toml` targeting `>=3.12` with minimal initial data processing libraries (`polars`, `pyarrow`, `pydantic`). Package installation and lockfile creation deferred until environment provisioning in M01. | **APPROVED** |
| **AD-004** | 2026-11-05 | Modeling | Modeling Unit and Horizon Definition | Prediction unit defined as composite key `(user_id, snapshot_at)`. Target is a binary indicator of at least one purchase in the subsequent 7 days `(snapshot_at, snapshot_at + 7d]`. | **APPROVED** |
| **AD-005** | 2026-11-05 | Methodology | Chronological Splitting & Leakage Prevention | Strict prohibition of random train/validation/test splits. Training, validation, and testing must use non-overlapping chronological intervals with label maturity buffers to prevent future-to-past leakage. | **APPROVED** |
| **AD-006** | 2026-11-05 | Governance | Three-Party Governance Protocol | Defined workflow: Antigravity implements $\to$ ChatGPT provides independent review recommendation $\to$ Project Owner decides final approval $\to$ Antigravity updates state. Milestones cannot be marked COMPLETED without explicit Project Owner approval. | **APPROVED** |
| **AD-007** | 2026-11-05 | Operations | Local Development Environment & Execution Policy | Authorized routine non-destructive operations (Git, uv/pip, venvs, test runs, internet dependency retrieval) on personal computer without per-step approval. Strict guardrails enforced: maintain exact 6-root repository structure, keep venvs/caches external where practical, assess resource needs before large downloads, and require explicit approval for destructive actions. | **APPROVED** |

---

## 2. Proposals Under Consideration (PR)

| Proposal ID | Date Proposed | Category | Title | Proposed Solution & Rationale | Target Milestone | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PR-001** | 2026-11-05 | Engineering | Data Processing Engine for Laptop Scale | Evaluate **Polars LazyFrame** vs **DuckDB** for out-of-core data transformations, aggregations, and feature extraction. Both offer high multi-threaded performance and low memory footprints suitable for 16 GB RAM laptops. | M01 / M02 | **PROPOSAL** |
| **PR-002** | 2026-11-05 | Methodology | Resource-Efficient User Cohort Sampling | Subsample active users based on an activity threshold in the lookback window, while preserving the natural purchase event distribution. Prevents zero-signal inactive user bloat. | M02 / M03 | **PROPOSAL** |
| **PR-003** | 2026-11-05 | Modeling | Model Architecture Selection | Prioritize **LightGBM** as primary gradient boosting engine due to its native handling of categorical features, low memory consumption (histogram-based splitting), and rapid CPU training. | M04 / M05 | **PROPOSAL** |
| **PR-004** | 2026-11-05 | Evaluation | Probability Calibration Technique | Evaluate Isotonic Regression vs Platt Scaling (Logistic Calibration) to output true posterior probabilities for downstream business thresholding. | M04 / M05 | **PROPOSAL** |
| **PR-005** | 2026-11-05 | Features | Point-in-Time Sliding Window Snapshots | Construct training datasets from multiple temporal snapshots spaced 7 or 14 days apart to provide the model with diverse behavioral contexts while strictly enforcing cutoff timestamps. | M03 | **PROPOSAL** |

---

## 3. Unverified Assumptions (UA)

| Assumption ID | Date Identified | Subject Area | Assumption Statement | Risk & Verification Plan | Verification Milestone | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **UA-001** | 2026-11-05 | Data Integrity | Timestamp Granularity & Coverage | Assumed that T-ECD interaction logs contain explicit, reliable UNIX or ISO timestamps sufficient for chronological ordering and 7-day lookahead calculation. | **M02** (Empirical Inspection) | **UNVERIFIED** |
| **UA-002** | 2026-11-05 | Data Integrity | User Identity Consistency | Assumed that `user_id` is stable across sessions and that no anonymous session stitching is required. | **M02** (Empirical Inspection) | **UNVERIFIED** |
| **UA-003** | 2026-11-05 | Data Semantics | Purchase Event Identification & Target Feasibility | Assumed that purchase interactions are clearly labeled (e.g., `behavior_type == 'buy'` or `behavior_type == 4`) without ambiguity, allowing the target to be constructed. | **M02** (Empirical Inspection) | **UNVERIFIED** |
| **UA-004** | 2026-11-05 | Feasibility | Personal Laptop Storage & RAM Viability | Assumed that the uncompressed T-ECD dataset can be handled on a 16 GB RAM laptop using streaming reads, subsampling, and Parquet compression without OOM crashes. | **M01** (Pre-check) / **M02** (Benchmark) | **UNVERIFIED** |
| **UA-005** | 2026-11-05 | Statistical | Target Event Prevalence & Lookahead Feasibility | Assumed that within a 7-day forward horizon, the purchase conversion rate among eligible users provides sufficient positive examples for gradient boosting without extreme sparsity breakdown. | **M02** (Empirical Inspection) | **UNVERIFIED** |
| **UA-006** | 2026-11-05 | Stationarity | Absence of Overwhelming Macro Anomalies | Assumed that the dataset does not solely represent extreme atypical shopping festivals (e.g., Double 11 peak day only) that would invalidate generalized 7-day propensity dynamics. | **M02** (Empirical Inspection) | **UNVERIFIED** |
