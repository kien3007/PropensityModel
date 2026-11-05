# Decision Log

This document records the architectural, methodological, and data decisions for the Customer Purchase Propensity project. It strictly categorizes decisions into **Approved Decisions**, **Proposals Under Consideration**, and **Unverified Assumptions**.

---

## 1. Approved Decisions (AD)

| Decision ID | Date | Category | Title | Summary & Rationale | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AD-001** | 2026-11-05 | Architecture | Strict 6 Root-Level Components | The repository root strictly contains only `docs/`, `notebooks/`, `src/`, `sample_data/`, `pyproject.toml`, and `README.md`. Prevents root clutter and enforces standard ML project organization. | **APPROVED** |
| **AD-002** | 2026-11-05 | Methodology | CRISP-DM Milestone Roadmap | Adopted standard CRISP-DM mapped across milestones M00 (Foundation), M01 (Ingestion), M02 (EDA), M03 (Preparation), M04 (Baselines), M05 (Advanced Modeling), and M06 (Deployment). | **APPROVED** |
| **AD-003** | 2026-11-05 | Environment | Python 3.12 Baseline with Deferred Installation | Configured `pyproject.toml` targeting `>=3.12` with minimal initial data processing libraries (`polars`, `pyarrow`, `pydantic`). Deferred package installation and lockfile creation to prevent premature environment bloating in M00. | **APPROVED** |
| **AD-004** | 2026-11-05 | Modeling | Modeling Unit and Horizon Definition | Prediction unit defined as composite key `(user_id, snapshot_at)`. Target is a binary indicator of at least one purchase in the subsequent 7 days `(snapshot_at, snapshot_at + 7d]`. | **APPROVED** |
| **AD-005** | 2026-11-05 | Methodology | Chronological Splitting & Leakage Prevention | Strict prohibition of random train/validation/test splits. Training, validation, and testing must use non-overlapping chronological intervals to mimic real-world deployment and prevent future-to-past leakage. | **APPROVED** |
| **AD-006** | 2026-11-05 | Governance | Antigravity & ChatGPT Dual-Agent Collaboration | Antigravity acts as Senior ML Engineer / Implementer; ChatGPT acts as Reviewer / Architect. Progression between milestones requires formal review sign-off via `docs/HANDOFF.md`. | **APPROVED** |
| **AD-007** | 2026-11-05 | Operations | Local Development Environment & Execution Policy | Authorized routine non-destructive operations (git operations, package installs via uv/pip, venv creation, script/notebook runs, internet dependency retrieval) on personal hardware without per-step approval. Strict guardrails enforced: maintain exact 6-root repository structure, keep venvs/caches external where practical, assess resource needs before large downloads, require explicit approval for destructive actions, and maintain strict milestone gates. | **APPROVED** |

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
| **UA-001** | 2026-11-05 | Data Integrity | Timestamp Granularity | Assumed that T-ECD interaction logs contain explicit, reliable UNIX or ISO timestamps sufficient for minute/second-level chronological ordering and 7-day lookahead calculation. | M01 / M02 | **UNVERIFIED** |
| **UA-002** | 2026-11-05 | Data Integrity | User Identity Consistency | Assumed that `user_id` is stable across sessions and that no anonymous session stitching is required. | M01 / M02 | **UNVERIFIED** |
| **UA-003** | 2026-11-05 | Data Semantics | Purchase Event Identification | Assumed that purchase interactions are clearly labeled (e.g., `behavior_type == 'buy'` or `behavior_type == 4`) without ambiguity between order placement, payment completion, or cancellation. | M01 / M02 | **UNVERIFIED** |
| **UA-004** | 2026-11-05 | Feasibility | Personal Laptop Storage & RAM Viability | Assumed that the uncompressed T-ECD dataset can be handled on a 16 GB RAM laptop using streaming reads, subsampling, and Parquet compression without system OOM crashes. | M01 | **UNVERIFIED** |
| **UA-005** | 2026-11-05 | Statistical | Target Event Prevalence | Assumed that within a 7-day forward horizon, the purchase conversion rate among eligible users is between 1% and 10%, providing sufficient positive examples for gradient boosting without extreme sparsity breakdown. | M02 | **UNVERIFIED** |
| **UA-006** | 2026-11-05 | Stationarity | Absence of Overwhelming Macro Anomalies | Assumed that the dataset does not solely represent extreme atypical shopping festivals (e.g., Double 11 peak day only) that would invalidate generalized 7-day propensity dynamics. | M02 | **UNVERIFIED** |

