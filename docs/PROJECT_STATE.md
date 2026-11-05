# Project State

## 1. Milestone Tracking

| Field | Current Value | Notes |
| :--- | :--- | :--- |
| **Active Milestone** | **M00 — Project Initialization** | Foundation & Repository Setup |
| **Milestone Status** | **REVIEW_PENDING** | Awaiting external review by ChatGPT |
| **CRISP-DM Phase** | Project Setup & Preliminary Business Understanding | Establishing operational & architectural ground rules |
| **Last Updated** | 2026-11-05 | M00 initialization complete |
| **Overall Health** | GREEN | On track; all M00 acceptance criteria satisfied locally |

---

## 2. Milestone Execution Status

| Milestone | Description | Status | Blockers / Dependencies |
| :--- | :--- | :--- | :--- |
| **M00** | Project Initialization & Foundation Setup | **REVIEW_PENDING** | None. Ready for ChatGPT review |
| **M01** | Data Acquisition & Ingestion Setup | NOT_STARTED | Requires M00 formal approval |
| **M02** | Exploratory Data Analysis & Data Quality | NOT_STARTED | Requires M01 completion |
| **M03** | Data Preparation & Feature Engineering | NOT_STARTED | Requires M02 completion |
| **M04** | Baseline Modeling & Evaluation Strategy | NOT_STARTED | Requires M03 completion |
| **M05** | Advanced Modeling, Tuning & Error Analysis | NOT_STARTED | Requires M04 completion |
| **M06** | Deployment Readiness & Project Packaging | NOT_STARTED | Requires M05 completion |

---

## 3. M00 Progress & Completed Deliverables

- [x] **Strict Root Structure Created**: Verified exactly 6 components (`docs/`, `notebooks/`, `src/`, `sample_data/`, `pyproject.toml`, `README.md`).
- [x] **pyproject.toml Initialized**: Targeted Python 3.12, declared minimal data processing dependencies (`polars`, `pyarrow`, `pydantic`), syntax verified using `py -3.12 tomllib`. Package installation deferred.
- [x] **Project Documentation Suite Created**:
  - `docs/PROJECT_MASTER_PLAN.md`: Full M00–M06 roadmap mapped to CRISP-DM.
  - `docs/PROJECT_STATE.md`: Real-time milestone tracker and state machine.
  - `docs/DECISION_LOG.md`: Formal demarcation of Approved Decisions, Proposals, and Unverified Assumptions.
  - `docs/MILESTONE_CHECKLIST.md`: Objective pass/fail criteria for each milestone.
  - `docs/BUSINESS_UNDERSTANDING.md`: Business value, stakeholder questions, KPI alignment, and hypotheses.
  - `docs/DATA_CONTRACT.md`: Preliminary prediction unit, 7-day target horizon, temporal leakage guards, and open data queries.
  - `docs/HANDOFF.md`: Protocol and structured template for Antigravity $\leftrightarrow$ ChatGPT collaboration.
- [x] **README.md Initialized**: Comprehensive repository documentation covering objectives, architecture, conventions, and status.

---

## 4. Confirmed Decisions (Approved)

1. **AD-001**: Strict 6 root-level components architecture. No extra root files or folders.
2. **AD-002**: CRISP-DM methodology organized into 7 sequential milestones (M00 to M06).
3. **AD-003**: Python 3.12 environment baseline; package installation and lockfile generation deferred to M01.
4. **AD-004**: Preliminary modeling definition: Prediction unit $(u_i, t_s)$, 7-day prediction window, chronological validation.
5. **AD-005**: Anti-leakage requirement: Strict prohibition of random splits; strictly point-in-time event filtering.
6. **AD-006**: Dual-agent review protocol: Antigravity acts as Implementer, ChatGPT acts as Reviewer/Architect. Advancement requires formal review sign-off in `HANDOFF.md`.
7. **AD-007**: Local Development Environment & Execution Policy: Authorization of routine, non-destructive operations (git, uv/pip, venvs, test execution, dependency retrieval) without per-step approval, while strictly preserving 6-root structure, keeping venvs/caches external where practical, assessing resource footprints before downloads, and requiring approval for destructive operations.

---

## 5. Pending Decisions & Decisions Requiring Validation

1. **PD-001**: Precise data acquisition source, version, and download pipeline for T-ECD (deferred to M01).
2. **PD-002**: Data processing engine selection between pure Polars vs DuckDB vs Pandas chunking (deferred to M01/M02).
3. **PD-003**: Laptop resource allocation and sampling strategy (active-user subsampling vs time-bounded slices) (deferred to M02/M03).
4. **PD-004**: Cutoff dates for train/validation/test chronological boundary splits (deferred to M02/M03).
5. **PD-005**: Primary ML framework selection (LightGBM vs XGBoost vs CatBoost) (deferred to M04/M05).

---

## 6. Next Actions

1. Expose M00 deliverables to ChatGPT for architectural and milestone review via `docs/HANDOFF.md`.
2. Await reviewer feedback, recommendations, or revision requests.
3. If changes are requested: Address comments within M00 scope and re-validate.
4. If approved: Update milestone status to `COMPLETED` and proceed to M01 preparation.

