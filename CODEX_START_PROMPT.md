# Codex — First Task (Copy-Paste Prompt)

You are the technical ML engineer and methodology assistant for a **Customer Purchase Propensity** project built with T-ECD Full on a Windows laptop (Ryzen 7 5700U, 16 GB RAM, 512 GB SSD), following all six CRISP-DM phases.

Before doing any edits, read and analyze:
- `AGENTS.md` (mandatory project policies)
- `PROJECT_MASTER_PLAN.md` (methodology, definitions, milestones, acceptance gates)
- existing repository tree and environment

Project objective: predict the probability that an eligible, previously active user records **at least one purchase/order event in the Retail domain in the next 7 days**. This is customer-level binary classification, not SKU-level recommendation and not repeat-purchase-only modeling. Whether the raw event represents *completed purchase* must be verified; do not fabricate event semantics. Features are based on user, recent behavior and product interactions, with a 24-hour gap before snapshot time. The evaluation population includes users who do **not** purchase.

For this task, implement **M00 (repository bootstrap) only** and create drafts for M01 artifacts, without initiating network data downloads or model training:
1. Inventory the current repository and environment in read-only mode, recording what exists.
2. Present an implementation plan containing files, expected commands, risks, and acceptance checks; wait for approval if a destructive action, dependency installation or network access is required.
3. Build a minimal, well-structured Python 3.12 project scaffold with `src/tecd_propensity/`, `tests/`, `configs/`, `docs/`, `data/` (gitignored), `reports/`, `harness/build-log.md` and `README.md`. Do not overwrite existing work.
4. Provide environment setup instructions using venv/pip on Windows. Start with minimal dependencies (DuckDB, Hugging Face hub, PyArrow, PyYAML, pytest); defer heavy ML stack until required. Create `.env.example` without real credentials.
5. Write `docs/01_business_understanding.md` and `docs/target_contract.md` as **DRAFT**, explicitly identifying decisions that require dataset audit (event mappings, timestamp semantics, date coverage and eligibility).
6. Add synthetic-data tests for configuration parsing, deterministic stable-hash user sampling, and timestamp boundaries as appropriate. Do not represent synthetic fixtures as real data results.
7. Run applicable tests and show their actual output, files changed and unresolved issues. Update `harness/build-log.md` and stop at Gate M00/M01. Do not start M02 unless explicitly instructed.

Essential constraints:
- Do not invent raw data schemas or numerical model results.
- Do not download the full T-ECD dataset (about 2.8 TB) or `marketplace/items.pq` (large file).
- Never derive eligible cohorts or user sampling from future purchase labels.
- Do not drop negative users, do not random-split rows, do not use a target date's events as features.
- Source code and notebook paths must be portable. No hardcoded credentials, no dataset files committed to Git.
- Ask for approval before large downloads, overwrites, deletes, Git pushes or external changes.

Output your completion summary as:
- Implemented files
- Assumptions verified versus unresolved
- Exact tests run and results
- RAM/disk impact if measured
- Decisions or blockers needing my approval
- Suggested next **single milestone**
