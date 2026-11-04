# AGENTS.md — T-ECD Customer Purchase Propensity

## Mission
Build an auditable, reproducible, laptop-friendly CRISP-DM ML project for **user-level 7-day Retail order/purchase propensity** using a representative subset of T-ECD Full. Read `PROJECT_MASTER_PLAN.md` before planning or editing. Work one milestone at a time. `docs/decisions.md` is the decision log; record uncertainty explicitly.

## Non-negotiable statistical/data contracts
1. Prediction unit: `(user_id, snapshot_at)`; binary `target=1` iff at least one *verified eligible Retail purchase/order event* in `[snapshot_at, snapshot_at+7 days)`; otherwise `0` only with complete label coverage. If completed status is unverifiable, call the label `recorded_retail_order`, not `completed_purchase`.
2. For snapshot `T`, features must use `event_ts < T-1 day` (24-hour gap), target starts at `T`. Check all timestamp boundaries. Source card requires at least 12-hour gap owing to temporal perturbation.
3. Primary eligible population is users with past activity based only on pre-cutoff event data, not future outcomes. Specify whether the model is for active users, not all 44 million users.
4. Sample USERS with reproducible stable hash + seed *before* examining outcomes. Preserve all their available events in selected domains/dates. Nested fractions must share the hash. Do not downsample events within selected users.
5. Do not drop users without target events. In particular, never reuse T-ECD recommendation baselines' target-positive filtering for binary classification. Keep natural label prevalence in evaluation.
6. Use temporal splits with non-overlapping label windows across split boundaries. Fit preprocessors and calibrators on training/calibration periods only. Never use a future event, future popularity or entire-dataset aggregate in a feature.
7. Raw `user_id` is an identifier/join key, not a predictive feature. `price`, region, category and other columns must be schema-verified; do not invent raw fields, label values or results.
8. Report metrics: PR-AUC vs prevalence, ROC-AUC, Precision/Lift@K, Brier/log-loss, calibration, backtesting and uncertainty. Never report test scores that weren't measured.

## Operational constraints
- Windows/Python 3.12, Ryzen 7 5700U, RAM 16 GB, SSD 512 GB. Prefer DuckDB/Polars lazy over loading all raw Parquet into pandas. Process one daily partition at a time; limit concurrency; log RAM, disk and bytes downloaded.
- Repo must run unit tests without T-ECD real data using **small synthetic fixtures**. Production-like data checks use a separately downloaded real pilot.
- No entire-Full `snapshot_download`, no enormous marketplace item catalog or Payments by default. First inventory exact paths, days, file sizes and network budget.
- Never push secrets or raw/sampled data to Git. Respect CC BY-NC-SA 4.0 license, attribution and non-commercial restrictions. Keep access tokens outside code/config via environment variables.
- Do not delete/overwrite raw cached files until checksums, row counts and outputs have been verified. Ask permission before large downloads, destructive cleanup, installs, remote writes or committing/pushing.
- Record source dataset revision, manifest, sampling fraction, hash seed, timestamp convention, Python/library versions, model seed and metrics in each experiment.

## Codex collaboration protocol
1. Read `PROJECT_MASTER_PLAN.md`, this `AGENTS.md`, `docs/decisions.md` (if present), relevant milestone context, and the current repo tree.
2. For every task, first give: purpose, files to change, known/unknown schema details, tests, acceptance criteria and risks. Keep plans read-only unless explicitly asked to implement.
3. Implement exactly one authorized milestone (M00–M14). Avoid broad refactors or installing heavy libraries prematurely.
4. Create maintainable functions under `src/tecd_propensity/`; notebooks are for exploration only. Include type hints, docstrings for public functions, structured logs and config-driven paths.
5. Create synthetic fixture tests for every timestamp, label, sampling and feature rule. Execute relevant tests and show actual output; do not claim unrun tests pass.
6. Update `harness/build-log.md` and `docs/decisions.md` with evidence, assumptions, blockers, changes, next milestone. Stop on failed data/quality gate; do not secretly proceed.
7. If a raw schema, purchase interpretation, cutoff convention or source URL is uncertain, **inspect the real data or official documentation first**. If still unresolved, state it and mark a GO/NO-GO issue.

## Definition of Done for each milestone
- Source changes minimal and reviewed; tests run and pass or blockers explicit; new data artifacts backed by source/seed/config; documentation reflects actual findings; memory/disk impact checked; previous phase gate not bypassed.

## References
- https://huggingface.co/datasets/t-tech/T-ECD
- https://github.com/kliakhnovich/tecd-baselines
- https://doi.org/10.1145/3770855.3817588
