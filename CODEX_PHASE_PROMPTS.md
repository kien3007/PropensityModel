# Codex phase-by-phase prompts — run ONE milestone per turn

## Reusable wrapper (include before each prompt)

```
Read AGENTS.md, PROJECT_MASTER_PLAN.md, docs/decisions.md (if any), and harness/build-log.md.
Do only the milestone named below. Begin with a read-only inspection and plan:
- objective, files, current evidence, assumptions needing verification,
- implementation steps and tests, acceptance gate and expected resource cost.
Do not invent schema/event semantics or claim unrun results.
Ask permission before downloads, destructive actions, external writes or broad installs.
After implementation run tests, show real results, update docs/decisions and build-log,
and STOP at this milestone. Do not automatically advance.
```

## M01 — Business definition

```
Milestone M01: Produce docs/01_business_understanding.md, docs/target_contract.md,
and docs/risk_register.md. Explain exact cohort, 7-day Retail order propensity,
24h gap, binary target, downstream use, success metrics, dataset and causal caveats.
Mark purchase event and timestamp mapping as unresolved until M03.
No real data downloading or training. Gate: terminology, scope, KPIs, assumptions reviewed.
```

## M02 — Source inventory and coverage

```
Milestone M02: Implement a read-only metadata audit with huggingface_hub for
repo t-tech/T-ECD and path dataset/full/. Inventory users.pq, retail/events,
marketplace/events, retail/items; enumerate available day IDs and file sizes,
continuous runs, overlap, missing partitions, estimated network bytes for candidate
90/180/270/365-day ranges. Write manifest CSV/Parquet and coverage report.
This must not download full parquet datasets; avoid snapshot_download on broad globs.
Implement pure functions and synthetic tests for interval/continuity logic.
Gate: verified paths, valid day-range candidates and budget; no inferred schemas.
```

## M03 — Real schema and target audit

```
Milestone M03: With explicit download budget approval, fetch users.pq and a
small set of Retail and Marketplace daily parquet partitions from the verified
manifest. Inspect actual Arrow schemas, data types, action_type distributions,
timestamps, user/item/brand key semantics, Retail order events, rows per day,
missingness, duplicate patterns and coverage. Map only verified raw fields to
normalized internal schema. Create data_dictionary.md and purchase_event_contract.md;
show sample queries and clear GO/NO-GO decision. If completed order cannot be
established, label a recorded Retail order event proxy instead of fabricating status.
Gate: verified target event semantics and valid label horizon coverage.
```

## M04 — Sampling implementation

```
Milestone M04: Implement deterministic user-level nested hash sampling from the
users catalog (0.25%, 0.5%, 1%, 2%) with explicit seed and cohort manifests.
Ensure same user stays in the same cohort across all domains/days, independent of
future labels. Unit tests: stable ID -> stable bucket; nested inclusion; zero
per-event sampling; seed and config audit. Estimate expected users vs observed active
users separately. No download of daily events beyond approved pilot.
Gate: repeatability and non-outcome-based selection demonstrated.
```

## M05 — Streaming ETL

```
Milestone M05: Implement restartable one-day-at-a-time Parquet ingestion for
Retail (and Marketplace if approved). Stage remote file in a temporary folder;
read and filter chosen users with DuckDB; normalize using M03 schema; record
source revision, rows, checksum and output stats; write compressed Parquet;
validate output before cleanup. Enforce RAM/threads/temp disk budget, idempotent
resume and structured errors. Synthetic tests for duplicate rows, missing schema,
partial downloads and selective event retention.
Gate: 3-day dry run and memory/download/retained size report.
```

## M06 — Sampling quality

```
Milestone M06: Produce reference vs sampled distribution comparisons from
users catalog and full-day streaming aggregates where feasible: user geography,
activity, event mix, purchase prevalence, user events/day, major cohort sizes.
Report what is a census reference vs an approximation; add confidence intervals
and investigate rare segments. Create a sampling_quality_report.md. Do not claim
representativeness based only on visually similar random event rows.
Gate: transparent evidence and limitations; whether sample percentage should grow.
```

## M07 — Population snapshots & labels

```
Milestone M07: Generate weekly (user_id,snapshot_at) snapshots for prior-active
users using ONLY eligible past event history. Use feature cutoff strictly earlier
than T-1 day and a future 7-day label window [T,T+7d). LEFT JOIN to Retail target
(events validated in M03); keep both 0 and 1, and mark unknown if label
partitions missing. Do not filter to future purchasers. Synthetic unit tests at
all boundaries, including missing-day and duplicate cases. Report #positives,
#negatives, prevalence, active users and weekly variation.
Gate: label contract verified, negative class retained, no leakage.
```

## M08 — Point-in-time features & splits

```
Milestone M08: Build 7/30/90-day features, including recency, activity,
cart/view/order history and aggregated product context from available verified
fields. Create feature dictionary: meaning, source, time cutoff and null handling.
Generate gold table and chronologically separated train/validation/calibration/test
with purge at split boundaries. Fit no transformations on validation/test.
Synthetic tests for future leakage, past-only popularity, ID uniqueness, missing
categories, unseen users, no target joined as predictor.
Gate: shape and quality checks, RAM-safe Parquet and splitting manifest.
```

## M09 — Baselines & LightGBM

```
Milestone M09: Build constant-prevalence, heuristic, Logistic Regression and
LightGBM baselines using identical chronological training and validation splits.
Measure ROC-AUC, PR-AUC vs prevalence, Precision@10%, Lift@10%, Brier, log-loss;
log model, seed, data manifest, CPU/RAM use. Ensure probability outputs 0..1.
No tuning on final test. Gate: baseline scorecard reproducible and tests pass.
```

## M10 — Tuning & calibration

```
Milestone M10: Run a resource-capped search on LightGBM with chronological
validation. Preserve calibration segment distinct from training/validation/test;
assess raw vs sigmoid/isotonic probability calibration via reliability diagrams,
Brier/log-loss and subgroup behavior. Do not calibrate on final test or use
balanced validation distribution. Gate: justified selected model, documented cost.
```

## M11 — Experiments / sampling proof

```
Milestone M11: Run ablations (profile/RFM; +behavior; +product context; optional
+Marketplace) holding cohort/dates constant. Run nested sample learning curve with
0.25%, 0.5%, 1%, 2% using common held-out validation; report positive counts,
PR-AUC, Lift@10%, calibration and clustered bootstrap confidence intervals, plus
RAM, storage and training cost. Temporal backtest on multiple periods. Never use
the locked final test for choosing data fraction, features or hyperparameters.
Gate: table, figures, uncertainty and written interpretation.
```

## M12 — Final test & model card

```
Milestone M12: Freeze chosen config and run exactly one locked final temporal
test after all selections are made. Report prevalence, PR-AUC/ROC-AUC,
Lift/Precision@K, Brier/log-loss, calibration, uncertainty, subgroup errors,
model limitations and data bias. Write model_card.md, docs/05_evaluation.md and
an explicit record of test date window and whether any test was previously viewed.
Gate: mentor-ready honest evaluation without test tuning.
```

## M13 — Batch deployment

```
Milestone M13: Create CLI batch scorer that validates as_of_date, derives the
same point-in-time features as training, checks schema, loads calibrated model,
and exports user_id/snapshot_at/purchase_probability/model_version in Parquet
or CSV. Test on synthetic users; include no-future-data proof and model reload
consistency. Optional lightweight Streamlit demo only after core batch works.
Gate: smoke test reproducible and instructions documented.
```

## M14 — Final CRISP-DM report and handoff

```
Milestone M14: Produce final CRISP-DM report mapped to 6 phases, architecture,
sampling protocol, data/label definitions, feature dictionary, ablations,
learning curve, locked test scorecard, calibration/lift visuals, resource budget,
limits (including anonymization, cohort definition, license), model card and
reproducibility guide. Prepare an evidence-backed mentor Q&A. Do not invent
numbers or outcomes. Gate: complete rerun from pinned configs on small fixture;
all six phase docs and citations present.
```
