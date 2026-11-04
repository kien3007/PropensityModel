# Milestone acceptance checklist — CRISP-DM T-ECD Project

Run one milestone at a time. Do not mark a gate passed without recorded evidence.

- [ ] M00 — Repo, Python 3.12 venv, tests, AGENTS.md, config, gitignore, build-log
- [ ] M01 — Business brief, eligibility, target contract draft, risk register reviewed
- [ ] M02 — HF metadata inventory: Retail/Marketplace partitions, day overlap, download budget
- [ ] M03 — Pilot schema, real event dictionary, order/purchase label verification, GO/NO-GO
- [ ] M04 — Deterministic nested user sampling with test proof (0.25/0.5/1/2%)
- [ ] M05 — Partition streaming ETL, source/output row counts, checksums, resume support
- [ ] M06 — Sample representativeness report vs reference: customer and behavior distributions
- [ ] M07 — Eligibility snapshots and 7-day labels; complete label coverage; negative class retained
- [ ] M08 — 7/30/90-day feature library, point-in-time tests, temporal/purged split
- [ ] M09 — Constant/heuristic/Logistic/LightGBM baselines; validation metrics
- [ ] M10 — Limited tuning, probability calibration on separate chronological window
- [ ] M11 — Feature ablation, nested sample learning curve, forward backtesting, confidence intervals
- [ ] M12 — Final locked test (one-time after selection), model card and honest limitations
- [ ] M13 — Batch inference CLI, model reload, schema and cutoff validation, demo
- [ ] M14 — Full CRISP-DM final report, reproducibility and mentor presentation

## Evidence bundle at each gate
- `source_revision`, `config`, `seed`, `code_commit`, `start/end dates`, `sampling_fraction`
- `data counts` and if relevant `positive/negative counts`, `target prevalence`, missing partition count
- `tests run` and actual result, performance/quality plots where needed
- `RAM peak`, `temporary disk`, download/retained bytes for data milestones
- `decision log` including assumptions, alternative options, limitations, approval

## Stop conditions
- Purchase event semantically unverifiable → use explicit `recorded_order` proxy or revise objective
- No complete label windows → no zero labels and no training
- Leakage test fails → fix data prep before training
- Representative sample cannot be supported with audit → no generalization claim
- Significant test tuning leak → revise evaluation; do not report contaminated test as unbiased
