# Enterprise Lambda Architecture: Household Purchase Propensity Platform

An end-to-end Enterprise Machine Learning system designed for household purchase propensity scoring on retail datasets (Dunnhumby *"The Complete Journey"* - 2,500 households, 2.59M transactions, 711 days).

Built with a strict **Lambda Architecture** (Batch Layer, Speed Layer, Serving Layer) to satisfy high-accuracy historical propensity while reacting within milliseconds to intraday shopping events. Designed **On-Premise first** with a **Zero-Downtime Cloud Migration Strategy** to AWS / GCP.

---

## 1. High-Level Architecture

The platform separates long-term lookback feature extraction from real-time intraday events, unifying them at inference time:

```mermaid
flowchart TD
    subgraph DataSources["Data Sources & Ingestion"]
        POS["POS Terminals / Web / App (CSV / CDC Stream)"]
        RawMaster["MinIO S3 / GCS (Immutable Master Dataset)"]
    end

    subgraph EventStreaming["High-Throughput Streaming Bus"]
        Kafka["Apache Kafka (KRaft Broker)<br/>Topic: pos-intraday-events"]
        Consumer["Kafka Stream Consumer Worker<br/>SpeedLayerProcessor"]
    end

    subgraph BatchLayer["Batch Layer (High Latency, High Accuracy)"]
        DuckDB["DuckDB Medallion Lakehouse<br/>Bronze -> Silver -> Gold"]
        PIT["Point-in-Time Feature Extractor<br/>(DAY <= t0, Zero Future Leakage)"]
        Feast["Feast Feature Store<br/>Offline Parquet + Online Store"]
        Champion["Calibrated LightGBM (28 Multi-Domain)<br/>Isotonic Calibration (ROC-AUC: 0.8600)"]
        BatchScores["Gold Mart: fct_weekly_batch_scores<br/>Deciles 1-10 & Marketing Actions"]
        Feedback["Ground-Truth Feedback Loop<br/>fct_model_attribution (Realized AUC)"]
    end

    subgraph SpeedLayer["Speed Layer (Low Latency, Real-Time Deltas)"]
        RedisDelta["Redis In-Memory Key-Value Store<br/>realtime:delta:{hh_key} (TTL 24h)"]
    end

    subgraph ServingLayer["Serving Layer & Canary Routing"]
        FastAPI["FastAPI Serving Gateway (Port 8000)"]
        Canary["Canary A/B Router<br/>(90% Champion / 10% Challenger)"]
        Fusion["Lambda Fusion Engine<br/>P_unified = Merge(P_batch, Delta_speed)"]
        Prometheus["Prometheus & Grafana<br/>Metrics & P99 Latency"]
    end

    POS --> RawMaster
    POS -.->|Stream Events| Kafka
    Kafka --> Consumer
    Consumer --> RedisDelta

    RawMaster --> DuckDB
    DuckDB --> PIT
    PIT --> Feast
    Feast --> Champion
    Champion --> BatchScores
    BatchScores -->|Attribution Join| Feedback
    DuckDB -->|Realized Sales (t0..t0+H)| Feedback
    BatchScores -->|Precomputed Baseline| Fusion

    RedisDelta -->|Live Deltas| Fusion
    FastAPI --> Canary
    Canary --> Fusion
    FastAPI -.->|Scrape /metrics| Prometheus
```

### Visual Excalidraw Diagram
- Interactive Excalidraw blueprint: [`docs/architecture/lambda_architecture.excalidraw`](file:///c:/Users/kienn/PropensityVSF/docs/architecture/lambda_architecture.excalidraw)
- Detailed Technical Specification: [`docs/architecture/LAMBDA_ARCHITECTURE_ONPREM_TO_CLOUD.md`](file:///c:/Users/kienn/PropensityVSF/docs/architecture/LAMBDA_ARCHITECTURE_ONPREM_TO_CLOUD.md)

---

## 2. Repository Structure

```
PropensityVSF/
├── docker-compose.yml              # On-premise infrastructure orchestration
├── pyproject.toml                  # Modern Python packaging & dependencies
├── .env.example                    # Template environment variables
├── docker/
│   ├── Dockerfile.serving          # Multi-stage production container for FastAPI Serving
│   ├── Dockerfile.pipeline         # Container for Lakehouse ETL, training & batch scoring
│   ├── prometheus.yml              # Prometheus scraping configuration (Serving API /metrics)
│   ├── grafana/                    # Grafana pre-provisioned data sources & MLOps dashboards
│   └── minio_init.sh               # MinIO S3 bucket provisioning script
├── dags/
│   ├── dag_weekly_batch_propensity.py    # Apache Airflow / Cloud Composer DAG for weekly batch scoring
│   ├── dag_monthly_model_retraining.py  # Apache Airflow DAG for monthly retraining & hot-reload
│   └── dag_ground_truth_feedback_loop.py # Apache Airflow DAG for weekly realized accuracy attribution
├── docs/
│   └── architecture/
│       ├── LAMBDA_ARCHITECTURE_ONPREM_TO_CLOUD.md  # Master architectural specification
│       └── lambda_architecture.excalidraw          # Interactive Excalidraw visual diagram
├── feature_repo/
│   ├── feature_store.yaml          # Feast Feature Store configuration (Offline Parquet + Online Store)
│   ├── entities.py                 # Feast Entity definitions (household_key)
│   └── features.py                 # 28 Multi-domain Point-in-Time FeatureView definitions
├── models/
│   ├── champion_model.joblib       # Active Champion LightGBM bundle (28 features, Isotonic calibrated, AUC: 0.8600)
│   ├── latest_model_manifest.json  # Manifest with metrics, validation scores & cutoffs
│   └── history/                    # Timestamped archives of promoted champion models
├── src/
│   ├── common/
│   │   └── logger.py               # Enterprise logging configuration
│   ├── config/
│   │   └── settings.py             # Pydantic v2 type-safe settings & environment validation
│   ├── features/
│   │   └── pit_extractor.py        # Point-in-Time feature extractor (strictly DAY <= t0)
│   ├── lakehouse/
│   │   └── engine.py               # DuckDB Parquet Medallion Lakehouse & DQ Framework
│   ├── feature_store/
│   │   └── client.py               # Feast FeatureStoreClient (PIT retrieval & online materialization)
│   ├── models/
│   │   └── predictor.py            # Champion model predictor, decile & urgency assignment
│   ├── monitoring/
│   │   ├── drift_detector.py       # DataDriftDetector (PSI, KS-test, Covariate shift audit)
│   │   ├── feedback_loop.py        # GroundTruthFeedbackLoop (Realized AUC, Brier, Lift attribution)
│   │   └── metrics.py              # Prometheus instrumentation (P50/P95/P99 latency, RPS, Deciles, PSI, Routing)
│   ├── pipelines/
│   │   ├── batch_pipeline.py       # Weekly batch scoring pipeline runner
│   │   └── train_pipeline.py       # Retraining & promotion pipeline (config: src/config/model_config.py)
│   ├── scheduler/
│   │   └── daemon.py               # Automated Scheduler Daemon (zero-dependency cron orchestrator)
│   ├── serving/
│   │   ├── api.py                  # FastAPI REST microservice endpoints & canary management
│   │   ├── canary_router.py        # A/B Testing & Canary Router (deterministic hash routing)
│   │   └── fusion_engine.py        # Lambda Serving Fusion Engine (P_batch + Delta_speed)
│   ├── speed_layer/
│   │   ├── delta_stream.py         # Real-time event ingestion & Redis delta management
│   │   └── kafka_bus.py            # High-throughput Kafka Event Producer & Consumer Stream Worker
│   └── training/
│       ├── dataset_builder.py      # Multi-cutoff temporal training data & target generator
│       ├── trainer.py              # LightGBM classifier & baseline Logistic Regression trainer
│       ├── calibrator.py           # None / Platt / Isotonic calibrators, chosen by cross-fitted ECE
│       ├── evaluator.py            # Metrics (ROC-AUC, Brier score, Log loss, Lift@10%)
│       └── model_registry.py       # Champion/Challenger quality gates & model bundle packager
└── tests/
    ├── test_canary_routing.py      # Canary Router & dynamic A/B weight allocation tests
    ├── test_feature_store.py       # Feast Feature Store (registry, online materialization & PIT retrieval)
    ├── test_features_pit.py        # Point-in-Time leakage & feature boundary tests
    ├── test_feedback_loop.py       # Ground-Truth Feedback Loop & Gold Lakehouse attribution tests
    ├── test_kafka_bus.py           # Kafka Event Producer, dual-dispatch & stream worker tests
    ├── test_lakehouse.py           # Bronze/Silver/Gold Lakehouse & Data Quality tests
    ├── test_lambda_fusion.py       # Lambda Serving fusion rules unit tests
    ├── test_monitoring.py          # Data Drift (PSI, KS-test) & Prometheus metrics tests
    ├── test_scheduler.py           # Scheduler daemon, triggers & Airflow DAGs tests
    ├── test_serving_api.py         # FastAPI REST endpoint & hot-reload integration tests
    └── test_training_pipeline.py   # Training, calibration, evaluator & registry gate tests
```

---

## 3. Quick Start

### Option A: Complete Docker Compose Stack (Recommended)

Start PostgreSQL 16 (OLTP), MinIO S3 (Lakehouse Master), Redis 7 (Speed/Serving Cache), FastAPI Serving API, and Batch Pipeline Runner:

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Build and launch all services
docker compose up -d

# 3. View service status
docker compose ps
```

Service URLs:
- **FastAPI Serving API**: [http://localhost:8000](http://localhost:8000) (Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs))
- **Prometheus Metrics Engine**: [http://localhost:9090](http://localhost:9090) (Scrapes `http://localhost:8000/metrics`)
- **Grafana MLOps Dashboard**: [http://localhost:3000](http://localhost:3000) (User: `admin` / `admin` - Pre-configured live dashboard)
- **Apache Kafka KRaft Broker**: `localhost:9092` (Topic: `pos-intraday-events`, KRaft cluster mode)
- **MinIO S3 Console**: [http://localhost:9001](http://localhost:9001) (User: `minioadmin` / `minioadmin`)
- **PostgreSQL OLTP**: `localhost:5432` (`propensity_oltp`)
- **Redis Cache**: `localhost:6379`

### Option B: Local Development Environment

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Retrain the champion definition from src/config/model_config.py
#    (28 features, 14-day horizon, fit/val/test cutoffs, calibration chosen on validation).
#    --no-promote: evaluate only; --force-promote: promote even if the gate fails.
python -m src.pipelines.train_pipeline --no-promote

# 2b. Check the active champion is reproducible from this repo (|dAUC| <= 0.01 on the OOT test cutoffs)
python -m scripts.reproduce_champion

# 2c. Second model: 21-day lapse risk of regular shoppers (registry: models/lapse/)
python -m src.pipelines.train_pipeline --target lapse
python -m scripts.registry --target lapse list

# 2d. Target & business-value analysis -> docs/analysis/TARGET_ANALYSIS.md
python -m scripts.target_analysis

# 3. Run the weekly batch scoring pipeline (also writes lapse_risk_day_<cutoff>.csv if a lapse champion exists)
python -m src.pipelines.batch_pipeline --cutoff 711

# 4. Start the FastAPI Serving Gateway
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```

---

## 4. Automated Scheduling & Production Orchestration

The platform provides dual-track scheduling designed for both standalone on-premise execution and managed cloud environments (Google Cloud Composer / AWS MWAA):

| Pipeline Job | Production Schedule | Orchestration Target | Description |
|---|---|---|---|
| **Weekly Batch Scoring** | Every Monday at 02:00 UTC (`0 2 * * 1`) | `src.pipelines.batch_pipeline` | Ingests new POS data, extracts 28 Point-in-Time features, scores 2,500 households, updates Gold Lakehouse mart, exports CRM audience, and clears intraday deltas. |
| **Monthly Model Retraining** | 1st of month at 03:00 UTC (`0 3 1 * *`) | `src.pipelines.train_pipeline` | Builds multi-cutoff temporal training data, retrains LightGBM candidate, evaluates the governance gate (base-rate-relative quality checks + non-inferiority vs. the active champion on the same OOT test rows), promotes atomically, and calls webhook to hot-reload serving. Roll back with `python -m scripts.registry rollback`. |
| **Daily Data Quality Audit** | Every day at 01:00 UTC (`0 1 * * *`) | `LakehouseEngine.run_data_quality_framework` | Verifies row counts, checks for negative pricing/quantity anomalies, validates temporal bounds, and ensures schema conformity. |
| **Weekly Ground-Truth Feedback** | Every Wednesday at 04:00 UTC (`0 4 * * 3`) | `GroundTruthFeedbackLoop.run_weekly_feedback` | Prospective 7-14 day attribution: joins past batch predictions with realized POS transactions, measures Realized ROC-AUC, Brier score, and Top Decile Lift, persisting to Gold Lakehouse `fct_model_attribution`. |

### Track 1: Lightweight Scheduler Daemon (On-Premise / Docker Compose)
Implemented with zero external dependencies in `src.scheduler.daemon.py`. Integrated directly into `docker-compose.yml` under the `scheduler` service:

```bash
# Run scheduler daemon locally (polls every 30s)
python -m src.scheduler.daemon --poll-interval 30 --serving-url http://localhost:8000

# Execute all jobs immediately once (smoke testing / verification)
python -m src.scheduler.daemon --run-once-all

# Execute a single on-demand job (batch | retrain | dq)
python -m src.scheduler.daemon --run-job batch

# Launch within Docker Compose
docker compose up -d scheduler
```

### Track 2: Enterprise Apache Airflow / Google Cloud Composer DAGs
Production-ready Airflow DAGs located in `dags/` featuring automatic retry policies, XCom parameter passing, and cross-platform fallbacks:
- `dags/dag_weekly_batch_propensity.py`: Orchestrates weekly batch scoring, Lakehouse sync, and marketing segment materialization.
- `dags/dag_monthly_model_retraining.py`: Orchestrates model retraining, governance verification, and calls `POST /v1/models/reload` on the FastAPI Serving Gateway for **zero-downtime hot-reloading** of the model bundle in memory.
- `dags/dag_ground_truth_feedback_loop.py`: Evaluates every target's champion on its latest matured cutoff, and retrains when the realized AUC degraded.

### Monitoring loop

| Signal | Produced by | Reference | Action |
|---|---|---|---|
| Feature and prediction drift (PSI / KS) | Batch pipeline, every run | The champion's own out-of-time test window (bundle `test_cutoffs`) | CRITICAL starts an out-of-cycle retraining |
| Realized AUC | Feedback job, latest cutoff with matured labels | The AUC the champion was accepted with | Drop > `REALIZED_AUC_DROP_ALERT` (0.05) starts a retraining |
| Fusion-rule calibration | Feedback job, from the API's served-prediction log | Realized purchase rate | Reports the dampening / boost values implied by the data |

- **Retraining:** a triggered retraining only produces a *candidate*. Promotion still goes through the governance gate. Set `AUTO_RETRAIN_ON_DEGRADATION=false` to only alert.
- **Getting reports to Prometheus:** the batch and feedback jobs run in their own processes. They publish their reports to `MonitoringState` (Redis, plus a JSON file), and the serving API turns them into gauges on every `/metrics` scrape.
- **Alerts:** defined in `docker/prometheus/alert_rules.yml` (drift, realized AUC drop, p99 latency, missing batch view, serving down, Kafka consumer lag) and routed by Alertmanager (`docker/alertmanager.yml`, port 9093). The rules are unit-tested with `promtool test rules docker/prometheus/alert_rules_test.yml` in CI.

```bash
python -m src.scheduler.daemon --run-job feedback   # realized AUC for purchase + lapse, retrain if degraded
```

---

## 5. REST API Reference

The Serving API fuses precomputed batch views with intraday streaming deltas.

**Where batch scores come from.** `src.pipelines.batch_pipeline` publishes each run to the shared
`BatchScoreStore`: a Redis hash per variant (champion / challenger), swapped in atomically with `RENAME`,
so every replica serves the same snapshot. Without Redis, the API loads the newest
`data/outputs/batch_manifest_day_*.json` artefacts at start-up.

**Security.**
- Protected endpoints require `X-API-Key` with a key from `SERVING_API_KEYS`: `POST /v1/speed/event`,
  `POST /v1/canary/config`, `GET /v1/batch/scores` and `POST /v1/models/reload`.
- With no keys configured, these endpoints are open in development and return `503` in production.
- Predictions and events are rate-limited per client (`RATE_LIMIT_PER_MINUTE`).
- `STRICT_INFRA=true` disables every in-memory fallback, so the service fails fast if Redis or an enabled Kafka is down.

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | – | Liveness: champion bundle, target, horizon, batch-view status |
| `GET /ready` | – | Readiness: `503` until champion batch scores are loaded |
| `POST /v1/predict/propensity` | – | Unified propensity for one household |
| `POST /v1/speed/event` | key | Intraday event (`pos_checkout`, `coupon_clip`, `app_open`); idempotent on `event_id` |
| `GET /v1/batch/scores` | key | Bulk export of batch views (`decile`, `limit`, `variant`) |
| `GET /v1/canary/config` / `POST` | – / key | Read / change the champion-challenger split |
| `GET /v1/models/champion` | – | Active champion contract |
| `POST /v1/models/reload` | key | Hot-reload the champion from the registry |
| `GET /metrics`, `GET /v1/metrics` | – | Prometheus / JSON metrics |

### Predict
```bash
curl -X POST http://localhost:8000/v1/predict/propensity \
  -H "Content-Type: application/json" -d '{"household_key": 100}'
```
```json
{
  "household_key": 100,
  "unified_propensity": 0.885,
  "batch_baseline_propensity": 0.885,
  "decile": 9,
  "urgency_tier": "URGENT",
  "marketing_action": "VIP_COUPON_HALO_UPSELL",
  "intraday_delta": {"intraday_trips_detected": 0, "intraday_coupons_clipped": 0, "intraday_spend": 0.0},
  "applied_fusion_rules": [],
  "model_variant": "champion",
  "routed_variant": "champion",
  "score_source": "batch",
  "batch_snapshot_day": 711
}
```

`score_source` tells you where the baseline came from:
- `batch`: the household's batch view.
- `cold_start`: the household has no batch view. The baseline is the population mean of the latest batch
  (or the champion's test base rate), `decile` is `null` and the action is `COLD_START_DEFAULT`.

`routed_variant` is the canary assignment, and `model_variant` is the batch view actually served. These
differ when the challenger has no published scores yet; the champion view is served instead.

### Ingest a speed event
```bash
curl -X POST http://localhost:8000/v1/speed/event \
  -H "Content-Type: application/json" -H "X-API-Key: $SERVING_API_KEY" \
  -d '{"household_key": 100, "event_type": "pos_checkout", "sales_value": 78.50, "event_id": "pos-123"}'
```

The next prediction for household 100 applies the intraday dampening rule
($P = P_{\text{batch}} \times$ `FUSION_PURCHASE_DAMPENING`). Re-sending the same `event_id` (a client retry
or a Kafka redelivery) is acknowledged without being applied twice.

The fusion coefficients (`FUSION_*` settings) are **heuristics**, not learned parameters. Recalibrate them
from feedback data.

### Batch export, canary, reload
```bash
curl -H "X-API-Key: $SERVING_API_KEY" "http://localhost:8000/v1/batch/scores?decile=10&limit=50"

# Challenger must predict the same target / horizon / feature set as the champion.
# Score it in the batch layer (CHALLENGER_MODEL_PATH or --challenger) so routed traffic gets its own view.
curl -X POST http://localhost:8000/v1/canary/config -H "Content-Type: application/json" \
  -H "X-API-Key: $SERVING_API_KEY" \
  -d '{"challenger_ratio": 0.20, "challenger_model_path": "/app/models/challenger_model.joblib"}'

curl -X POST -H "X-API-Key: $SERVING_API_KEY" http://localhost:8000/v1/models/reload
```

### Load test
`python -m scripts.load_test` starts uvicorn, waits for `/ready` and reports p50 / p95 / p99. The Phase 4
target is predict p99 < 50 ms. Measured locally with one worker, the memory backend and the client on the
same machine:

| Concurrency | p50 | p95 | p99 |
|---|---|---|---|
| 16 | 26.2 ms | 42.3 ms | 47.3 ms |
| 4 | 8.5 ms | 30.6 ms | 31.9 ms |

---

## 6. Automated Testing & Verification

The test suite validates data quality, point-in-time leakage prevention, speed-batch fusion logic, scheduler triggers, Airflow DAGs, and API endpoints:

```bash
# Run complete test suite
python -m pytest tests/ -v

# What CI runs (no real data / no live infrastructure needed)
ruff check .
python -m pytest -m "not integration"
```

Test markers: `requires_data` (self-skips when the Dunnhumby CSVs are absent), `integration` (needs the
local Feast registry / live infrastructure). Pipeline tests use the synthetic dataset in `tests/fixtures/`.

**Test Coverage Summary:**
1. `tests/test_canary_routing.py`: Validates deterministic MD5 hash customer bucketing, sticky 90/10 split, and runtime dynamic A/B weight configuration endpoints.
2. `tests/test_feature_store.py`: Validates Feast Feature Store registration, Online Store materialization, and Point-in-Time historical feature retrieval.
3. `tests/test_features_pit.py`: Validates strictly $DAY \le t_0$, zero future leakage, and correct feature computation across 2,500 households.
4. `tests/test_feedback_loop.py`: Validates prospective ground-truth attribution (7-14 days), Realized ROC-AUC, Brier score, Top Decile Lift, and Gold Lakehouse persistence.
5. `tests/test_kafka_bus.py`: Validates Kafka event producer dispatch, graceful fallback to Redis, and stream consumer worker.
6. `tests/test_lakehouse.py`: Validates Bronze Parquet conversion, Silver star schema views, Gold marts, and automated Data Quality assertions.
7. `tests/test_lambda_fusion.py`: Validates the mathematical rules of the Serving Layer (intraday dampening, coupon intent surge, app nudge).
8. `tests/test_monitoring.py`: Validates Model & Data Drift Monitor (numerical PSI, categorical PSI, two-sample KS-test, drift reports) and Prometheus metrics exporter.
9. `tests/test_scheduler.py`: Validates `ScheduledTask` triggers (interval, daily, weekly, monthly), scheduler execution, and Apache Airflow DAG structures.
10. `tests/test_serving_api.py`: Validates batch-view serving, cold start, fusion rules from settings, API-key auth (401 / fail-closed 503), rate limiting (429), event idempotency, readiness and canary variant serving.
14. `tests/test_serving_infra.py`: Validates the Redis batch store (atomic snapshot swap, shared across replicas), bootstrap from batch artefacts, Redis-backed idempotent speed layer, STRICT_INFRA fail-fast, canary contract checks and challenger scoring in the batch pipeline.
11. `tests/test_training_pipeline.py`: Validates leakage-safe splits, labeling, calibration selection, evaluation metrics, governance gate and the end-to-end retraining pipeline (bundle schema v2).
12. `tests/test_feature_contract.py`: Validates point-in-time correctness of the 28 features and train/serve parity (CSV vs Lakehouse loaders, training vs batch scoring).
13. `tests/test_predictor.py`: Validates champion bundle schema checks (feature set, horizon, ambiguous bundles) and scoring.

---

## 7. Zero-Downtime Cloud Migration Strategy

The system is designed with strict storage and compute decoupling, enabling a phased, zero-downtime migration to GCP or AWS:

| Component | On-Premise Stack | GCP Migration Target | AWS Migration Target | Migration Method |
|---|---|---|---|---|
| **Master Dataset** | MinIO (S3 API) | Google Cloud Storage (GCS) | Amazon S3 | `rclone` / Storage Transfer Service (Active-Active) |
| **Lakehouse Engine** | DuckDB + Parquet | Google BigQuery | Snowflake / Redshift | `dbt-bigquery` adapter swap; zero SQL rewrite |
| **Streaming / Speed** | Redis 7 + In-Memory | Google Cloud Memorystore | Amazon ElastiCache Redis | Redis replication or dual-write from gateway |
| **Event Bus** | Kafka / Direct REST | Cloud Pub/Sub | Amazon Kinesis / MSK | MirrorMaker 2 (Zero loss) |
| **Serving API** | FastAPI on Docker | Google Kubernetes Engine (GKE) | Amazon EKS | Identical Docker images; Blue/Green DNS cutover |

For complete rollback criteria, data validation checks, and operational runbooks, see [`docs/architecture/LAMBDA_ARCHITECTURE_ONPREM_TO_CLOUD.md`](file:///c:/Users/kienn/PropensityVSF/docs/architecture/LAMBDA_ARCHITECTURE_ONPREM_TO_CLOUD.md).

---

## 8. License
Proprietary & Confidential - Built for Enterprise Household Purchase Propensity Scoring.
