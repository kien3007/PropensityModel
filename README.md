# Customer Purchase Propensity V2 – Dunnhumby

Dự án CRISP-DM hoàn chỉnh nhằm dự báo xác suất một hộ gia đình phát sinh ít nhất
một basket trong **7 ngày kế tiếp**, phục vụ weekly customer prioritization khi
nguồn lực vận hành hoặc contact capacity có giới hạn.

## Executive result

Mô hình được chọn qua ba rolling-origin backtest và đánh giá trên temporal holdout:

| Metric | Test | 95% bootstrap CI |
|---|---:|---:|
| PR-AUC | 0,8506 | 0,8409–0,8612 |
| ROC-AUC | 0,8303 | 0,8226–0,8384 |
| Brier score | 0,1679 | 0,1637–0,1717 |
| Lift @ top 10% | 1,819x | 1,779–1,863x |

Nhóm top 10% có purchase rate **96,80%**, so với tỷ lệ nền **53,22%**. Kết quả
đủ để đề xuất controlled pilot, nhưng không chứng minh coupon tạo ra incremental
purchase; điều đó cần randomized treatment/control.

## Start here

- [Notebook end-to-end từ dữ liệu thô](notebooks/customer_purchase_propensity_end_to_end.ipynb):
  bài phân tích theo mạch EDA → baseline → temporal cross-validation → evaluation.
- [Executive report](docs/EXECUTIVE_REPORT_V2.md): kết quả và đề xuất cho quản lý.
- [Technical report](docs/TECHNICAL_REPORT_V2.md): đầy đủ sáu phase CRISP-DM.
- [Project charter](docs/PROJECT_CHARTER_V2.md): decision, scope, KPI và governance.
- [Model card](docs/MODEL_CARD_V2.md): intended use, limitations và monitoring.
- [Presentation storyline](docs/PRESENTATION_STORYLINE_V2.md): cấu trúc 12 slide.
- [Analysis notebook](notebooks/customer_purchase_propensity_crisp_dm_v2.ipynb):
  71 cell với EDA, target validation, backtest, uncertainty và deployment.

## Reproduce the project

Yêu cầu Python 3.10 trở lên.

```powershell
python src/audit_dunnhumby_data.py
python src/run_propensity_v2.py train
python src/run_propensity_v2.py score
python src/build_propensity_notebook.py
python src/validate_propensity_v2.py
python -m pytest -q
```

Notebook end-to-end đã có sẵn output. Để tạo và chạy lại notebook sau khi thay đổi
source, dùng `python src/build_propensity_end_to_end.py` và sau đó
`python src/run_propensity_end_to_end.py` (cần IPython).

Nếu cài package ở editable mode, có thể dùng `propensity-v2 train` và
`propensity-v2 score`.

## V2 architecture

```text
src/
├── audit_dunnhumby_data.py       # Full eight-table data audit
├── build_propensity_notebook.py  # Builds the 71-cell review notebook
├── run_propensity_v2.py          # Local CLI entry point
└── propensity_v2/
    ├── config.py                 # Target, windows and feature contract
    ├── data.py                   # Raw loading and household-day preparation
    ├── features.py               # Point-in-time snapshot feature store
    ├── modeling.py               # Backtest, selection, calibration, evaluation
    └── cli.py                    # Train and score workflows
```

V2 tạo 170 feature thuộc các nhóm purchase cadence, basket, monetary, promotion,
category, brand, daypart, diversity, lifecycle và demographics. `QUANTITY` không
được dùng vì fuel/miscellaneous sales có scaled quantity không tương thích với
grocery item counts.

## Key evidence

`artifacts/propensity_v2/data_understanding` chứa inventory, quality audit,
household/department/weekly summaries và target-horizon analysis.

`artifacts/propensity_v2/model` chứa:

- `metrics_v2.json`: run metadata và holdout metrics;
- `rolling_backtest_metrics.csv`: kết quả từng model trên từng fold;
- `bootstrap_confidence_intervals.csv`: uncertainty theo household cluster;
- `top_k_business_metrics.csv`: capacity/precision/recall/lift trade-off;
- `test_metrics_by_snapshot.csv` và `test_metrics_by_segment.csv`;
- `grouped_feature_importance.csv`: explainability theo feature family;
- `feature_drift_psi.csv`: monitoring evidence;
- `latest_scores_v2.csv`: probability, rank và percentile tại ngày 711.

## Model governance

Histogram Gradient Boosting regularized là champion; full logistic regression là
challenger vì chỉ kém khoảng 0,0022 rolling PR-AUC nhưng dễ giải thích hơn. Quyết
định production cuối cùng cần xét thêm latency, maintenance và pilot economics.

Propensity score chỉ trả lời “ai có khả năng mua”. Nó không trả lời “ai sẽ mua
thêm vì được tác động”. Không dùng model để auto-couponing trước khi có uplift
experiment và contribution-margin guardrails.
