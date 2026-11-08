# Model Card – Household Purchase Propensity V2

## Model details

| Field | Value |
|---|---|
| Model | Histogram Gradient Boosting regularized |
| Prediction | Any basket in next 7 days |
| Unit | Household-snapshot |
| Training snapshots | Day 180–537 |
| Calibration snapshots | Day 551–621 |
| Test snapshots | Day 635–698 |
| Latest scored snapshot | Day 711 |
| Feature count | 170 |
| Random seed | 42 |

## Intended use

- Weekly ranking for finite-capacity audience or operational prioritization.
- Scenario planning using expected purchasers and top-K trade-offs.
- Input to a controlled treatment/control pilot.

## Out-of-scope use

- Automatic coupon allocation or causal targeting.
- Credit, employment, insurance or other high-impact eligibility decisions.
- Product-level recommendation or purchase-value prediction.
- Real-time use without retraining and latency validation.

## Training and evaluation design

Household behavior is aggregated point-in-time over 7/30/90/180-day windows.
Households enter the cohort only after their first observed purchase. Model family
selection uses three rolling-origin folds. Primary partitions have a 7-day embargo
and the final test is not used for model selection, thresholding or calibration.

## Performance

| Metric | Test | 95% CI |
|---|---:|---:|
| PR-AUC | 0,8506 | 0,8409–0,8612 |
| ROC-AUC | 0,8303 | 0,8226–0,8384 |
| Brier | 0,1679 | 0,1637–0,1717 |
| Lift@10% | 1,819x | 1,779–1,863x |

Confidence intervals use 300 household-cluster bootstrap samples, preserving the
fact that a household appears at multiple test snapshots.

## Explainability

Grouped permutation importance identifies purchase cadence and assortment/store
diversity as the leading signal families. Feature importance is predictive, not
causal. Correlated feature windows make individual importance less reliable than
family-level importance.

## Data limitations

- Demographics cover 32,04% of households and are missing-not-at-random.
- `DAY` lacks calendar-date mapping.
- Raw `QUANTITY` is excluded due incompatible scale in fuel/miscellaneous sales.
- Observational campaign/coupon fields do not establish treatment effect.
- Results are specific to households already observed in this dataset.

## Segment considerations

Performance and calibration vary with recency/activity. The demographic-known
cohort has much higher purchase prevalence, likely reflecting data collection
selection. Do not interpret this difference as demographic causality or use it to
exclude households without demographic records.

## Monitoring and retraining

- Validate schema, freshness, uniqueness and join coverage every batch.
- Monitor rolling behavior PSI; cumulative lifetime-feature PSI is informational.
- When labels mature, monitor PR-AUC, Brier, calibration and lift@K by snapshot.
- Review segment metrics and business guardrails.
- Retrain on approved cadence or after sustained performance/calibration decline.

## Human oversight

Business owners select capacity/action and approve the pilot. The model provides
ranking evidence, not the final customer-treatment decision. A randomized holdout
is required before claiming incremental business value.
