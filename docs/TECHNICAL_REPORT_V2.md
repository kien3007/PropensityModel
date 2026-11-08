# Technical Report V2 – CRISP-DM Customer Purchase Propensity

## Phase 1 – Business Understanding

### Objective

Ước lượng tại snapshot `s`:

\[
P(Y_{h,s}=1\mid X_{h,\le s}),\quad
Y_{h,s}=\mathbb{1}[\text{có basket trong }(s,s+7]]
\]

Prediction unit là `(household_key, snapshot_day)`. Primary use case là ranking
cho weekly audience prioritization. Project charter, success gates và governance
nằm trong `PROJECT_CHARTER_V2.md`.

### Why 7 days

| Horizon | Mean target rate | Range qua snapshot |
|---:|---:|---:|
| 7 ngày | 53,05% | 49,78–57,14% |
| 14 ngày | 68,26% | 65,80–71,26% |
| 30 ngày | 80,96% | 78,69–82,99% |

Horizon 7 ngày giữ target cân bằng hơn, phù hợp weekly decision cadence và cho
phép hành động nhanh. Horizon dài hơn làm phần lớn hộ trở thành positive, giảm giá
trị phân biệt cho audience selection.

## Phase 2 – Data Understanding

### Data inventory

| Table | Rows | Vai trò |
|---|---:|---|
| transaction_data | 2.595.732 | Item-level purchase behavior |
| product | 92.353 | Department, brand, commodity |
| hh_demographic | 801 | Household attributes |
| campaign_table | 7.208 | Household campaign assignment |
| campaign_desc | 30 | Campaign windows |
| coupon | 124.548 | Coupon-product mapping |
| coupon_redempt | 2.318 | Observed redemption |
| causal_data | 36.786.524 | Product-store-week display/mailer |

Transaction có 2.500 hộ, 276.484 basket, 92.339 sản phẩm, 582 store và ngày
1–711. Product join coverage là 100%; campaign phủ 63,36% hộ; demographic phủ
32,04%. Coupon redemption khớp household/coupon/campaign window.

### Important semantic findings

- `QUANTITY` có maximum 89.638. Có 23.136 dòng quantity >100, chủ yếu KIOSK-GAS
  và MISC SALES; đây là scaled measure, không phải comparable item count.
- V2 loại `QUANTITY`, thay bằng line items, basket count và product diversity.
- GROCERY chiếm 50,81% sales; DRUG GM 13,10%; PRODUCE 6,92%; MEAT 6,81%;
  KIOSK-GAS 6,75%.
- `DAY` không có mapping sang calendar date; không đưa ra claim seasonality thực tế.

Evidence tái lập nằm trong `artifacts/propensity_v2/data_understanding`.

## Phase 3 – Data Preparation

### Point-in-time rules

- Feature windows dùng `(s-window, s]`.
- Target dùng `(s, s+7]`.
- Household chỉ vào cohort từ lần mua đầu tiên đã quan sát.
- Snapshot cách nhau 7 ngày, từ ngày 180 đến 698.
- Train/validation/test có embargo một horizon.
- Không dùng campaign outcome hoặc future redemption làm feature.

### Feature families

170 feature được xây từ các cửa sổ 7/30/90/180 ngày:

- cadence: recency, active days, interpurchase gap và acceleration;
- basket: frequency, line items, average basket value;
- monetary: spend, spend per active day và trend;
- promotion: discount rate, coupon rate, discounted-line rate;
- category: grocery/fresh/drug-GM/fuel/other spend share;
- brand: private-label share;
- context: morning/afternoon/evening basket share;
- diversity: product, department và store diversity;
- lifecycle và demographic availability.

Modeling table có 187.305 household-snapshot. Không có duplicate key hoặc missing
feature; demographic thiếu được giữ dưới category `Unknown`.

## Phase 4 – Modeling

### Validation design

Ba expanding-window rolling-origin fold được dùng để chọn model. Mỗi fold có một
embargo 7 ngày trước validation. Sau khi khóa model family:

- primary train: ngày 180–537, 129.834 dòng;
- primary validation: ngày 551–621, 27.480 dòng;
- temporal test: ngày 635–698, 24.994 dòng;
- embargo: ngày 544 và 628.

### Candidates

- Behavioral logistic: recency, active days và recent basket count.
- Full logistic: toàn bộ feature, numeric scaling và demographic one-hot.
- HGB shallow.
- HGB regularized với native categorical handling.

Rolling backtest chọn HGB regularized theo mean PR-AUC. Xác suất được Platt-scale
trên primary validation. Threshold 0,3721 chỉ phục vụ F1 reporting; production
nên dùng rank/capacity.

## Phase 5 – Evaluation

### Holdout results

| Metric | Estimate | 95% household-cluster bootstrap CI |
|---|---:|---:|
| PR-AUC | 0,8506 | 0,8409–0,8612 |
| ROC-AUC | 0,8303 | 0,8226–0,8384 |
| Brier | 0,1679 | 0,1637–0,1717 |
| Lift@10% | 1,819x | 1,779–1,863x |
| F1 | 0,7740 | — |
| Precision tại threshold | 0,7013 | — |
| Recall tại threshold | 0,8635 | — |

PR-AUC theo 10 test snapshots nằm trong khoảng 0,8182–0,8718; Brier nằm trong
0,1608–0,1758. Kết quả không phụ thuộc vào một tuần duy nhất.

### Calibration

Decile thấp nhất dự báo 8,44%, quan sát 8,40%; decile cao nhất dự báo 96,63%,
quan sát 96,80%. Model hơi over-predict ở một số decile giữa/cao, nên Brier và
calibration cần tiếp tục theo dõi sau pilot.

### Segment behavior

- Recency 0–7 ngày: positive rate 69,82%, PR-AUC 0,8982.
- Recency 61+ ngày: positive rate 8,97%, PR-AUC 0,1523.
- Demographic-known group có positive rate 75,36%, nhưng đây nhiều khả năng là
  selection bias của data collection; không diễn giải demographic gây ra purchase.
- Nhóm activity trung bình có Brier cao hơn; đây là nhóm cần ưu tiên error analysis.

### Explainability

Grouped permutation cho thấy cadence và diversity chi phối dự báo. Individual
importance thấp do feature windows tương quan mạnh; vì vậy grouped importance là
diễn giải chính, không dùng một feature đơn lẻ để kết luận causal.

## Phase 6 – Deployment

### Commands

```powershell
python src/audit_dunnhumby_data.py
python src/run_propensity_v2.py train
python src/run_propensity_v2.py score
python src/validate_propensity_v2.py
python -m pytest -q
```

Latest scoring tại ngày 711 xuất 2.500 household với probability, flag, rank và
percentile. Probability median là 0,5409; percentile 90 là 0,9282.

### Monitoring

- Input: schema, null, row count, duplicate key, day freshness và join coverage.
- Rolling behavior: PSI và distribution theo 7/30/90/180 ngày.
- Lifetime/tenure PSI chỉ informational vì cumulative features tăng tự nhiên.
- Khi label mature: PR-AUC, Brier, calibration, lift@K và segment metrics.
- Business: incremental margin/contact, cost và negative-contact guardrails.

### Deployment decision

Go cho controlled weekly batch pilot. No-go cho auto-coupon optimization cho đến
khi có randomized experiment hoặc uplift design.
