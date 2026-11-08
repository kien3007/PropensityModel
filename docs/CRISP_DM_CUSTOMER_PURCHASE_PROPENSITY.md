# Customer Purchase Propensity theo CRISP-DM

Ngày hoàn tất: 06-11-2026  
Dữ liệu: Dunnhumby – The Complete Journey trong `sample_data/dunnhumby`

## Tóm tắt điều hành

Bài toán dự báo cho mỗi hộ gia đình tại ngày snapshot `s`:

\[
P(\text{có ít nhất 1 basket trong }(s,s+7]\mid\text{hành vi đến hết ngày }s)
\]

Mô hình được chọn là Histogram Gradient Boosting, sau khi so sánh với Logistic
Regression trên validation theo thời gian. Holdout cuối cùng đạt PR-AUC 0,8498,
ROC-AUC 0,8295 và Brier score 0,1684. Nhóm 10% hộ có xác suất cao nhất có tỷ lệ
mua thực tế 96,84%, cao gấp 1,82 lần tỷ lệ nền 53,22%.

## Phase 1 – Business Understanding

### Mục tiêu kinh doanh

Xếp hạng hộ gia đình theo khả năng mua bất kỳ sản phẩm nào trong 7 ngày tới để:

- ưu tiên audience cho thông điệp, dự báo nhu cầu và chăm sóc khách hàng;
- giảm liên hệ không cần thiết với nhóm có propensity thấp;
- tạo một probability score có thể dùng với các ngưỡng/capacity khác nhau.

Đơn vị dự báo là `(household_key, snapshot_day)`. Kết quả là xác suất từ 0 đến 1,
không phải dự báo sản phẩm cụ thể hay giá trị giỏ hàng.

### Tiêu chí thành công

- Kỹ thuật: PR-AUC và ROC-AUC cao hơn baseline ngẫu nhiên; Brier score và bảng
  calibration đủ tốt để diễn giải xác suất; temporal holdout không leakage.
- Kinh doanh: lift@top-10% > 1,5 và probability score xuất được cho toàn bộ hộ.
- Vận hành: một lệnh huấn luyện, một lệnh batch scoring, artifact có version và
  kiểm thử cutoff.

### Rủi ro và giới hạn

- Propensity cao không đồng nghĩa khách cần khuyến mãi. Gửi coupon cho nhóm này
  có thể trợ giá cho hành vi vốn đã xảy ra.
- `DAY` là chỉ số ngày, không phải ngày lịch; mô hình chưa học mùa vụ thực tế.
- Chỉ 32,07% household-snapshot có demographic; behavior vì vậy là nguồn tín hiệu chính.
- Dữ liệu không có exposure/response experiment thích hợp để suy luận causal.

## Phase 2 – Data Understanding

### Inventory và grain

| Thành phần | Kết quả quan sát |
|---|---:|
| Dòng transaction | 2.595.732 |
| Hộ gia đình | 2.500 |
| Khoảng ngày | 1–711 |
| Household-day có hoạt động | 225.533 |
| Household-snapshot có demographic | 32,07% |

`transaction_data.csv` ở grain dòng sản phẩm trong basket. `BASKET_ID` nhận diện
giao dịch và `household_key` nhận diện hộ. Dữ liệu được gom về household-day trước
khi tạo feature để không đếm một basket nhiều lần.

### Chất lượng dữ liệu

- Không thiếu `household_key` hoặc `DAY`.
- Không có dòng mua trùng hoàn toàn trên khóa hộ/basket/ngày/sản phẩm cùng các
  giá trị quantity, sales và discount được sử dụng.
- Modeling table không có duplicate `(household_key, snapshot_day)`, không thiếu
  feature và target chỉ nhận 0/1.
- Target rate toàn bộ snapshot là 53,05%; test là 53,22%. Accuracy đơn thuần vì
  vậy không phải metric phù hợp.

## Phase 3 – Data Preparation

### Snapshot và chống leakage

Tạo 75 snapshot cách nhau 7 ngày, từ ngày 180 đến 698, tổng 187.305 dòng. Mỗi
snapshot chỉ chứa các hộ đã xuất hiện tính đến cutoff. Quy tắc thời gian:

- feature window: `(s-window, s]`;
- label window: `(s, s+7]`;
- household chỉ đủ điều kiện sau lần mua đầu tiên quan sát được;
- chỉ snapshot có label đã trưởng thành mới được dùng khi train/evaluate;
- chừa một snapshot 7 ngày giữa train–validation và validation–test.

Split thực tế:

| Tập | Snapshot | Số dòng |
|---|---:|---:|
| Train | 180–537 | 129.834 |
| Embargo | 544 | 2.500 (không dùng) |
| Validation | 551–621 | 27.480 |
| Embargo | 628 | 2.500 (không dùng) |
| Test | 635–698 | 24.994 |

### Feature engineering

Các cửa sổ 7/30/90/180 ngày gồm spend, quantity, số basket, tổng discount, số
ngày active, average basket value và discount rate. Feature bổ sung gồm recency,
tenure, lifetime spend/baskets, lifetime average basket value và xu hướng 30 ngày
so với 30 ngày trước. Demographic được điền `Unknown`; không dùng campaign/coupon
future state để tránh leakage.

## Phase 4 – Modeling

Hai candidate dùng cùng train set:

1. Logistic Regression: baseline dễ giải thích, numeric được chuẩn hóa và
   categorical one-hot.
2. Histogram Gradient Boosting: học quan hệ phi tuyến, numeric median imputation
   và categorical ordinal encoding.

Chọn model theo PR-AUC validation, dùng Brier score làm tie-breaker. Gradient
boosting thắng với PR-AUC 0,8604 so với 0,8577. Sau chọn model, Platt scaling được
fit trên validation để hiệu chỉnh xác suất; threshold 0,3826 tối đa F1 trên chính
validation. Test không tham gia chọn model, calibration hay threshold.

## Phase 5 – Evaluation

### Kết quả holdout

| Metric | Test |
|---|---:|
| Positive rate | 0,5322 |
| ROC-AUC | 0,8295 |
| PR-AUC | 0,8498 |
| Brier score | 0,1684 |
| Log loss | 0,5053 |
| Precision @ threshold | 0,7071 |
| Recall @ threshold | 0,8528 |
| F1 @ threshold | 0,7731 |
| Precision @ top 10% | 0,9684 |
| Lift @ top 10% | 1,8197x |

Calibration theo decile bám tương đối sát: decile thấp nhất dự báo trung bình
8,67% và quan sát 8,40%; decile cao nhất dự báo 96,48% và quan sát 96,84%.

### Giải thích mô hình

Permutation importance trên test cho thấy các feature chính:

| Feature | Giảm PR-AUC khi tráo feature |
|---|---:|
| active_days_90d | 0,0444 |
| active_days_30d | 0,0144 |
| recency_days | 0,0114 |
| active_days_180d | 0,0037 |
| baskets_30d | 0,0012 |

Kết luận phù hợp với domain: nhịp mua gần đây và độ mới của lần mua cuối quan
trọng hơn demographic. Importance mô tả đóng góp dự báo, không phải quan hệ causal.

### Quyết định go/no-go

**Go cho batch ranking/pilot có kiểm soát.** Chưa go cho tự động phân bổ coupon vì
cần uplift experiment, cost matrix và guardrail về contact policy. Hiệu quả kinh
doanh thật cần A/B test chứ không suy ra từ offline metrics.

## Phase 6 – Deployment

### Batch scoring

```powershell
python src/propensity_pipeline.py train
python src/propensity_pipeline.py score
```

Scoring gần nhất tại ngày 711 đã tạo xác suất cho 2.500 hộ. Xác suất trung bình
0,5368; median 0,5396; percentile 90 là 0,9291. Dùng ranking/capacity (ví dụ top
10%) cho chiến dịch hữu hạn; chỉ dùng threshold 0,3826 nếu chi phí false positive
và false negative gần với F1 objective.

### Contract đầu ra

`latest_scores.csv` gồm:

- `household_key`: khóa hộ;
- `snapshot_day`: cutoff dữ liệu;
- `purchase_probability_next_7d`: xác suất đã calibration;
- `high_propensity_flag`: 1 nếu vượt threshold được chọn trên validation.

### Monitoring

Theo mỗi lần scoring cần theo dõi:

- data freshness, schema, null, duplicate key và số hộ;
- drift của recency, active days, spend và phân phối score;
- khi label trưởng thành: PR-AUC, Brier, calibration decile, lift@K theo thời gian;
- coverage/performance theo household tenure và nhóm demographic khi đủ mẫu;
- retrain khi performance giảm đáng kể hoặc định kỳ theo chu kỳ dữ liệu.

Rollback bằng cách giữ model bundle trước đó và so sánh shadow score. Không dùng
model nếu raw day không tăng, schema thay đổi hoặc calibration drift vượt ngưỡng
được phê duyệt.

## Tái lập

Config, split, phiên bản runtime và toàn bộ metric nằm trong
`artifacts/propensity/metrics.json`. Model bundle chứa preprocessing, model,
calibrator, threshold, feature contract và cấu hình thời gian. Kiểm thử tự động
xác minh cutoff và embargo bằng `python -m pytest -q`.
