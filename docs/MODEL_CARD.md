# Model Card: Propensity champions

> Cập nhật: 2026-11-26. Số liệu lấy từ `models/latest_model_manifest.json` và `models/lapse/latest_model_manifest.json`.
> Mọi chỉ số là kết quả trên tập test **out-of-time** (cutoff 650 và 680), tức dữ liệu model chưa thấy lúc train và lúc chọn calibration.

## 1. Tổng quan

| | `purchase` | `lapse` |
|---|---|---|
| Câu hỏi | Hộ có mua ít nhất 1 basket hợp lệ trong 14 ngày tới không? | Hộ đang mua đều đặn có *ngừng* mua trong 21 ngày tới không? |
| Đối tượng được chấm | Toàn bộ 2.500 hộ | Hộ có ≥ 1 basket trong 30 ngày **và** ≥ 4 basket trong 90 ngày (khoảng 1.850 hộ) |
| Nhãn | 1 nếu có basket với tổng chi > 0 trong (t0, t0 + 14] | 1 nếu **không** có basket hợp lệ nào trong (t0, t0 + 21] |
| Bundle | `lgbm_purchase_multidomain_v2_h14_20261126_113544_071807` | `lgbm_lapse_multidomain_v2_h21_20261126_113618_388050` |
| Registry | `models/` | `models/lapse/` |
| Dùng cho | Serving API (`/v1/predict/propensity`), batch CRM export | Danh sách retention trong batch (`lapse_risk_day_<cutoff>.csv`) |

Cả hai đều là LightGBM trên **27 feature multi-domain** (`multidomain_v2`, xem [src/features/contract.py](../src/features/contract.py)):
- recency và chu kỳ mua;
- tần suất mua trong các cửa sổ 14 / 30 / 60 / 90 / 180 ngày;
- chi tiêu;
- tỷ lệ hàng tươi sống;
- campaign và coupon;
- mức độ trung thành với cửa hàng;
- 7 biến demographic dạng categorical.

Mọi feature được tính **point-in-time**: chỉ dùng dữ liệu có `DAY ≤ t0`. Cửa sổ "N ngày" lấy đúng N ngày (`t0 − N < DAY ≤ t0`).

**Khác với `multidomain_v1`** (champion trước 2026-11-26, 28 feature):
- Bỏ `primary_store_id`: đây là mã cửa hàng nhưng bị đưa vào model như một con số, nên model có thể học thứ tự vô nghĩa giữa các mã.
- Sửa biên cửa sổ: v1 dùng `DAY ≥ t0 − N`, nên cửa sổ "30 ngày" thực tế lấy 31 ngày.
- Cả hai thay đổi lấy từ projectphase3. Code vẫn tính được v1, nên champion v1 trong history vẫn chấm điểm và rollback được.

Training đọc feature từ **Feast feature store** bằng join point-in-time, trên các snapshot mà batch hoặc backfill đã ghi theo từng cutoff. Có test chứng minh feature đọc từ store trùng khớp từng giá trị với feature tính trực tiếp. Mỗi phiên bản feature có file lịch sử riêng (`household_features_<version>.parquet`).

## 2. Dữ liệu và cách chia

- **Nguồn:** Dunnhumby "The Complete Journey". 2.500 hộ, 2,59 triệu dòng giao dịch, DAY 1–711. Fingerprint dữ liệu là `385c047b9baa0a81` (giống nhau khi đọc từ CSV hay từ Lakehouse).
- **Cách chia theo thời gian** (`TrainingConfig`). Khoảng cách giữa các tập luôn ≥ horizon, để cửa sổ nhãn của tập trước không chồng lên tập sau:

  | Tập | Cutoff | Dùng để |
  |---|---|---|
  | fit | 200, 230, …, 500 (11 mốc) | Train |
  | val | 530, 560, 590 | Early stopping và chọn calibration |
  | test | 650, 680 | Đánh giá out-of-time, chạy gate, làm mốc tham chiếu cho drift |

- **Số dòng (fit / val / test):** purchase 27.500 / 7.500 / 5.000; lapse 19.830 / 5.506 / 3.699.
- **Base rate (fit / val / test):** purchase 0,676 / 0,691 / 0,701; lapse 0,111 / 0,094 / 0,104.

## 3. Huấn luyện

- **LightGBM:** `learning_rate` 0,03, `max_depth` 3, `num_leaves` 8, `min_child_samples` 400, bagging 0,7, `colsample` 0,95, tối đa 1.000 cây, early stopping 50 vòng theo log loss trên tập val.
  - Bộ tham số cây nông này lấy từ projectphase3 (random search 30 cấu hình, kiểm lại bằng backtest 7 fold). Trước đây dùng `max_depth` 6, `num_leaves` 31.
  - Purchase dừng ở vòng 289, lapse ở vòng 157.
- **Calibration:** chọn giữa không calibrate / Platt / Isotonic theo ECE cross-fit 5 fold trên tập val:
  - Purchase chọn **Isotonic**: ECE 0,0043, so với 0,0092 (Platt) và 0,0120 (không calibrate).
  - Lapse chọn **Platt**: ECE 0,0046, so với 0,0074 (Isotonic) và 0,0188 (không calibrate).
- **Tái lập:** `python -m scripts.reproduce_champion --target purchase|lapse` train lại theo đúng công thức ghi trong bundle (bộ feature, tham số) rồi so với champion trên cùng các dòng test. Cả hai: ΔAUC = 0,0000.
  - Hai champion được train khi các thay đổi của v2 chưa được commit (git SHA `5ee576f-dirty`). Kết quả tái lập xác nhận code đã commit cho ra cùng model.

## 4. Kết quả trên tập test out-of-time

| Chỉ số | purchase | lapse |
|---|---|---|
| ROC-AUC | **0,861** | **0,819** |
| PR-AUC | 0,931 | 0,291 |
| Brier skill score | 0,351 | 0,129 |
| ECE | 0,015 | 0,010 |
| Lift@10% | 1,41× (tối đa có thể đạt: 1,43×) | 3,03× (tối đa có thể đạt: 9,6×) |
| Base rate | 0,701 | 0,104 |

**So với champion v1 trên cùng các dòng test** (gate không-kém-hơn, bootstrap theo household):

| | purchase | lapse |
|---|---|---|
| ΔAUC (v2 − v1) | +0,0013, CI95 [−0,0003; +0,0028] | +0,0020, CI95 [−0,0026; +0,0069] |
| Brier skill score v1 → v2 | 0,347 → 0,351 | 0,128 → 0,129 |
| ECE v1 → v2 | 0,019 → 0,015 | 0,009 → 0,010 |
| Số dòng so sánh | 5.000 | 3.699 (hộ đủ điều kiện ở cả hai bộ feature) |

Danh sách target theo từng mức năng lực chiến dịch: xem [analysis/TARGET_ANALYSIS.md](analysis/TARGET_ANALYSIS.md).

**Feature quan trọng nhất** (theo tỷ trọng gain):
- Purchase: `baskets_count_60d` 33%, `baskets_count_90d` 25%, `baskets_count_30d` 8%.
- Lapse: `baskets_count_90d` 32%, `baskets_count_60d` 16%, `std_inter_trip` 14%.

**Feedback ngoài mẫu:** AUC thực tế đo ở cutoff có nhãn đã đủ ngày gần nhất.
- Purchase ở cutoff 697: 0,859.
- Lapse ở cutoff 690: 0,809.

Cả hai đều nằm trong ngưỡng chấp nhận (giảm không quá 0,05 so với AUC lúc được chấp nhận).

## 5. Gate chấp nhận

**Gate chất lượng** (`GateConfig`, chạy trên tập test out-of-time):

| Tiêu chí | purchase | lapse |
|---|---|---|
| ROC-AUC | ≥ 0,75 | ≥ 0,70 |
| Brier skill score | ≥ 0,20 | ≥ 0,05 |
| Lift | lift chuẩn hoá ≥ 0,85 | lift@10% ≥ 2,0 |
| PR-AUC gain | ≥ 0,60 | ≥ 0,10 |
| ECE | ≤ 0,03 | ≤ 0,03 |

Ngưỡng của lapse được chốt *trước* khi train model lapse đầu tiên.

**So sánh với champion đang chạy:** ứng viên chỉ được promote nếu cận dưới của khoảng tin cậy 95% của ΔAUC (bootstrap ghép cặp, lấy mẫu lại theo hộ) ≥ −0,005. Nếu champion có target, horizon hoặc feature set khác thì hai model không so sánh được, và việc promote bị chặn trừ khi dùng `force_promote`.

## 6. Cách dùng đúng

- **Purchase:** dùng để *xếp hạng* và phân tier (decile, urgency) cho lịch liên hệ.
  - Ở horizon 14 ngày, 70% hộ đằng nào cũng mua. Top decile gần như toàn là khách mua đều đặn (precision khoảng 1,0).
  - Vì vậy **không được** coi chi tiêu của top decile là doanh thu tăng thêm nhờ chiến dịch.
- **Lapse:** danh sách retention nên xếp theo **p(lapse) × chi tiêu 90 ngày** (`expected_spend_at_risk`), đúng như batch đang xuất.
  - Danh sách `RETENTION_OFFER` là 10% hộ có giá trị kỳ vọng cao nhất.
- **Xác suất đã được calibrate,** nên có thể dùng trực tiếp làm giá trị kỳ vọng.
  - Isotonic tạo ra các bậc giá trị, nên nhiều hộ có cùng xác suất.
  - Decile được tính theo thứ hạng (rank), không theo ngưỡng xác suất.

## 7. Giới hạn và rủi ro

- **Không phải model nhân quả.** Lift và các con số giá trị chỉ đo *mối liên hệ*. Muốn chứng minh hiệu quả chiến dịch cần holdout hoặc thiết kế uplift.
- **Thời gian:**
  - `DAY` không ánh xạ được về ngày lịch thật, nên không mô hình hoá được mùa vụ.
  - Ngày 278 và 643 có dữ liệu bất thường (đã ghi nhận ở phần EDA lưu trữ).
- **Demographic** chỉ có cho 801 trên 2.500 hộ; các hộ còn lại nhận giá trị `UNKNOWN`. Model có dùng demographic, nên cần lưu ý về công bằng khi dùng điểm số để phân bổ ưu đãi.
- **Điểm cập nhật mỗi tuần.** API trả đúng điểm batch của champion, không điều chỉnh theo sự kiện trong ngày.
- **Lapse:** định nghĩa "hộ mua đều đặn" và ngưỡng 21 ngày là quyết định nghiệp vụ, chưa được kiểm chứng bằng hành vi rời bỏ dài hạn.

## 8. Truy vết

Mỗi bundle schema v2 lưu: `target`, `feature_set_version`, danh sách feature, `horizon_days`, các cutoff, `data_fingerprint`, `git_sha`, tham số LightGBM, `best_iteration`, phương pháp và báo cáo calibration, metric của tập val và tập test, cùng phiên bản thư viện.

Lịch sử promote và rollback được ghi ở `models/history/promotions.jsonl` (purchase) và `models/lapse/history/promotions.jsonl` (lapse). Xem quy trình vận hành trong [RUNBOOK.md](RUNBOOK.md).
