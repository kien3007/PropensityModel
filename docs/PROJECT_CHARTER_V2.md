# Project Charter V2 – Household Purchase Propensity

## 1. Executive intent

Xây dựng một hệ thống chấm điểm hàng tuần để ước lượng xác suất mỗi hộ gia đình
phát sinh ít nhất một basket trong 7 ngày kế tiếp. Score được dùng để **ưu tiên**
nguồn lực có giới hạn như audience selection, CRM contact planning và demand
planning; không tự động đồng nhất với quyết định phát coupon.

## 2. Business decision

| Thành phần | Định nghĩa |
|---|---|
| Decision owner | CRM/Marketing Operations hoặc bộ phận vận hành sử dụng audience |
| Decision | Chọn hộ nào được ưu tiên khi capacity nhỏ hơn toàn bộ customer base |
| Decision frequency | Hàng tuần |
| Prediction unit | Một `household_key` tại một `snapshot_day` |
| Outcome | Có ít nhất một basket trong `(snapshot_day, snapshot_day + 7]` |
| Primary output | Probability, rank và percentile |
| Default operating mode | Chọn top-K theo capacity, không dùng threshold cố định |

## 3. Value hypothesis

Nếu chỉ có thể phục vụ hoặc liên hệ một phần customer base, propensity ranking sẽ:

- tăng tỷ lệ household có giao dịch trong audience được ưu tiên;
- giảm số contact cần thiết để tiếp cận cùng một số lượng người mua;
- hỗ trợ lập kế hoạch nguồn lực theo expected number of purchasers;
- tạo nền móng cho A/B test và uplift modeling sau này.

Offline lift chỉ chứng minh khả năng **xếp hạng người sẽ mua**, không chứng minh
contact/coupon tạo ra giao dịch. Giá trị incremental phải được đo bằng randomized
treatment/control experiment.

## 4. Success criteria

### Model gate

- Rolling-origin PR-AUC ổn định và tốt hơn behavioral baseline.
- Temporal holdout PR-AUC ≥ 0,83; lift@10% ≥ 1,70.
- Brier score ≤ 0,18 và calibration decile không sai lệch hệ thống nghiêm trọng.
- Không có leakage qua feature cutoff, cohort eligibility hoặc target maturity.

### Operational gate

- Batch scoring xuất đúng một dòng cho mỗi household đủ điều kiện.
- Dữ liệu mới được kiểm tra schema, freshness, key uniqueness và range.
- Score có model version, snapshot, probability, rank và percentile.
- Có rollback model và monitoring plan.

### Business gate

- Pilot có randomized holdout.
- Đo incremental purchase rate, incremental margin, contact cost và unsubscribe.
- Chỉ scale khi lower confidence bound của giá trị incremental lớn hơn chi phí.

## 5. Scope

### In scope

- Dự báo any-purchase trong 7 ngày.
- Transaction, product, demographic và promotion-derived behavioral features.
- Rolling backtest, calibration, uncertainty, segment và drift analysis.
- Weekly batch scoring và pilot design.

### Out of scope hiện tại

- Dự báo sản phẩm cụ thể hoặc basket value.
- Causal effect của coupon/campaign.
- Real-time serving.
- Suy luận seasonality theo ngày lịch vì dữ liệu chỉ có chỉ số `DAY`.

## 6. Stakeholders và governance

| Vai trò | Trách nhiệm |
|---|---|
| Business sponsor | Phê duyệt use case, capacity và success economics |
| Product/CRM owner | Thiết kế action và treatment/control pilot |
| Data owner | Xác nhận semantics, freshness và quyền sử dụng dữ liệu |
| Model owner | Feature pipeline, training, validation, monitoring và model card |
| Risk reviewer | Kiểm tra privacy, segment impact và contact policy |

Mọi thay đổi target, horizon, eligibility hoặc source semantics phải tạo model
version mới và chạy lại temporal validation.

## 7. Key risks

| Rủi ro | Guardrail |
|---|---|
| Propensity bị hiểu nhầm là uplift | Gắn caveat trên report; bắt buộc randomized pilot |
| Demographic coverage chỉ 32,04% | `Unknown` là nhóm riêng; không loại hộ thiếu demographic |
| Quantity không cùng đơn vị | Không sử dụng `QUANTITY` trong V2 |
| Temporal leakage | Cutoff tests, label maturity và embargo 7 ngày |
| Model phức tạp chỉ tăng rất ít | Duy trì logistic challenger và so sánh chi phí vận hành |
| Drift giả ở lifetime features | Phân loại là structural; alert chủ yếu trên rolling features |
