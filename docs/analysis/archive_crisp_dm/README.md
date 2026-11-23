# Lưu trữ: phân tích CRISP-DM / EDA (POC giai đoạn 1)

Thư mục này khôi phục phần phân tích dữ liệu và tài liệu POC bị xoá ở commit `14d0407`
("complete enterprise lambda architecture"). Nội dung lấy nguyên trạng từ commit `120ade9`
(2026-11-11, "hoàn thiện slide"), là trạng thái ngay trước khi bị xoá. Mục đích là giữ lại
bằng chứng phân tích để truy vết. Đây **không** phải luồng chạy hiện tại.

| Thư mục | Nội dung |
|---|---|
| `notebooks/` | 4 notebook chính theo CRISP-DM (Data Understanding → Deployment POC), đã có sẵn output |
| `notebooks/research/` | 14 notebook thí nghiệm (baseline, feature, calibration, causal, HPO, giá trị kinh doanh) |
| `docs/` | Hướng dẫn trình bày EDA và feature, rà soát đề xuất nâng cấp CRISP-DM, từ điển dữ liệu |
| `presentations/` | Slide tổng quan dữ liệu Dunnhumby (LaTeX, PDF, figures) và ghi chú chọn bộ dữ liệu |

## Khác biệt so với hệ thống hiện tại

| | POC lưu trữ | Hiện tại (`src/`) |
|---|---|---|
| Target | Mua trong **7 ngày** | `purchase` 14 ngày; `lapse` 21 ngày |
| Model | LightGBM 18 feature + biến chu kỳ | LightGBM 28 feature multi-domain |
| Code | `src/propensity_v2`, `artifacts/` (đã bị xoá) | `src/features`, `src/training`, `src/pipelines` |

- Các notebook import `src/propensity_v2` và đọc `artifacts/`. Cả hai đều không còn trong repo,
  nên notebook **không chạy lại được**. Hãy xem output đã lưu sẵn trong notebook.
- Nếu cần chạy lại, checkout commit `120ade9`. Thư mục `artifacts/` (khoảng 10 MB) cũng nằm ở commit đó.
- Số liệu mô hình trong lưu trữ là của target 7 ngày, nên không so sánh trực tiếp được với champion hiện tại.

Phân tích target và giá trị kinh doanh của hệ thống hiện tại nằm ở [../TARGET_ANALYSIS.md](../TARGET_ANALYSIS.md).

Những giới hạn đã ghi nhận ở POC vẫn đúng với hệ thống hiện tại:
- `DAY` không ánh xạ được về ngày lịch thực.
- Ngày 278 và 643 có dữ liệu bất thường.
- Lift chỉ phản ánh việc xếp hạng, không chứng minh doanh thu tăng thêm.
