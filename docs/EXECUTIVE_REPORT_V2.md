# Executive Report – Customer Purchase Propensity V2

## Executive summary

Dự án đã xây dựng một hệ thống xếp hạng 2.500 hộ gia đình theo xác suất phát sinh
basket trong 7 ngày tiếp theo. Mô hình được kiểm chứng bằng ba rolling-origin
backtest và một temporal holdout hoàn toàn tách biệt.

Mô hình Histogram Gradient Boosting regularized đạt trên holdout:

| KPI | Kết quả | 95% bootstrap CI |
|---|---:|---:|
| PR-AUC | 0,8506 | 0,8409–0,8612 |
| ROC-AUC | 0,8303 | 0,8226–0,8384 |
| Brier score | 0,1679 | 0,1637–0,1717 |
| Lift tại top 10% | 1,819x | 1,779–1,863x |

Kết luận: mô hình đủ tốt cho **controlled pilot về audience prioritization**.
Chưa đủ bằng chứng để khẳng định ROI của coupon hoặc tác động incremental.

## Business interpretation

Trong 24.994 household-snapshot ở test, tỷ lệ mua nền là 53,22%. Nếu nguồn lực
chỉ cho phép ưu tiên một phần population:

| Capacity | Precision | Lift | Người mua được capture | Contact tránh được |
|---:|---:|---:|---:|---:|
| Top 5% | 98,16% | 1,845x | 1.227 | 23.744 |
| Top 10% | 96,80% | 1,819x | 2.420 | 22.494 |
| Top 20% | 92,14% | 1,731x | 4.606 | 19.995 |
| Top 30% | 87,84% | 1,651x | 6.587 | 17.495 |
| Top 50% | 77,90% | 1,464x | 9.735 | 12.497 |

Đây là hiệu quả ranking trên dữ liệu quan sát, không phải số giao dịch do chiến
dịch tạo thêm. Capacity nên được chọn theo chi phí contact, biên lợi nhuận và mục
tiêu reach, thay vì dùng threshold F1 như một quy tắc kinh doanh cố định.

## What drives the prediction

Grouped permutation importance cho thấy:

- purchase cadence làm giảm PR-AUC 0,0720 khi bị phá vỡ;
- assortment/store diversity làm giảm PR-AUC 0,0556;
- basket frequency/size giảm 0,0015;
- monetary, promotion, demographic, category và brand đóng góp nhỏ sau khi đã
  biết cadence và diversity.

Thông điệp quản lý quan trọng: **nhịp mua và độ đa dạng điểm mua gần đây quan
trọng hơn hồ sơ demographic**. Điều này cũng làm solution dễ mở rộng cho khách
thiếu demographic.

## Model choice và engineering judgment

| Model | Rolling PR-AUC mean | Std |
|---|---:|---:|
| HGB regularized | 0,85885 | 0,00810 |
| HGB shallow | 0,85832 | 0,00781 |
| Full logistic | 0,85664 | 0,00765 |
| Behavioral logistic baseline | 0,85076 | 0,00831 |

HGB regularized là champion vì có PR-AUC tốt nhất ở backtest. Full logistic nên
được giữ làm challenger: chênh lệch chỉ khoảng 0,0022 PR-AUC, nên nếu chi phí
giải thích/vận hành của champion cao, logistic là phương án thực tế hợp lý.

## Recommended pilot

1. Score hàng tuần và chia household thành propensity bands.
2. Trong từng band đủ lớn, random assignment treatment/control để tránh selection bias.
3. Không đưa nhóm control vào contact; giữ nguyên eligibility và contact policy.
4. Chạy tối thiểu 6–8 chu kỳ để quan sát biến động theo tuần.
5. Primary KPI: incremental purchase rate và incremental contribution margin.
6. Guardrails: contact cost, unsubscribe/complaint, discount cost và segment impact.
7. Chỉ scale khi confidence interval của net incremental value vượt 0.

## Decision requested from management

- Xác nhận action cụ thể mà score sẽ ưu tiên.
- Cung cấp capacity mỗi tuần và unit economics.
- Phê duyệt randomized holdout.
- Chỉ định business owner và model owner cho pilot.

Không nên duyệt auto-couponing chỉ từ kết quả propensity hiện tại.
