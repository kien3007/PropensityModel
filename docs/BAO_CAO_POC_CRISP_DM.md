# Báo cáo POC: Household Purchase Propensity

## Tóm tắt

Project xây dựng một quy trình dự báo xác suất household có **ít nhất một basket được ghi nhận trong 7 ngày sau thời điểm chốt** dựa trên lịch sử Dunnhumby. POC gồm phân tích dữ liệu, chuẩn bị snapshot theo thời gian, baseline, mô hình, đánh giá và scoring giả lập. Kết quả phục vụ chứng minh tính khả thi trên dữ liệu nghiên cứu; chưa chứng minh tác động của một chiến dịch hoặc doanh thu tăng thêm.

Ứng viên POC là **LightGBM với 18 feature giao dịch và 5 feature chu kỳ mua**, được chốt trước bước đánh giá test trong notebook calibration. Phiên bản bàn giao fit model trên các cutoff 180–439; cutoff 453–481 dùng fit sigmoid, 495–537 dùng lựa chọn calibration theo quy tắc đã định. Kết quả chọn giữ xác suất thô. Phiên bản này không được refit trên toàn bộ train.

## 1. Business Understanding

- **Tình huống giả định:** retailer cần ước lượng khả năng household quay lại mua trong tuần tới để hỗ trợ dự báo hoặc sắp xếp ưu tiên.
- **Đơn vị ra quyết định:** một household tại một cutoff.
- **Target:** `1` nếu có bất kỳ basket trong `(cutoff, cutoff+7]`, bao gồm fuel, miscellaneous và zero-sales; `0` nếu không có basket được ghi nhận.
- **Phạm vi kết luận:** dự báo giao dịch được ghi nhận tại retailer trong dữ liệu. Household có thể mua tại nơi khác; xác suất tự mua không phải uplift do liên hệ.
- **Thước đo POC:** Brier, log loss, ROC-AUC, lift top 10% theo từng cutoff và tính ổn định qua thời gian. Chưa có ngưỡng thành công hay chi phí thực tế được nghiệp vụ xác nhận.

## 2. Data Understanding

Nguồn gồm 2.595.732 dòng transaction, 276.484 basket và 2.500 household trong ngày chỉ số 1–711; 801 household có hồ sơ demographic. Dòng transaction là dòng sản phẩm, không đồng nghĩa một chuyến mua. Product join và tính nhất quán của basket được kiểm tra trong notebook EDA.

- `DAY` chưa có ánh xạ đáng tin cậy sang ngày lịch thực.
- Ngày 278 và 643 mỗi ngày có một basket, cách nhau 365 ngày; nguyên nhân và mức độ thiếu nguồn chưa được xác nhận.
- `QUANTITY` được giữ trong nguồn nhưng không tổng hợp xuyên department vì chưa xác minh đơn vị/quy đổi.
- Thời điểm sẵn có trong lịch sử của demographic và marketing chưa được xác minh; chúng không được dùng trong pipeline POC.
- Các household-snapshot của cùng hộ không độc lập. Nhóm có lịch sử ngắn trong validation có quá ít household để khái quát.

Tài liệu: [Notebook EDA](../notebooks/phan_tich_du_lieu_propensity.ipynb), [data dictionary](DUNNHUMBY_DATA_DICTIONARY.xlsx).

## 3. Data Preparation

Mỗi snapshot chỉ bao gồm household đã xuất hiện đến cutoff; lịch sử dùng `(s-w,s]`, target dùng `(s,s+7]`. Bộ 18 biến gồm recency, thời gian quan sát và basket, sales, số dòng sản phẩm, số ngày hoạt động trong các cửa sổ 7/30/90/180 ngày. Năm biến ứng viên bổ sung đo chu kỳ giữa các ngày mua và khả năng tính chu kỳ.

Các cutoff train là 180–537, validation 551–621 và test 635–698; 544 và 628 là embargo. Train có 129.834, validation 27.480 và test 24.994 household-snapshot. Các cửa sổ lịch sử có thể chồng lấn giữa các lần dự báo; nhãn train của một giai đoạn trưởng thành trước cutoff đầu giai đoạn tiếp theo.

Tài liệu: [Data Preparation](../notebooks/chuan_bi_du_lieu_propensity.ipynb), [hợp đồng dữ liệu](../artifacts/data_preparation/manifest.json).

## 4. Modeling

Baseline nhóm recency ước lượng tỷ lệ mua trong train theo các khoảng 0–7, 8–14, 15–30, 31–60, 61–90 và 91+ ngày. Logistic Regression và LightGBM sử dụng 18 feature được so sánh trên validation; các nhóm feature bổ sung được thử riêng. Lift top 10% tính theo từng cutoff; đồng điểm ở ngưỡng được xử lý bằng kỳ vọng chọn ngẫu nhiên.

Feature chu kỳ giúp Brier và log loss giảm trong cả bốn fold rolling backtest trên train cho cả hai model, nhưng lift không tăng đều. Việc chọn feature đã chịu ảnh hưởng của validation nên rolling backtest không phải xác nhận độc lập.

Tài liệu: [baseline](../notebooks/research/baseline_propensity_recency.ipynb), [thử nghiệm feature](../notebooks/research/thu_nghiem_feature_propensity.ipynb), [rolling backtest](../notebooks/research/rolling_backtest_chu_ky_mua.ipynb).

## 5. Evaluation

Các pipeline LightGBM trong bảng dưới đây fit đến cutoff 439, không phải phiên bản thăm dò trước đó fit đến 537. Xác suất thô được giữ theo quy tắc calibration đã đặt trước. Test đã từng được xem trong quá trình phát triển, nên chỉ là kết quả tham khảo cho POC.

| Tập | Feature | Brier ↓ | Log loss ↓ | ROC-AUC ↑ | Lift@10% tuần ↑ |
|---|---|---:|---:|---:|---:|
| Validation | 18 | 0,16560 | 0,49883 | 0,83497 | 1,78599 |
| Validation | 18 + chu kỳ | 0,16552 | 0,49858 | 0,83520 | 1,79139 |
| Test đã xem | 18 | 0,16833 | 0,50547 | 0,82995 | 1,82242 |
| Test đã xem | 18 + chu kỳ | 0,16840 | 0,50546 | 0,82977 | 1,81804 |

Chênh lệch giữa hai phương án nhỏ; không dùng test để chọn lại feature. Phân tích độ nhạy loại các cửa sổ train liên quan đến ngày bất thường cho kết quả validation tương đối ổn định trong các kịch bản đã thử, nhưng không xác nhận dữ liệu đầy đủ hay loại bỏ mọi tác động lịch sử.

Trong kịch bản validation tiếp cận 10% mỗi cutoff, nhóm được xếp hạng bởi LightGBM + chu kỳ có tỷ lệ mua ghi nhận 96,58%, so với 70,82% kỳ vọng theo recency có xử lý đồng điểm. Đây là đánh giá xếp hạng trên 2.750 **lượt household–cutoff**, không phải giao dịch tăng thêm do can thiệp; baseline và LightGBM đều fit tới cutoff 439.

Tài liệu: [calibration và đánh giá](../notebooks/research/calibration_va_danh_gia_poc.ipynb), [kịch bản nghiệp vụ](../notebooks/research/danh_gia_gia_tri_kinh_doanh_poc.ipynb).

## 6. Deployment giả lập

Quy trình POC: kiểm tra dữ liệu đến cutoff → tính feature point-in-time → áp dụng pipeline đã lưu → xuất household, cutoff, xác suất và thứ hạng. [Notebook scoring](../notebooks/deployment_gia_lap_propensity.ipynb) đối chiếu 18 feature và xác suất tại cutoff 551 với artifact tham chiếu cho 2.498 household. Scoring cutoff 711 tạo [2.500 dự báo mẫu](../artifacts/deployment_poc/household_scores_day_711.csv) cho ngày 712–718; nguồn không có nhãn để xác minh giai đoạn này.

Nếu có dữ liệu mới, cần kiểm tra độ mới và chất lượng nguồn, khóa, tỷ lệ thiếu, phân bố feature/điểm; khi nhãn đủ 7 ngày, theo dõi calibration, Brier, log loss, ROC-AUC và lift theo cutoff. Ngưỡng cảnh báo và lịch cập nhật cần được thiết kế cho môi trường vận hành; notebook hiện không phải production service.

## Kết luận và hướng tiếp theo

POC cho thấy dữ liệu lịch sử có tín hiệu dự báo hành vi mua được ghi nhận và có thể tạo danh sách xác suất có thể tái lập. Chưa chứng minh can thiệp marketing tạo giao dịch tăng thêm, chưa xác nhận hiệu quả trên khách hàng công ty và chưa kiểm chứng độc lập bằng dữ liệu tương lai. Các bước sau POC là xác minh nguồn cập nhật, đánh giá trên dữ liệu mới, thống nhất mục tiêu và thiết kế đo tác động nếu thực hiện chiến dịch.
