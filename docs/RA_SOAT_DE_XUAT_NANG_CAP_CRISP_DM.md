# Rà soát đề xuất nâng cấp CRISP-DM

Tài liệu ghi nhận cách áp dụng các đề xuất vào **POC hiện tại**. Pipeline V2 cũ sử dụng hợp đồng feature khác, gồm demographic chưa xác minh thời điểm sẵn có; không tự động đưa các thay đổi vào pipeline đó hoặc thay đổi mô hình đã đánh giá. Validation được dùng thăm dò; test không được đọc trong notebook kiểm tra mới.

## Những điểm cần sửa trước khi sử dụng

1. **Không suy ROI từ propensity.** Chênh lệch buyer giữa top model và chọn ngẫu nhiên là chênh lệch theo xếp hạng quan sát, không phải giao dịch do liên hệ tạo ra. Thiếu tác động can thiệp, chi phí và margin thực, không thể tính ROI thực hay phê duyệt pilot từ các giá trị đó. Với chính phép tính giả định 109 × 8,5 USD và chi phí 250 × 7 USD, tỷ lệ là khoảng **−47%**, không phải +47%; dù vậy đây cũng không phải ROI nhân quả hợp lệ.
2. **Không nhầm dòng với household.** Artifact legacy `artifacts/propensity_v2/model/top_k_business_metrics.csv` từng gộp household-snapshot; trong file cũ, hàng 250 contacts là **1%**, còn 10% là 2.500 dòng, không phải 250 household độc lập. Thư mục artifact legacy đã xóa trong đợt dọn repo; không trộn các số cũ với pipeline POC mới.
3. **Power ví dụ:** với p0=0,5322, p1=0,58542, alpha hai phía 0,05, power 80%, tỷ lệ chia đều và giả định độc lập, xấp xỉ hai tỷ lệ cho khoảng **1.366 hộ mỗi nhóm**, không phải 2.500. Với 1.250/hai nhóm, MDE xấp xỉ **5,56 điểm phần trăm**, không phải khoảng tin cậy bỏ thành phần power. Các giá trị phụ thuộc giả định và không cho phép tuyên bố chắc chắn đủ/không đủ hộ cho mọi pilot. Baseline nhóm top propensity có thể khác baseline toàn tập; liên hệ lặp và cụm ảnh hưởng cỡ mẫu.
4. **Không gọi product–store–week là tiếp xúc cá nhân đã xác nhận.** `causal_data` chỉ cho biết bối cảnh quảng cáo/trưng bày ở cấp sản phẩm/cửa hàng/tuần. Cần kiểm tra cardinality khóa, mã `0`, missing, thời điểm dữ liệu sẵn có và phạm vi join. Không dùng `x != ''`: mã `0` và NaN có thể bị tính nhầm là exposure. Không có bằng chứng code hiện tại hỗ trợ cột `feature` được đề xuất.
5. **Không chọn feature từ test hoặc một lần importance âm.** Permutation importance biến động và phụ thuộc tương quan, phép hoán vị, phân phối. Cần ablation và rolling validation; không có căn cứ tự động xóa 65 feature hay khẳng định không giảm performance. Pipeline POC mới có 18 feature gốc, không phải bộ 170 feature legacy.
6. **Không fit và chọn calibrator trên cùng kết quả in-sample.** So sánh Platt/isotonic/raw cần fit trên calibration, chọn/đánh giá trên giai đoạn khác theo thời gian. POC hiện chọn giữa raw và Platt như notebook calibration; isotonic có thể được thử ở một thiết kế mới, không dùng lại test đã xem để tuning.

## Đối chiếu từng nhóm đề xuất

| Đề xuất | Đánh giá / xử lý |
|---|---|
| Break-even, business value curve | Đã có kịch bản hòa vốn giả định tách biệt uplift và propensity trong notebook nghiệp vụ. Chưa có unit economics hoặc causal uplift thực tế; không gán ROI. |
| A/B power | Đã thêm phép tính normal minh họa trong notebook kiểm tra mới; điều kiện và MDE được nêu rõ. |
| `causal_data` | Chưa tích hợp vì cần audit khóa, mã, tính sẵn có lịch sử và không được mô tả như tiếp xúc đã xác nhận. Có thể khảo sát như bối cảnh sản phẩm đã mua. |
| DAY modulo 7 | Có thể xem vị trí ngày tương đối, không gán thứ Hai/cuối tuần hay calendar seasonality nếu không có mốc lịch. Không mặc định feature có giá trị dự báo. |
| Cohort | Có thể phân tích mô tả, nhưng first purchase trong panel không chắc là gia nhập thật; zero sau kết thúc quan sát không được tính như đã rời bỏ. Không mặc định suy giảm 30–40%. |
| Missing indicators | Phân biệt số đếm thực sự bằng 0 với tỷ lệ không xác định do mẫu số bằng 0; cờ `rate > 0` không phân biệt hai trường hợp. POC cadence đã dùng thiếu + cờ và imputer fit riêng trên train. |
| Gap distribution | POC đã có median và std của gap theo ngày hoạt động trong 180 ngày, cùng cờ đủ khoảng cách. `abs(last_gap - mean_gap)` không phải standard deviation và không chứng minh độ đều. |
| Trend 7/30 và 90/180 | Đã thử 7 ngày so với 23 ngày trước và 90 ngày so với 90 ngày trước, chuẩn hóa theo ngày, không chồng lấn; thiếu khi lịch sử chưa đủ. Validation chỉ là thăm dò. |
| Feature selection | Chưa tự động xóa; cần rolling ablation trên train và báo cáo độ bất định, chi phí triển khai. |
| LightGBM | Đã được dùng trong các notebook modeling mới, dependency tùy chọn. Không thêm vào V2 legacy như thể hai pipeline tương đương. |
| HPO/early stopping | Chưa chạy. Nếu thực hiện cần temporal folds nội bộ, embargo phù hợp, kiểm soát số thử nghiệm và tách validation/test; code đề xuất early stopping chưa hoàn chỉnh. Không mặc định cải thiện hoặc tốc độ gấp 3–5 lần. |
| Isotonic | Có thể thử trên calibration riêng và chọn trên giai đoạn khác; không chọn theo điểm trên chính dữ liệu đã fit. |
| ECE/MCE | Đã tính trên validation cho 10/15/20 bin trong notebook kiểm tra. Chỉ số nhạy với bin và số mẫu; vẫn báo Brier, log loss và reliability diagram. |
| Rank stability | Đã tính Spearman và top-10% overlap trên household chung giữa các cutoff validation. Không gán ngưỡng tốt/xấu cố định; thay đổi thứ hạng có thể phản ánh hành vi thực. |
| Fairness | `has_demographics` là cờ có hồ sơ, không phải protected attribute. Chọn top 10% riêng cho từng nhóm sẽ ép selection rate gần bằng nhau và không đánh giá chính sách chung; phải dùng cùng quy tắc lựa chọn và định nghĩa equal opportunity với mẫu đủ lớn. Coverage 801/2500 và thiếu thời điểm sẵn có tạo rủi ro selection bias. |

## Kết quả kiểm tra thêm trên validation

Xem [notebook kiểm tra đề xuất](../notebooks/research/kiem_tra_de_xuat_nang_cap_crisp_dm.ipynb) và [artifact](../artifacts/research/crisp_dm_review/). ECE 15-bin của pipeline 18 + chu kỳ khoảng 0,0172; giá trị thay đổi theo số bin. Spearman trung bình giữa các cutoff khoảng 0,977, top-10% overlap khoảng 0,853; đây là mô tả, không chứng minh mô hình tốt hay xấu.

Thử trend trên validation cho LightGBM: Brier 0,164389 → 0,164359, log loss 0,495405 → 0,495249, ROC-AUC 0,836662 → 0,836698; lift trung bình 1,786582 → 1,783194. Các khác biệt nhỏ và dùng validation đã được xem nhiều lần; chưa đủ bằng chứng đưa trend vào pipeline POC chính.

## Kiểm tra bổ sung sau đề xuất cập nhật

- [Rolling ablation và isotonic](../notebooks/research/rolling_trend_va_isotonic.ipynb): trend cải thiện Brier trong 2/4 fold LightGBM và 3/4 fold Logistic; chưa nhất quán. Với LightGBM 18 feature, isotonic được fit trên cutoff 453–481 và đánh giá riêng 495–537 có Brier 0,165163 và log loss 0,497544, cao hơn raw 0,164990 và 0,496930. Đây là đánh giá thăm dò trên giai đoạn chọn, chưa đủ để thay pipeline.
- [Audit causal context](../notebooks/research/audit_causal_context.ipynb): quét 36.786.524 dòng; trong transaction DAY ≤ 537, 357.651/1.722.750 khóa product–store–week khớp ít nhất một dòng causal, tương đương 20,76% khóa và 22,14% dòng transaction. Có 426 khóa khớp nhiều dòng causal, liên quan 587 dòng transaction. Kiểm tra 852 dòng liên quan cho thấy **tất cả 426 khóa có hai cặp mã khác nhau**: 425 khóa khác display, một khóa khác cả display và mailer; không thể tự deduplicate bằng cách giữ một dòng. Không khớp chưa được xác định là không quảng cáo. Theo User Guide trang 6, `display=0` là không trưng bày và `mailer=0` là không có trong quảng cáo. Chưa xác minh thời điểm dữ liệu sẵn có.
- Không suy ra “người được chọn sẽ mua dù có hay không contact” từ propensity: quan sát không cung cấp cả hai kết quả phản thực. Tương tự, “tiết kiệm coupon” cần quy tắc chi phí và phân bổ thực tế; chọn trúng buyer nhiều hơn không đủ đo tác động can thiệp.
- Xu hướng Brier giảm nhẹ trong khi lift giảm là hai chỉ số thay đổi khác hướng; Brier chịu ảnh hưởng cả calibration và discrimination, không đủ cơ sở xác định nguyên nhân. Ngưỡng tương quan 0,7 không tự chứng minh feature dư thừa. POC cadence không có biến legacy `activity_acceleration_30_vs_90`.
- Pipeline POC đã chốt **giữ xác suất raw**, không phải Platt: trên giai đoạn chọn, cải thiện log loss của Platt nhỏ hơn ngưỡng 0,001 đã đặt trước. So sánh isotonic bổ sung là thăm dò cho model 18 feature. Isotonic tạo ít mức điểm hơn nhưng điều đó không tự chứng minh overfit; số cutoff fit calibration là 5, không phải 53.

## Phân tích quy tắc gộp và thử nghiệm HPO

- [Phân tích quy tắc causal](../notebooks/research/phan_tich_quy_tac_causal.ipynb): trong 426 khóa nhiều mã, quy tắc ANY so với ALL khác nhau ở cả 426 khóa display và ở một khóa mailer; tương ứng 587 và một dòng transaction train. Hai quy tắc chỉ mô tả các dòng có sẵn; chưa xác minh trạng thái thực, exposure hay thời điểm dữ liệu được phát hành. Khoảng ngày/tuần trong bảng không đủ để xác định lag. Chưa tích hợp vào mô hình.
- [HPO theo thời gian](../notebooks/research/hpo_temporal_propensity.ipynb): 30 trial Optuna, bốn fold train, chọn mean Brier; kiểm tra validation sau search, không đọc test. Cấu hình chọn có CV Brier 0,163401 so với 0,163583 của cấu hình gốc, nhưng điểm tốt nhất trên cùng fold dùng để chọn có thể lạc quan.
- Trên validation, model HPO fit toàn train có Brier 0,164038 so với 0,164026 cấu hình gốc; log loss 0,494343 so với 0,494306; ROC-AUC 0,837335 so với 0,837327; lift tuần 1,794736 so với 1,791297. Chênh lệch nhỏ, các chỉ số không đồng thuận và validation đã dùng nhiều lần. Chưa đủ lý do ghi đè pipeline POC. Hai model thử nghiệm này fit đến cutoff 537, khác pipeline bàn giao fit đến 439.

## Kiểm tra bản tổng hợp cập nhật

- **Cỡ mẫu A/B:** 1.366 hộ mỗi nhóm là kết quả xấp xỉ cho baseline **53,22%**, tăng tương đối 10% lên 58,542%, alpha hai phía 0,05 và power 80%. Nếu baseline giả định là **15%** và tăng tương đối 10% lên 16,5%, cùng công thức cho khoảng **9.257 hộ mỗi nhóm**. Hai kịch bản giả định hộ độc lập, chia đều và không tính hao hụt; không quyết định cỡ mẫu pilot thực tế. Chi tiết tại [notebook kiểm tra](../notebooks/research/kiem_tra_de_xuat_nang_cap_crisp_dm.ipynb) và [bảng kịch bản](../artifacts/research/crisp_dm_review/power_scenarios_baseline.csv).
- **Test đã được xem:** notebook HPO không đọc test, nhưng test của project đã được đánh giá trước đó. Không mô tả test chung là chưa từng đọc hoặc hoàn toàn độc lập.
- **Cutoff HPO:** cả cấu hình gốc và HPO trong cùng thí nghiệm đều fit đến cutoff **537**, không phải 539; pipeline POC bàn giao fit model đến cutoff 439. So sánh validation hai cấu hình HPO là cùng phạm vi fit, nhưng khác pipeline bàn giao.
- **Không khẳng định gần tối ưu hoặc delta chỉ là noise:** 30 trial trong search space giới hạn không chứng minh không có vùng tốt hơn. Chênh lệch nhỏ cần được đánh giá theo độ bất định, mục tiêu kinh doanh và dữ liệu chưa dùng để lựa chọn; không có ngưỡng Brier 0,001 áp dụng chung cho mọi quyết định.
- **Unit economics và tác động:** dataset có `SALES_VALUE`, nhưng không cung cấp đầy đủ chi phí, contribution margin hoặc tác động nhân quả của liên hệ. Chỉ cost, margin và coupon redemption chưa đủ suy ra ngưỡng propensity có expected incremental value dương; cần ước lượng tác động can thiệp và chi phí liên quan.
- **Thiết kế pilot:** không mặc định phải loại household từng xuất hiện trong train. Cần xác định mục tiêu tổng quát hóa: dự báo tương lai cho household đã biết hay cho household mới. Phân tầng, cách randomize, thời lượng và xử lý liên hệ lặp phụ thuộc thiết kế, không mặc định 4–6 tuần là đủ.
- **Causal data:** 20,76% là tỷ lệ **khóa** transaction khớp causal, còn tỷ lệ **dòng transaction** khớp là 22,14%. Không suy ra transaction không khớp là không có promotion, hoặc quy đổi tỷ lệ khớp thành tỷ lệ household đã tiếp xúc.
