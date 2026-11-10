# Hướng dẫn trình bày EDA và bảo vệ lựa chọn feature

Tài liệu này dành cho buổi trao đổi với mentor. Mỗi nhận định phân biệt **quan sát trong dữ liệu**, **giả thuyết nghiệp vụ** và **kết quả đã kiểm tra**. Không coi liên hệ mô tả là nguyên nhân hay chứng minh giá trị kinh doanh.

## 1. Câu chuyện trong khoảng 5 phút

### Câu hỏi và đơn vị quan sát

> “Em muốn dự báo tại một thời điểm chốt, hộ gia đình có phát sinh ít nhất một basket được ghi nhận trong 7 ngày kế tiếp không. Em chọn household vì dữ liệu định danh ở cấp hộ, và chọn basket vì một chuyến mua có thể chứa nhiều dòng sản phẩm.”

Dữ liệu có 2.595.732 dòng transaction, 276.484 basket và 2.500 household. Một dòng transaction giống một dòng sản phẩm trên hóa đơn. Ví dụ một basket gồm 20 dòng không được đếm thành 20 chuyến mua.

### Kiểm tra dữ liệu trước khi phân tích

Trong kiểm tra nguồn, `PRODUCT_ID` duy nhất ở catalog; 100% dòng transaction tìm được product, và basket ID không gắn với nhiều household hoặc nhiều ngày. Điều đó giúp việc nối product không nhân dòng trong dữ liệu hiện tại.

Có 801/2.500 household có bản ghi demographic. Không thể suy từ đó rằng hộ không có hồ sơ không mua; cũng chưa xác nhận hồ sơ sẵn có tại từng cutoff. `QUANTITY` có giá trị lớn tập trung ở fuel và miscellaneous, nhưng guide chưa xác minh quy ước đơn vị riêng; giữ cột nguồn, không tổng hợp xuyên department.

### Hành vi và bằng chứng cho feature

Trong 129.834 household-snapshot train, nhóm recency 0–7 ngày có tỷ lệ target dương **70,47%**, còn nhóm từ 91 ngày là **6,56%**. Theo số ngày hoạt động trong 90 ngày, nhóm 1–3 ngày có tỷ lệ **20,59%**, nhóm 19 ngày trở lên là **89,87%**. Các nhóm khác nhau về nhiều yếu tố và cùng hộ xuất hiện nhiều lần: đây là liên hệ quan sát, không chứng minh nhân quả.

Từ đó, em chọn recency và tần suất làm nhóm feature cơ bản. Em thử thêm chu kỳ mua để phân biệt hai hộ cùng recency nhưng khác thói quen: hộ thường mua 7 ngày/lần và hộ thường mua 30 ngày/lần. Feature chu kỳ cải thiện Brier và log loss ở bốn fold train trong thử nghiệm, nhưng lift không tăng đều; trên test đã xem, kết quả hai phương án rất gần nhau.

### Chuyển sang chuẩn bị dữ liệu

Tại cutoff `s`, feature chỉ dùng lịch sử đến hết `s`; target dùng basket trong `(s, s+7]`. Ngày 1–179 vẫn dùng làm lịch sử; cutoff bắt đầu ở ngày 180 để có cửa sổ 180 ngày của toàn nguồn. Household có thể xuất hiện muộn hơn, nên có feature thời gian đã được quan sát.

### RFM để mô tả, không tự động phân loại khách hàng

Tại cutoff 537 có 2.498 household đã xuất hiện: R là số ngày từ basket gần nhất; F là số basket trong `(447,537]`; M là tổng `SALES_VALUE` trong cùng 90 ngày. Có 184 hộ không có basket trong cửa sổ này. Đây là một lát cắt mô tả; mọi hộ trong lát cắt đã xuất hiện ít nhất 90 ngày trước cutoff, nhưng điều đó không xác nhận nguồn hoàn chỉnh.

Nhóm R 0–7 ngày và F từ 10 basket trở lên có 864/1.096 hộ mua trong 7 ngày tiếp theo (78,83%). Trong các hộ có sales dương, Q1 có 146/579 hộ mua tiếp và median 3 basket, còn Q4 có 495/579 hộ mua tiếp và median 29 basket. Vì tần suất khác nhau, không quy chênh lệch này chỉ cho M. Các tứ phân vị là mô tả tại cutoff, không phải ngưỡng phân khúc dùng trong pipeline.

## 2. Từ EDA đến từng nhóm feature

| Nhóm / feature | Lý do đề xuất | Cách tạo và sử dụng | Bằng chứng / giới hạn |
|---|---|---|---|
| `recency_days` | Lần mua gần nhất có thể phản ánh mức độ hoạt động hiện tại. | Cutoff trừ ngày có basket gần nhất đã ghi nhận. Giá trị 0 nghĩa là có basket tại cutoff. | EDA có chênh lệch mô tả giữa nhóm 0–7 và 91+ ngày; baseline recency tạo mốc so sánh. Không coi là nguyên nhân. |
| `baskets_{7,30,90,180}d` | Đo số chuyến mua gần đây và dài hơn. | Đếm basket trong `(s-w, s]`; một ngày có thể có nhiều basket. | Cửa sổ là lựa chọn thiết kế ban đầu, không phải con số được nhà cung cấp xác nhận là tối ưu. |
| `active_days_{7,30,90,180}d` | Phân biệt hộ mua rải trên nhiều ngày với hộ có nhiều basket cùng ngày. | Đếm số ngày có ít nhất một basket trong `(s-w, s]`. | EDA cho thấy tỷ lệ mua tiếp khác nhau theo nhóm ngày hoạt động. Không suy ra tăng activity sẽ gây mua tiếp. |
| `line_items_{7,30,90,180}d` | Mô tả quy mô giao dịch qua số dòng sản phẩm. | Đếm dòng transaction trong cửa sổ. | Là **số dòng**, không phải số đơn vị sản phẩm. Chưa có thí nghiệm loại riêng nhóm để khẳng định giá trị tăng thêm của nhóm này. |
| `sales_{7,30,90,180}d` | Mô tả giá trị giao dịch được ghi nhận. | Tổng `SALES_VALUE` của các dòng trong cửa sổ. | Guide mô tả số tiền retailer nhận, có thể khác số tiền khách thực trả; không phải lợi nhuận. Chưa có ablation riêng để khẳng định tác dụng của nhóm sales. |
| `observed_tenure_days` | Hộ mới xuất hiện có ít lịch sử hơn, ảnh hưởng cách hiểu các cửa sổ. | `s − ngày đầu hộ xuất hiện trong dữ liệu`. | Không phải thời gian từ lúc hộ trở thành khách hàng ngoài đời. Giá trị 0 nghĩa là lần đầu quan sát ngay tại cutoff. |
| `cadence_median_gap_180d` | Ước lượng khoảng cách điển hình giữa các ngày mua. | Trung vị các khoảng cách liên tiếp giữa những **ngày có basket** trong `(s-180, s]`. | Cần ít nhất hai ngày hoạt động để có một khoảng cách; nếu không đủ thì thiếu. |
| `cadence_gap_std_180d` | Mô tả mức dao động của khoảng cách mua, bên cạnh trung vị. | Độ lệch chuẩn mẫu của các khoảng cách; cần ít nhất hai khoảng cách. | Thiếu khi chưa đủ ba ngày hoạt động. Không tự gọi hộ là “trung thành” chỉ từ feature này. |
| `cadence_recency_ratio` | Đặt recency vào bối cảnh nhịp mua của chính hộ. | `recency_days / cadence_median_gap_180d` khi median tính được. | Không xác định nếu chưa có gap; không xem missing là 0. |
| `cadence_has_gap`, `cadence_has_two_gaps` | Phân biệt không đủ dữ liệu với giá trị gap đo được. | Cờ lần lượt cho biết có ít nhất một hoặc hai khoảng cách. | Là chỉ báo độ đầy đủ lịch sử quan sát, không khẳng định lịch sử ngoài dữ liệu đầy đủ. |

**Xử lý missing trong POC:** Với các biến chu kỳ không đủ dữ liệu, giữ `NaN` và dùng cờ tương ứng. LightGBM nhận giá trị thiếu; trong các thử nghiệm Logistic, imputer học median trên train của từng lần fit. Không fit imputer trên toàn bộ train–validation–test.

## 3. Ví dụ tính tay tại cutoff 180

Giả sử household A được ghi nhận lần đầu ở ngày 171 và có các basket:

| Ngày | Basket | Số dòng transaction | Tổng `SALES_VALUE` |
|---:|---:|---:|---:|
| 171 | B1 | 2 | 20 |
| 175 | B2 | 2 | 30 |
| 180 | B3 | 3 | 25 |
| 180 | B4 | 1 | 5 |

Tại cutoff 180:

- `recency_days = 0` và `observed_tenure_days = 9`.
- Cửa sổ 7 ngày là `(173, 180]`, tức ngày 174–180: `baskets_7d = 3`, `active_days_7d = 2`, `line_items_7d = 6`, `sales_7d = 60`.
- Các ngày hoạt động trong cửa sổ 180 ngày là 171, 175 và 180; hai gap là 4 và 5 ngày. `cadence_median_gap_180d = 4,5`; độ lệch chuẩn mẫu khoảng 0,707; hai cờ đủ gap đều bằng 1.
- `cadence_recency_ratio = 0 / 4,5 = 0`.
- Target bằng 1 nếu có **bất kỳ basket** trong ngày 181–187. Basket ngày 180 thuộc lịch sử; basket ngày 187 thuộc target; basket ngày 188 nằm ngoài target.

Ví dụ là dữ liệu tự dựng để giải thích công thức, không phải một household thật trong dataset.

## 4. Những câu mentor có thể hỏi

**Vì sao chọn recency?** EDA cho thấy các nhóm recency có tỷ lệ mua tiếp khác nhau rõ trong train; baseline recency tạo mốc để kiểm tra feature khác có bổ sung giá trị. Đây là liên hệ mô tả.

**Vì sao có cả số basket và số ngày hoạt động?** Chúng đo hai khía cạnh: hộ có 3 basket trong 1 ngày có `baskets = 3`, `active_days = 1`; hộ có 3 basket trong 3 ngày có `active_days = 3`. Không giả định hai feature hoàn toàn độc lập.

**Vì sao dùng bốn cửa sổ 7/30/90/180?** Chúng cho các góc nhìn gần và dài hơn trong phạm vi dữ liệu. Đây là lựa chọn thiết kế ban đầu, chưa chứng minh từng độ dài là tối ưu. Các cửa sổ lồng nhau và có thể tương quan.

**Vì sao cutoff bắt đầu ở ngày 180?** Để dữ liệu nguồn có đủ cửa sổ dài nhất. Các ngày 1–179 vẫn dùng làm lịch sử; không phải mọi household đã xuất hiện đủ 180 ngày.

**Vì sao dùng mọi basket, kể cả fuel và zero-sales?** Đó là định nghĩa sự kiện được chọn. So với target mọi basket trong train EDA, yêu cầu basket có tổng sales dương thay đổi 33 nhãn; chỉ xét department `GROCERY` thay đổi 5.309 nhãn. `GROCERY` không bao gồm toàn bộ thực phẩm.

**Vì sao không dùng `QUANTITY`?** Guide mô tả là lượng mua nhưng không xác nhận đơn vị/quy đổi riêng giữa mọi department. Giá trị lớn tập trung ở fuel và miscellaneous không tự chứng minh lỗi; chưa đủ cơ sở để cộng xuyên nhóm.

**Vì sao không dùng demographic và marketing?** Chưa xác minh thời điểm thông tin có sẵn tại từng cutoff; demographic chỉ phủ một phần hộ. `causal_data` ở cấp product–store–week, không chứng minh hộ đã xem quảng cáo; có khóa nhiều mã và availability chưa rõ.

**Hai ngày 278 và 643 có phải lỗi?** Ta chỉ quan sát mỗi ngày có một basket; nguyên nhân chưa được xác nhận. Đã đánh dấu cửa sổ có khả năng bị ảnh hưởng và làm một số kiểm tra độ nhạy; không thể kết luận mọi nhãn trong cửa sổ đó sai.

**Vì sao không giữ mọi feature hợp lý?** Trend, thành phần basket và discount đã được thử riêng, nhưng chưa cho cải thiện nhất quán. Một giả thuyết tốt vẫn cần kiểm tra point-in-time, thiếu dữ liệu và lợi ích bổ sung.

**Có thể kết luận coupon tạo thêm mua hàng không?** Không. Propensity mô tả khả năng mua được ghi nhận; cần thiết kế đánh giá can thiệp phù hợp để ước lượng giao dịch tăng thêm.

**Test có độc lập không?** Tập test được chia theo thời gian nhưng đã từng được xem trong quá trình phát triển; kết quả chỉ là tham khảo POC, không phải xác nhận hoàn toàn độc lập.

## 5. Đường dẫn để đối chiếu

- [Notebook EDA](../notebooks/phan_tich_du_lieu_propensity.ipynb): tính toán và biểu đồ nguồn của các tỷ lệ mô tả.
- [Notebook Data Preparation](../notebooks/chuan_bi_du_lieu_propensity.ipynb): target, cửa sổ và feature cơ bản.
- [Thử nghiệm feature](../notebooks/research/thu_nghiem_feature_propensity.ipynb) và [rolling backtest chu kỳ](../notebooks/research/rolling_backtest_chu_ky_mua.ipynb): kiểm tra giá trị bổ sung.
- [Calibration và đánh giá POC](../notebooks/research/calibration_va_danh_gia_poc.ipynb): pipeline cuối fit đến cutoff 439, raw probability; test đã từng được xem.
