# Speaker Notes – Vì sao chọn Dunnhumby?

Deck chính: `dunnhumby_dataset_selection.tex`  
Thời lượng khuyến nghị: 18–25 phút, chưa gồm Q&A.

## Slide 1 – Title

“Hôm nay tôi chưa tập trung vào thuật toán. Tôi muốn bảo vệ một quyết định quan
trọng hơn: tại sao bộ dữ liệu được chọn thực sự phù hợp với customer purchase
propensity và với một quyết định kinh doanh có thể triển khai.”

## Slide 2 – Thông điệp chính

Nhấn ba ý: đúng grain, đủ context và dựng được target theo thời gian. Quy mô dữ
liệu là lợi thế, nhưng không phải lý do chính.

## Slide 3 – Bài toán cần dữ liệu gì?

Giải thích household-snapshot là đơn vị dự báo. Nếu dataset chỉ có một dòng mỗi
khách hoặc không có timestamp, không thể mô phỏng việc chấm điểm tại nhiều thời
điểm và dự báo hành vi tương lai một cách đúng nghĩa.

## Slide 4 – Tiêu chí lựa chọn

Nói rõ tiêu chí được đặt ra trước modeling để tránh cherry-picking. Trọng số phản
ánh use case của project: business/temporal fit quan trọng hơn việc dataset dễ xử lý.

## Slide 5 – So sánh lựa chọn

- Instacart mạnh cho next-product/reorder, nhưng mục tiêu competition là sản phẩm
  nào có trong next order, không phải xác suất household mua trong một horizon.
- UCI Online Retail dễ dùng nhưng chỉ có sáu feature, là non-store gift retailer
  và có nhiều wholesale customers; marketing context hạn chế.
- Các score 1–5 là assessment theo tiêu chí của project, không phải xếp hạng phổ quát.

## Slide 6 – Mô tả chính thức

Nhấn rằng chính Dunnhumby mô tả dataset gồm household transactions khoảng hai
năm, toàn bộ store purchases và direct-marketing history. Vì vậy project không
“ép” dataset vào một use case xa lạ.

## Slide 7 – Kiến trúc dữ liệu

Không cần dùng mọi bảng ngay trong champion model. Giá trị là có customer view
đa chiều và đường mở rộng sang campaign/uplift khi business question trưởng thành.

## Slide 8 – Quy mô audit thực tế

Phân biệt số liệu audit với mô tả trên website. Đây là các con số đo trực tiếp từ
file trong project: 2,596 triệu transaction lines, 276 nghìn baskets và product
join 100%.

## Slide 9 – Target point-in-time

Chỉ vào timeline: feature kết thúc tại cutoff; label bắt đầu sau cutoff. Household
chỉ xuất hiện sau first purchase. Đây là cơ sở chống temporal leakage.

## Slide 10 – Horizon

7 ngày không được chọn tùy ý. 14/30 ngày làm target positive quá cao trong cohort
frequent shoppers. 7 ngày vừa khớp weekly action vừa giữ khả năng phân biệt.

## Slide 11 – Feature richness

Mỗi nhóm feature phải trả lời một giả thuyết kinh doanh. Không lập luận rằng nhiều
feature luôn tốt hơn. Điểm mạnh của dataset là có thể bắt đầu từ RFM và mở rộng mà
không thay prediction grain.

## Slide 12 – Business fit

Phân biệt rõ propensity và uplift. Propensity phù hợp để ưu tiên capacity hoặc
planning. Nếu quyết định là cấp coupon, cần treatment/control trước khi nói ROI.

## Slide 13 – Hạn chế

Chủ động nói hạn chế trước khi người nghe hỏi. Đặc biệt:

- cohort frequent shoppers làm external validity bị giới hạn;
- demographic missing không ngẫu nhiên;
- `QUANTITY` fuel không phải item count;
- không có calendar date mapping;
- campaign data quan sát không đủ cho causal inference.

## Slide 14 – Bằng chứng hậu nghiệm

Kết quả model chỉ xác nhận dataset có signal; không phải lý do ban đầu để chọn.
Lift top 10% nghĩa là nhóm được xếp cao có purchase rate gấp khoảng 1,82 lần nền,
không có nghĩa campaign tạo thêm 1,82 lần purchase.

## Slide 15 – Kết luận

Nếu bị giới hạn thời gian, đọc lại sáu lý do trên slide. Câu chốt:

> Dunnhumby tạo được cầu nối hoàn chỉnh từ lịch sử hành vi household đến một
> quyết định ưu tiên khách hàng có thể kiểm chứng theo thời gian.

## Slide 16 – Bước tiếp theo

Đề nghị hội đồng xác nhận dataset và target framing. Để chứng minh business value
thực tế, cần thêm action, weekly capacity, contact cost, margin và randomized pilot.

## Các câu hỏi có thể được hỏi

### Vì sao không chọn dataset lớn hơn?

Quy mô không thay thế target fit. Dataset rất lớn nhưng thiếu stable customer ID,
timestamp hoặc marketing context vẫn không giải quyết đúng propensity use case.

### 2.500 hộ có quá ít không?

Đơn vị modeling là household-snapshot: 75 snapshots tạo 187.305 dòng point-in-time.
Tuy nhiên household correlation đã được tôn trọng bằng temporal split và
household-cluster bootstrap; không giả định 187 nghìn dòng là độc lập hoàn toàn.

### Có dùng causal_data không?

Đã audit 36,8 triệu dòng. Champion hiện ưu tiên behavioral signal để giữ target
contract rõ. Causal media data chỉ nên join khi exposure use case được định nghĩa,
tránh thêm feature chỉ vì dữ liệu tồn tại.

### Tại sao demographic chỉ 32% vẫn dùng dataset này?

Propensity dựa chủ yếu vào observed behavior. Missing demographic được giữ thành
nhóm `Unknown`; không loại 68% household. Kết quả cũng cho thấy cadence/diversity
quan trọng hơn demographic.

### Dataset này có chứng minh coupon hiệu quả không?

Không. Nó hỗ trợ nghiên cứu campaign history và tạo hypothesis, nhưng incremental
effect cần randomized treatment/control hoặc một thiết kế causal hợp lệ.

## Cách biên dịch

Deck dùng Unicode tiếng Việt và cần XeLaTeX:

```powershell
xelatex presentations/dunnhumby_dataset_selection.tex
xelatex presentations/dunnhumby_dataset_selection.tex
```

Chạy hai lần để cập nhật tổng số slide. Nếu biên dịch từ thư mục `presentations`,
chỉ cần dùng tên file. Thay `[Điền tên]`, `[Điền phòng ban / công ty]` và
`[Điền ngày]` trước khi trình bày.
