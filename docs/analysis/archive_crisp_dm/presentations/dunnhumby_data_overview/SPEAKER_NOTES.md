# Gợi ý thuyết trình — Dunnhumby Data Understanding

Thời lượng tham khảo: 20–25 phút, cộng phần hỏi đáp. Các số liệu trong bài được tính trên 8 CSV local; không suy rộng ra toàn bộ khách hàng của retailer.

1. **Mở đầu:** Giới thiệu phạm vi là tìm hiểu dữ liệu, chuẩn bị dữ liệu và trực quan hóa; mục tiêu hôm nay chưa phải so sánh mô hình.
2. **Mục tiêu:** Người nghe cần trả lời được dữ liệu có những bảng nào, mỗi bảng ở grain gì và chúng ta đã kiểm tra trước khi join như thế nào.
3. **Toàn cảnh:** 2.500 hộ, 276.484 basket, 2.595.732 dòng giao dịch và 711 chỉ số ngày. Basket không phải household, còn dòng transaction không phải basket.
4. **Tám bảng:** Giới thiệu vai trò từng bảng. `causal_data` là tên nguồn và phản ánh display/mailer theo product–store–week, không phải bằng chứng nhân quả.
5. **Grain:** Minh họa 1 hộ có nhiều giỏ và một giỏ có nhiều sản phẩm. Chỉ đếm `BASKET_ID` phân biệt nếu muốn biết số basket.
6. **Quan hệ:** Product có thể nối many-to-one sau khi xác nhận khóa. Coupon và campaign có thể gây nhân bản dòng nếu join sai grain.
7. **Quy trình:** Từ inventory và schema đến quality, integrity, aggregation và visualization. Giữ raw CSV và ghi lại mọi quyết định làm sạch.
8. **Missing:** Không có null ở 12 cột transaction, nhưng chỉ 801 hộ có demographic. Left join giúp không loại các hộ thiếu demographic.
9. **Duplicate:** Zero exact duplicate transaction không có nghĩa tất cả khóa nghiệp vụ đều duy nhất. 5.164 exact duplicate coupon mapping cần được đánh giá theo mục đích sử dụng.
10. **Integrity:** 100% transaction products khớp catalog; basket ID nhất quán household/day. Giải thích `many_to_one` và kiểm tra row count trước/sau join.
11. **Quantity:** 23.136 dòng quantity >100 tập trung ở fuel/miscellaneous. Chưa kết luận là sai; cần data owner xác nhận đơn vị trước khi cộng.
12. **Zero sales:** Có 942 basket tổng sales bằng 0. Hiện giữ lại khi mô tả. Định nghĩa purchase hợp lệ cần được business xác nhận.
13. **Chuỗi thời gian:** Lượng basket tăng trong giai đoạn đầu rồi ổn định hơn. Ngày 278 và 643 chỉ có một basket mỗi ngày; có thể do nhiều nguyên nhân, cần kiểm tra thêm.
14. **Thời điểm tham gia panel:** Trung vị lần đầu quan sát là ngày 69. Đó không nhất thiết là ngày household bắt đầu mua tại retailer; không xem thời gian trước đó là không mua.
15. **Basket theo hộ:** Trung vị 79, trung bình 110,6 trên toàn bộ thời gian. Số trung bình cao hơn trung vị vì phân bố lệch phải. Có hộ thực hiện nhiều basket trong ngày.
16. **Dòng sản phẩm theo basket:** Trung vị 5 dòng, trung bình 9,39. Các đuôi trên 60 được gộp lại ở rìa biểu đồ để nhìn rõ, nhưng không bị xóa khỏi dữ liệu.
17. **Department:** Grocery chiếm khoảng 50,8% tổng doanh số ghi nhận. `any basket` rộng hơn grocery do có fuel và miscellaneous.
18. **Brand:** National 72,2% và private 27,8% theo sales trong bộ dữ liệu. Đó không phải thị phần trên toàn thị trường.
19. **Bảng bổ trợ:** Nêu 63,36% campaign assignment, 2.318 redemptions và 36,79 triệu dòng media. Coverage không đồng nghĩa exposure và dữ liệu quan sát không chứng minh hiệu quả chiến dịch.
20. **Kết luận:** Xin xác nhận định nghĩa purchase, ý nghĩa quantity, lịch thực và nguyên nhân hai ngày bất thường; đồng thời kiểm tra quyền sử dụng trước khi chia sẻ dữ liệu.

## Câu hỏi thường gặp

**Vì sao 2.500 hộ có tới 276.484 basket?** Vì dữ liệu quan sát trong 711 ngày; một hộ mua nhiều lần, thậm chí có nhiều basket cùng ngày. Basket là ID giao dịch, không phải số hộ.

**Có nên xóa 942 basket doanh số 0?** Chưa đủ bằng chứng. Cần xác định business definition và kiểm tra lý do zero-sales; có thể báo cáo song song theo nhiều định nghĩa.

**Hai ngày 278 và 643 có phải lỗi hay ngày đóng cửa?** Chưa thể kết luận chỉ từ CSV. Đề nghị đối chiếu lịch cửa hàng hoặc data owner trước khi suy diễn.

**Demographic thiếu 68% thì có dùng được không?** Có thể mô tả có điều kiện trên 801 hộ và giữ `Unknown` khi nối. Không nên giả định mẫu có demographic đại diện cho toàn bộ 2.500 hộ.
