# Dunnhumby — Data Understanding, Preparation & Visualization

34 slide chính và 9 slide phụ lục từ điển field, bằng tiếng Việt. Nội dung đi từ cấu trúc nguồn đến hết Data Understanding và EDA cho bài toán Customer Purchase Propensity; chưa trình bày kết quả modeling. Thiết kế được điều chỉnh từ [deepmancer/latex-presentation-template](https://github.com/deepmancer/latex-presentation-template): tỷ lệ 16:9, màu xanh đậm RGB (0,33,71), tiêu đề và footer có đường kẻ. Template gốc theo MIT License; xem `TEMPLATE_LICENSE.txt`.

## Biên dịch

Dùng XeLaTeX để hỗ trợ Unicode tiếng Việt. Từ thư mục project:

```powershell
cd presentations/dunnhumby_data_overview
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Trong Overleaf, upload thư mục này, chọn `main.tex` và compiler **XeLaTeX**. Các biểu đồ PDF trong `figures/` và ảnh EDA trong `eda_figures/` đã được tạo sẵn; không cần dữ liệu gốc để biên dịch. Điền tên/ngày ở đầu file `main.tex` và trong footline.

Nếu Windows chưa nhận lệnh `xelatex`, thêm thư mục chứa XeLaTeX của MiKTeX vào `PATH` rồi chạy lại hai lệnh trên.

Bản đã biên dịch để xem ngay: `main.pdf`.

## Cập nhật biểu đồ

Từ root của repo, với pandas, NumPy và matplotlib:

```powershell
python presentations/dunnhumby_data_overview/build_figures.py
```

Biểu đồ tính trực tiếp từ CSV local và tạo `figures/summary.json`. Biểu đồ có giới hạn trục hiển thị ở phần đuôi, không thay đổi dữ liệu dùng tính thống kê.

Các biểu đồ EDA mới được trích từ output đã lưu của [notebook phân tích](../../notebooks/phan_tich_du_lieu_propensity.ipynb). Khi notebook thay đổi, chạy lại notebook rồi xuất ảnh dùng cho slide bằng:

```powershell
python presentations/dunnhumby_data_overview/build_eda_figures.py
```
