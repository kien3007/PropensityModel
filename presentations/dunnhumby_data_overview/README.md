# Dunnhumby — Data Understanding, Preparation & Visualization

20 slide chính và 9 slide phụ lục từ điển field, bằng tiếng Việt. Thiết kế được điều chỉnh từ [deepmancer/latex-presentation-template](https://github.com/deepmancer/latex-presentation-template): tỷ lệ 16:9, màu xanh đậm RGB (0,33,71), tiêu đề và footer có đường kẻ. Template gốc theo MIT License; xem `TEMPLATE_LICENSE.txt`.

## Biên dịch

Dùng XeLaTeX để hỗ trợ Unicode tiếng Việt. Từ thư mục project:

```powershell
cd presentations/dunnhumby_data_overview
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Trong Overleaf, upload thư mục này, chọn `main.tex` và compiler **XeLaTeX**. Các biểu đồ PDF đã được tạo sẵn trong `figures/`; không cần dữ liệu gốc để biên dịch. Điền tên/ngày ở đầu file `main.tex` và trong footline.

Nếu Windows chưa nhận lệnh `xelatex` trong PATH nhưng MiKTeX đã được cài tại vị trí mặc định của máy này, chạy từ thư mục bài trình bày:

```powershell
& 'C:\Users\kienn\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe' -disable-installer -interaction=nonstopmode -halt-on-error main.tex
& 'C:\Users\kienn\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe' -disable-installer -interaction=nonstopmode -halt-on-error main.tex
```

Bản đã biên dịch để xem ngay: `main.pdf`.

## Cập nhật biểu đồ

Từ root của repo, với pandas, NumPy và matplotlib:

```powershell
python presentations/dunnhumby_data_overview/build_figures.py
```

Biểu đồ tính trực tiếp từ CSV local và tạo `figures/summary.json`. Biểu đồ có giới hạn trục hiển thị ở phần đuôi, không thay đổi dữ liệu dùng tính thống kê.
