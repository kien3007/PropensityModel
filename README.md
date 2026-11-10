# Household Purchase Propensity — Dunnhumby POC

POC theo CRISP-DM dự báo xác suất một household có **ít nhất một basket được ghi nhận trong 7 ngày sau cutoff**, dựa trên lịch sử giao dịch. Đây là nghiên cứu khả thi và scoring giả lập, chưa đo tác động của coupon hoặc doanh thu tăng thêm.

## Bắt đầu từ đây

1. [Báo cáo POC theo CRISP-DM](docs/BAO_CAO_POC_CRISP_DM.md): mục tiêu, phương pháp, kết quả và giới hạn.
2. [Slide PDF](presentations/bao_cao_poc_propensity.pdf) và [nguồn LaTeX](presentations/bao_cao_poc_propensity.tex).
3. [Data dictionary](docs/DUNNHUMBY_DATA_DICTIONARY.xlsx).
4. [Notebook deployment giả lập](notebooks/deployment_gia_lap_propensity.ipynb) và [dự báo mẫu tại cutoff 711](artifacts/deployment_poc/household_scores_day_711.csv).
5. [Hướng dẫn báo cáo EDA và bảo vệ feature](docs/HUONG_DAN_BAO_CAO_EDA_VA_FEATURE.md): giải thích bằng chứng, cách tính và câu hỏi thường gặp.

## Luồng notebook chính

Chạy notebook từ thư mục gốc repo hoặc `notebooks`, theo thứ tự dưới đây nếu cần tái tạo artifact. Notebook đã có sẵn output để xem.

| Bước | Notebook | Vai trò |
|---|---|---|
| Data Understanding | [Phân tích dữ liệu](notebooks/phan_tich_du_lieu_propensity.ipynb) | Chất lượng nguồn, hành vi và giới hạn |
| Data Preparation | [Chuẩn bị dữ liệu](notebooks/chuan_bi_du_lieu_propensity.ipynb) | Tạo train/validation/test và kiểm tra point-in-time |
| Modeling & Evaluation | [Mô hình và đánh giá](notebooks/modeling_va_danh_gia_propensity.ipynb) | Baseline, hai model, feature chu kỳ và kết quả pipeline POC |
| Deployment POC | [Scoring giả lập](notebooks/deployment_gia_lap_propensity.ipynb) | Đối chiếu phép tính và dự báo tại cutoff 711 |

Các notebook thí nghiệm được giữ trong [notebooks/research](notebooks/research/); output tương ứng nằm tại [artifacts/research](artifacts/research/). Chúng được giữ để truy vết, không nằm ngang hàng với bốn notebook chính.

## Dữ liệu và môi trường

Đặt các CSV Dunnhumby trong `sample_data/dunnhumby/`. Dữ liệu nguồn không được đưa vào Git. Cần Python 3.10+, các thư viện trong `pyproject.toml`, Jupyter/IPython để mở notebook và LightGBM để chạy các notebook mô hình. Có thể cài dependency của project bằng:

```powershell
python -m pip install -e ".[modeling,dev]"
```

Nếu LightGBM được cài cục bộ trong `.deps`, các notebook liên quan tự thêm thư mục đó vào đường dẫn Python. Để chạy kiểm thử:

```powershell
python -m pytest -q
```

Notebook Modeling & Evaluation huấn luyện lại baseline, Logistic Regression và LightGBM 18 feature; nó **đọc** kết quả backtest và calibration đã lưu thay vì tự chạy lại các thí nghiệm đó. Nếu các artifact này chưa tồn tại, chạy các notebook nghiên cứu được liên kết trong notebook chính: `rolling_backtest_chu_ky_mua.ipynb` và `calibration_va_danh_gia_poc.ipynb`.

Để chạy scoring, cần `artifacts/data_preparation/`, `artifacts/calibration_final_poc/fixed_pipelines.joblib` và file dự báo validation trong cùng thư mục để đối chiếu. Artifact dạng joblib chỉ nên nạp từ nguồn nội bộ đáng tin cậy. Cutoff 711 dự báo khoảng 712–718; dữ liệu nguồn không có nhãn để xác minh khoảng đó.

## Phiên bản POC

Ứng viên bàn giao là LightGBM **18 feature + chu kỳ**, fit trên cutoff 180–439. Cutoff 453–481 dùng fit calibration và 495–537 dùng lựa chọn calibration; quy tắc đã chốt giữ xác suất thô. Test đã từng được xem trong quá trình phát triển và chỉ là kết quả tham khảo POC. Xem [báo cáo](docs/BAO_CAO_POC_CRISP_DM.md) trước khi trích dẫn chỉ số.

`src/propensity_v2/prepared_data.py` xác thực hợp đồng dữ liệu đã chuẩn bị cho bốn notebook POC. Các script và tài liệu của pipeline V2 trước đây đã được loại khỏi luồng chính. Artifact POC hiện tại được giữ để đối chiếu và chạy demo.

## Giới hạn quan trọng

- `DAY` không có ánh xạ lịch thực đáng tin cậy. Hai ngày 278 và 643 bất thường; nguyên nhân chưa được xác nhận.
- Không tổng hợp `QUANTITY` xuyên department khi chưa xác minh đơn vị; demographic và marketing không nằm trong pipeline POC.
- Lift phản ánh xếp hạng mua được ghi nhận, không chứng minh giao dịch tăng thêm do liên hệ hay coupon.
- Không có căn cứ coi POC là production service hoặc dự báo cutoff 711 là kết quả đã được xác thực.
