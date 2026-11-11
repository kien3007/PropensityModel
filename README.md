# Household Purchase Propensity — Dunnhumby

Dự án theo CRISP-DM nhằm dự báo xác suất một hộ có **ít nhất một giỏ hàng được ghi nhận trong 7 ngày sau mốc chốt**, dựa trên lịch sử giao dịch. Hiện phần trình bày tập trung vào tìm hiểu và khám phá dữ liệu; mô hình và chạy thử dự báo được giữ để tiếp tục phát triển. Dự án chưa đo tác động của phiếu giảm giá hay doanh thu tăng thêm.

## Bắt đầu từ đây

1. [Slide phân tích dữ liệu (PDF)](presentations/dunnhumby_data_overview/main.pdf) và [nguồn LaTeX](presentations/dunnhumby_data_overview/main.tex): nội dung trình bày đến hết EDA, chưa đi vào kết quả mô hình.
2. [Notebook phân tích dữ liệu](notebooks/phan_tich_du_lieu_propensity.ipynb): số liệu, biểu đồ và cách diễn giải.
3. [Từ điển dữ liệu](docs/DUNNHUMBY_DATA_DICTIONARY.xlsx).
4. [Hướng dẫn trình bày EDA và giải thích biến đầu vào](docs/HUONG_DAN_BAO_CAO_EDA_VA_FEATURE.md).

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

Ứng viên thử nghiệm là LightGBM **18 biến giao dịch + biến chu kỳ**, học trên các mốc 180–439. Các mốc 453–481 dùng thử hiệu chỉnh xác suất, 495–537 dùng lựa chọn cách hiệu chỉnh; quy tắc đã chốt giữ xác suất gốc. Tập kiểm tra cuối đã từng được xem trong quá trình phát triển, nên kết quả chỉ mang tính tham khảo. Xem [notebook mô hình và đánh giá](notebooks/modeling_va_danh_gia_propensity.ipynb) trước khi trích dẫn chỉ số.

`src/propensity_v2/prepared_data.py` xác thực hợp đồng dữ liệu đã chuẩn bị cho bốn notebook POC. Các script và tài liệu của pipeline V2 trước đây đã được loại khỏi luồng chính. Artifact POC hiện tại được giữ để đối chiếu và chạy demo.

## Giới hạn quan trọng

- `DAY` không có ánh xạ lịch thực đáng tin cậy. Hai ngày 278 và 643 bất thường; nguyên nhân chưa được xác nhận.
- Không tổng hợp `QUANTITY` xuyên department khi chưa xác minh đơn vị; demographic và marketing không nằm trong pipeline POC.
- Lift phản ánh xếp hạng mua được ghi nhận, không chứng minh giao dịch tăng thêm do liên hệ hay coupon.
- Không có căn cứ coi POC là production service hoặc dự báo cutoff 711 là kết quả đã được xác thực.
