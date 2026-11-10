# Notebook nghiên cứu bổ trợ

Luồng trình bày chính gồm bốn notebook ở thư mục `notebooks/`. Các notebook tại đây lưu lại quy trình và bằng chứng cho những lựa chọn trong POC; không phải tất cả đều cần chạy mỗi lần xem báo cáo.

- **Baseline và model ban đầu:** `baseline_propensity_recency`, `modeling_logistic_propensity`, `modeling_lightgbm_propensity`.
- **Feature và độ nhạy:** `thu_nghiem_feature_propensity`, `rolling_backtest_chu_ky_mua`, `rolling_trend_va_isotonic`, `hpo_temporal_propensity`.
- **Đánh giá và bối cảnh:** `danh_gia_mo_hinh_validation`, `danh_gia_gia_tri_kinh_doanh_poc`, `kiem_tra_de_xuat_nang_cap_crisp_dm`.
- **Causal data:** `audit_causal_context`, `phan_tich_quy_tac_causal`; chưa đưa vào pipeline POC.
- **Pipeline cuối:** `calibration_va_danh_gia_poc` tạo pipeline và kết quả đã chốt; không cần chạy lại nếu artifact bàn giao còn nguyên.

Chạy từ thư mục gốc, `notebooks/` hoặc `notebooks/research/`. Các notebook có thể ghi lại artifact nghiên cứu tương ứng. Test đã được xem trong dự án; không dùng các notebook này để trình bày test như một tập hoàn toàn độc lập.
