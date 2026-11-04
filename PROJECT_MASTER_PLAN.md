# PROJECT MASTER PLAN — CUSTOMER PURCHASE PROPENSITY (T-ECD FULL SUBSET)

**Phiên bản:** 1.0 — 04/11/2026  
**Phương pháp:** CRISP-DM, với quality gates và workflow Codex  
**Môi trường:** Windows, Python 3.12, Ryzen 7 5700U, RAM 16 GB, SSD 512 GB  
**Trạng thái:** Kế hoạch; chưa audit schema và chưa huấn luyện mô hình. Mọi thông số mẫu và KPI là giả thuyết/đề xuất cần kiểm chứng.

## A. Executive decision — những điều không được nhầm

**Bài toán**: Dự đoán `P(customer phát sinh ít nhất một sự kiện mua hàng hợp lệ trong Retail trong 7 ngày tiếp theo | thông tin có sẵn ở thời điểm dự đoán)`. Không dự đoán SKU, không bắt buộc khách hàng từng mua trước đó, không phải dự đoán mua lại. `prediction_unit = (user_id, snapshot_at)`.

**Giới hạn ý nghĩa**: Bản đầu chỉ đo purchase/order trong *Retail*. Không gọi đây là xác suất mua *bất kỳ sản phẩm trên toàn hệ sinh thái*, vì dữ liệu ở các domain không cùng nghĩa và Payments bao gồm giao dịch không phải mua hàng hóa. Cụm từ **completed purchase** chỉ được dùng nếu audit xác nhận event/status tương ứng; nếu không, định nghĩa trung thực là **recorded Retail order event**.

**Phương án dữ liệu**: T-ECD Full, khởi đầu `users.pq`, `retail/events`, tùy tính hữu ích thêm `marketplace/events`, `retail/items.pq`; hoãn `payments`, `marketplace/items.pq` và embeddings lớn. Ưu tiên 180–365 ngày có coverage hợp lệ; không cố giữ 365 ngày nếu Retail không đủ ngày. Thu thập từng partition ngày; giảm user bằng deterministic hash; giữ **toàn bộ event** của user đã chọn trong vùng thời gian.

**Sampling**: Bắt đầu 0,25%–0,5% *tất cả user catalog*, rồi thử 1%, 2% khi cần. Các tỷ lệ là vòng thử, **không** phải cam kết số người hoạt động hoặc số nhãn dương. Tạo tập mẫu lồng nhau bằng một hash/seed cố định; không tuyển người dựa trên future purchase. Định nghĩa người dùng đủ điều kiện tại snapshot bằng tương tác *quá khứ* trong 30 ngày (tùy audit). Không cân bằng nhãn trên validation/test.

**Tài nguyên**: Giữ RAM thực tế dưới khoảng 12 GB; DuckDB external processing trên SSD, streaming từng ngày; huấn luyện tabular bằng CPU. Mức mẫu dữ liệu giữ lại và bảng gold phải được quyết định qua pilot; 5–15 GB không phải giới hạn tải mạng có thể đảm bảo.

**Đánh giá**: Time-based backtesting, PR-AUC (so với prevalence baseline), ROC-AUC, Precision@K/Lift@K, Brier score/Log loss và calibration. Lập learning curve theo kích thước mẫu và feature ablation. Chốt final test chỉ một lần sau khi chọn cấu hình; không chọn model theo test.

**Giá trị với mentor**: chứng minh (1) nhãn đúng nghĩa, (2) mẫu có tính đại diện, (3) dữ liệu không rò rỉ tương lai, (4) lợi ích của hành vi so với baseline, (5) chất lượng xác suất, (6) tác động khi tăng cỡ mẫu, (7) hệ thống tái lập trên 16 GB.

## B. Nguồn chính thức và tính pháp lý

- Dataset card: https://huggingface.co/datasets/t-tech/T-ECD
- Cấu trúc file Full: https://huggingface.co/datasets/t-tech/T-ECD/tree/main/dataset/full
- Paper: https://doi.org/10.1145/3770855.3817588
- Baseline implementation: https://github.com/kliakhnovich/tecd-baselines
- CRISP-DM: https://www.ibm.com/docs/en/spss-modeler/saas?topic=dm-crisp-help-overview
- Codex project guide: https://developers.openai.com/cookbook/examples/codex/iterating-development-workflows-with-codex

T-ECD được công bố theo giấy phép **CC BY-NC-SA 4.0**: cần ghi nguồn, xem xét giới hạn mục đích thương mại và chia sẻ tương tự nếu phân phối dữ liệu phái sinh. Không commit Parquet gốc, dữ liệu mẫu hoặc token vào Git. Dữ liệu đã ẩn danh/biến đổi; không khẳng định là log giao dịch nguyên bản 100%. Bản Full gồm nhiều domain và tổng lịch sử 1300+ ngày nhưng **mỗi domain có chiều dài lịch sử khác nhau**. Tác giả yêu cầu ít nhất **12 giờ gap** giữa lịch sử và prediction; dự án chọn gap mặc định 24 giờ. Repository baselines dùng target cuối cửa sổ và **loại user không có target**, nên không được tái dùng nguyên logic lọc này cho binary propensity.

## C. Contract bất biến và các điểm phải audit

### C1. Dữ liệu ở dạng chuẩn hóa (tên cột bên dưới là schema nội bộ do ta thiết kế, KHÔNG mặc định là cột raw)

- `users`: `user_id`, `region`, các demographic cluster hoặc thuộc tính **thực sự hiện diện**.
- `events_normalized`: `user_id`, `domain`, `event_ts`, `event_day`, `event_type`, `item_id`, `brand_id`, `price`, `quantity`, `source_file` (field có thể null nếu domain không hỗ trợ).
- `product_catalog`: metadata/category/brand của Retail **nếu tồn tại và join được**.
- `snapshot_population`: `user_id`, `snapshot_at`, `eligible_reason`, `sample_bucket`.
- `labels`: `user_id`, `snapshot_at`, `target_7d`, `target_event_count`, `label_window_start`, `label_window_end`.
- `features`: `(user_id, snapshot_at)` + 7/30/90-day customer, behavior, aggregate product context features.
- `model_table`: labels LEFT JOIN features, bảo toàn user không mua.

### C2. Mốc thời gian định nghĩa chính xác

- `snapshot_at = T` (00:00 UTC hoặc timezone chuẩn hóa được audit và lưu trong config).
- Feature lookback 90 ngày: `[T-91 ngày, T-1 ngày)`. Tính 7/30/90 ngày bằng điểm kết thúc `T-1 ngày`.
- Leakage gap: `[T-1 ngày, T)`; không dùng event trong gap để tính feature hoặc xác định eligibility.
- Label horizon 7 ngày: `[T, T+7 ngày)`; `target=1` nếu có ít nhất một **Retail purchase/order event hợp lệ**.
- `target=0` chỉ khi tất cả partition Retail trong label window tồn tại, đầy đủ và có thể quan sát.
- Snapshot tạo mỗi 7 ngày, nhưng audit overlap nhãn và split phải dựa trên **khoảng event_ts thực**, không dựa vào thứ tự dòng.
- Để có `full 90d lookback + 1d gap + 7d target`, cần ít nhất 98 ngày dữ liệu liên tục; với 180 ngày lý thuyết có khoảng 12 weekly snapshots, với 365 ngày khoảng 39 snapshots (thực tế phụ thuộc ngày hợp lệ).
- Phải khóa nghĩa `order`, `purchase`, `transaction`, `checkout`, `status` trước khi dùng nhãn. Nếu không xác nhận completed, đổi tên bài toán thành **Retail order propensity** và mô tả hạn chế.

### C3. Tập khách hàng / population

- Primary cohort: user trong catalog và **có ít nhất một tương tác Retail hoặc Marketplace** trong `[T-31 ngày, T-1 ngày)`; định nghĩa này thể hiện *active-user purchase propensity*, không phải tất cả khách hàng hệ sinh thái.
- Secondary diagnostic cohorts: từng mua trước T / chưa từng mua trước T; hoạt động ít/vừa/nhiều; region; new vs established; segment missing metadata.
- Không dùng `tương tác xảy ra sau T` để xác định ai được đưa vào mẫu.
- Không sử dụng raw `user_id`, `future purchase count`, future popularity hay timestamp của label làm features.

## D. CRISP-DM PHASE 0 — Repository foundation & Codex governance

Đây là giai đoạn kỹ thuật chuẩn bị phục vụ sáu phase chính.

**Tasks**
1. Tạo Git repository, `.gitignore`, `.env.example`, `README.md`, `AGENTS.md`, `PROJECT_MASTER_PLAN.md`, `docs/decisions.md`.
2. Python 3.12 virtual environment. Ban đầu cài: `duckdb`, `pyarrow`, `huggingface_hub`, `pyyaml`, `psutil`, `pytest`, `ruff`; bổ sung `polars`, `pandas`, `scikit-learn`, `lightgbm`, `matplotlib`, `shap` sau khi pilot. Ghim dependencies bằng lockfile hoặc pinned requirements khi đã xác minh cài được trên máy.
3. Windows/PowerShell: `py -3.12 -m venv .venv`; `./.venv/Scripts/python.exe -m pip install ...`. Có thể chuyển sang `uv` sau, nhưng không coi là bắt buộc.
4. Tạo `config/base.yaml` giữ: HF repo/revision, domains, day range, user hash seed, sampling fraction, gap, lookbacks, horizon, snapshot cadence, maximum RAM/threads, test periods.
5. Chạy Codex theo từng issue/milestone; yêu cầu đọc `AGENTS.md` và `PROJECT_MASTER_PLAN.md`; kế hoạch trước khi code; unit tests, evidence, docs cập nhật sau mỗi milestone.
6. Thiết lập logger, seed, run ID, manifest outputs, disk free checks, file checksum, dry-run.

**Deliverables:** repository scaffold; working environment; config; README; sample-run smoke test; CI chạy unit tests không cần tải dữ liệu Full.

**Gate 0:** bootstrap thành công trên laptop; `pytest` qua; `.gitignore` không cho dữ liệu/token lọt Git; Codex chạy được read-only inventory trước khi sửa.

## E. PHASE 1 — Business Understanding

### E1. Business question

“Trong số các khách hàng hoạt động, ai có khả năng phát sinh ít nhất một đơn hàng Retail trong 7 ngày tới, để hỗ trợ ưu tiên chăm sóc/đề xuất theo propensity score?” Không khẳng định rằng ưu tiên tiếp thị sẽ **tăng** số đơn; propensity là dự đoán hành vi tự nhiên, không phải uplift/causal impact.

### E2. Outputs và stakeholders

- Data science: calibrated `purchase_proba` (0..1), score timestamp, model version.
- Business: bảng rank theo xác suất, top decile, tỷ lệ mua thực tế theo decile, số khách hàng phát sinh đơn.
- Mentor: business brief, mathematical target, data limitations, baseline comparison, risk register.

### E3. KPI / acceptance proposal (không đặt mục tiêu phi thực tế trước khi biết base rate)

- **Must-have**: target rõ và có đủ nhãn âm/dương; không leakage; PR-AUC cao hơn prevalence baseline; Brier/Log loss có cải thiện so với constant probability; Lift@10% > 1 trên validation và final test.
- **Stretch**: cải thiện nhất quán qua 2–3 validation windows, stability trước thay đổi cỡ mẫu, calibration tốt, nhóm demographic/region không có suy giảm bất thường.
- Không hứa AUC ≥ 0.9; dự báo kết quả trước audit là vô căn cứ.

### E4. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| `order` không chứng minh đã thanh toán/hoàn tất | Target sai nghĩa | Kiểm tra event semantics; đổi tên nếu cần |
| Retail coverage < 180/365 ngày | Thiếu snapshot | Chọn 90/180 ngày, hoặc Retail-only baseline |
| Ít nhãn dương | Metric bất ổn | Mở rộng tỷ lệ user/horizon, báo cáo confidence interval |
| Sample bias/cold-start | Đánh giá lạc quan | Hash user độc lập target, audit strata, cohort metrics |
| Label thiếu vì missing partition | Negative sai | Coverage check trước khi gán 0 |
| Timestamp perturbation | Leakage | Gap 24h + point-in-time tests |
| Memory/network/SSD | Không xử lý được | Per-file processing, spill to disk, cache cleanup có kiểm tra |
| Data licensing | Không được phân phối | Attribution, không commit raw, ghi giấy phép |

**Deliverable:** `docs/01_business_understanding.md`, `docs/target_contract.md`, `docs/risk_register.md`.

**Gate 1:** Mentor có thể đọc một trang và biết chính xác population, đơn vị dự đoán, target, horizon, giá trị sử dụng, hạn chế. Chưa tuyên bố “completed purchase” khi chưa xác thực.

## F. PHASE 2 — Data Understanding

### F1. Audit metadata trước khi tải nặng

1. Dùng Hugging Face API liệt kê **đúng path** `dataset/full/users.pq`, `dataset/full/retail/events/{day}.pq`, `dataset/full/marketplace/events/{day}.pq`, `dataset/full/retail/items.pq`; lưu `revision`, `path`, `day`, `size`, `sha` nếu API cung cấp.
2. Đếm min/max/continuity của ngày theo domain, ngày giao nhau, missing gaps, tổng bytes cần download cho mỗi kịch bản 90/180/270/365 ngày.
3. Tạo inventory JSON/CSV và biểu đồ coverage. Không suy luận ngày data từ số thứ tự file nếu chưa so với timestamps.
4. Đầu tiên chỉ lấy `users.pq` (109 MB), 3–7 ngày Retail files và, nếu khả thi, 3–7 ngày Marketplace files; mở rộng lên pilot 14 ngày nếu kiểm chứng tốt.
5. Không dùng `snapshot_download` với pattern chung `dataset/full/`, vì sẽ kéo cả repository hàng TB.

### F2. Schema profiling và event semantics

- Đọc schema thực từng file `Parquet`: dtypes, null fraction, min/max, categorical values của `action_type`/`subdomain`.
- Lấy mẫu vài người để xem chuỗi event và phân biệt `view`, `cart`, `order`, `transaction`, `purchase` nếu có.
- Xác định Purchase Event Contract: domain, raw action value, trạng thái, quy tắc duplicate/dedup (nếu có), độ trễ, ngày có nhãn hợp lệ.
- Kiểm tra ý nghĩa `price`/`count` (có phải tiền thật/đơn vị thật hay đã perturb), logic event/item/brand IDs, format `user_id`, timezone/timestamp, giá trị âm/zero bất thường.
- Kiểm tra event-item join Retail items, tỷ lệ unmatched. Kiểm tra metadata users (`region`, clusters) có coverage đủ và tỷ lệ missing.
- Ghi rõ `Marketplace` có dùng cho **features**, không tự động cho **target**.

### F3. EDA phải có

- Event count theo ngày/domain/action; active users theo ngày; user activity histogram/log scale; purchase rate theo tuần.
- Distinct users/items/brands; events/user; recency/cart/price distributions.
- Sequential paths: view→cart→order, xác định **chỉ để mô tả**, không bắt buộc chain hoàn chỉnh.
- Region distribution, missingness, invalid values; product category coverage nếu có.
- Cohort quan tâm: never-purchased-before, existing buyers, new active vs returning active.
- Coverage + user overlap giữa Retail và Marketplace.
- Phân biệt observed event data, unavailable future data, missing partition và thực sự không có giao dịch.

**Deliverables:** `data/manifests/source_inventory.parquet`, `docs/02_data_understanding.md`, `docs/data_dictionary.md`, `docs/purchase_event_contract.md`, `reports/figures/coverage.png`, profiling metrics.

**Gate 2 — GO/NO-GO:** Có purchase proxy hợp lệ, đủ khoảng ngày liên tục và đủ nhãn; xác nhận rõ cột event/ts/id. Nếu không: ghi quyết định chuyển sang `recorded_order_propensity`, thay đổi horizon, giảm target domain hoặc thay đổi dataset; không viết nhãn bừa.

## G. PHASE 3 — Data Preparation (trọng tâm dự án)

### G1. Chọn temporal windows một cách có kiểm chứng

1. Ưu tiên tối đa 365 ngày **thực sự có Retail labels liên tục**; chọn đoạn có event đủ phổ biến, tránh cửa sổ ngày bị missing.
2. Marketplace có thể có ít ngày hơn Retail: có 2 chế độ (a) Retail-only main model trên full valid Retail range; (b) Retail+Marketplace model trên overlap range. Khi so sánh hai feature sets, dùng **cùng snapshots/cohort** để tránh attribution sai.
3. Pilot 14 ngày chỉ phục vụ xác thực, chưa đủ cho model 90d history + 7d horizon.
4. Chọn tần suất snapshot 7 ngày, lookback 7/30/90, gap 24h, label 7d. Chốt time origin, handling end boundary theo nửa mở `[start,end)`.

### G2. Deterministic user sampling

1. Mẫu gốc chọn từ `users.pq` bằng `stable_hash(user_id + seed)` với threshold, không dùng `ORDER BY RANDOM()` trên event.
2. Tạo nested samples 0,25%, 0,5%, 1%, 2%; **giữ nguyên seed** và các cohort lồng nhau; không đụng đến target để quyết định inclusion.
3. Với user đã chọn, giữ **tất cả các event** trong days và domains được chọn. Không chỉ lấy 10% event và coi frequency là thật.
4. Khi xử lý ngày: quét `Parquet` bằng DuckDB/Polars lazy → thu reference daily aggregates (nếu khả thi) → join với user sample → ghi `data/silver/events/...parquet` → verify count/schema/size → dọn raw tạm khi an toàn.
5. `sample_bucket` phải ổn định giữa runs, vẽ retention/count by cohort; đặt user partition `experiment_cohort` nếu muốn test generalization to unseen users.
6. Báo cáo network bytes scanned/downloaded khác hẳn retained Parquet bytes. Không hứa hash 1% sẽ cắt download xuống 1%.

### G3. Sampling quality / representativeness

- So sánh phân phối **ở cấp khách hàng** mẫu với users catalog (`region`, cluster, nếu có).
- Đối chiếu events/day, activity buckets, purchase prevalence, shopping frequency, rate of missing info với dữ liệu tham chiếu từ stream/toàn bộ ngày đã scan.
- Dùng absolute standardized mean differences với biến numeric; tỷ lệ phân khúc và sai khác phần trăm điểm cho categorical. Nếu dùng PSI, ghi rõ cách bin/cutoff, không coi PSI=0.1 là định luật.
- Tính uncertainty bằng bootstrap theo user, không bootstrap độc lập từng snapshot vì một user lặp lại nhiều lần.
- Nếu rare strata quá nhỏ, giữ random main sample và tạo exploratory **stratified diagnostic sample** riêng; nếu thay sampling theo strata thì phải ghi inclusion probabilities và weighting, không lấy test đã oversample làm bằng chứng prevalence thực.
- Learning curve dùng cùng validation set, không chọn fraction dựa trên final test.

### G4. Tạo population và nhãn

1. Tạo weekly snapshots có toàn bộ feature/label horizon và Retail coverage hợp lệ.
2. Chọn eligible user có hoạt động **trước T-1 ngày** trong lookback 30 ngày (hoặc 90 ngày nếu baseline population quá nhỏ; phải document).
3. Sử dụng LEFT JOIN từ eligible snapshots sang future Retail purchase events; không lọc bỏ không có purchase.
4. Label `1` nếu ít nhất 1 event thỏa contract trong `[T,T+7d)`; `0` nếu không và quan sát đầy đủ; nếu thiếu coverage đánh dấu `unknown` và loại khỏi training/evaluation với lý do.
5. Một user có thể có nhiều dòng snapshot. Nhớ rằng các dòng không độc lập thống kê; khi CI bootstrap theo user.
6. Nếu `action_type` không chứng minh trạng thái hoàn tất, dùng `retail_order_next_7d` thay vì `completed_purchase_next_7d`.
7. Lưu bảng label và metadata về source version, raw action mapping, valid target window.

### G5. Feature engineering — roadmap

**Level A: Basic user + activity**
- `region`, `demographic_cluster` nếu có, missing indicator.
- `active_days_7/30/90d`, `events_7/30/90d`, `days_since_last_event`, `distinct_sessions` nếu raw có session ID.
- `views_7/30/90d`, `clicks_7/30/90d`, `carts_7/30/90d`, `orders_30/90d` (chỉ events trước feature cutoff).

**Level B: RFM + purchase intent**
- `days_since_last_order`, `order_count_30/90d`, `spent_on_orders_30/90d` nếu số tiền có nghĩa.
- `cart_view_ratio_30d` với smoothing; `cart_to_order_ratio_past_90d`, `last_cart_recency`.
- `cart_count_7d / cart_count_30d`, `views_7d / views_30d`, hành vi tăng/giảm.
- `days_since_first_seen`, `weeks_active_90d`, `visit_days_weekend_ratio` nếu temporal jitter cho phép ở độ phân giải này.

**Level C: Aggregated product context**
- `unique_items_viewed_30d`, `unique_brands_interacted_30d`, `unique_categories_30d` nếu catalog có category.
- `avg/min/max/std_viewed_item_price`, `avg_cart_item_price`, `price_std` (ưu tiên robust stats), `expensive_items_share` so với mức tương tác của chính user.
- `top_category_share`, `brand_concentration`, `category_entropy`, `repeated_view_same_item_count`.
- `user_recent_items_popularity` chỉ dùng popularity **đã biết trước T-1d**; không tính popularity bằng toàn bộ dataset.

**Level D: Optional cross-domain**
- Marketplace clicks/views, exposure diversity; offers engagement nếu coverage đủ.
- Payments chỉ mở nếu chứng minh uplift rõ và chi phí phù hợp; không nhầm `payment` với Retail order.

**Chống leakage ở mọi feature**: data cutoff cố định, point-in-time join, tính popularity/history theo ngày, không fit imputers/encoders trên future/test, không đưa `user_id` trực tiếp làm predictor, không dùng thành phần được tính từ toàn bộ lịch sử bao gồm tương lai. Price/quantity có thể bị nhiễu do privacy; chỉ tạo biến nếu audit hợp lý.

### G6. Train/validation/test và Gold table

- Time-based split theo **snapshot_at**, không random split rows.
- Chọn train sớm hơn, validation sau đó và test cuối cùng; bảo đảm `label_window_end` train không lấn qua validation/test prediction window (purged split).
- Ví dụ **chỉ minh họa**: 65% weekly snapshot dates đầu cho train; 15% tiếp theo cho validation; 20% cuối cho test, điều chỉnh khi biết số dates và positive events.
- Giữ một tập **calibration** riêng sau training và trước test, hoặc tách validation thành tuning/calibration; không calibrate trên final test.
- Lưu `gold_features.parquet`, `gold_labels.parquet`, split manifest, data card, feature lineage và distribution.
- Cần kiểm thử cả trường hợp user vừa tham gia, không có order lịch sử, activity rất thấp.

**Deliverables:** `docs/03_data_preparation.md`, `docs/sampling_design.md`, `docs/sampling_quality_report.md`, `docs/feature_dictionary.md`, `data/gold/*.parquet`, quality tests và split manifests.

**Gate 3:** Không missing labels ngụy tạo; `target` có cả 0,1; time-bound tests pass; sample reproducible; no future fields; feature table vừa RAM và có reasonable positives.

## H. PHASE 4 — Modeling

### H1. Models theo thứ tự

1. **No-skill baseline**: constant purchase prevalence dự đoán trên training window; làm mốc Brier/Log loss, PR-AUC.
2. **Heuristic**: score dựa trên `days_since_last_cart`, activity hoặc RFM; giúp giải thích lợi ích với business.
3. **Logistic Regression**: xử lý scale/one-hot trên train; kiểm tra interpretability.
4. **LightGBM CPU**: mô hình chính, dữ liệu dạng tabular, kiểm soát overfit.
5. **CatBoost/XGBoost**: chỉ là challenger nếu tài nguyên đủ; không cần deep learning để dự án thuyết phục.

### H2. Training design

- Chạy pilot 50K–200K rows rồi 200K–500K, mở rộng 0,5–2M nếu ổn về RAM/positive count.
- Chọn feature types numeric/categorical, missing handling, high-cardinality guard. Dùng `float32`, `int32` thích hợp, categorical encoding nhất quán.
- Giới hạn CPU threads 4–6, phân bổ DuckDB memory 4–6 GB và temp spill riêng, theo dõi peak RAM/disk. Mục tiêu process không vượt 12 GB.
- Random seeds cố định; lưu file list, commit SHA, config hash, library versions, model parameters, wall clock, peak memory, sample share.
- Tuning vừa phải (20–40 trials hoặc grid nhỏ), tối ưu validation PR-AUC/Lift và kiểm tra Log loss; early stopping đúng dữ liệu thời gian.
- Nếu downsample class 0 chỉ trong train, lưu sampling fraction/weights, calibrate lại trên set tự nhiên; baseline mặc định **không downsample target**.

### H3. Calibration

- Score thực chất là xác suất; dùng reliability diagram, Brier, log loss, ECE (với số bins/weight được ghi rõ).
- Thử sigmoid/Platt hoặc isotonic dựa trên **calibration window riêng**, chỉ giữ nếu test/backtest chứng minh có lợi.
- Không mặc định threshold 0.5; percentile rank/top-K thường hữu dụng hơn cho targeting với ngân sách giới hạn.

### H4. Model registry nhẹ

- Lưu `model.pkl`/native LightGBM, `preprocessor`, `feature_schema.json`, `calibrator`, `training_manifest.json`, `metrics.json`.
- `predict` không nhận các cột chưa có khi scoring, và từ chối schema sai.

**Deliverables:** `docs/04_modeling.md`, ML training scripts, baselines, tuned model, experiment registry, calibration curves.

**Gate 4:** Baseline reproducible; LightGBM training trên laptop không crash; metrics trên validation có cải thiện thực; predictions 0..1; không leakage; model/pipeline có thể reload.

## I. PHASE 5 — Evaluation (trọng tâm bảo vệ trước mentor)

### I1. Scorecard và baseline

- PR-AUC + **prevalence (random-rank PR baseline)**; ROC-AUC.
- `Precision@5%`, `Precision@10%`, `Recall@10%`, `Lift@5%`, `Lift@10%`, gain chart.
- Calibration curve, Brier score, log loss. Báo cáo prevalence từng period.
- Confidence intervals bằng bootstrap cluster theo user hoặc temporal block; không đánh lừa bằng hàng triệu repeated snapshots như các IID records.
- Reporting segment: recent/non-buyers, repeat buyers, low/medium/high activity, region, new/established users; ghi số mẫu từng segment và bất định.
- Error analysis false positives/false negatives, distribution shift, ranking stability.

### I2. Experiments bắt buộc

- **Ablation 1:** Profile + basic RFM.
- **Ablation 2:** A + browsing/cart.
- **Ablation 3:** B + aggregated product category/brand/price.
- **Optional Ablation 4:** C + Marketplace features (chỉ trên cohort, dates giống nhau).
- **Sample learning curve:** model cùng pipeline trên nested cohorts 0,25% → 0,5% → 1% → 2%; cùng validation timeline, đếm positive count, confidence intervals, CPU/RAM cost.
- **Temporal backtest:** tối thiểu 2–3 forward validation windows nếu dữ liệu cho phép. Final test khóa và chạy **sau khi** chọn sampling/features/hyperparams.
- **Horizon sensitivity:** so 7 vs 14 ngày hoặc 30 ngày nếu còn dư nguồn dữ liệu; đây là bài toán thay đổi chứ không được chọn chỉ vì metric đẹp.
- **Population sensitivity:** chỉ người đã mua vs chưa từng mua; kiểm tra cold-start, không loại nhóm khó khi báo cáo KPI chung.
- **Leakage tests:** deliberate future-shift sanity test, timestamp assertions, no full-period popularity, label no-overlap, train-fitted preprocessors.
- **Proxy check:** nếu label thực chất là `order`, thảo luận vì sao không đại diện tuyệt đối cho completed paid purchase.

### I3. Business translation

- Nếu target rate nền = q (đo từ test), so `Top 10% observed purchase rate` với q và `Lift@10%`. Không tự đặt mức ROI; không có cost/revenue data đầy đủ thì mô tả **ranking effectiveness**.
- Propensity không đo incremental effect của gửi voucher/notification. Nếu muốn uplift cần thêm dữ liệu treatment/control.

**Deliverables:** `docs/05_evaluation.md`, scorecard CSV/Markdown, lift & calibration plots, ablation table, learning curve, segment evaluation, `model_card.md`.

**Gate 5:** Một final test khóa sẵn, mọi quyết định tuning đã được ghi trước; có khoảng tin cậy, calibration và so sánh baseline; kết luận giới hạn và negative results minh bạch.

## J. PHASE 6 — Deployment (batch demo, không ép production cloud)

### J1. Inference pipeline

- CLI chạy theo `as_of_date`: lấy thông tin có trước cutoff, validate schema, sinh đúng một dòng `user_id, snapshot_at, propensity_score, model_version` cho mỗi eligible user.
- `batch_predict` chạy cho 1 nhóm user hoặc synthetic fixture, xuất Parquet/CSV nhỏ; không cần online API ngay.
- Đảm bảo feature computation của inference dùng cùng function/config đã dùng training, và không cần dữ liệu tương lai.
- Nếu UI cần: Streamlit demo nhập user ID ẩn danh và snapshot ngày hợp lệ, xem score/nhóm feature quan trọng; không dự đoán trên ngày không có đủ lịch sử/nhãn đánh giá.

### J2. Model monitoring proposal

- Data quality: day partitions missing, null/change schema, active user count, event volume.
- Model scoring: propensity distribution, segment coverage, rate top decile, feature drift.
- Khi có labels trễ: PR-AUC, Brier, calibration drift và purchase prevalence theo tuần.
- Trigger retraining chỉ là **đề xuất**; không có dữ liệu live nên không giả vờ đã chạy production monitoring.

### J3. Final research package

- `README.md` quickstart, environment, data download manifest and license, reproducible commands.
- `project_report.pdf` hoặc `report.md` + slide deck nếu mentor yêu cầu; CRISP-DM decision log, diagrams, methods, experiments, result tables và limitations.
- `model_card.md`, `data_card.md`, `reproducibility.md`, sampling quality report, screenshots/demo.
- Re-run on a small fixture từ clean venv, verify model outputs and no hardcoded local paths.

**Gate 6:** Một người khác có thể clone repo, chạy tests, đọc documentation, tải dữ liệu được phép và tái hiện pilot/baseline từ config mà không cần sửa source code thủ công.

## K. Cấu trúc repository khuyến nghị

```text
tecd-customer-purchase-propensity/
├── AGENTS.md
├── PROJECT_MASTER_PLAN.md
├── CODEX_START_PROMPT.md
├── README.md
├── pyproject.toml              # hoặc requirements.txt có pinned versions
├── .env.example
├── .gitignore
├── configs/
│   ├── base.yaml
│   ├── pilot.yaml
│   ├── baseline.yaml
│   └── final.yaml
├── docs/
│   ├── 01_business_understanding.md
│   ├── 02_data_understanding.md
│   ├── 03_data_preparation.md
│   ├── 04_modeling.md
│   ├── 05_evaluation.md
│   ├── 06_deployment.md
│   ├── target_contract.md
│   ├── data_dictionary.md
│   ├── sampling_design.md
│   ├── sampling_quality_report.md
│   ├── feature_dictionary.md
│   ├── risk_register.md
│   └── decisions.md
├── data/                        # .gitignore toàn bộ artifacts lớn
│   ├── raw_tmp/
│   ├── manifests/
│   ├── silver/
│   └── gold/
├── src/tecd_propensity/
│   ├── audit.py
│   ├── acquisition.py
│   ├── schemas.py
│   ├── sampling.py
│   ├── events.py
│   ├── labels.py
│   ├── snapshots.py
│   ├── features.py
│   ├── splits.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── quality.py
│   └── config.py
├── tests/
│   ├── fixtures/             # synthetic only
│   ├── test_sampling.py
│   ├── test_labels.py
│   ├── test_temporal_leakage.py
│   ├── test_features.py
│   ├── test_splits.py
│   └── test_inference.py
├── notebooks/                 # exploration only, source logic trong src/
├── experiments/
├── models/                     # ignore binary weights unless small release
├── reports/
│   ├── figures/
│   └── tables/
└── harness/
    ├── milestones/
    └── build-log.md
```

## L. Milestone board — hành động theo thứ tự

| ID | CRISP-DM | Milestone | Evidence required | Gate |
|---|---|---|---|---|
| M00 | Setup | Repo + Python + AGENTS.md + test harness | env smoke test, gitignore check | pass |
| M01 | Business | Problem, eligible cohort, target contract draft | business brief, risks, decisions | approve |
| M02 | Data Understanding | Metadata inventory, coverage, download budget | source inventory & plot | pass |
| M03 | Data Understanding | Pilot schema/event type audit | dictionary, raw action mapping, purchase evidence | GO/NO-GO |
| M04 | Prep | Stable user cohort sampling + unit tests | sample manifests, reproducibility proof | pass |
| M05 | Prep | Day-by-day ETL to sampled Parquet | row counts, checksums, memory log | pass |
| M06 | Prep | Reference stats & sample quality | distributions, rare segments, drift measures | approve |
| M07 | Prep | Snapshot+label generator | label table, coverage tests, no leakage | GO/NO-GO |
| M08 | Prep | Feature library + temporal split | feature dictionary, tests, gold table | pass |
| M09 | Model | Baselines + LightGBM | experiments, validation metrics | pass |
| M10 | Model | Tuning + calibration | best config, calibration report | pass |
| M11 | Evaluation | Ablation, sampling curve, backtest | plots, tables, CIs | approve |
| M12 | Evaluation | Locked final test & model card | fixed test scorecard, error analysis | pass |
| M13 | Deployment | Batch inference + demo | reproducible predictions and smoke tests | pass |
| M14 | Report | CRISP-DM report & mentor presentation | reproducibility package and limitations | complete |

**Scope discipline**: Mỗi Codex turn nên làm đúng **một** milestone hoặc một vertical slice có thể test. Trước khi sửa: Codex đọc tài liệu, liệt kê files sẽ thay đổi, assumptions, validation. Sau khi sửa: chạy tests, xuất evidence, cập nhật docs/build-log và dừng nếu gate chưa đạt.

## M. Practical hardware & resource policy

- `threads=4` khởi điểm cho DuckDB và LightGBM; tăng nếu benchmark thấy ổn.
- DuckDB memory limit thử 4–6 GB, đặt `temp_directory` có ít nhất 20–30 GB free space trong thời gian ETL; điều chỉnh theo peak thực.
- Không `pandas.read_parquet()` hàng chục GB thành DataFrame; xử lý daily parquet, projection pushdown, `COPY TO PARQUET` và lưu ít cột hữu dụng.
- Tải **một partition ngày một lần**, kiểm tra trước khi xóa raw temp; Hugging Face cache có thể tăng dung lượng đĩa đáng kể, đo và quản lý rõ.
- Luôn đo download bytes, output bytes, processing time, peak RAM, temporary disk, #event/#user/#purchase và cost per million rows.
- Có resume manifest: không tải lại ngày đã thành công với cùng source revision/hash, không ghi đè file tốt nếu process lỗi.
- Không chạy concurrent downloads 10 workers trên SSD cá nhân ngay từ đầu; bắt đầu 1–2, benchmark.
- Nếu pip gặp 403/MediaTypeBlocked, dừng cài và kiểm tra môi trường network/proxy/index; không tự suy ra lỗi do phiên bản scipy.

## N. Data quality & test contracts (Codex phải viết các kiểm thử này)

1. Stable hash: cùng user/seed luôn cùng bucket; nested 0,25% ⊂ 0,5% ⊂ 1%.
2. Nhãn: event trước T không tạo label dương; purchase sau `T+7d` không tạo label dương; không mua nhưng đầy đủ coverage => `0`; thiếu ngày => `unknown`.
3. Leakage: feature event phải có `event_ts < T-1d`; mọi tập train/val/test phải tuân điểm thời gian; không join future category popularity.
4. Completeness: tất cả partition label window tồn tại; schema/raw mappings phù hợp; số dòng trước/sau ETL được giải thích.
5. Identity: `(user_id,snapshot_at)` duy nhất trong gold; không có duplicate row sau joins; event duplicates xử lý theo hợp đồng.
6. Sampling: tần suất hoạt động của selected user được giữ nguyên trong window; no event-level downsampling.
7. Fit/transform: encoder/imputer fit trên train; validation/test chỉ transform.
8. Calibration/inference: proba thuộc [0,1]; model reload tạo predictions bằng nhau trong tolerance; feature schema check.
9. Negative class: không bị xóa bởi inner join hoặc baseline recommendation original.
10. Metadata: mọi báo cáo có thời gian, dữ liệu nguồn, revision, fraction, seed, config và model version.

## O. Diễn giải kết quả và câu hỏi mentor có thể hỏi

- **Vì sao 7 ngày?** Chốt như horizon nghiệp vụ khởi đầu, kiểm tra 14/30 ngày như sensitivity, không chọn 7 ngày vì score đẹp.
- **Vì sao lấy mẫu user thay vì event?** Để không phá hủy frequency, sequence, conversion behavior.
- **Mẫu có đại diện không?** Trình bày bảng phân phối users, cohort, event và tỷ lệ mua so với reference; báo cáo độ bất định.
- **Vì sao không dùng toàn bộ Payments?** Domain khác nghĩa, nặng, và không cần thiết cho first stage; chứng minh marginal value trước khi mở rộng.
- **Vì sao bạn có thể cho xác suất 0.7?** Đánh giá calibration và Brier, không chỉ ROC-AUC; nếu miscalibrated thì phải hiệu chỉnh.
- **Bạn biết model tốt hơn baseline?** So sánh prevalence model, heuristic, Logistic Regression, LightGBM trên cùng temporal validation.
- **Bạn tránh leakage ra sao?** Time cutoff + gap + unit tests + purged splits + feature lineage.
- **Tính tổng quát đến đâu?** Chỉ suy luận chắc cho sampled active Retail/Marketplace users trong phân phối dữ liệu T-ECD; không khẳng định áp dụng mọi người dùng/dataset thật bên ngoài.
- **Dữ liệu T-ECD khác log gốc?** Ẩn danh, timestamp/numerical perturbations và event edits; report limitations và citation paper.
- **Propensity có đảm bảo marketing ROI?** Không. Dự đoán purchase tự nhiên, không ước lượng tác động nguyên nhân của marketing.

## P. Cách sử dụng Codex cho mỗi task

**Template yêu cầu cho từng milestone:**

> Hãy đọc AGENTS.md, PROJECT_MASTER_PLAN.md và docs/decisions.md. Thực hiện CHỈ milestone Mxx. Đầu tiên trình bày current state, assumptions, file paths thay đổi, tiêu chí nghiệm thu, cách test và rủi ro. Không tự đoán schema T-ECD, không tạo giả data evidence. Triển khai source code dưới src/, test với synthetic fixtures, ghi output documentation và build-log. Nếu cần tải dữ liệu/network/ghi đè/xóa file lớn, hãy hỏi quyền trước. Kết thúc bằng test commands, results thực tế, files changed, unresolved blockers và next gate; không tự động chuyển milestone tiếp theo.

**Khi gate không đạt:** quay lại business/data understanding để sửa giả định; không chạy tiếp tới model và chọn thuật toán để che lỗi dữ liệu.

## Q. Definition of Done — toàn dự án

- Business objective/target/eligible population và caveats rõ ràng.
- Có data lineage và source revision; dữ liệu sampled tái tạo được bằng seed + config.
- Có sampled distribution audit và learning curve chứng minh sự hợp lý của quy mô.
- Label chính xác theo định nghĩa, negative giữ nguyên, no leakage (unit tests pass).
- Có feature dictionary và at least 3 ablations.
- Baselines + calibrated LightGBM, test locked và backtest; uncertainty/segment evaluation.
- Batch inference chạy từ model đã lưu; reproducibility docs/small synthetic fixtures.
- Mentor report theo đủ sáu pha CRISP-DM; source citations + license + limitations; toàn bộ code không phụ thuộc đường dẫn cá nhân cứng.

## R. First 5 actions — bắt đầu ngay

1. Đặt `AGENTS.md`, `PROJECT_MASTER_PLAN.md`, `CODEX_START_PROMPT.md` vào repo; tạo venv; không tải dữ liệu Full.
2. Giao Codex M00 + M01: repo setup, business brief, target contract **draft**.
3. Giao Codex M02: chỉ gọi HF metadata để inventory Retail/Marketplace days và budget cho 180/365 ngày.
4. Giao Codex M03: tải vài files nhỏ thực tế, kiểm tra schema/action types/label semantics; báo cáo GO/NO-GO.
5. Chỉ sau khi chốt gate, giao Codex M04/M05: stable cohort hash và streamed per-day ETL. Chưa huấn luyện model trước khi có reliable labels.

---

**Một câu mô tả project cuối cùng:** *Developing a reproducible customer purchase propensity pipeline from T-ECD's anonymized cross-domain events, using representative user-level sampling, leakage-safe temporal features, calibrated classification, and rigorous sampling-quality evaluation under limited local compute.*
