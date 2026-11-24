# Kế hoạch hoàn thiện PropensityVSF

> Ngày lập: 2026-11-23 · Trạng thái: **Phase 0–7 đã hoàn thành** (2026-11-23)

## Bối cảnh

Đánh giá hiện trạng (HEAD `ea0e615`):

- Khung kiến trúc Lambda đã đủ các lớp. 34/34 test pass.
- **Champion không tái lập được trong repo:**
  - Model được train ở project ngoài (`projectphase3`), dùng 28 feature, horizon 14 ngày.
  - [train_pipeline.py](../src/pipelines/train_pipeline.py) lại train 23 feature cũ, horizon 7 ngày.
- **Governance gate không thể đạt:** `min_lift_10 = 1.50`, trong khi với base rate 0.70 thì lift tối đa chỉ ≈ 1.43.
- **README và code không khớp:** README ghi Isotonic, nhưng code chỉ có Platt.
- **Hai nguồn dữ liệu song song:** `extract_multidomain_features` ([pit_extractor.py](../src/features/pit_extractor.py)) đọc thẳng CSV, không đi qua Lakehouse.
- **Bài toán gần như hiển nhiên:** base rate 70%, decile 1 có tỷ lệ mua 99%.
- **Mất phần EDA / CRISP-DM:** bị xoá ở commit `14d0407`; slide bị xoá ở `120ade9`.
- **Serving còn thiếu:**
  - API không có xác thực.
  - Luật fusion là con số cố định.
  - Hộ không có trong cache bị gán mặc định 0.5 mà không báo.
  - Chưa có CI.

## Nguyên tắc

- Làm theo thứ tự: tính đúng của ML → khả năng tái lập → vận hành → bảo mật và production.
- Mỗi phase kết thúc khi test pass và có một tiêu chí nghiệm thu đo được.

## Tiến độ

> **Quyết định 2026-11-23:** chỉ dùng LightGBM. Pipeline và báo cáo không còn train hay so sánh với các baseline (logistic regression, luật heuristic, chọn ngẫu nhiên). Các con số baseline trong ghi chú Phase 1–3 bên dưới chỉ là số liệu lịch sử. Gate vẫn giữ phần so sánh model mới với champion đang chạy (cùng là LightGBM) trước khi promote.

### Bổ sung: Feast làm kho lịch sử feature (hướng B)
- **Trước đây** Feast không được pipeline nào gọi tới. Mỗi lần materialize lại ghi đè snapshot, nên không có lịch sử. Kiểu dữ liệu trong Feast lệch với contract, registry phải tạo tay bằng `feast apply`, và đường dẫn bị cố định theo thư mục repo.
- **Hiện tại:**
  - Feature view được sinh từ contract (21 `Float64` + 7 `String`). Contract chuẩn hoá mọi feature số về `float64`; trước đây cùng một feature có thể là `int64` hoặc `float64` tuỳ dữ liệu.
  - Mỗi cutoff một snapshot, ghi thêm vào kho lịch sử một cách atomic.
  - Kiểm tra chất lượng trước khi ghi hoặc chấm điểm (`src/features/quality.py`): khoảng giá trị, category hợp lệ, tính nhất quán giữa các cửa sổ thời gian, số household, tỷ lệ giá trị 0 so với snapshot trước. Ngưỡng được hiệu chỉnh trên 17 snapshot thật và cả 17 đều qua. Bước này tự viết thay vì dùng Evidently vì cái cần ở đây là kết quả đạt hay không đạt để chặn snapshot.
  - Quan sát snapshot bằng Evidently (`src/feature_store/observability.py`): mỗi lần batch, so snapshot mới với snapshot trước. Gồm drift từng feature (Wasserstein/Jensen-Shannon), giá trị ngoài khoảng, cột hằng số và thống kê. Kết quả là báo cáo HTML, file JSON và gauge cho Grafana, kèm 3 alert. Ngưỡng được hiệu chỉnh trên 16 cặp snapshot thật: 0–4 trên 28 feature bị drift, giá trị ngoài khoảng dưới 0,3%. Evidently chỉ để quan sát, không chặn batch; việc chặn vẫn do bước kiểm tra chất lượng.
  - Không dùng online store (serving dùng điểm batch trong Redis). TTL 7 ngày: dữ liệu quá hạn thì báo lỗi, không âm thầm dùng.
  - Registry được đăng ký từ code; thư mục `feature_repo/` đã xoá.
  - Batch ghi snapshot mỗi lần chạy. Training mặc định đọc qua join point-in-time và tự backfill cutoff còn thiếu.
  - CLI `scripts.feature_store`.
- **Kiểm chứng:**
  - Đọc từ store và tính trực tiếp cho kết quả giống hệt nhau (có test).
  - Train qua Feast cho AUC 0,8593, giống hệt khi tính trực tiếp; mất 20s so với 10s.
  - Cả hai champion vẫn tái lập được (ΔAUC = 0).
  - Smoke test e2e 15/15 với dữ liệu thật.

### Phase 0: đã xong
- CI: [.github/workflows/ci.yml](../.github/workflows/ci.yml) chạy `ruff` (bộ rule ghim trong `pyproject.toml`) và `pytest -m "not integration"` kèm coverage.
- Marker `slow`, `requires_data` (tự skip khi thiếu CSV thật) và `integration`.
- Fixture dữ liệu tổng hợp: [tests/fixtures/synthetic_dunnhumby.py](../tests/fixtures/synthetic_dunnhumby.py).
- Config tập trung: [src/config/model_config.py](../src/config/model_config.py).
- Redis thăm dò kết nối một lần, không retry, khi khởi tạo. Thời gian chờ khi không có Redis giảm từ khoảng 26s xuống 0.5s; cả bộ test giảm từ 106s xuống khoảng 26s.
- Sửa `.gitignore`: pattern `data/` đang ignore luôn `src/data/`.

### Phase 1: đã xong
- Champion được tái lập trong repo. Lệnh `python -m scripts.reproduce_champion` cho kết quả trên cùng 5.000 dòng test (cutoff 650, 680):

  | Metric | Champion | Train lại | Δ |
  |---|---|---|---|
  | ROC-AUC | 0.8596 | 0.8593 | −0.0003 |
  | PR-AUC | 0.9304 | 0.9307 | +0.0003 |
  | Brier | 0.1363 | 0.1368 | +0.0005 |
  | Lift@10% | 1.414 | 1.417 | +0.003 |

- Loader CSV và Lakehouse cho fingerprint và feature giống hệt nhau trên dữ liệu thật.
- Bundle schema v2 lưu: feature set, horizon, các cutoff, fingerprint dữ liệu, git SHA, phương pháp calibration.
- Predictor từ chối bundle không khớp contract hoặc không khai báo horizon.

### Phase 2: đã xong
- Gate chất lượng tính tương đối theo base rate ([model_config.py](../src/config/model_config.py) `GateConfig`). Kết quả trên tập test thật:

  | Tiêu chí | Ngưỡng | Ứng viên | Luật recency |
  |---|---|---|---|
  | ROC-AUC | ≥ 0.75 | 0.859 | 0.789 |
  | Brier skill score | ≥ 0.20 | 0.347 | −0.87 |
  | Lift chuẩn hoá (lift−1)/(lift_max−1) | ≥ 0.85 | 0.980 | 0.697 |
  | PR-AUC gain (PR-AUC−base)/(1−base) | ≥ 0.60 | 0.768 | 0.557 |
  | ECE | ≤ 0.03 | 0.019 | 0.444 |

- So sánh với champion trên cùng các dòng test ([champion_challenger.py](../src/training/champion_challenger.py)), bootstrap ghép cặp, lấy mẫu lại theo hộ ([statistics.py](../src/training/statistics.py)). Điều kiện promote: cận dưới khoảng tin cậy 95% của ΔAUC ≥ −0.005. Với dữ liệu thật: ΔAUC = −0.0003, CI [−0.0015, +0.0010], nên ứng viên được coi là không kém hơn.
- Nếu champion có target khác (horizon hoặc feature set khác) thì hai model **không so sánh được**, và việc promote bị chặn, trừ khi dùng `force_promote`.
- Promote theo kiểu atomic (ghi file tạm rồi rename). Mọi champion đều được lưu ở `models/history/`, có audit log `promotions.jsonl`. Champion cũ (chưa có trong registry) được tự động lưu lại trước khi bị thay. Rollback bằng `python -m scripts.registry rollback [--to VERSION]`.
- DAG hằng tháng không bao giờ dùng `force_promote`, và log rõ tiêu chí nào chặn việc promote.
- **Chưa promote champion mới:** ứng viên đã qua gate, nhưng tôi chạy với `--no-promote` để không thay file `models/champion_model.joblib` đang được track trong git. Lần chạy retrain tự động đầu tiên (hoặc chạy `python -m src.pipelines.train_pipeline`) sẽ promote nó, và champion cũ được lưu vào history.

### Phase 3: đã xong
Chi tiết và số liệu đầy đủ nằm ở [analysis/TARGET_ANALYSIS.md](analysis/TARGET_ANALYSIS.md), sinh ra bằng lệnh `python -m scripts.target_analysis`.

- **Phân tích target (3.1).** Base rate của target purchase là 0.33 / 0.53 / 0.68 / 0.80 ở horizon 3 / 7 / 14 / 28 ngày. Ở horizon 14 ngày, tần suất mua gần như quyết định nhãn: hộ có từ 20 basket trở lên trong 90 ngày thì 97% sẽ mua, hộ không có basket nào thì 12%.
- **Target thứ hai là `lapse` (3.2)** ([targets.py](../src/features/targets.py)).
  - Định nghĩa: hộ mua đều đặn (ít nhất 1 basket trong 30 ngày và ít nhất 4 basket trong 90 ngày) nhưng không mua gì trong 21 ngày tiếp theo.
  - Base rate 0.106 trên khoảng 1.850 hộ đủ điều kiện.
  - Gate riêng (`LAPSE_TRAINING_CONFIG`), ngưỡng được chốt *trước* khi train model đầu tiên.
  - Champion đầu tiên ở `models/lapse/`: AUC 0.816, lift@10% 3.02 lần, ECE 0.010.
  - Tái lập được (ΔAUC = 0). DAG hằng tháng và scheduler retrain cả hai target.
- **Đánh giá theo năng lực vận hành (3.3).**
  - Lapse: xếp hạng theo **p(lapse) × chi tiêu 90 ngày** giữ được số tiền gấp khoảng 2 lần so với xếp hạng chỉ theo p(lapse), và gấp 1.4–1.6 lần so với luật quá hạn. Batch xuất `lapse_risk_day_<cutoff>.csv` theo thứ hạng giá trị này: 186 hộ được gắn `RETENTION_OFFER` ở ngày 711.
  - Purchase: mọi model đều chọn đúng nhóm khách hàng mua nhiều và đều đặn (precision khoảng 1.0). Chi tiêu của nhóm này không phải doanh thu tăng thêm.
- **Lưu trữ EDA/CRISP-DM (3.4).** Notebook, tài liệu, slide và từ điển dữ liệu được khôi phục vào [analysis/archive_crisp_dm/](analysis/archive_crisp_dm/) từ commit `120ade9`. Không lấy `artifacts/`; các notebook không chạy lại được, nhưng vẫn xem được output đã lưu.
- **Chưa đạt trọn tiêu chí nghiệm thu "hơn baseline có ý nghĩa".** Model vượt rõ luật heuristic và chọn ngẫu nhiên. Nhưng **logistic regression gần ngang LightGBM** ở cả hai target: lapse AUC 0.809 so với 0.816, purchase 0.856 so với 0.859; về giá trị tiền của target purchase, logistic regression còn tốt hơn. Cần giữ logistic regression làm baseline bắt buộc. Bước tiếp theo để chứng minh giá trị là thiết kế uplift / holdout.

### Phase 4: đã xong
- **Lỗi lớn phát hiện thêm:** API trước đây **không bao giờ nạp điểm batch** (`_batch_cache` chỉ được nạp trong test). Ở môi trường thật, mọi request đều trả 0.5 và decile 5. Điểm challenger cũng không bao giờ được dùng.
- **Kho điểm batch dùng chung** ([batch_store.py](../src/serving/batch_store.py)):
  - Batch pipeline ghi điểm vào Redis, mỗi biến thể (champion / challenger) một hash, và đổi sang bản mới bằng `RENAME` atomic. Mọi replica đọc chung một bản.
  - Khi không có Redis, API nạp điểm từ artifact của lần batch mới nhất lúc khởi động.
  - Có endpoint `/ready`, trả 503 khi chưa có điểm batch.
- **Luật fusion** lấy từ `settings.FUSION_*` và được ghi rõ là heuristic (4.1).
- **Cold-start** trả rõ `score_source="cold_start"` cùng xác suất trung bình của toàn bộ hộ, không trả 0.5 im lặng (4.2). Ràng buộc `household_key ≤ 2500` đã được bỏ.
- **Canary (4.4).**
  - Batch chấm điểm thêm challenger (`--challenger` / `CHALLENGER_MODEL_PATH`), và traffic được route sang challenger sẽ nhận điểm của challenger.
  - Response trả về cả `routed_variant` (biến thể được route) và `model_variant` (điểm thực sự được dùng).
  - Challenger phải có cùng target, horizon và feature set với champion.
- **`STRICT_INFRA` (4.5).** Khi bật, Redis hoặc Kafka không kết nối được thì service lỗi ngay, không âm thầm fallback. Sự kiện có `event_id` và được xử lý idempotent (đánh dấu trong Redis bằng `SET NX` có TTL).
- **Bảo mật (4.6).**
  - Các endpoint ghi, admin và xuất hàng loạt yêu cầu `X-API-Key`. Ở production nếu chưa cấu hình key thì chặn hẳn (503).
  - Rate limit theo client, trả 429 khi vượt ngưỡng.
  - Scheduler và DAG gửi kèm key khi gọi hot-reload.
- **Nghiệm thu.**
  - Có test cho 401, 503, 429, cold-start, idempotency và Redis store (dùng `fakeredis`).
  - Load test `python -m scripts.load_test` (1 worker, backend memory, client chạy cùng máy): p99 của predict là **47.3 ms** ở 16 request đồng thời và 31.9 ms ở 4, đạt mục tiêu < 50 ms nhưng sát ngưỡng ở 16. Ở production nên chạy nhiều worker.
- **Chưa làm:** rate limit hiện chỉ áp dụng trong từng tiến trình, chưa chia sẻ giữa các replica (việc này thuộc về gateway). Docker compose chưa được chạy thử end-to-end (Phase 6).

### Phase 5: đã xong
- **Bốn lỗi gốc đã sửa.**
  1. Drift trước đây được so với snapshot Gold của *chính ngày đang chấm điểm*, nên luôn ra `STABLE`.
  2. Batch ghi gauge trong tiến trình riêng nên Prometheus không bao giờ thấy.
  3. Feedback loop đọc thẳng CSV, tính nhãn theo từng dòng giao dịch (khác định nghĩa lúc train), và scheduler gọi với horizon 7 ngày trong khi champion dự đoán 14 ngày.
  4. Job feedback của scheduler gọi `run_weekly_feedback`, một hàm không tồn tại.
  - Ngoài ra `build_gold_layer()` bị crash khi chưa có bảng demographics.
- **Drift (5.1)** được so với cửa sổ test out-of-time của champion ([reference.py](../src/monitoring/reference.py)). Ở cutoff 711: `STABLE`, PSI lớn nhất 0.02, PSI của phân phối dự đoán 0.003.
- **Feedback (5.2)** ([feedback_loop.py](../src/monitoring/feedback_loop.py)):
  - Dùng cùng DataLoader và cùng định nghĩa nhãn như lúc train; horizon lấy từ champion.
  - Lấy dự đoán từ bảng Gold, nếu không có thì chấm lại (replay).
  - So AUC thực tế với AUC lúc champion được chấp nhận. Kết quả thật: purchase ở cutoff 697 đạt 0.857 so với 0.859; lapse ở cutoff 690 đạt 0.812 so với 0.816. Cả hai đều `HEALTHY`.
  - Chính sách retrain ([retraining_policy.py](../src/monitoring/retraining_policy.py)) tự kích hoạt retrain khi drift ở mức CRITICAL hoặc AUC giảm quá 0.05. Việc promote vẫn phải qua gate.
- **Cảnh báo (5.3):**
  - 7 luật trong `docker/prometheus/alert_rules.yml`; Alertmanager đã có trong compose.
  - Metric lag của Kafka consumer, có exporter riêng ở cổng 9101.
  - Dashboard Grafana thêm 8 panel về chất lượng model.
  - `promtool check` và `promtool test rules` đều pass, và đã được đưa vào CI.
- **Hiệu quả fusion (5.4):** API ghi log các dự đoán đã trả về (Redis stream). Job feedback so từng luật với tỷ lệ mua thực tế và đưa ra hệ số mà dữ liệu gợi ý.
- **Nghiệm thu:** test `test_simulated_drift_is_critical_and_retrains` mô phỏng drift (chi tiêu ×6, mất 2/3 basket); kết quả là drift `CRITICAL_DRIFT` và scheduler kích hoạt retrain. Unit test bằng promtool xác nhận các cảnh báo bật đúng lúc.
- **Giới hạn:** Dunnhumby không có sự kiện intraday thật, nên việc đánh giá luật fusion mới chỉ được kiểm chứng bằng dữ liệu giả lập, chưa hiệu chỉnh được hệ số thật.

### Phase 6: đã xong
- **Docker stack trước đây không chạy được.**
  - Dockerfile chạy `pip install .` khi chưa có `src/`, và `.dockerignore` loại luôn `README.md` mà pyproject cần.
  - `COPY sample_data/` copy một symlink trỏ ra ngoài build context.
  - Serving dùng file model đóng sẵn trong image, trong khi scheduler lại mount `./models` của host. Hot-reload không thấy champion mới, và container ghi đè lên file model đang được git track.
  - Mật khẩu viết thẳng trong compose; Kafka không có healthcheck.
  - Image MinIO cộng đồng không còn tải được.
- **Compose viết lại (6.2, 6.3).**
  - Mọi service đều có healthcheck và `depends_on: service_healthy`.
  - Secret bắt buộc lấy từ `.env` (`${VAR:?}`); thiếu là compose từ chối chạy.
  - Registry, lakehouse và outputs nằm trong named volume; dữ liệu thô mount chỉ-đọc.
  - Batch và scheduler dùng chung Redis với serving.
  - MinIO và Postgres (không được code dùng) chuyển sang profile `storage`.
- **Smoke test end-to-end (6.1)** bằng `python -m scripts.e2e_smoke` (hoặc `make e2e`), chạy trong project compose cô lập:
  - Kiểm tra: build, services healthy, batch, train, feedback, `/ready` lấy từ Redis, predict, sự kiện qua Kafka làm đổi điểm, xác thực 401, `/metrics`, Prometheus scrape, 7 luật cảnh báo, Alertmanager.
  - Kết quả: **14/14 PASS với dữ liệu thật** (315s, đã gồm build); **13/13 PASS với dữ liệu tổng hợp** (60s).
  - CI chạy hằng đêm với dữ liệu tổng hợp (`.github/workflows/e2e.yml`).
- **6.4:** không có file `__pycache__`, `.duckdb` hay `.db` nào bị git track.

### Phase 7: đã xong
- Viết mới [MODEL_CARD.md](MODEL_CARD.md) (2 target, dữ liệu, metric, baseline, gate, cách dùng đúng, giới hạn) và [RUNBOOK.md](RUNBOOK.md) (job, retrain / promote / rollback, xử lý từng cảnh báo, xoay vòng key, sao lưu).
- README viết lại theo đúng trạng thái hiện tại: kiến trúc, cấu trúc thư mục, quick start bằng Docker, lịch job, test, giới hạn. Các link `file:///` đã được thay bằng link tương đối.
- Tài liệu kiến trúc được ghi chú rõ là "thiết kế mục tiêu", và thông số model đã được cập nhật.

### Phát hiện mới (đầu vào cho Phase 3)
- Gate cũ **không thể đạt**: lift@10% tối đa của tập test là 1.426, trong khi ngưỡng là 1.50. Đã sửa ở Phase 2.
- Logistic regression trên feature số đạt AUC 0.8556, chỉ kém LightGBM (0.8593) 0.004. Luật recency đơn giản đạt AUC 0.789. Điều này củng cố Phase 3: bài toán hiện tại gần như đã bão hoà.
- Calibration gần như không cải thiện: ECE trên test là 0.0189 (thô) so với 0.0193 (isotonic).
- Tier `URGENT` trước đây đọc feature cũ `cadence_recency_ratio` (không tồn tại với model 28 feature, nên luôn bằng 0). Giờ đã đọc `urgency_ratio`. Ở cutoff 711 có 1.430 hộ được gắn `URGENT`.

---

## Phase 0: Nền móng (1–2 ngày)

Mục tiêu: có lưới an toàn trước khi sửa code.

| # | Việc | File / vị trí |
|---|---|---|
| 0.1 | Thêm CI bằng GitHub Actions: `ruff`, `pytest`, đo coverage | `.github/workflows/ci.yml` (mới) |
| 0.2 | Đánh dấu test chậm hoặc test cần hạ tầng bằng `@pytest.mark.slow` / `integration`; CI mặc định chạy phần nhanh | `pyproject.toml`, `tests/` |
| 0.3 | Tạo bộ dữ liệu fixture nhỏ (khoảng 50 hộ, 100 ngày) để test không phụ thuộc `sample_data/` (đang bị gitignore) | `tests/fixtures/` |
| 0.4 | Gom mọi hằng số nghiệp vụ (horizon, cutoff, ngưỡng gate) vào một file config | `src/config/settings.py` hoặc `config/model.yaml` |

**Nghiệm thu:** CI chạy xanh trên mỗi PR, thời gian dưới 3 phút.

---

## Phase 1: Đúng và tái lập được model (3–5 ngày), ưu tiên cao nhất

Mục tiêu: champion phải train lại được 100% từ repo này.

| # | Việc | File |
|---|---|---|
| 1.1 | Định nghĩa một feature contract duy nhất: danh sách 28 feature, kiểu dữ liệu và nhóm categorical. Bỏ `MODEL_FEATURES` cũ hoặc chuyển nó thành `LEGACY_` | [pit_extractor.py](../src/features/pit_extractor.py) |
| 1.2 | Tách I/O khỏi logic: `extract_multidomain_features(tx, products, campaigns, ..., cutoff)` nhận DataFrame đầu vào, không tự đọc CSV | `pit_extractor.py` |
| 1.3 | Viết `DataLoader` đọc từ Lakehouse (bảng Silver). Đây là nguồn duy nhất cho cả train lẫn batch | [engine.py](../src/lakehouse/engine.py), `src/data/loader.py` (mới) |
| 1.4 | `DatasetBuilder` dùng extractor 28 feature; horizon lấy từ config (mặc định 14 ngày) | [dataset_builder.py](../src/training/dataset_builder.py) |
| 1.5 | Chia dữ liệu theo thời gian: fit / calib / test, với các cutoff giống manifest hiện tại (fit 200→500, val 530–590, test 650–680). Có khoảng trống (gap) ≥ horizon giữa các tập | [train_pipeline.py](../src/pipelines/train_pipeline.py) |
| 1.6 | Thêm `IsotonicCalibrator` và chọn giữa Platt / Isotonic theo ECE trên tập calib. Lưu loại calibrator vào bundle | [calibrator.py](../src/training/calibrator.py), [predictor.py](../src/models/predictor.py) |
| 1.7 | Bundle lưu đủ: `features`, `categorical_features`, `horizon_days`, `data_hash`, `git_sha`, metrics của test. Xoá trường `source_project` | [model_registry.py](../src/training/model_registry.py) |
| 1.8 | Predictor kiểm tra schema: nếu feature hoặc horizon không khớp contract thì báo lỗi rõ ràng, không đoán bundle theo tên | `predictor.py` |
| 1.9 | Train lại champion, rồi so sánh với model cũ: AUC trên tập test lệch không quá ±0.01 | `scripts/reproduce_champion.py` (mới) |

**Nghiệm thu:**
- Chạy `python -m src.pipelines.train_pipeline` sinh ra một champion có AUC ≈ 0.86 trên cùng cutoff test.
- Có test kiểm tra train/serve skew: cùng một hộ, feature lúc train và lúc batch trùng nhau.

---

## Phase 2: Governance và Champion/Challenger thật (2–3 ngày)

| # | Việc | File |
|---|---|---|
| 2.1 | Đổi gate lift thành ngưỡng tương đối, ví dụ `lift_10 ≥ 0.9 × (1 / base_rate)`, hoặc dùng normalized lift. Bổ sung `PR-AUC` và `ECE ≤ 0.03` | `model_registry.py` |
| 2.2 | Thêm gate so sánh: challenger chỉ được promote nếu không kém champion quá ε trên cùng tập test. Có thể dùng bootstrap CI | `model_registry.py`, [evaluator.py](../src/training/evaluator.py) |
| 2.3 | Promote theo kiểu atomic (ghi file tạm rồi rename) và có lệnh rollback: `registry.rollback(version)` | `model_registry.py` |
| 2.4 | DAG retrain gọi gate mới, không gọi `force_promote` khi chạy tự động | [dag_monthly_model_retraining.py](../dags/dag_monthly_model_retraining.py) |

**Nghiệm thu:** có test cho các trường hợp gate pass, gate fail và champion tốt hơn challenger. Champion hiện tại phải qua được gate mới.

---

## Phase 3: Giá trị bài toán (3–5 ngày, có thể làm song song với Phase 4)

Mục tiêu: model mang lại quyết định có giá trị, không chỉ "đoán ai sẽ đi chợ".

| # | Việc |
|---|---|
| 3.1 | Phân tích lại target: báo cáo base rate theo horizon 7/14/28 ngày và theo phân khúc |
| 3.2 | Chọn thêm **một** target có giá trị hơn và train song song làm model thứ hai. Gợi ý: lapse/churn (không mua trong 28 ngày dù thường xuyên mua) hoặc mua category X |
| 3.3 | Đánh giá theo năng lực vận hành: precision@K và doanh thu kỳ vọng ở top K hộ, đặt cạnh baseline recency |
| 3.4 | Khôi phục phần EDA / CRISP-DM từ commit `0a03cc4` vào `docs/analysis/` và `notebooks/`, rồi cập nhật lại slide nếu cần |

**Nghiệm thu:** có báo cáo so sánh model với baseline recency ở mức lift theo năng lực vận hành; model phải hơn baseline có ý nghĩa.

---

## Phase 4: Serving và Speed layer (3–4 ngày)

| # | Việc | File |
|---|---|---|
| 4.1 | Đưa luật fusion vào config (hệ số, TTL). Ghi nhận rõ trong code và tài liệu đây là heuristic. Về sau hiệu chỉnh bằng dữ liệu feedback (Phase 5) | [fusion_engine.py](../src/serving/fusion_engine.py) |
| 4.2 | Hộ không có trong cache: trả `404` hoặc `score_source="cold_start"` với xác suất bằng base rate, không trả mặc định 0.5 mà không báo | `fusion_engine.py`, [api.py](../src/serving/api.py) |
| 4.3 | Batch cache: nạp bằng vector hoá thay vì `iterrows`; đọc từ Redis hoặc Gold để nhiều replica dùng chung | `fusion_engine.py` |
| 4.4 | Batch pipeline chấm điểm thêm bằng challenger, để canary thực sự so được hai model | [batch_pipeline.py](../src/pipelines/batch_pipeline.py), [canary_router.py](../src/serving/canary_router.py) |
| 4.5 | Speed layer: cấu hình `STRICT_INFRA=true` ở production để tắt fallback bộ nhớ trong. Kafka consumer xử lý idempotent theo `event_id` | [delta_stream.py](../src/speed_layer/delta_stream.py), [kafka_bus.py](../src/speed_layer/kafka_bus.py) |
| 4.6 | Xác thực API: dùng API key hoặc JWT cho các endpoint ghi và admin (`/v1/canary/config`, `/v1/models/reload`, `/v1/speed/event`); thêm rate limit | `api.py`, `settings.py` |

**Nghiệm thu:** test API có trường hợp `401` và cold-start; p99 dưới 50 ms đo bằng `locust` hoặc `k6` cho 2.500 hộ.

---

## Phase 5: Monitoring và feedback loop khép kín (2–3 ngày)

| # | Việc | File |
|---|---|---|
| 5.1 | Drift chạy định kỳ: so PSI và KS giữa snapshot mới và tập train của champion (lấy từ bundle), xuất ra Prometheus | [drift_detector.py](../src/monitoring/drift_detector.py) |
| 5.2 | Feedback loop tính realized AUC/lift sau khi hết horizon; dữ liệu này là đầu vào kích hoạt retrain khi suy giảm | [feedback_loop.py](../src/monitoring/feedback_loop.py), DAG |
| 5.3 | Đặt cảnh báo Grafana / Alertmanager: PSI > 0.2, realized AUC giảm > 0.05, p99 latency, Kafka lag | `docker/grafana/`, `docker/prometheus.yml` |
| 5.4 | Đo hiệu quả fusion: so tỷ lệ chuyển đổi giữa hộ có và không áp dụng luật speed layer, từ đó hiệu chỉnh lại hệ số ở mục 4.1 | `feedback_loop.py` |

**Nghiệm thu:** mô phỏng drift (dịch chuyển dữ liệu) thì cảnh báo bắn và DAG retrain được kích hoạt.

---

## Phase 6: Hạ tầng và kiểm thử end-to-end (2–3 ngày)

| # | Việc |
|---|---|
| 6.1 | Smoke test chạy `docker compose up`: ingest → train → batch → gọi API → bắn event Kafka → kiểm tra score thay đổi. Chạy trong CI ở job `integration`, theo lịch nightly |
| 6.2 | Thêm healthcheck và `depends_on: condition: service_healthy` cho tất cả service trong [docker-compose.yml](../docker-compose.yml) |
| 6.3 | Quản lý secret qua `.env` và Docker secrets; không để mật khẩu mặc định trong compose |
| 6.4 | Gỡ các file `__pycache__`, `.duckdb`, `.db` khỏi repo nếu đang được track; kiểm tra lại `.gitignore` |

**Nghiệm thu:** một lệnh `make e2e` chạy xanh từ máy trắng.

---

## Phase 7: Tài liệu (1 ngày, nhưng cập nhật dần sau mỗi phase)

- Đồng bộ README với code: calibration, horizon, số feature, metric mới.
- Thêm `docs/RUNBOOK.md`: cách retrain, rollback, xử lý cảnh báo.
- Thêm `docs/MODEL_CARD.md`: target, dữ liệu, metric, giới hạn và rủi ro.

---

## Lộ trình tổng

| Tuần | Phase | Kết quả |
|---|---|---|
| 1 | 0 + 1 | CI chạy, champion tái lập được |
| 2 | 2 + 4 (song song với 3) | Gate đúng, API an toàn, canary hoạt động thật |
| 3 | 3 + 5 | Target có giá trị, vòng feedback khép kín |
| 4 | 6 + 7 | Chạy E2E được, tài liệu hoàn chỉnh |

Tổng khoảng **3–4 tuần** cho một người.

## Rủi ro chính

- Chuyển sang extractor 28 feature với đầu vào DataFrame có thể làm metric lệch so với model gốc ở `projectphase3`. Mục 1.9 dùng để phát hiện sớm việc này.
- Dữ liệu Dunnhumby là dữ liệu tĩnh, nên speed layer và feedback loop chỉ kiểm chứng được bằng dữ liệu replay hoặc giả lập.
