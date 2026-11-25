# Runbook vận hành

Dành cho người trực vận hành nền tảng propensity: cách chạy job, retrain, promote, rollback và xử lý từng cảnh báo. Thông tin về model nằm ở [MODEL_CARD.md](MODEL_CARD.md).

## 1. Thành phần và địa chỉ

| Service (compose) | Cổng | Kiểm tra |
|---|---|---|
| `serving-api` | 8000 | `GET /health` (liveness), `GET /ready` (503 cho tới khi có batch view) |
| `scheduler` | – | Log: `docker compose logs -f scheduler` |
| `redis` | 6379 | `redis-cli ping`. Chứa batch view và trạng thái monitoring |
| `prometheus` / `alertmanager` / `grafana` | 9090 / 9093 / 3000 | Luật cảnh báo: `docker/prometheus/alert_rules.yml` |

**Volume dùng chung:**
- `models_data`: registry champion.
- `lakehouse_data`: DuckDB và các file Parquet lớp Bronze.
- `outputs_data`: artifact của batch và báo cáo monitoring.

Dữ liệu thô được mount chỉ-đọc từ `RAW_DATA_DIR`.

**Khởi động:**
```bash
cp .env.example .env        # điền RAW_DATA_DIR, SERVING_API_KEYS, GRAFANA_ADMIN_PASSWORD
docker compose up -d --build
docker compose --profile tools run --rm batch-pipeline      # nạp batch view lần đầu -> /ready = 200
python -m scripts.e2e_smoke                                 # kiểm tra toàn bộ stack (project cô lập)
```

Mọi lệnh `python -m ...` bên dưới chạy được trên máy, hoặc trong container qua `docker compose --profile tools run --rm batch-pipeline <lệnh>`.

## 1b. Feature store (Feast)

- **Vị trí:** `FEATURE_STORE_DIR`; trong docker là `/app/data/lakehouse/feature_store`, nằm trong volume `lakehouse_data`.
  - Kho lịch sử (offline): `offline/household_features.parquet`, mỗi cutoff một snapshot.
  - Không có online store: serving dùng điểm batch tính sẵn trong Redis, không đọc feature trực tiếp.
  - Registry: `registry.db`, được đăng ký tự động từ code.
- **Batch** mỗi lần chạy ghi thêm snapshot của cutoff đó. Training mặc định (`TRAINING_FEATURE_SOURCE=feast`) đọc feature bằng join point-in-time. Cutoff nào chưa có snapshot thì được tính và ghi bổ sung một lần.
- **Kiểm tra:** `python -m scripts.feature_store status`. Sau mỗi lần batch, ngày cutoff mới phải xuất hiện trong danh sách.
- **Lỗi `MissingSnapshotError` ("no snapshot within 7 days")** nghĩa là feature được yêu cầu ở thời điểm cách snapshot gần nhất quá TTL. Cách xử lý:
  - Chạy `backfill --cutoffs <ngày>`, hoặc chạy lại batch.
  - Không tăng TTL để che lỗi, vì như vậy model sẽ âm thầm dùng feature cũ.
- **Khi sửa cách tính feature** (logic trong `pit_extractor.py`):
  - Ghi đè lịch sử bằng `python -m scripts.feature_store backfill --training --overwrite`, hoặc xoá thư mục rồi backfill lại.
  - Sau đó chạy `scripts.reproduce_champion`.
  - Nếu đổi danh sách feature thì phải tăng `FEATURE_SET_VERSION`; tên feature view sẽ đổi theo.
- **Lỗi `FeatureQualityError`** nghĩa là snapshot không qua bước kiểm tra chất lượng. Khi đó snapshot không được ghi vào store và batch dừng, không chấm điểm cũng không publish; điểm của lần chạy trước vẫn được giữ nguyên. Cách xử lý:
  - Đọc danh sách vi phạm trong log. Ví dụ: giá trị âm, category lạ, số household giảm quá 10%, tỷ lệ giá trị 0 của một feature thay đổi quá 25 điểm %.
  - Kiểm tra dữ liệu nguồn (Bronze, Silver) trước khi sửa code.
  - Nếu category mới là hợp lệ (upstream đổi cách mã hoá), cập nhật `ALLOWED_CATEGORIES` trong `src/features/quality.py` rồi retrain.
  - `FEATURE_QUALITY_CHECKS=false` chỉ dùng tạm thời khi đã xác nhận dữ liệu đúng.
- **Bỏ qua feature store khi cần:** `--feature-source builder` (training tính lại trực tiếp), hoặc `FEATURE_STORE_ENABLED=false` (batch không ghi snapshot).
- **Sao lưu:** kho lịch sử là nguồn của dữ liệu train, nên hãy sao lưu cùng `models_data`. Nếu mất, có thể dựng lại bằng `backfill` từ lớp Bronze.

- **Quan sát snapshot bằng Evidently:**
  - Mỗi lần chạy, batch so snapshot mới với snapshot trước đó và ghi ra `data/outputs/reports/feature_store/snapshot_<day>.html`. Mở file trong trình duyệt để xem phân phối hai snapshot chồng lên nhau theo từng feature, drift và thống kê.
  - Trên Grafana, hàng *Feature Store* hiển thị:
    - kết quả kiểm tra chất lượng;
    - tỷ lệ feature bị drift;
    - điểm drift của từng feature;
    - tỷ lệ giá trị ngoài khoảng của từng feature.
  - Bước này chỉ để quan sát: Evidently lỗi thì batch vẫn chạy tiếp.
- **Cảnh báo của feature store:**
  - `PropensityFeatureQualityFailed` (critical): snapshot bị kiểm tra chất lượng chặn lại, batch đã dừng. Xử lý như mục lỗi `FeatureQualityError` ở trên.
  - `PropensityFeatureSnapshotDrift` (warning): hơn 25% feature bị drift so với snapshot trước. Trên dữ liệu thật, mức này thường chỉ 0–14%. Cách xử lý:
    - Mở báo cáo HTML của snapshot đó.
    - Nếu là thay đổi thật của hành vi (mùa vụ, campaign) thì không cần làm gì.
    - Nếu là lỗi dữ liệu nguồn thì chạy lại batch sau khi sửa.
  - `PropensityFeatureOutOfRange` (warning): hơn 5% giá trị của một feature nằm ngoài khoảng của snapshot trước. Hay gặp khi đơn vị dữ liệu nguồn bị đổi.
- **Xem xu hướng qua nhiều snapshot:**
  - Chạy `python -m scripts.feature_store profile --all --no-html` để tạo bản tóm tắt cho mọi snapshot.
  - Sau đó chạy `python -m scripts.feature_store trend --feature <tên>`.

## 2. Lịch job

| Job | Lịch | Chạy tay |
|---|---|---|
| Chấm điểm batch, kèm kiểm tra drift | Thứ Hai 02:00 UTC | `python -m src.scheduler.daemon --run-job batch` |
| Retrain cả hai target | Ngày 1 hằng tháng, 03:00 UTC | `python -m src.scheduler.daemon --run-job retrain` |
| Kiểm tra chất lượng dữ liệu | Hằng ngày 01:00 UTC | `python -m src.scheduler.daemon --run-job dq` |
| Feedback (AUC thực tế) | Thứ Tư 04:00 UTC | `python -m src.scheduler.daemon --run-job feedback` |

Job batch và job feedback **tự kích hoạt retrain** khi drift ở mức `CRITICAL_DRIFT` hoặc AUC thực tế giảm quá 0,05. Đặt `AUTO_RETRAIN_ON_DEGRADATION=false` nếu chỉ muốn nhận cảnh báo. Retrain chỉ tạo ra ứng viên; việc promote vẫn phải qua gate.

## 3. Retrain, promote, rollback

```bash
# Chỉ đánh giá, không động vào registry
python -m src.pipelines.train_pipeline --target purchase --no-promote
# Promote nếu qua gate (gate chất lượng + so sánh với champion)
python -m src.pipelines.train_pipeline --target purchase
# Xem các phiên bản (* = champion đang chạy) và rollback
python -m scripts.registry --target purchase list
python -m scripts.registry --target purchase rollback [--to VERSION] --reason "realized AUC drop"
# Báo serving nạp lại champion (cần API key)
curl -X POST -H "X-API-Key: $KEY" http://localhost:8000/v1/models/reload
```

- **Đọc lý do bị chặn:** xem `governance_gate` trong `data/outputs/training_manifest_<run>.json`.
  - `quality.checks.*.pass = false`: ứng viên chưa đạt chất lượng.
  - `champion_comparison.status = "compared"` kèm `passed = false`: ứng viên kém champion đang chạy.
  - `champion_comparison.status = "incomparable"`: target, horizon hoặc feature set đã đổi. Đây là quyết định có chủ đích: kiểm tra lại, rồi mới promote bằng `--force-promote`. Mọi lần promote đều được ghi vào `promotions.jsonl`.
- **Kiểm tra khả năng tái lập** (nên chạy trước khi sửa feature): `python -m scripts.reproduce_champion --target purchase`.
- **Sau khi promote purchase:** commit `models/champion_model.joblib` và `models/latest_model_manifest.json`, để repo luôn khớp với registry đang chạy.

## 4. Xử lý cảnh báo

### `PropensityNoBatchView` (critical): serving không có batch view
**Biểu hiện:**
- `/ready` trả 503.
- Mọi câu trả lời có `score_source = "cold_start"`.

**Cách xử lý:**
1. Xem log của lần batch gần nhất: `docker compose logs scheduler | grep -i batch`.
2. Chạy lại: `docker compose --profile tools run --rm batch-pipeline`.
3. Kiểm tra Redis: `redis-cli HLEN batch:champion:scores`. Giá trị phải bằng 2500.

### `PropensityFeatureDriftHigh` (warning) / `PropensityDriftCritical` (critical)
1. Mở `data/outputs/reports/drift_purchase_latest.json` và xem `feature_details` có PSI cao.
2. Phân biệt drift do dữ liệu hỏng (ingest lỗi, cột đổi kiểu) với drift do hành vi thật:
   - Chạy `--run-job dq`.
   - So số dòng của lớp Bronze.
3. Nếu dữ liệu hỏng, **tắt tự động retrain** (`AUTO_RETRAIN_ON_DEGRADATION=false`), sửa dữ liệu rồi chạy batch lại. Nếu là hành vi thật, để retrain tự chạy và đọc kết quả gate.

### `PropensityRealizedAucDrop` (critical): AUC thực tế giảm quá 0,05
1. Mở `data/outputs/reports/feedback_attribution_<target>_day_<cutoff>.json` và xem `decile_breakdown`.
2. Nếu ứng viên retrain cũng không qua gate, cân nhắc rollback về phiên bản trước (mục 3) và báo đội model.
3. `prediction_source = "replay"` nghĩa là dự đoán được chấm lại chứ không lấy từ bảng Gold. Hãy kiểm tra xem batch của cutoff đó có thực sự đã chạy không.

### `PropensityPredictLatencyP99High` (warning): p99 > 50 ms
- Kiểm tra độ trễ của Redis. Mỗi container chỉ chạy 1 worker (để metric và rate limit đúng); cần thêm năng lực thì tăng số container `serving-api`, không tăng `--workers`.
- Load test: `python -m scripts.load_test`. Mốc tham chiếu (1 worker, 16 request đồng thời, trong container Linux): p99 khoảng 28 ms, khoảng 1.100 request/giây.

### `PropensityServingDown`
- `docker compose ps serving-api` và `docker compose logs serving-api`.
- Khi `STRICT_INFRA=true`, API dừng khởi động nếu Redis không kết nối được. Lỗi `InfrastructureUnavailable` trong log là dấu hiệu của trường hợp này.

## 5. Thao tác bảo mật

- **Xoay vòng API key** (không gây gián đoạn):
  1. Thêm key mới vào `SERVING_API_KEYS` (nhiều key cách nhau bằng dấu phẩy).
  2. `docker compose up -d serving-api scheduler`.
  3. Cập nhật key ở phía client.
  4. Bỏ key cũ và khởi động lại các service đó một lần nữa.
- Ở production, nếu `SERVING_API_KEYS` rỗng thì các endpoint ghi và admin trả 503 (chặn hẳn). Đây là hành vi có chủ đích.

## 6. Sao lưu và khôi phục

- `models_data`: registry, gồm toàn bộ history. Sao lưu trước mỗi lần promote thủ công:
  ```bash
  # volume = <compose project>_models_data; project mặc định là tên thư mục (propensityvsf)
  docker run --rm -v propensityvsf_models_data:/m -v "$PWD":/b alpine tar czf /b/models.tgz -C /m .
  ```
- `lakehouse_data` có thể dựng lại từ CSV thô (ingest Bronze là thao tác lặp lại được). `redis_data` chỉ chứa trạng thái tạm thời: batch view sẽ được ghi lại ở lần batch kế tiếp.
- **Khôi phục champion:** ưu tiên `scripts.registry rollback`. Nếu mất cả volume, image vẫn chứa champion đã commit trong repo.

## 7. Giới hạn đã biết

- Rate limit đếm theo từng container (mỗi container 1 worker). Chạy nhiều container thì mỗi container có bộ đếm riêng; cần giới hạn chung thì đặt ở gateway.
- Điểm chỉ cập nhật mỗi tuần. Khách vừa mua hôm nay vẫn giữ điểm cũ đến lần batch sau; nếu cần, loại khách vừa mua ngay trong công cụ chạy chiến dịch.
