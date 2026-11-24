# Kiến trúc hệ thống

Đọc từ trên xuống: sơ đồ 1 là bức tranh tổng thể; các sơ đồ 2–5 đi vào từng luồng.
Mỗi sơ đồ chỉ trả lời một câu hỏi.

## 1. Tổng quan: hệ thống gồm những khối nào?

```mermaid
flowchart LR
    data[("Dữ liệu Dunnhumby<br/>Lakehouse DuckDB")]
    fs["Feature store<br/>(Feast)"]
    train["Training<br/>(hằng tháng)"]
    registry[("Model registry")]
    batch["Batch scoring<br/>(hằng tuần)"]
    redis[("Redis")]
    speed["Speed layer<br/>(Kafka)"]
    api["Serving API<br/>(FastAPI)"]
    crm(["CRM / ứng dụng"])
    mon["Monitoring<br/>Prometheus + Grafana"]

    data --> fs --> train --> registry --> batch
    fs --> batch
    batch --> redis
    speed --> redis
    redis --> api --> crm
    api -.-> mon
```

| Khối | Làm gì |
|---|---|
| Lakehouse | Nạp CSV, làm sạch theo các lớp Bronze → Silver → Gold, lưu điểm batch |
| Feature store | Lưu lịch sử 28 feature của từng household theo mỗi ngày cutoff |
| Training | Train LightGBM; chỉ promote nếu model mới qua governance gate |
| Batch scoring | Mỗi tuần chấm điểm toàn bộ household rồi đẩy vào Redis |
| Speed layer | Nhận sự kiện mua hàng trong ngày, điều chỉnh điểm ngay |
| Serving API | Trả điểm = điểm batch + điều chỉnh trong ngày |
| Monitoring | Drift, AUC thực tế, cảnh báo, dashboard |

## 2. Mỗi tuần: điểm số được tạo ra như thế nào?

```mermaid
flowchart TD
    A["Nạp dữ liệu mới<br/>Bronze → Silver → Gold"] --> B["Tính 28 feature<br/>tại ngày cutoff"]
    B --> C{"Kiểm tra<br/>chất lượng"}
    C -- "không đạt" --> X["Dừng batch<br/>giữ điểm tuần trước<br/>+ alert critical"]
    C -- "đạt" --> D["Ghi snapshot<br/>vào Feast"]
    D --> E["Chấm điểm bằng champion<br/>(+ challenger, lapse)"]
    E --> F["Đẩy điểm vào Redis<br/>(swap atomic)"]
    E --> G["Lưu Gold + xuất CSV / CRM"]
    D -.-> H["Evidently: so với<br/>snapshot tuần trước"]
    E -.-> I["Drift audit: so với<br/>dữ liệu lúc train"]
```

Đường liền là luồng chính. Đường nét đứt chỉ để quan sát: nếu lỗi thì batch vẫn chạy tiếp.

## 3. Mỗi tháng: model mới được train và promote như thế nào?

```mermaid
flowchart TD
    A["Đọc feature từ Feast<br/>theo từng cutoff (point-in-time)"] --> B["Gắn nhãn từ dữ liệu<br/>sau cutoff (14 / 21 ngày)"]
    B --> C["Train LightGBM<br/>+ calibration"]
    C --> D{"Governance gate"}
    D -- "đạt chất lượng<br/>và không kém champion" --> E["Promote<br/>thành champion mới"]
    D -- "không đạt" --> F["Giữ champion cũ"]
    E --> G["Batch tuần sau<br/>dùng model mới"]
```

Governance gate kiểm tra 2 điều:
1. Chất lượng tuyệt đối: AUC, Brier, lift, ECE vượt ngưỡng.
2. Không kém champion hiện tại: kiểm định bằng bootstrap.

Hai model chạy song song là `purchase` (mua trong 14 ngày tới) và `lapse` (khách quen sắp rời bỏ, 21 ngày).

## 4. Khi có request: API trả điểm như thế nào?

```mermaid
sequenceDiagram
    participant POS as Quầy POS / web
    participant API as Serving API
    participant K as Kafka
    participant W as Consumer
    participant R as Redis
    participant CRM as CRM

    POS->>API: sự kiện mua hàng (event_id)
    API->>K: đẩy vào topic
    K->>W: đọc sự kiện
    W->>R: ghi điều chỉnh trong ngày (mỗi event_id 1 lần)

    CRM->>API: xin điểm household 7
    API->>R: điểm batch + điều chỉnh trong ngày
    R-->>API:
    API-->>CRM: điểm cuối + nhóm mức độ khẩn cấp
```

Ví dụ: household vừa mua sáng nay thì xác suất mua thêm trong vài ngày tới bị giảm xuống (dampening).
Household chưa có điểm batch thì nhận điểm mặc định (cold start).

## 5. Làm sao biết model đang xuống cấp?

```mermaid
flowchart LR
    A["Drift so với<br/>dữ liệu lúc train"] --> P{"Retraining<br/>policy"}
    B["AUC thực tế<br/>khi đã có nhãn"] --> P
    P -- "drift nghiêm trọng<br/>hoặc AUC giảm > 0,05" --> T["Train lại<br/>ngoài lịch"]
    A --> M["Prometheus<br/>→ Grafana + Alertmanager"]
    B --> M
    C["Kiểm tra chất lượng<br/>+ Evidently"] --> M
```

| Câu hỏi | Công cụ |
|---|---|
| Dữ liệu tuần này có hỏng không? | Kiểm tra chất lượng: chặn batch nếu dữ liệu xấu |
| Snapshot này khác tuần trước bao nhiêu? | Evidently: báo cáo HTML + Grafana |
| Dữ liệu có khác lúc train không? | Drift audit (PSI, KS) |
| Model còn dự đoán đúng không? | Feedback loop (AUC thực tế) |

## Lịch chạy

| Job | Khi nào |
|---|---|
| Kiểm tra chất lượng Lakehouse | Hằng ngày 01:00 |
| Batch scoring | Thứ Hai 02:00 |
| Feedback (AUC thực tế) | Thứ Tư 04:00 |
| Train lại | Ngày 1 hằng tháng 03:00, hoặc khi retraining policy yêu cầu |
