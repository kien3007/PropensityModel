"""Build an executable, narrative-first propensity notebook from raw data."""
from __future__ import annotations

import json
from pathlib import Path


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)}


def code(s: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s.strip().splitlines(True)}


cells = [
    md('''# Customer Purchase Propensity — từ dữ liệu thô đến kiểm định theo thời gian

**Câu hỏi:** Tại ngày snapshot `s`, một household đã được quan sát có xác suất bao nhiêu để phát sinh ít nhất một basket được ghi nhận trong 7 ngày tiếp theo?

Notebook này theo mạch của notebook churn tham khảo: đặt bài toán → kiểm tra từng bảng → hiểu panel thời gian → dựng nhãn → chia thời gian → tạo feature → phân tích feature → baseline → rolling validation → chọn mô hình → calibration → đánh giá và ứng dụng.

**Phạm vi:** dữ liệu của 2.500 frequent shoppers tại một retailer; kết quả không đại diện cho toàn bộ người tiêu dùng. Một basket được ghi nhận có thể chứa fuel hoặc các dòng giá trị bằng 0; đây là proxy *any recorded basket*, chưa phải định nghĩa giao dịch hợp lệ đã được data owner xác nhận.

**Lưu ý holdout:** tập test ở project này đã được mở trong V2 trước đây. Nó được giữ ngoài mọi bước lựa chọn trong notebook này, nhưng không còn là bằng chứng hoàn toàn mới đối với toàn bộ quá trình dự án.'''),
    md(r'''# 1. Business understanding và định nghĩa target

Hệ thống weekly ranking có thể hỗ trợ ưu tiên contact hoặc lập kế hoạch nhu cầu khi capacity hữu hạn. Score là xác suất mua quan sát được, không đo tác động tăng thêm của coupon.

Với household `h` và cutoff `s`:

$$Y_{h,s}=1 \quad\text{nếu có ít nhất một basket trong }(s,s+7].$$

Feature chỉ dùng dữ liệu tới hết ngày `s`. Household chỉ đủ điều kiện sau lần đầu được quan sát và label phải có đủ 7 ngày theo dõi. Dữ liệu không chứng minh household không mua ở nơi khác.'''),
    code('''from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             precision_recall_curve, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
if not (ROOT / 'src' / 'propensity_v2').exists():
    raise FileNotFoundError('Chạy notebook từ project root hoặc thư mục notebooks')
sys.path.insert(0, str(ROOT / 'src'))
from propensity_v2.config import PropensityV2Config
from propensity_v2.data import load_transactions, enrich_transactions, build_daily_features, load_demographics
from propensity_v2.features import build_modeling_table
from propensity_v2.modeling import temporal_split, rolling_origin_folds, fit_calibrator, calibrate

RAW = ROOT / 'sample_data' / 'dunnhumby'
cfg = PropensityV2Config()
target = cfg.target_column
sns.set_theme(style='whitegrid', palette='deep')
pd.set_option('display.max_columns', 30)
print('Project:', ROOT, '| Target:', target)'''),
    md('''## Leakage: vì sao snapshot là bắt buộc?

Nếu tính recency đến ngày 711 rồi dự đoán hộ có mua trong 7 ngày cuối hay không, `recency < 7` gần như trực tiếp chứa đáp án. Cần tách thời gian: `[..., s]` cho feature và `(s, s+7]` cho label. Chia ngẫu nhiên các snapshot sau khi tính feature không thay thế được kiểm định tương lai.'''),
    md('''# 2. Data understanding — tám bảng nguồn

Kiểm tra schema, độ phủ và khóa trước khi join. `causal_data.csv` có khoảng 36,8 triệu dòng nên được đọc theo chunks; dataset này được audit, chưa đưa vào champion vì thời điểm exposure và availability cần được xác minh.'''),
    code('''files = sorted(RAW.glob('*.csv'))
inventory = []
tables = {}
for path in files:
    if path.name in ('transaction_data.csv', 'causal_data.csv'):
        continue
    frame = pd.read_csv(path)
    tables[path.stem] = frame
    inventory.append({'table': path.stem, 'rows': len(frame), 'columns': len(frame.columns),
                      'missing_cells': int(frame.isna().sum().sum()),
                      'exact_duplicate_rows': int(frame.duplicated().sum())})
transactions = load_transactions(RAW)
inventory.append({'table': 'transaction_data', 'rows': len(transactions),
                  'columns': len(transactions.columns),
                  'missing_cells': int(transactions.isna().sum().sum()),
                  'exact_duplicate_rows': int(transactions.duplicated().sum())})
causal_rows = sum(len(chunk) for chunk in pd.read_csv(RAW / 'causal_data.csv',
                  usecols=['PRODUCT_ID'], chunksize=1_000_000))
inventory.append({'table': 'causal_data', 'rows': causal_rows, 'columns': 5,
                  'missing_cells': np.nan, 'exact_duplicate_rows': np.nan})
display(pd.DataFrame(inventory).sort_values('rows', ascending=False))'''),
    md('''`NaN` trong inventory của causal data nghĩa là chưa tính ở cell này, không phải bằng 0. Không xóa duplicate coupon mapping một cách tùy tiện trước khi hiểu semantics.'''),
    code('''display(transactions.head())
display(transactions.dtypes.to_frame('dtype').T)
display(transactions.isna().sum().to_frame('nulls').T)
print('Rows / households / baskets:', len(transactions),
      transactions.household_key.nunique(), transactions.BASKET_ID.nunique())
print('Day range:', transactions.DAY.min(), '–', transactions.DAY.max())
print('Exact duplicate transaction rows:', transactions.duplicated().sum())'''),
    md('''## Kiểm tra product, household và promotion trước khi nối

Product phải là many-to-one theo `PRODUCT_ID`. Demographic chỉ là enrichment tùy chọn: file không cung cấp thời điểm hiệu lực, vì vậy không thể khẳng định thông tin đã sẵn có tại từng cutoff. Campaign assignment cũng không đồng nghĩa hộ đã thực sự nhìn thấy thông điệp.'''),
    code('''product = tables['product']
demo = tables['hh_demographic']
campaign = tables['campaign_table']
campaign_desc = tables['campaign_desc']
coupon = tables['coupon']
redemption = tables['coupon_redempt']
checks = {
    'product duplicate keys': product.PRODUCT_ID.duplicated().sum(),
    'transaction product match rate': transactions.PRODUCT_ID.isin(product.PRODUCT_ID).mean(),
    'demographic household coverage': transactions.household_key.drop_duplicates().isin(demo.household_key).mean(),
    'campaign household coverage': transactions.household_key.drop_duplicates().isin(campaign.household_key).mean(),
    'redemption coupon match rate': redemption.COUPON_UPC.isin(coupon.COUPON_UPC).mean(),
}
redemption_windows = redemption.merge(campaign_desc[['CAMPAIGN','START_DAY','END_DAY']],
                                     on='CAMPAIGN', how='left', validate='many_to_one')
checks['redemption in campaign window'] = redemption_windows.DAY.between(
    redemption_windows.START_DAY, redemption_windows.END_DAY).mean()
display(pd.Series(checks, name='value').to_frame())
assert checks['product duplicate keys'] == 0
assert checks['transaction product match rate'] == 1'''),
    md('''**Diễn giải:** product join được kiểm chứng trên dữ liệu; demographic coverage và marketing coverage là các đặc điểm của mẫu, không chứng minh đại diện cho population. Các bảng coupon/campaign chỉ tạo giả thuyết, không chứng minh tác động nhân quả.'''),
    md('''## Sanity check: quantity không đồng nhất giữa department

Giá trị rất lớn tập trung ở fuel/miscellaneous. Không cộng `QUANTITY` xuyên department như một số lượng món duy nhất; notebook sử dụng basket, số dòng và spend.'''),
    code('''large_quantity = transactions.loc[transactions.QUANTITY > 100, ['PRODUCT_ID','QUANTITY']].merge(
    product[['PRODUCT_ID','DEPARTMENT']], on='PRODUCT_ID', how='left', validate='many_to_one')
display(large_quantity.groupby('DEPARTMENT').QUANTITY.agg(['size','median','max']).sort_values('size', ascending=False).head(10))
print('Rows QUANTITY > 100:', len(large_quantity), '| max:', transactions.QUANTITY.max())'''),
    md('''# 3. Data exploration — panel thời gian và nhãn

Trước khi gán nhãn âm, cần kiểm tra coverage của cửa sổ outcome. Bản thân việc không có basket trong retailer này không đồng nghĩa household không mua ở nơi khác.'''),
    code('''basket = transactions.groupby(['household_key','BASKET_ID','DAY'], as_index=False).agg(
    spend=('SALES_VALUE','sum'), lines=('PRODUCT_ID','size'))
daily_volume = basket.groupby('DAY').agg(baskets=('BASKET_ID','nunique'),
                                         households=('household_key','nunique'))
first_day = transactions.groupby('household_key').DAY.min()
fig, ax = plt.subplots(1, 2, figsize=(14,4))
daily_volume.baskets.plot(ax=ax[0], title='Basket theo ngày')
first_day.hist(bins=35, ax=ax[1]); ax[1].set_title('Ngày đầu tiên quan sát household')
plt.tight_layout(); plt.show()
display(first_day.describe(percentiles=[.25,.5,.75,.95]).to_frame('first_observed_day'))'''),
    md('''Việc hộ xuất hiện dần tạo left-censoring: ngày mua đầu tiên quan sát được không nhất thiết là ngày khách hàng gia nhập thực tế. `tenure` trong notebook chỉ là observed tenure. Không suy ra absence đầu panel là churn.'''),
    md('''## Dựng bảng point-in-time

Sử dụng cùng feature builder với project để xử lý transaction và product. Dòng lệnh dưới đây xây lại từ raw data; không đọc modeling table tính sẵn. Feature demographics sẽ bị loại khỏi các mô hình trong notebook.'''),
    code('''enriched = enrich_transactions(transactions, RAW)
daily, basket_features = build_daily_features(enriched)
household_ids = np.sort(daily.household_key.unique())
demographics = load_demographics(RAW, household_ids)
snapshots, profile = build_modeling_table(daily, demographics, cfg)
print(profile)
print('Duplicate household/snapshot:', snapshots.duplicated(['household_key','snapshot_day']).sum())
assert snapshots.duplicated(['household_key','snapshot_day']).sum() == 0
assert set(snapshots[target].unique()) <= {0, 1}'''),
    md('''## Kiểm chứng nhãn trên các ví dụ thô

Chọn vài household-snapshot và tính lại số basket tương lai trực tiếp từ transaction. Những assert này kiểm tra ranh giới `(s, s+7]` cùng eligibility, không dựa vào feature builder để tự xác nhận chính nó.'''),
    code('''examples = snapshots.sample(12, random_state=42)[['household_key','snapshot_day',target]]
for row in examples.itertuples(index=False, name=None):
    hh, cutoff, label = row
    history = transactions[(transactions.household_key == hh) & (transactions.DAY <= cutoff)]
    future = transactions[(transactions.household_key == hh) &
                          (transactions.DAY > cutoff) & (transactions.DAY <= cutoff + cfg.horizon_days)]
    assert not history.empty
    assert int(future.BASKET_ID.nunique() > 0) == label
display(examples)
print('12 kiểm tra label và eligibility độc lập: PASS')'''),
    md('''## Horizon và target prevalence

So sánh 7/14/30 ngày chỉ trên các cutoff có đủ label cho **cả ba** horizon. Target rate không phải thước đo duy nhất để chọn horizon; weekly decision cadence và chi phí hành động là giả định kinh doanh cần xác nhận.'''),
    code('''last_day = int(daily.DAY.max())
common_days = list(range(180, last_day - 30 + 1, 7))
hh_day = daily[['household_key','DAY','baskets']]
matrix = (hh_day.pivot(index='household_key', columns='DAY', values='baskets')
          .reindex(columns=range(last_day + 1), fill_value=0).fillna(0).to_numpy())
cum = np.cumsum(matrix, axis=1)
first = np.where(matrix > 0, np.arange(last_day + 1), last_day + 1).min(axis=1)
horizon_records = []
for h in (7,14,30):
    for s in common_days:
        eligible = first <= s
        labels = (cum[:, s+h] - cum[:, s])[eligible] > 0
        horizon_records.append({'horizon': h, 'snapshot_day': s, 'rate': labels.mean(),
                                'households': int(eligible.sum())})
horizon_rates = pd.DataFrame(horizon_records)
display(horizon_rates.groupby('horizon').rate.agg(['mean','min','max','std']))
sns.lineplot(data=horizon_rates, x='snapshot_day', y='rate', hue='horizon', palette='deep')
plt.title('Positive rate trên cùng cutoff trưởng thành'); plt.ylabel('Tỷ lệ có basket'); plt.show()'''),
    md('''# 4. Chia train / validation / test theo thời gian

Mỗi tập gồm nhiều snapshot tuần; giữa các tập chừa một snapshot. Label training phải trưởng thành trước snapshot validation/test. Cùng household có thể xuất hiện ở nhiều giai đoạn: kết quả đo khả năng dự báo tương lai cho nhóm khách đã biết, không đo generalization sang household hoàn toàn mới.'''),
    code('''train_mask, val_mask, test_mask, split = temporal_split(snapshots, cfg)
display(pd.DataFrame({k:v for k,v in split.items() if isinstance(v, dict)}).T)
print('Embargo snapshots:', split['embargo_days'])
assert split['train']['snapshot_max'] + cfg.horizon_days < split['validation']['snapshot_min']
assert split['validation']['snapshot_max'] + cfg.horizon_days < split['test']['snapshot_min']
train = snapshots.loc[train_mask].copy()
val = snapshots.loc[val_mask].copy()
test = snapshots.loc[test_mask].copy()
print('Train target rate:', train[target].mean(), '| Validation:', val[target].mean())'''),
    md('''# 5. Feature engineering và EDA trên train

Các nhóm feature gồm recency/cadence, frequency, spend, basket size, category, brand, discount và diversity. Để tránh phụ thuộc vào thông tin nhân khẩu học không có ngày hiệu lực, mô hình của notebook chỉ dùng feature hành vi. Toàn bộ biểu đồ và phép chọn dưới đây dùng **train**.'''),
    code('''features = ['recency_days','active_days_30d','active_days_90d','baskets_30d',
            'spend_30d','avg_basket_value_30d','line_items_30d',
            'discount_rate_30d','private_label_spend_share_90d',
            'grocery_spend_share_90d','fuel_spend_share_90d',
            'product_day_diversity_per_active_day_90d','recency_to_cadence']
assert all(f in train.columns for f in features)
display(train[features].describe(percentiles=[.05,.5,.95]).T)
display(train.groupby(target)[features].median().T.rename(columns={0:'Không mua',1:'Có mua'}))'''),
    code('''fig, axes = plt.subplots(2, 3, figsize=(16,8))
for ax, col in zip(axes.flat, features[:6]):
    upper = train[col].quantile(.99)
    sns.histplot(data=train.assign(plot_value=train[col].clip(upper=upper)), x='plot_value',
                 hue=target, stat='density', common_norm=False, bins=30,
                 element='step', fill=False, ax=ax)
    ax.set_title(col + ' (cắt hiển thị tại p99)')
plt.tight_layout(); plt.show()'''),
    md('''Histogram được cắt ở p99 chỉ để dễ nhìn; model vẫn dùng feature gốc. Sự khác nhau giữa hai phân phối là association trên train, chưa phải quan hệ nhân quả.'''),
    code('''fig, ax = plt.subplots(figsize=(11,8))
sns.heatmap(train[features].corr(), center=0, cmap='RdBu_r', vmin=-1, vmax=1, ax=ax)
ax.set_title('Tương quan feature trên train'); plt.tight_layout(); plt.show()'''),
    md('''## Một ví dụ kiểm tra confounding: fuel và tần suất mua

So sánh tỷ lệ mua giữa household có/không fuel, sau đó chia theo mức activity 90 ngày. Đây là phép kiểm tra mô tả; stratification không điều chỉnh mọi confounder và không xác lập tác động của fuel.'''),
    code('''eda = train[['active_days_90d','fuel_spend_90d',target]].copy()
eda['fuel_buyer'] = eda.fuel_spend_90d.gt(0)
eda['activity_band'] = pd.qcut(eda.active_days_90d, 4, duplicates='drop')
print('Association thô:')
display(eda.groupby('fuel_buyer')[target].agg(['mean','size']))
print('Theo nhóm activity:')
display(eda.groupby(['activity_band','fuel_buyer'], observed=True)[target].agg(['mean','size']))
print('Chênh lệch thô cần được đọc cùng activity và số mẫu; không diễn giải là quan hệ nhân quả.')'''),
    md('''# 6. Baseline và rolling-origin cross-validation

Baseline một biến xếp hạng theo **recency thấp hơn → propensity cao hơn**. Baseline này không cần fit. So sánh với logistic regression và histogram gradient boosting trên cùng ba temporal folds; preprocessing chỉ fit trên phần train của từng fold. PR-AUC được hiểu cùng prevalence, ROC-AUC là thước đo bổ sung.'''),
    code('''folds = rolling_origin_folds(snapshots, split['validation']['snapshot_min'])
models = {
    'logistic': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                              LogisticRegression(C=0.3, max_iter=500, random_state=42)),
    'hgb_regularized': make_pipeline(SimpleImputer(strategy='median'),
        HistGradientBoostingClassifier(max_iter=120, learning_rate=.07, max_leaf_nodes=15,
                                       min_samples_leaf=80, l2_regularization=5,
                                       early_stopping=False, random_state=42))}
records = []
for fold in folds:
    ft = snapshots[snapshots.snapshot_day.isin(fold['train_days'])]
    fv = snapshots[snapshots.snapshot_day.isin(fold['validation_days'])]
    assert max(fold['train_days']) + cfg.horizon_days < min(fold['validation_days'])
    for name, model in [('recency_rule', None), *models.items()]:
        if model is None:
            score = -fv.recency_days.to_numpy()
        else:
            fitted = clone(model).fit(ft[features], ft[target])
            score = fitted.predict_proba(fv[features])[:,1]
        records.append({'fold':fold['fold'], 'model':name,
                        'train_end':max(fold['train_days']), 'val_start':min(fold['validation_days']),
                        'val_end':max(fold['validation_days']), 'prevalence':fv[target].mean(),
                        'pr_auc':average_precision_score(fv[target], score),
                        'roc_auc':roc_auc_score(fv[target], score)})
cv_results = pd.DataFrame(records)
display(cv_results)
display(cv_results.groupby('model')[['pr_auc','roc_auc']].agg(['mean','std']).sort_values(('pr_auc','mean'), ascending=False))
means = cv_results.groupby('model').pr_auc.mean()
print(f"PR-AUC trung bình: recency={means['recency_rule']:.4f}, logistic={means['logistic']:.4f}, HGB={means['hgb_regularized']:.4f}.")'''),
    code('''sns.pointplot(data=cv_results, x='fold', y='pr_auc', hue='model', markers='o')
plt.title('Rolling-origin PR-AUC: cùng khoảng validation'); plt.ylabel('Average precision'); plt.show()'''),
    md('''Chọn theo mean PR-AUC trên ba folds, với tie-breaker là mô hình đơn giản hơn. Ba folds không đủ để khẳng định sai khác nhỏ có ý nghĩa thống kê; bảng theo từng fold cho thấy variation và tính phụ thuộc thời gian.'''),
    code('''cv_mean = cv_results[cv_results.model != 'recency_rule'].groupby('model').pr_auc.mean()
winner = cv_mean.sort_values(ascending=False).index[0]
if abs(cv_mean['logistic'] - cv_mean['hgb_regularized']) < 0.002:
    winner = 'logistic'
print('Model khóa trước validation/test:', winner)
champion = clone(models[winner]).fit(train[features], train[target])'''),
    md('''# 7. Calibration trên validation

Probability ranking và calibration là hai yêu cầu khác nhau. Fit Platt scaling trên validation; báo cáo validation sau calibration chỉ là in-sample diagnostic cho calibrator. Ngưỡng 0.5 dưới đây chỉ minh họa, không phải ngưỡng lợi nhuận tối ưu.'''),
    code('''p_val_raw = champion.predict_proba(val[features])[:,1]
calibrator = fit_calibrator(p_val_raw, val[target].to_numpy())
p_val = calibrate(calibrator, p_val_raw)
display(pd.DataFrame({'uncalibrated':[brier_score_loss(val[target],p_val_raw)],
                      'calibrated_in_sample':[brier_score_loss(val[target],p_val)]},
                     index=['validation Brier']))'''),
    md('''# 8. Temporal holdout — đánh giá sau khi khóa model

Trong notebook này test không tham gia chọn feature/model/calibration; vì đã được xem ở V2 trước đây, không gọi đây là untouched test ở cấp toàn dự án. Cần một giai đoạn dữ liệu mới cho xác nhận độc lập.'''),
    code('''p_test = calibrate(calibrator, champion.predict_proba(test[features])[:,1])
y_test = test[target].to_numpy()
recency_test = -test.recency_days.to_numpy()
test_metrics = pd.DataFrame({
    'prevalence':[y_test.mean()],
    'PR-AUC':[average_precision_score(y_test,p_test)],
    'ROC-AUC':[roc_auc_score(y_test,p_test)],
    'Brier':[brier_score_loss(y_test,p_test)],
    'recency PR-AUC':[average_precision_score(y_test,recency_test)]})
display(test_metrics)
print('Đây là đánh giá trên household đã biết; cần một giai đoạn mới để kiểm định độc lập.')'''),
    code('''precision, recall, _ = precision_recall_curve(y_test,p_test)
fpr, tpr, _ = roc_curve(y_test,p_test)
fig, ax = plt.subplots(1,2,figsize=(12,4))
ax[0].plot(recall,precision); ax[0].axhline(y_test.mean(),ls='--',color='gray',label='prevalence')
ax[0].set(title='Precision–recall',xlabel='Recall',ylabel='Precision'); ax[0].legend()
ax[1].plot(fpr,tpr); ax[1].plot([0,1],[0,1],'--',color='gray')
ax[1].set(title='ROC',xlabel='False positive rate',ylabel='True positive rate')
plt.tight_layout(); plt.show()'''),
    code('''prediction = test[['household_key','snapshot_day',target,'recency_days','active_days_90d']].copy()
prediction['probability'] = p_test
prediction['decile'] = pd.qcut(prediction.probability, 10, duplicates='drop')
calibration = prediction.groupby('decile', observed=True).agg(
    n=(target,'size'), predicted=('probability','mean'), observed=(target,'mean'))
display(calibration)
plt.plot(calibration.predicted,calibration.observed,'o-',label='model')
plt.plot([0,1],[0,1],'--',color='gray',label='ideal')
plt.xlabel('Mean predicted probability'); plt.ylabel('Observed rate'); plt.legend(); plt.show()'''),
    md('''## Capacity: chọn top-K theo từng tuần

Tính top-K riêng cho từng snapshot để phản ánh capacity của chiến dịch tuần. Một household có thể xuất hiện nhiều tuần, do đó số contact là household-snapshot, không phải số người duy nhất hay số giao dịch tăng thêm.'''),
    code('''capacity_rows = []
for fraction in (.05,.10,.20,.30,.50,1.0):
    ordered = prediction.sort_values(['snapshot_day','probability'], ascending=[True,False])
    selected = pd.concat([g.head(max(1, int(np.ceil(len(g)*fraction))))
                          for _, g in ordered.groupby('snapshot_day')])
    base = prediction[target].mean()
    capacity_rows.append({'capacity':fraction, 'contacts':len(selected),
                          'buyers_observed':int(selected[target].sum()),
                          'precision':selected[target].mean(),
                          'recall':selected[target].sum()/prediction[target].sum(),
                          'lift':selected[target].mean()/base})
capacity = pd.DataFrame(capacity_rows)
display(capacity)
capacity.plot(x='capacity',y=['precision','recall'],marker='o',title='Trade-off theo capacity mỗi tuần')
plt.ylabel('Tỷ lệ'); plt.show()'''),
    md('''## Ổn định theo thời gian và theo nhóm recency

PR-AUC phụ thuộc prevalence nên cần đọc cùng tỷ lệ positive của từng tuần. Phân tích nhóm có tính mô tả, các snapshot của cùng household có liên quan với nhau.'''),
    code('''weekly = pd.DataFrame([{'snapshot_day': day, 'rows':len(g),
    'prevalence':g[target].mean(), 'pr_auc':average_precision_score(g[target],g.probability),
    'brier':brier_score_loss(g[target],g.probability)}
    for day, g in prediction.groupby('snapshot_day')]).set_index('snapshot_day')
display(weekly)
weekly[['prevalence','pr_auc','brier']].plot(marker='o', figsize=(10,4),title='Holdout theo snapshot')
plt.show()
prediction['recency_band'] = pd.cut(prediction.recency_days,[-1,7,14,30,60,np.inf])
display(prediction.groupby('recency_band', observed=True).agg(
    rows=(target,'size'), observed=(target,'mean'), predicted=('probability','mean')))'''),
    md('''# 9. Đưa kết quả vào vận hành và giới hạn

- Ranking top-K hỗ trợ phân bổ capacity; xác suất cao không đồng nghĩa tác động của coupon cao.
- Trước pilot: xác nhận định nghĩa basket hợp lệ, data availability tại cutoff, chi phí contact và contribution margin.
- Randomized treatment/control cần thiết để đo incremental purchase/margin.
- Monitor freshness, schema, calibration, PR-AUC và lift theo thời gian khi label mature.
- Dataset là panel frequent shoppers của một retailer; không tự động suy rộng sang khách mới hoặc toàn thị trường.
- Cần dữ liệu tương lai chưa từng dùng trong quá trình phát triển để kiểm định cuối độc lập.

**Tái lập:** chạy các cell theo thứ tự từ thư mục project root hoặc `notebooks`; đầu vào là tám CSV trong `sample_data/dunnhumby`. Lần chạy đầy đủ đọc file causal khoảng 664 MB và huấn luyện 6 mô hình qua temporal folds.'''),
]


def main() -> None:
    out = Path('notebooks/customer_purchase_propensity_end_to_end.ipynb')
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
                'language': 'python', 'name': 'python3'},
                'language_info': {'name': 'python', 'version': '3.10'}},
                'nbformat': 4, 'nbformat_minor': 5}
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'Built {out}: {len(cells)} cells')


if __name__ == '__main__':
    main()
