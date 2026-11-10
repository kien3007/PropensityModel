import ast
import json
from pathlib import Path

import pandas as pd


def test_rfm_uses_history_only_and_retains_inactive_households():
    notebook=json.loads(Path('notebooks/phan_tich_du_lieu_propensity.ipynb').read_text(encoding='utf-8'))
    source=next(''.join(c['source']) for c in notebook['cells']
                if c['cell_type']=='code' and 'rfm_cutoff = 537' in ''.join(c['source']))
    ast.parse(source)
    basket=pd.DataFrame({'household_key':[1,1,1,1,2], 'BASKET_ID':[1,2,3,4,5],
                         'DAY':[447,448,537,538,447], 'sales':[50.,20.,0.,99.,10.]})
    panel=pd.DataFrame({'household_key':[1,2], 'snapshot_day':[537,537],
                        'recency_days':[0,90], 'target_any_basket':[1,0]})
    first_day=pd.Series({1:447,2:447})
    scope={'pd':pd,'basket':basket,'panel':panel,'first_day':first_day,'display':lambda *args:None}
    exec(compile(source,'<notebook>','exec'),scope)
    result=scope['rfm'].set_index('household_key')
    assert result.loc[1,'F_baskets_90d']==2
    assert result.loc[1,'M_sales_90d']==20
    assert result.loc[1,'active_days_90d']==2
    assert result.loc[2,'F_baskets_90d']==0
    assert result.loc[2,'M_sales_90d']==0
