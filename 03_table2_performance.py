# -*- coding: utf-8 -*-
"""3단계 — 표 2: 방사선 판독문 포함 여부에 따른 성능 (valid)

- AUROC·AUPRC: 조건별 valid 전체 기준.
- 성능 차이(포함 − 제외)의 95% CI: 환자 단위 클러스터 부트스트랩 2,000회.
  같은 환자의 입원은 함께 뽑히며, 두 조건은 같은 재표본에서 비교.
- 입력: RESULT_DIR/predictions_valid.pkl (02_train_models.py)
"""
import pickle, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, average_precision_score

from config import RESULT_DIR

R = pickle.load(open(RESULT_DIR / 'predictions_valid.pkl', 'rb'))
base, fix = R['BASE'], R['FIX_d0']
y, subj = base['y_va'], base['subj_va']
assert np.array_equal(y, fix['y_va']) and np.array_equal(subj, fix['subj_va'])
p0, p1 = base['p_va'], fix['p_va']

def ci(v): return (np.percentile(v, 2.5), np.percentile(v, 97.5))

def boot_pair(y, subj, p0, p1, B=2000, seed=7):
    rs = np.random.RandomState(seed); u = np.unique(subj); pos = {s: np.where(subj == s)[0] for s in u}
    IDX = [np.concatenate([pos[s] for s in rs.choice(u, len(u), replace=True)]) for _ in range(B)]
    IDX = [i for i in IDX if y[i].min() != y[i].max()]
    da = np.empty(len(IDX)); dp = np.empty(len(IDX))
    for j, i in enumerate(IDX):
        da[j] = roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p0[i])
        dp[j] = average_precision_score(y[i], p1[i]) - average_precision_score(y[i], p0[i])
    return da, dp

print(f'valid n={len(y):,} (환자 {len(np.unique(subj)):,}명, 양성 {int(y.sum())})\n')
print(f'{"조건":<16}{"AUROC":>8}{"AUPRC":>8}')
print(f'{"방사선 제외":<16}{roc_auc_score(y, p0):>8.4f}{average_precision_score(y, p0):>8.4f}')
print(f'{"방사선 전체 포함":<16}{roc_auc_score(y, p1):>8.4f}{average_precision_score(y, p1):>8.4f}')

da, dp = boot_pair(y, subj, p0, p1)
print(f'\nΔAUROC {da.mean():+.4f} [{ci(da)[0]:+.4f}, {ci(da)[1]:+.4f}]')
print(f'ΔAUPRC {dp.mean():+.4f} [{ci(dp)[0]:+.4f}, {ci(dp)[1]:+.4f}]')
