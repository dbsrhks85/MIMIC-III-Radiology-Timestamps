# -*- coding: utf-8 -*-
"""2단계 — 방사선 판독문 제외/포함 모델 학습, valid 예측 저장 (표 2)

- 대상: 예측 대상 입원 26,122건 (두 조건 동일). 조건 간 차이는 입력 텍스트뿐.
    BASE    방사선 판독문 제외
    FIX_d0  CHARTTIME이 입원 후 24시간 이내인 방사선 판독문 전부 포함
- 모델: 단어(1,2-gram)·문자(char_wb 3~5-gram) TF-IDF 각각 로지스틱 회귀, 두 확률의 평균.
- 입력: WORK_DIR/cohort.pkl, notes_24h.pkl (01_build_cohort.py)
- 출력: RESULT_DIR/predictions_valid.pkl
"""
import time, pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from config import WORK_DIR, RESULT_DIR
T0 = time.time()

W = pd.read_pickle(WORK_DIR / 'notes_24h.pkl')
C = pd.read_pickle(WORK_DIR / 'cohort.pkl')

nonrad_ok = W[(~W.is_rad) & W.ST.notna() & (W.ST <= W.idx)]
rad       = W[W.is_rad]
BASE_HADM = set(nonrad_ok.HADM_ID)
print(f'예측 대상 입원 {len(BASE_HADM):,}', flush=True)

def build_text(parts):
    """노트 DataFrame 리스트 → {HADM_ID: 시간순 결합 텍스트}"""
    d = pd.concat(parts, ignore_index=True).sort_values(['HADM_ID','CT'])
    return d.groupby('HADM_ID').ctext.apply(' '.join).to_dict()

txt_base = build_text([nonrad_ok])
txt_rad  = build_text([nonrad_ok, rad])
CONFIGS = {
    'BASE':   txt_base,
    'FIX_d0': {h: txt_rad[h] for h in BASE_HADM},
}

CGRID = [0.1, 0.3, 0.5, 1.0, 2.0]
VECT = dict(
    word=dict(ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True, max_features=200_000),
    char=dict(analyzer='char_wb', ngram_range=(3,5), min_df=5, max_df=0.95, sublinear_tf=True, max_features=100_000),
)
def fit_branch(kind, Xtr_txt, ytr, Xva_txt, yva):
    v = TfidfVectorizer(**VECT[kind])
    Xtr = v.fit_transform(Xtr_txt); Xva = v.transform(Xva_txt)
    res = []
    for Cv in CGRID:
        lr = LogisticRegression(C=Cv, class_weight='balanced', solver='liblinear', max_iter=2000)
        lr.fit(Xtr, ytr)
        res.append((Cv, average_precision_score(yva, lr.predict_proba(Xva)[:,1]), lr))
    bC, bAP, bM = max(res, key=lambda t: t[1])
    return bM.predict_proba(Xva)[:,1]

R = {}
for name, txt in CONFIGS.items():
    t1 = time.time()
    d = C[C.HADM_ID.isin(BASE_HADM)].copy()
    d['text'] = d.HADM_ID.map(lambda h: txt.get(h, ''))
    tr, va = (d[d.split==s] for s in ['train','valid'])
    ytr = tr.y_inhosp.to_numpy().copy()     # .copy(): pandas 3.x 읽기전용 배열 → liblinear 오류 방지
    yva = va.y_inhosp.to_numpy().copy()
    p = [fit_branch(k, tr.text.tolist(), ytr, va.text.tolist(), yva) for k in ['word','char']]
    R[name] = dict(y_va=yva, p_va=(p[0]+p[1])/2, subj_va=va.SUBJECT_ID.to_numpy().copy())
    print(f'{name:<7} valid AUROC {roc_auc_score(yva, R[name]["p_va"]):.4f}  '
          f'AUPRC {average_precision_score(yva, R[name]["p_va"]):.4f}  ({time.time()-t1:.0f}s)', flush=True)

pickle.dump(R, open(RESULT_DIR / 'predictions_valid.pkl', 'wb'))
print(f'저장 완료 ({time.time()-T0:.0f}s)')
