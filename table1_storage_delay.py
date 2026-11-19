# -*- coding: utf-8 -*-
"""표 1·그림 1 — 카테고리별 저장 지연 (STORETIME − CHARTTIME, 입원 후 24시간 노트)

- 대상: 24시간 노트 중 방사선 외, STORETIME 있는 노트.
- 지연(50%)·지연(95%) 단위: 시간. 노트 1,000건 미만 카테고리는 비교에서 제외(*).
- 코호트·24시간 창: 01_build_cohort.py와 동일.
"""
import time
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

from config import P
t0 = time.time()

# ── 코호트 (01_build_cohort.py와 동일) ───────────────────────────────
adm = pd.read_csv(P('ADMISSIONS'), low_memory=False)
pat = pd.read_csv(P('PATIENTS'),   low_memory=False)
for c_ in ['ADMITTIME','DISCHTIME','DEATHTIME']:
    adm[c_] = pd.to_datetime(adm[c_], errors='coerce')
pat['DOB'] = pd.to_datetime(pat.DOB, errors='coerce')
c = adm.merge(pat[['SUBJECT_ID','DOB']], on='SUBJECT_ID', how='left')
m = c.ADMITTIME.notna() & c.DOB.notna()
age = (c.loc[m,'ADMITTIME'].values.astype('datetime64[D]')
       - c.loc[m,'DOB'].values.astype('datetime64[D]')).astype('timedelta64[D]').astype(float)/365.2425
c['AGE'] = np.nan; c.loc[m,'AGE'] = age; c.loc[c.AGE > 89, 'AGE'] = 90
c = c[c.AGE >= 18].copy()
c['INDEX_TIME'] = c.ADMITTIME + pd.Timedelta(hours=24)
c = c[(c.DISCHTIME.isna() | (c.DISCHTIME >= c.INDEX_TIME))
    & (c.DEATHTIME.isna() | (c.DEATHTIME >= c.INDEX_TIME))].copy()
c['HADM_ID'] = c.HADM_ID.astype('int64')
print(f'코호트 {len(c):,} (기대 48,741) {"✅" if len(c)==48741 else "⚠️"}')
admit  = dict(zip(c.HADM_ID, c.ADMITTIME))
indexm = dict(zip(c.HADM_ID, c.INDEX_TIME))
cohort = set(c.HADM_ID)

# ── 24시간 노트 ──────────────────────────────────────────────────────
USE = ['HADM_ID','CHARTTIME','STORETIME','CATEGORY','ISERROR']
buf = []
for i, ck in enumerate(pd.read_csv(P('NOTEEVENTS'), usecols=USE, chunksize=200_000, low_memory=False)):
    ck = ck[ck.HADM_ID.notna()]
    ck = ck[ck.ISERROR != 1]
    ck = ck[ck.CATEGORY.str.strip() != 'Discharge summary'].copy()
    ck['CT'] = pd.to_datetime(ck.CHARTTIME, errors='coerce')
    ck = ck[ck.CT.notna()]
    if ck.empty: continue
    ck['HADM_ID'] = ck.HADM_ID.astype('int64')
    ck = ck[ck.HADM_ID.isin(cohort)]
    if ck.empty: continue
    ck['adm'] = ck.HADM_ID.map(admit); ck['idx'] = ck.HADM_ID.map(indexm)
    w = ck[(ck.CT >= ck.adm) & (ck.CT <= ck.idx)].copy()
    if w.empty: continue
    w['ST'] = pd.to_datetime(w.STORETIME, errors='coerce')
    w['cat'] = w.CATEGORY.str.strip()
    buf.append(w[['cat','CT','ST']])
    if (i+1) % 5 == 0: print(f'  chunk {i+1} ({time.time()-t0:.0f}s)', flush=True)

W = pd.concat(buf, ignore_index=True)
nr = W[(W.cat != 'Radiology') & W.ST.notna()].copy()
nr['delay_h'] = (nr.ST - nr.CT).dt.total_seconds()/3600

rows = []
for cat, sub in nr.groupby('cat'):
    rows.append(dict(category=cat, n=len(sub),
                     delay_p50_h=sub.delay_h.quantile(.50),
                     delay_p95_h=sub.delay_h.quantile(.95),
                     excluded='*' if len(sub) < 1000 else ''))
tbl = pd.DataFrame(rows).sort_values('n', ascending=False).reset_index(drop=True)
print(f'\n방사선 외 · STORETIME 있는 24시간 노트 {len(nr):,}건')
print('\n[카테고리별 저장 지연, 시간]  (* n<1,000, 비교 대상에서 제외)')
print(tbl.round(4).to_string(index=False))
print(f'\n완료 ({time.time()-t0:.0f}s)')
