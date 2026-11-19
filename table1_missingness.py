# -*- coding: utf-8 -*-
"""표 1 — 카테고리별 STORETIME 결측률 (입원 후 24시간 노트)

- 코호트·24시간 창: 01_build_cohort.py와 동일 (ISERROR 제외, 퇴원요약 제외).
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

# ── 24시간 노트, 카테고리별 STORETIME 결측 ───────────────────────────
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
    buf.append(w[['cat','ST']])
    if (i+1) % 5 == 0: print(f'  chunk {i+1} ({time.time()-t0:.0f}s)', flush=True)

W = pd.concat(buf, ignore_index=True)
g = W.groupby('cat').agg(n=('cat','size'), st_miss=('ST', lambda s: s.isna().sum()))
g['miss_pct'] = (g.st_miss/g.n*100).round(1)
g = g.sort_values('n', ascending=False)

n_miss = int(g.st_miss.sum())
print(f'\n24시간 노트 {len(W):,}건 중 STORETIME 결측 {n_miss:,}건 ({n_miss/len(W)*100:.1f}%)')
print('\n[카테고리별 STORETIME 결측률]')
print(g.to_string())
print(f'\n완료 ({time.time()-t0:.0f}s)')
