# -*- coding: utf-8 -*-
"""1단계 — 코호트와 입원 후 24시간 노트 구성 (2절)

- 코호트: 18세 이상, 입원 24시간 시점 이전 퇴원·사망 제외. 환자 단위 70/15/15 분할.
- 노트: CHARTTIME이 입원 후 24시간 이내인 노트. 방사선 판독문도 함께 보관.
- 출력: WORK_DIR/cohort.pkl, WORK_DIR/notes_24h.pkl (노트 원문 포함 — 공유 금지)
"""
import re, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

from config import P, WORK_DIR
t0 = time.time()

# ── 1) 코호트 ────────────────────────────────────────────────────────
adm = pd.read_csv(P('ADMISSIONS'), low_memory=False)
pat = pd.read_csv(P('PATIENTS'),   low_memory=False)
for c_ in ['ADMITTIME','DISCHTIME','DEATHTIME']:
    adm[c_] = pd.to_datetime(adm[c_], errors='coerce')
pat['DOB'] = pd.to_datetime(pat.DOB, errors='coerce')
pat['DOD'] = pd.to_datetime(pat.DOD, errors='coerce')
c = adm.merge(pat[['SUBJECT_ID','GENDER','DOB','DOD']], on='SUBJECT_ID', how='left')
n0 = len(c)

c['AGE'] = np.nan
m = c.ADMITTIME.notna() & c.DOB.notna()
age = (c.loc[m,'ADMITTIME'].values.astype('datetime64[D]')
       - c.loc[m,'DOB'].values.astype('datetime64[D]')).astype('timedelta64[D]').astype(float)/365.2425
c.loc[m,'AGE'] = age
c.loc[c.AGE > 89, 'AGE'] = 90
c = c[c.AGE >= 18].copy()
n_adult = len(c)
print(f'전체 입원 {n0:,} → 18세 이상 {n_adult:,}', flush=True)

c['INDEX_TIME'] = c.ADMITTIME + pd.Timedelta(hours=24)
c = c[(c.DISCHTIME.isna() | (c.DISCHTIME >= c.INDEX_TIME))
    & (c.DEATHTIME.isna() | (c.DEATHTIME >= c.INDEX_TIME))].copy()
print(f'24시간 시점 전 퇴원·사망 제외 → {len(c):,}', flush=True)

# 라벨: 입원 후 168시간(7일) 이내 원내 사망
c['t_death_h'] = (c.DEATHTIME - c.ADMITTIME).dt.total_seconds()/3600
c['y_inhosp']  = (c.DEATHTIME.notna() & (c.t_death_h <= 168)).astype(int)

# 환자 단위 분할 70/15/15
subj = c.SUBJECT_ID.drop_duplicates().to_numpy().copy()
rng = np.random.RandomState(42); rng.shuffle(subj)
n = len(subj); tr = int(n*0.70); va = int(n*0.85)
trs, vas = set(subj[:tr]), set(subj[tr:va])
c['split'] = np.where(c.SUBJECT_ID.isin(trs), 'train',
              np.where(c.SUBJECT_ID.isin(vas), 'valid', 'test'))
c['HADM_ID'] = c.HADM_ID.astype('int64')
print('분할 환자 수: ' + ' / '.join(f'{s} {c[c.split==s].SUBJECT_ID.nunique():,}'
                                for s in ['train','valid','test']), flush=True)
print(f'코호트 {len(c):,} / 환자 {c.SUBJECT_ID.nunique():,}  '
      f'{"✅" if (len(c)==48741 and c.SUBJECT_ID.nunique()==37182) else "⚠️ 불일치"}', flush=True)

admit  = dict(zip(c.HADM_ID, c.ADMITTIME))
indexm = dict(zip(c.HADM_ID, c.INDEX_TIME))
cohort = set(c.HADM_ID)

# ── 2) 입원 후 24시간 노트 ───────────────────────────────────────────
DEID = re.compile(r'\[\*\*.*?\*\*\]'); WS = re.compile(r'\s+')
def clean(t):
    if not isinstance(t, str): return ''
    t = DEID.sub(' DEID_TOKEN ', t)
    t = re.sub(r'[-_=*]{3,}', ' ', t)
    t = t.replace('\r',' ').replace('\n',' ').replace('\t',' ')
    return WS.sub(' ', t).strip().lower()

USE = ['HADM_ID','CHARTTIME','STORETIME','CATEGORY','ISERROR','TEXT']
buf = []
for i, ck in enumerate(pd.read_csv(P('NOTEEVENTS'), usecols=USE, chunksize=200000, low_memory=False)):
    ck = ck[ck.HADM_ID.notna()]
    ck = ck[ck.ISERROR != 1]
    ck = ck[ck.CATEGORY != 'Discharge summary']
    ck = ck.copy()
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
    w['is_rad'] = w.CATEGORY.str.strip() == 'Radiology'
    w['ctext'] = w.TEXT.map(clean)
    w = w[w.ctext != '']
    buf.append(w[['HADM_ID','CATEGORY','CT','ST','is_rad','adm','idx','ctext']])
    if (i+1) % 5 == 0: print(f'  chunk {i+1} ({time.time()-t0:.0f}s)', flush=True)

W = pd.concat(buf, ignore_index=True).sort_values(['HADM_ID','CT']).reset_index(drop=True)

# 예측 대상: 방사선 외 노트 중 STORETIME이 24시간 시점 이전인 노트가 있는 입원
base = W[(~W.is_rad) & W.ST.notna() & (W.ST <= W.idx)]
print(f'예측 대상 입원 {base.HADM_ID.nunique():,} (노트 {len(base):,})  '
      f'{"✅" if len(base)==126029 and base.HADM_ID.nunique()==26122 else "⚠️ 불일치"}', flush=True)

W.to_pickle(WORK_DIR / 'notes_24h.pkl')
c.to_pickle(WORK_DIR / 'cohort.pkl')
print(f'저장 완료 ({time.time()-t0:.0f}s)')
