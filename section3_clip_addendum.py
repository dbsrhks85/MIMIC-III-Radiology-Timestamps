# -*- coding: utf-8 -*-
"""3절 — 같은 검사(Clip#)에 속한 방사선 판독문의 CHARTTIME 비교, 원본–addendum 157쌍

- 대상: 전체 방사선 판독문 중 HADM_ID·CHARTTIME이 있는 노트.
- 같은 입원·같은 검사 번호(Clip#)로 묶음.
- 원본–addendum 쌍: 묶음 안에 원본이 정확히 1건, addendum이 1건 이상인 경우만 사용.
- 출력: RESULT_DIR/section3_clip_addendum.json (집계값만)
"""
import re, time, json
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

from config import P, RESULT_DIR
t0 = time.time()
CLIP_RE = re.compile(r'Clip # \[\*\*Clip Number \(Radiology\) (\d+)\*\*\]')
ADDENDUM_RE = re.compile(r'addendum', re.I)

USE = ['ROW_ID','HADM_ID','CATEGORY','CHARTTIME','TEXT']
buf = []
for i, ck in enumerate(pd.read_csv(P('NOTEEVENTS'), usecols=USE, chunksize=200_000, low_memory=False)):
    ck = ck[ck.CATEGORY.str.strip() == 'Radiology']
    if ck.empty: continue
    ck = ck[ck.HADM_ID.notna()].copy()
    if ck.empty: continue
    ck['HADM_ID'] = ck.HADM_ID.astype('int64')
    ck['CT'] = pd.to_datetime(ck.CHARTTIME, errors='coerce')
    ck = ck[ck.CT.notna()]
    ck['clip_id'] = ck.TEXT.str.extract(CLIP_RE, expand=False)
    ck['is_addendum'] = ck.TEXT.str.contains(ADDENDUM_RE, regex=True, na=False)
    buf.append(ck[['ROW_ID','HADM_ID','CT','clip_id','is_addendum']])
    if (i+1) % 5 == 0: print(f'  chunk {i+1} ({time.time()-t0:.0f}s)', flush=True)

R = pd.concat(buf, ignore_index=True)
ok = R.clip_id.notna()
Rc = R[ok]

n_multi_groups = n_multi_notes = n_same_ct_notes = 0
n_pairs = n_pairs_same_ct = n_pairs_diff_row = 0
for key, sub in Rc.groupby(['HADM_ID', 'clip_id']):
    if len(sub) < 2: continue
    # 같은 검사 묶음 안의 CHARTTIME 일치
    n_multi_groups += 1; n_multi_notes += len(sub)
    if sub.CT.nunique() == 1: n_same_ct_notes += len(sub)
    # 원본–addendum 쌍
    if not sub.is_addendum.any(): continue
    orig = sub[~sub.is_addendum]
    if len(orig) != 1: continue
    o = orig.iloc[0]
    for _, a in sub[sub.is_addendum].iterrows():
        n_pairs += 1
        n_pairs_same_ct  += int(a.CT == o.CT)
        n_pairs_diff_row += int(a.ROW_ID != o.ROW_ID)

res = dict(radiology_notes=len(R), clip_parsed=int(ok.sum()),
           multi_note_groups=n_multi_groups, notes_in_multi_groups=n_multi_notes,
           notes_same_charttime=n_same_ct_notes,
           pairs=n_pairs, pairs_same_charttime=n_pairs_same_ct, pairs_different_row_id=n_pairs_diff_row)

print(f'\n방사선 판독문 {len(R):,}건 중 Clip# 확인 {ok.sum():,}건 ({ok.mean()*100:.1f}%)')
print(f'한 검사에 판독문이 여러 건인 묶음 {n_multi_groups:,}개 (노트 {n_multi_notes:,}건)')
print(f'  묶음 내 CHARTTIME 동일: {n_same_ct_notes:,} / {n_multi_notes:,} ({n_same_ct_notes/n_multi_notes*100:.2f}%)')
print(f'원본–addendum 쌍 {n_pairs}개')
print(f'  CHARTTIME 일치: {n_pairs_same_ct} / {n_pairs} ({n_pairs_same_ct/n_pairs*100:.0f}%)')
print(f'  서로 다른 행(ROW_ID): {n_pairs_diff_row} / {n_pairs}')

json.dump(res, open(RESULT_DIR / 'section3_clip_addendum.json', 'w'), indent=1)
print(f'\n완료 ({time.time()-t0:.0f}s)')
