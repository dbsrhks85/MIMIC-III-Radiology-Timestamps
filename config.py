# -*- coding: utf-8 -*-
"""경로 설정. 환경변수로 지정, 없으면 저장소 안 기본 폴더 사용.

  MIMIC_DIR   MIMIC-III v1.4 CSV 폴더 (ADMISSIONS, PATIENTS, NOTEEVENTS)
  WORK_DIR    중간 산출물 (노트 원문 포함 — 공유·커밋 금지)
  RESULT_DIR  집계 결과 pkl/json
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent

MIMIC_DIR  = Path(os.environ.get('MIMIC_DIR',  REPO / 'data' / 'mimic-iii'))
WORK_DIR   = Path(os.environ.get('WORK_DIR',   REPO / 'work'))
RESULT_DIR = Path(os.environ.get('RESULT_DIR', REPO / 'results'))

for _d in (WORK_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def P(name):
    """테이블 이름 → CSV 경로. 'X.csv/X.csv'(압축 해제 폴더) 구조도 지원."""
    nested = MIMIC_DIR / f'{name}.csv' / f'{name}.csv'
    return nested if nested.exists() else MIMIC_DIR / f'{name}.csv'
