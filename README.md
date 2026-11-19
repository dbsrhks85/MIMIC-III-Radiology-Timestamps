# MIMIC-III 방사선 판독문 타임스탬프의 정량적 특성화

MIMIC-III NOTEEVENTS의 두 시각 필드(CHARTTIME, STORETIME)를 방사선 판독문 중심으로 분석합니다.

1. 노트 종류별 STORETIME 결측률
2. 같은 영상 검사(Clip#)에 속한 판독문의 CHARTTIME 비교 (원본–addendum 157쌍)
3. 노트 종류별 저장 지연(STORETIME − CHARTTIME)
4. 방사선 판독문 포함 여부에 따른 예측 성능 비교

## 데이터

- [MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/)를 사용합니다. PhysioNet 자격 인증과 데이터 사용 동의(DUA)가 필요하며, **이 저장소에는 데이터가 포함되어 있지 않습니다.**
- 사용 테이블: `ADMISSIONS.csv`, `PATIENTS.csv`, `NOTEEVENTS.csv` (압축을 푼 CSV)
- 실행 중 만들어지는 `work/`, `results/`에는 MIMIC에서 추출한 데이터가 들어 있습니다. DUA에 따라 **공유하거나 커밋하지 마세요** (`.gitignore`로 제외되어 있습니다).

## 설치

```bash
pip install -r requirements.txt   # Python 3.12
```

## 경로 설정

경로는 `config.py`가 환경변수로 읽습니다. 지정하지 않으면 저장소 안의 기본 폴더를 씁니다.

| 환경변수 | 내용 | 기본값 |
|---|---|---|
| `MIMIC_DIR` | MIMIC-III CSV 폴더 | `data/mimic-iii/` |
| `WORK_DIR` | 중간 산출물 | `work/` |
| `RESULT_DIR` | 결과 | `results/` |

```bash
export MIMIC_DIR=/path/to/mimic-iii-clinical-database-1.4
```

## 실행

```bash
# 표 2 — 순서대로 실행
python 01_build_cohort.py          # 코호트·입원 후 24시간 노트 구성
python 02_train_models.py          # 방사선 판독문 제외/포함 모델 학습
python 03_table2_performance.py    # AUROC·AUPRC, 성능 차이와 95% CI

# 표 1·그림 1
python table1_missingness.py       # STORETIME 결측률
python table1_storage_delay.py     # 저장 지연

# 3절
python section3_clip_addendum.py   # Clip#·addendum CHARTTIME 비교
```

`table1_*`, `section3_*`는 원자료만 읽으므로 따로 실행할 수 있습니다.

## 분석 설정

- **코호트**: 18세 이상, 입원 24시간 시점 이전 퇴원·사망 제외 → 입원 48,741건(환자 37,182명). 환자 단위 무작위 분할 70/15/15.
- **예측 대상**: 입원 후 24시간 노트 중 방사선 판독문을 제외하고 저장 시각이 24시간 시점 이전인 노트가 있는 입원 26,122건(노트 126,029건).
- **예측 과제**: 입원 후 24시간 노트로 7일 이내 원내 사망 예측.
- **모델**: 단어·문자 단위 TF-IDF + 로지스틱 회귀.
- **평가**: 검증 집합(valid)에서 AUROC·AUPRC. 성능 차이의 95% 신뢰구간은 환자 단위 부트스트랩 2,000회.

## 논문 수치 대응표

| 스크립트 | 논문 위치 | 수치 |
|---|---|---|
| `01_build_cohort.py` | 2절 | 입원 58,976 → 50,765 → 48,741건(환자 37,182명), 분할 환자 26,027/5,577/5,578명, 예측 대상 입원 26,122건(노트 126,029건) |
| `03_table2_performance.py` | 표 2 | AUROC/AUPRC: 방사선 제외 0.9023/0.4686, 전체 포함 0.9078/0.4783 · ΔAUROC +0.0055 [−0.0029, +0.0137] · ΔAUPRC +0.0096 [−0.0113, +0.0301] (valid 입원 3,874건, 환자 3,207명, 양성 216) |
| `table1_missingness.py` | 1~3절, 표 1 | 24시간 노트 197,747건 중 STORETIME 결측 66,397건(33.6%), 카테고리별 n·결측률 |
| `table1_storage_delay.py` | 3절, 표 1, 그림 1 | 카테고리별 지연(50%)·지연(95%) |
| `section3_clip_addendum.py` | 3절 | 방사선 판독문 378,920건 중 Clip# 333,712건(88.1%), 여러 건이 달린 검사 15,458건(노트 30,916건) 중 CHARTTIME 일치 99.99%(30,914/30,916), 원본–addendum 157쌍 CHARTTIME 일치 100%·모두 서로 다른 행(ROW_ID) |

그림 1은 `table1_storage_delay.py`의 지연(95%) 값으로 별도 작성했습니다.

## 인용

```
윤관, 김유섭, "MIMIC-III 방사선 판독문 타임스탬프의 정량적 특성화,"
한국정보처리학회 ACK 2026, 2026.
```

지도교수: 김유섭 교수님

## 사사

본 연구는 2026년도 과학기술정보통신부 및 정보통신기획평가원의 ‘SW중심대학사업’ 지원을 받아 수행되었음(2024-0-00064)

## 라이선스

코드는 [MIT License](LICENSE)를 따릅니다. MIMIC-III 데이터의 이용은 PhysioNet 데이터 사용 동의를 따릅니다.
