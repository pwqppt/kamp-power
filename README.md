# KAMP 전력 예측 연구 — 시작 안내

이 저장소는 여러 AI agent와 컴퓨터 사이에서 이어가는 연구의 기준 저장소입니다. **먼저 `PROJECT_STATUS.md`, `HANDOFF.md`, `docs/RESEARCH_PLAN.md`를 읽으세요.**

> 공개 저장소 주의: `data/original.csv`는 대회에서 제공된 원본을 보존한 파일입니다. 재배포 허용 조건을 아직 확인하지 못했습니다. 저장소 공개를 대회 데이터의 별도 이용 허락으로 해석하면 안 되며, 권리자 요청이나 대회 규정 확인 결과에 따라 데이터 파일을 내릴 수 있습니다. 코드·문서의 라이선스도 아직 별도로 부여하지 않았습니다.

## 연구 목표
전날 16시에 다음 날 96개 15분 구간의 전력값을 예측하고, 실패 조건을 분석한 뒤, 제한적인 부하 이동이 비용 면에서 유리한 조건을 검토합니다. 실제 설비의 제어 가능성과 생산량 보존은 이 데이터만으로 검증할 수 없습니다.

## 폴더 안내
| 경로 | 역할 |
|---|---|
| `PROJECT_STATUS.md` | 단계별 완료 상태와 완료 기준 |
| `HANDOFF.md` | 새 AI/작업자가 바로 이어받을 내용 |
| `docs/RESEARCH_PLAN.md` | 전체 연구 목표, 단계, 체크포인트 운영 규칙 |
| `docs/AGENT_START_PROMPT.md` | 다른 AI 계정에 복사할 clone·인수인계 프롬프트 |
| `AGENTS.md`, `CLAUDE.md` | 작업 규칙과 읽기 순서 |
| `docs/DECISIONS.md` | 확정 조건, 가정, 변경 원칙 |
| `docs/DATA_TREATMENT.md` | 결측치·이상치 처리와 영향 |
| `docs/NEXT_STAGE.md` | 경제성 분석 구현 조건 |
| `data/original.csv` | 제공 ZIP에서 추출한 원본, 수정 금지 |
| `data/quarter_hour.csv` | 하나의 시간축으로 변환한 데이터 |
| `audit.py`, `experiment.py`, `diagnostics.py` | 01~03 단계 실행 코드 |
| `visualize.py` | 현재 결과를 한글 그림으로 생성 |
| `outputs/` | 모델, 예측 CSV, 비교 실험표, 진단 결과 |
| `figures/` | 보고서용 PNG/SVG와 시각화 검사 결과 |
| `logs/` | 단계별 실행 기록 |
| `results_review.ipynb` | 저장된 결과 검토용 미실행 notebook (현재 환경 커널 소켓 제한) |

## 환경과 실행
Python 3.12를 사용했습니다. 다른 컴퓨터의 가상환경은 다시 만드세요. `.venv`는 배포하지 않습니다.

```bash
python -m venv .venv
# macOS/Linux/WSL
source .venv/bin/activate
# Windows PowerShell에서는 .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_stage.py check
```

결과 검토만 하려면 재학습할 필요가 없습니다. 단계별 재실행은 다음과 같습니다. 기존 산출물은 덮어쓰므로 실험을 변경할 때는 폴더를 복사하거나 별도 실험 폴더를 사용하세요.

```bash
python run_stage.py 01
python run_stage.py 02
python run_stage.py 03
python run_stage.py figures
```

04 경제성 코드와 05 최종 제출 보고서는 아직 구현하지 않았습니다. 현 단계 결과를 최종 성능이나 검증된 절감액으로 표현하지 마세요. 숫자의 단위도 원본 설명에서 확인되지 않았으므로 현재 그래프는 ‘원자료 단위’를 사용합니다.

단계가 끝날 때마다 `PROJECT_STATUS.md`와 `HANDOFF.md`를 고친 뒤 커밋·태그·push합니다. 자세한 명령과 중단 시 기록 방식은 `docs/RESEARCH_PLAN.md`에 있습니다.
