# H1 결과: 미채택

설정과 채택 조건: [PEAK_V3_PROTOCOL.md](PEAK_V3_PROTOCOL.md). 설정 재탐색 없음.

| 지표 | 4~6월 기존 결합 | 4~6월 H1 | 7월 H1 사후 평가 |
|---|---:|---:|---:|
| 평균 MAE | 16.910845 | 17.107678 | 10.935968 |
| 일최대 MAE | 19.992784 | 20.633839 | 20.441619 |
| 피크일 재현율 | 0.295455 | 0.340909 | 0.850000 |
| 구간 피크 재현율 | 0.167553 | 0.140957 | 0.468326 |

미충족 조건: mean_mae_preserved, daily_peak_mae_improved, interval_recall_not_worse.
월별 악화와 모든 오경보·미탐 수는 `outputs/peak_v3/H1/fold_scores.csv`에 보존했다.
전체 예측과 불확실성 구간은 같은 폴더의 `predictions.csv.gz`, `bootstrap.json`에 있다.
7월 결과로 판정을 바꾸지 않았다. 반복 사용된 과거 데이터이므로 독립 검증이나 배포 승인이 아니다.

다음 작업: H2: 고정 비대칭 손실. 실행: `.venv/Scripts/python.exe peak_study.py H2`.
