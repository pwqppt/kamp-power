# H2 결과: 미채택

설정과 채택 조건: [PEAK_V3_PROTOCOL.md](PEAK_V3_PROTOCOL.md). 설정 재탐색 없음.

| 지표 | 4~6월 기존 결합 | 4~6월 H2 | 7월 H2 사후 평가 |
|---|---:|---:|---:|
| 평균 MAE | 16.910845 | 17.153770 | 10.887849 |
| 일최대 MAE | 19.992784 | 20.392214 | 20.561440 |
| 피크일 재현율 | 0.295455 | 0.409091 | 0.750000 |
| 구간 피크 재현율 | 0.167553 | 0.159574 | 0.486425 |

미충족 조건: mean_mae_preserved, daily_peak_mae_improved, interval_recall_not_worse.
월별 악화와 모든 오경보·미탐 수는 `outputs/peak_v3/H2/fold_scores.csv`에 보존했다.
전체 예측과 불확실성 구간은 같은 폴더의 `predictions.csv.gz`, `bootstrap.json`에 있다.
7월 결과로 판정을 바꾸지 않았다. 반복 사용된 과거 데이터이므로 독립 검증이나 배포 승인이 아니다.

다음 작업: H3: 과거 OOF 잔차 보정. 실행: `.venv/Scripts/python.exe -X utf8 peak_study.py H3`.
