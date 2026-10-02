# v3 원격 업로드 상태

## 최신: 권한 해결, 가설별 push 완료

사용자가 collaborator 권한을 추가한 후 다음 네 브랜치의 push가 성공했다. 원격 저장소는 `pwqppt/kamp-power`이며 main과 출발 브랜치는 수정하지 않았다.

| 브랜치 | 종료 커밋 |
|---|---|
| experiments/peak-diagnosis-v3 | e81dd1c |
| experiments/peak-v3-h1-weighted | 2db0d35 |
| experiments/peak-v3-h2-asymmetric | 7296041 |
| experiments/peak-v3-h3-calibration | d3e7ecf |

종합 브랜치 `experiments/peak-v3-summary`는 최종 검사 후 push한다. 아래는 최초 거부 이력으로 현재 권한 상태가 아니다.

## 최초 실패 이력

2026-10-02. 공개 저장소 clone은 성공했으나 `git push -u origin experiments/peak-diagnosis-v3`는 GitHub 403으로 실패했다. 서버 응답: `Permission to pwqppt/kamp-power.git denied to joo-hyun`.

현재 PC의 인증 계정에 쓰기 권한이 없다. 사용자에게 저장소 쓰기 권한 부여 또는 권한 있는 계정의 Git 인증을 요청했다. 인증정보는 저장소나 대화에 넣지 않는다. 로컬의 진단·가설별 브랜치와 커밋은 보존하며 원격 완료로 표시하지 않는다. 권한 해결 전에는 동일 요청 반복으로 해결할 수 없다.

권한 해결 후 각 로컬 브랜치를 동일한 이름으로 push하고 원격 SHA를 로컬과 대조한다. 원본 실험 브랜치와 main은 변경하지 않는다. 최종 브랜치 목록은 PEAK_V3_RESULTS.md에 기록한다.
