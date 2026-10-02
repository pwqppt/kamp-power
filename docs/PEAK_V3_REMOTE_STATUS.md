# v3 원격 업로드 상태

2026-10-02. 공개 저장소 clone은 성공했으나 `git push -u origin experiments/peak-diagnosis-v3`는 GitHub 403으로 실패했다. 서버 응답: `Permission to pwqppt/kamp-power.git denied to joo-hyun`.

현재 PC의 인증 계정에 쓰기 권한이 없다. 사용자에게 저장소 쓰기 권한 부여 또는 권한 있는 계정의 Git 인증을 요청했다. 인증정보는 저장소나 대화에 넣지 않는다. 로컬의 진단·가설별 브랜치와 커밋은 보존하며 원격 완료로 표시하지 않는다. 권한 해결 전에는 동일 요청 반복으로 해결할 수 없다.

권한 해결 후 각 로컬 브랜치를 동일한 이름으로 push하고 원격 SHA를 로컬과 대조한다. 원본 실험 브랜치와 main은 변경하지 않는다. 최종 브랜치 목록은 PEAK_V3_RESULTS.md에 기록한다.
