# 다른 AI Agent에 그대로 붙여넣을 프롬프트

공개 저장소: `https://github.com/pwqppt/kamp-power`

```text
다음 GitHub 저장소의 KAMP 제조 전력 예측 프로젝트를 기존 연구 설계와 결과를 보존하면서 이어서 진행해 줘.

저장소: https://github.com/pwqppt/kamp-power

1. 먼저 터미널에서 저장소를 clone하고 프로젝트 루트로 이동해.
   git clone https://github.com/pwqppt/kamp-power.git
   cd kamp-power

2. 코드 수정 전에 반드시 다음 파일을 순서대로 읽어.
   AGENTS.md
   PROJECT_STATUS.md
   HANDOFF.md
   docs/RESEARCH_PLAN.md
   docs/DECISIONS.md
   docs/DATA_TREATMENT.md
   docs/NEXT_STAGE.md

3. README의 방법으로 Python 가상환경을 만들고 requirements.txt를 설치한 뒤 아래 검사를 실행해.
   python run_stage.py check

4. 이미 완료된 CP00~CP03을 이유 없이 다시 돌리거나 테스트 결과를 보고 기존 모델·전처리를 재선택하지 마. 기존 결과의 핵심은 LightGBM이 테스트 MAE를 줄였지만 피크일 재현율은 SeasonalNaive보다 낮았고, 일 단위 bootstrap에서 우월성이 확정되지 않았다는 것이다.

5. 현재 다음 작업은 CP04A다. 2026년 적용 가능한 한전 공식 공급약관·산업용 요금표와 근로기준법상 야간/연장/휴일 비용을 공식 출처로 확인하고, 시행일·계약종별·적용조건·미확인 값을 구조화한 근거표와 설정 파일을 만들어. CSV의 전력 단위와 공장/설비 정보가 확인되지 않으면 추측하지 말고 조건부 가정으로 유지해.

6. CP04A를 완료한 뒤 CP04B를 진행해. 예측값만으로 0/5/10/15% 유연부하를 ±30/60분 이동시키고, 실제 평가값을 보고 이동안을 다시 최적화하지 마. 전일 에너지 보존, 비음수, 이동 범위를 검증해. 무조정/SeasonalNaive/고정 AI/완전예측 참고를 같은 조정기로 비교하고, 기본요금·에너지요금·추가 노동비·운영비를 분리해. 손실이면 조정하지 않는 것이 결론이 될 수 있다.

7. 한 단계씩만 진행해. 각 단계가 끝날 때 코드, 결과 CSV, 그림, 출처, 가정, 검증 로그를 저장하고 PROJECT_STATUS.md와 HANDOFF.md를 갱신해. 다음 형식으로 checkpoint를 남겨.
   git add .
   git commit -m "CP04A: ..."
   git tag -a cp04a -m "..."
   git push origin main --follow-tags

8. 작업이 중단되기 전에도 재현 비용이 큰 결과가 있으면 WIP 커밋을 push하고 HANDOFF.md에 마지막 성공 명령, 실패 원인, 다음 실행 명령을 적어. 원본 데이터와 기존 산출물을 임의로 삭제하거나 덮어쓰지 마.

9. 한글 시각화는 NanumGothic을 사용하고, 날짜와 15/30/45/60분 열은 하나의 15분 시간축으로 표시해. 축·범례·글자 겹침, 단위, 표본 수를 검사하고 PNG와 SVG를 모두 저장해.

10. 공개 저장소에 비밀키, 토큰, 개인정보를 넣지 마. 원본 데이터의 공개 재배포 허용 여부가 아직 확인되지 않았으므로 LICENSE로 데이터 권리를 주장하지 말고 README의 데이터 주의 문구를 유지해.

먼저 읽은 내용을 바탕으로 현재 상태, 이번에 진행할 단일 체크포인트, 예상 산출물을 짧게 보고한 다음 실제 작업을 시작해. 설명만 하고 멈추지 말고 완료 조건까지 실행해.
```
