# 결과물 제작 소스

연구 재현은 루트 `run_submission.py all`과 제출 ZIP으로 실행한다. 이 디렉터리는 PDF/PPTX 제작 소스이며 연구 모델의 실행 의존성과 분리했다.

- `../make_report.py`: reportlab/Pillow가 있는 Python으로 공식 6장 보고서를 생성. Windows HY신명조/맑은고딕 사용. 공식 휴먼명조는 환경에 없어 대체했음을 보고서에 명시했다.
- `build_slides.mjs`: `@oai/artifact-tool`로 편집 가능한 native 표·차트·텍스트를 생성. 실행 전 `RESEARCH_ROOT`, `PRESENTATION_SKILL_ROOT`, `ARTIFACT_PYTHON`, `RUNTIME_NODE_MODULES` 설정. 런타임의 node_modules를 Node 모듈 경로로 연결한다. 기존 최종 파일을 덮어쓰지 않으므로 새 버전을 만들 때 파일명을 바꾼다.
- `export_slide_pdf.py`: 최종 PPTX를 재수입해 렌더링한 13개 PNG를 동일 비율 PDF로 결합. PDF는 시각 사본이며 편집은 PPTX에서 한다.

최종 검사 기록: `docs/SELF_REVIEW_V4.md`, `outputs/submission/quality_verification.json`. 런타임이 내보내는 구조검사 성공은 실제 PowerPoint에서의 실행 검증을 의미하지 않는다. 이 환경에서는 artifact-tool 렌더와 PDF 렌더를 확인했다.
