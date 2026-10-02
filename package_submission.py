"""Build a blind, portable research source archive; optional clean reproduction tree."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parent
S=ROOT/'outputs/submission'

README='''# 전력 예측 및 피크 분석 재현

## 환경과 실행
Python 3.13.1에서 검증. 새 가상환경에서 다음 명령을 실행한다.
```
python -m pip install -r requirements.txt
python -X utf8 run_submission.py all
```
Jupyter에서는 reproduce.ipynb를 순서대로 실행한다. 전체 연구는 수분 이상 걸린다. CPU 모델만 사용하며 네트워크는 패키지 설치 외에 필요 없다. 현재 저장된 결과는 expected_results에 보존되어 재실행 결과와 독립적으로 비교할 수 있다.

## 구성
- data/original.csv: 공식 학습용 원자료, 변경하지 않는다.
- outputs/adaptive_v2: 과거 실행과의 차이 진단을 위한 이전 결과 두 파일. 모델 훈련·새 후보 채택에 사용하지 않는다.
- outputs/submission/test_predictions.csv: 고정된 모델의 8/1~9/14 4,320개 구간 예측.
- test_daily_peak_predictions.csv: 별도 일최대 출력. 96구간의 최대와 같을 필요는 없다.
- expected_results: 전체 설정·결과·실패·그림. 전체 실행으로 outputs에 새 결과 생성.
- docs: 사전 설계와 결과·외부자료 출처·평가 추적표.

## 해석
D-1 16시 이후 실측 기온·생산량·전력은 입력에 금지한다. 4~6월 expanding 검증으로 판단하고 7월은 이미 본 사후 진단이다. 8~9월도 과거 연구에서 이미 사용했으므로 새 독립 테스트가 아니다. H6는 구간 곡선의 성능 변화 없이 별도 일최대 MAE를 개선한다. 새 현장 검증·통계적 우위는 미확정이다. 이동 시뮬레이션의 생산량 보존은 조건부 장부이며 설비별 실행 가능성의 증거가 아니다. 요금은 2026 단가를 2021 부하에 가상 적용한다.

## 제출 형식
이 ZIP은 연구 재현용이다. 보고서·발표자료는 별도 PDF/PPTX로 제공한다. 팀 정보·서명·실제 설문 완료 화면은 참가자가 제공해야 한다. 원문 공지/양식의 예시 개인 정보와 계정·컴퓨터 경로는 포함하지 않는다. FILE_MANIFEST.json은 배포 파일의 SHA256이다.
'''

def build(stage):
    stage.mkdir(parents=True,exist_ok=False)
    names=['audit','experiment','context_features','residual_experiment','continue_research','peak_study','check_peak_study','peak_report','research_v4','peak_v4','peak_head','economics_v4','run_submission','final_evidence']
    for name in names:shutil.copy2(ROOT/(name+'.py'),stage)
    shutil.copy2(ROOT/'requirements.txt',stage)
    (stage/'data').mkdir();shutil.copy2(ROOT/'data/original.csv',stage/'data')
    for d in ['context_v2','adaptive_v2','submission']:(stage/'outputs'/d).mkdir(parents=True)
    for name in ['selected_july.csv','fold_scores.csv']:shutil.copy2(ROOT/'outputs/adaptive_v2'/name,stage/'outputs/adaptive_v2'/name)
    for name in ['test_predictions.csv','test_daily_peak_predictions.csv','test_export.json']:shutil.copy2(S/name,stage/'outputs/submission'/name)
    for name in ['reproduction_verification.json','quality_verification.json']:
        if (S/name).exists():shutil.copy2(S/name,stage/'outputs/submission'/name)
    (stage/'docs').mkdir()
    for pattern in ['PEAK_V3*','PEAK_V4*','PEAK_HEAD*','H6_RESULTS*','RESEARCH_V4*','EXTERNAL_DATA_REGISTER*','EVALUATION_TRACEABILITY*','SELF_REVIEW_V4*']:
        for f in (ROOT/'docs').glob(pattern):
            if f.name=='PEAK_V3_REMOTE_STATUS.md':continue  # Account history is not submission evidence.
            shutil.copy2(f,stage/'docs'/f.name)
    for name in ['HANDOFF.md','PROJECT_STATUS.md']:(stage/name).write_text('# 연구 재현 기록\n\n결과 해석 및 제약은 README.md와 docs를 참조한다.\n',encoding='utf8')
    for d in ['peak_v3','peak_v4','research_v4']:shutil.copytree(ROOT/'outputs'/d,stage/'expected_results'/d)
    (stage/'README.md').write_text(README,encoding='utf8')
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# 전체 연구 재현\nREADME의 고정 패키지를 먼저 설치한다. 아래 셀은 원자료부터 재실행하고, 두 번째 셀은 결과를 표시한다. 이미 관측된 검증기간을 새 독립 테스트로 해석하지 않는다.'),
        nbformat.v4.new_code_cell("import subprocess, sys\nfrom pathlib import Path\nlog = Path('reproduction.log')\nwith log.open('w', encoding='utf8') as f:\n    subprocess.run([sys.executable, '-X', 'utf8', 'run_submission.py', 'all'], stdout=f, stderr=subprocess.STDOUT, check=True)\nprint('전체 재현 완료. reproduction.log 참조.')"),
        nbformat.v4.new_code_cell("import pandas as pd\ndisplay(pd.read_csv('outputs/peak_v3/comparison.csv'))\ndisplay(pd.read_csv('outputs/peak_v4/H6/scores.csv'))")],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nbformat.write(nb,stage/'reproduce.ipynb')
    executed=ROOT/'work/reproduce-check/reproduce_executed.ipynb'
    if executed.exists():
        done=nbformat.read(executed,as_version=4)
        assert all(c.get('execution_count') is not None for c in done.cells if c.cell_type=='code')
        assert not any(o.output_type=='error' for c in done.cells if c.cell_type=='code' for o in c.get('outputs',[]))
        nbformat.write(done,stage/'reproduce_executed.ipynb')
    verification=S/'reproduction_verification.json'
    if verification.exists():shutil.copy2(verification,stage)
    manifest={f.relative_to(stage).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(stage.rglob('*')) if f.is_file()}
    (stage/'FILE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    with zipfile.ZipFile(S/'research_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(stage.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(stage).as_posix())
    print('Packaged',len(manifest),'files',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args();build(Path(a.stage))
