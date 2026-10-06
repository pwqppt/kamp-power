"""Build Korean figures and a source-backed interpretation from saved results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import koreanize_matplotlib
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/input_reliability_v1';FIG=ROOT/'figures/input_reliability_v1';FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':11,'svg.fonttype':'none','axes.unicode_minus':False})
names={'recent':'최근 전력 + 달력','F0':'기본 전력 이력','F1':'전력 이력 + 최근 통계','production':'최근 통계 + 생산량','temperature':'최근 통계 + 기온','humidity':'최근 통계 + 습도','weather':'최근 통계 + 기온·습도','personnel':'최근 통계 + 인원','all':'최근 통계 + 부가정보 전체'}
order=list(names);pooled=pd.read_csv(OUT/'pooled_metrics.csv');reg=pooled[pooled.task.eq('regression')];clf=pooled[pooled.task.eq('classification')&pooled.scope.eq('45_120')]
fig,axs=plt.subplots(1,2,figsize=(12,6.5),sharey=True)
for ax,period,title in zip(axs,['AprJun','July'],['4~6월 검증 · 91일','7월 재사용 진단 · 29일']):
 q=reg[reg.period.eq(period)].set_index('model').loc[order];colors=['#1b70a6' if n=='F0' else '#d28632' if n=='F1' else '#aebfca' for n in order]
 ax.barh(np.arange(len(order)),q.MAE,color=colors,height=.63)
 for i,v in enumerate(q.MAE):ax.text(v+.07,i,f'{v:.3f}',va='center',fontsize=10)
 ax.set_xlim(0,11.5);ax.set_title(title,pad=14);ax.set_xlabel('평균 절대오차 (원자료 단위, 작을수록 좋음)');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
axs[0].set_yticks(range(len(order)),[names[n] for n in order]);axs[0].invert_yaxis()
fig.suptitle('입력을 추가해도 전력값 예측이 항상 좋아지지는 않았다',fontsize=16,y=.97)
fig.text(.28,.025,'모든 구성: 동일 LightGBM · 향후 15~120분 예측 · 미래 실측 입력 없음\n기본 전력 이력 = 최근 8구간 + 전일·전주 동일시각 + 달력',fontsize=10,ha='left')
fig.subplots_adjust(left=.26,right=.97,top=.85,bottom=.17,wspace=.16)
fig.savefig(FIG/'input_ablation_mae.png',dpi=160);fig.savefig(FIG/'input_ablation_mae.svg');plt.close(fig)

def mdtable(df):
 def fmt(v):return f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v)
 lines=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
 lines += ['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]
 return '\n'.join(lines)
t=reg.pivot(index='model',columns='period',values='MAE').loc[order].reset_index();t['model']=t.model.map(names);t=t.rename(columns={'model':'입력 구성','AprJun':'4~6월 MAE','July':'7월 MAE'})
ct=clf[clf.model.isin(['F0','F1','production','all'])].copy();ct['model']=ct.model.map(names)
for col in ['recall','FPR','precision']:ct[col]*=100
ct=ct[['period','model','AP','recall','FPR','precision']].rename(columns={'period':'기간','model':'입력','recall':'재현율 %','FPR':'오경보율 %','precision':'정밀도 %'})
r=pd.read_csv(OUT/'retrospective_strata.csv');r=r[r.field.eq('repeated_anywhere')].copy();r['period']=np.where(r.month.lt(7),'AprJun','July');r['sae']=r.n*r.MAE
z=r.groupby(['period','model','value']).agg(n=('n','sum'),days=('target_days','sum'),sae=('sae','sum')).reset_index();z['MAE']=z.sae/z.n;z.to_csv(OUT/'repeat_strata_pooled.csv',index=False)
# Independent arithmetic checks against stored monthly summaries, no retraining.
monthly=pd.read_csv(OUT/'regression_monthly.csv');monthly=monthly[monthly.scope.eq('15_120')]
for period,months in [('AprJun',[4,5,6]),('July',[7])]:
 for name in order:
  a=monthly[monthly.month.isin(months)&monthly.model.eq(name)];v=np.average(a.MAE,weights=a.n);b=reg[reg.period.eq(period)&reg.model.eq(name)].MAE.iloc[0];assert abs(v-b)<1e-10
assert monthly.groupby('month').n.nunique().eq(1).all()
sel=json.loads((OUT/'selection.json').read_text());assert sel['regression']=='F0' and sel['classification_45_120']=='F1'
checks={'monthly_to_pooled_MAE':'PASS','equal_rows_across_candidates':'PASS','pre_July_selection_file':'PASS','source_manifest_checks':json.loads((OUT/'manifest.json').read_text())['checks'],'training_runs':72,'stress_predictions':48,'note':'72 fits = 9 inputs x 4 months x regression/classification; prior aggregation failure repaired from saved predictions without refitting'}
(OUT/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
report='''# 어떤 입력을 믿고 예측할 수 있는가 — 입력 기여 검증 v1

## 결론

**현재 자료에서는 전력 이력을 핵심 입력으로 유지하는 것이 가장 근거가 있다.** 생산량·기온·습도·인원 기록이 틀렸다고 판정한 것은 아니다. 정보의 물리적 진실성과 예측에 추가로 도움이 되는지는 다른 질문이다. 본 실험은 후자를 같은 모델/시점에서 비교했다.

회귀와 경보에 필요한 특징은 다르다. 전력값 회귀는 기본 전력 이력(F0)이 선택됐고, 준비시간 이후 개별 고부하 구간 분류는 최근 통계가 추가된 F1이 선택됐다. 하나의 입력 집합을 모든 목적에 강제로 사용하지 않는다. 실시간 자동운영을 확정한 단계는 아니다.

## 설계와 재현

- LightGBM 파라미터/seed42 고정, 9입력 × 4개월 × 회귀/분류 = 72회 학습. 출력 차이는 모델 파라미터가 아닌 입력 집합 비교다.
- 회귀: 매정시 이후15~120분 8구간. 입력은 그시점까지의 최근전력/완료된 시간별기록. 준비30분 이후에는 끝시각45~120분 6구간을 별도평가.
- F0: 최근8개 전력,전일/전주 동일시각 전력,달력,예측거리. F1: F0+1/3/6/24시간 전력 평균/최대/표준편차/변화율 등. 각 부가정보는 F1에 값/변화/결측/최근관측나이4개를 추가. 기상은 기온/습도만 검증했고 풍속/강수량은 포함하지 않았다.
- 원본 SHA256 보존, 시간오류7/13·15 타깃제외 및전력이력결측,결측은LightGBM 자체분기,정상일수있는피크는보존. 미래입력교란누수검사통과. 모든후보는같은평가행.
-4~6월 선택 규칙을7월출력전에적용:회귀는풀링MAE최저의1%이내중최소특징,분류는풀링AP최고의1%이내중최소특징. AP는고부하구간을높은점수로정렬하는능력이며높을수록좋음. 여러월 AP풀링은월별점수척도차이를포함하므로월별CSV도함께보존.
- 직접분류는월시작이전마지막28일을경보임계값보정에사용,그보다이전자료만학습하고재학습하지않음. 보정정상점수95백분위초과시경보(정상구간오경보5%목표). 실제미래오경보5%보장은아님. 확률자체보정/준비시간이후새고부하진입사건모델은이번에하지않았다.
- 7월은 이미관찰한재사용진단이다. 최종독립테스트가아니다.8~9월새평가없음. 출력월경계를넘는타깃은제외해과거4~6월MAE9.220과이번9.221에작은차이가있다.
- 정상/결측/부가정보의현장입수시점,전력단위/계측범위/증강원인은미확인. 원화절감/실제원인효과추론없음.

## 1. 전력값 예측

'''+mdtable(t)+'''

최근전력만으로도 예측할 수 있지만 전일·전주 정보를 보탠 F0가 검증전체와7월에서 더 좋았다. F1은 F0보다 검증MAE가나빠졌고, 부가정보를추가한6구성도F0를넘지못했다. 단, **부가정보를F0에직접추가하는모든조합을검증한것은아니므로 어떤상황에서도쓸모없다는결론은금지**한다.

기온추가는동일F1 대비4/5/6월모두MAE가소폭감소했고7월도8.463→8.331이다. 그러나4~6월일별오차차의7일블록95%bootstrap구간은[-0.232,0.020]으로0을포함한다. 모든부가정보의기여가0이라고할근거도없고,주요성능개선이라고할근거도부족하다.

생산량추가는F1 대비4~6월9.769→9.926으로악화했다.7일블록일별오차차구간[0.010,0.333]이지만,이미살펴본자료재사용·여러비교·제한된월수·고정seed의한계가있어일반적인생산량무용론으로해석하지않는다.

## 2. 고부하 경보는 회귀와 별도 평가

대상은 준비시간30분 이후 **각15분구간(끝시각45~120분)** 이다. 창안에한번이라도고부하가되는사건이나새진입경보재현율과다르다. 회귀값을임계값과비교한42%재현율과도다른별도분류결과다.

'''+mdtable(ct)+'''

F1의4~6월AP0.737은F0의0.540보다높아검증규칙에따라F1을경보용후보로선택했다. 그러나7월에는F0가AP0.804/FPR3.90%,F1이AP0.769/FPR5.25%여서F1이모든측면에서우세하지않다. F1은재현율76.0%로F0의73.5%보다높지만오경보도늘어난다. 추가정보전체는7월재현율75.1%/FPR5.22%로F1대비명확한운영상우위를보이지않았다.

AP의개선은한임계값에서의정밀도개선을보장하지않는다.4~6월F1정밀도36.96%는경보중오경보가많다는뜻이므로재현율만으로성과를홍보하지않는다.

## 3. 반복 날짜에서만 부가정보가 문제인가

전력곡선의전체1~9월동일상대유무를사후태그로사용했다. 이것은현장에서미리알수있는입력이나선택조건이아니다.

|4~6월평가대상|날짜수|F1 MAE|F1+생산량 MAE|변화|
|---|---:|---:|---:|---:|
|반복날짜|75|10.021|10.242|+0.221 악화|
|비반복날짜|16|8.589|8.442|-0.146 개선|

생산량추가효과가집단에따라달라져연계문제의가능성을계속점검할이유는있다. 하지만4월평가30일전체가반복집단이고비반복16일은5~6월에있다. **반복효과와월/운전조건차이를분리하지못했으므로연결오류의증거로확정할수없다.** 날씨추가는반복/비반복집단모두소폭개선됐다(9.998/8.230). “반복때문에모든부가정보가해롭다”는설명은현재결과와맞지않는다.

7월반복2일과비반복27일비교는표본불균형이심하다.학습과동일곡선유무태그의별도월별결과도CSV에보존했다. 원자료왜곡이주원인이라는가설은아직확정되지않았다.

## 4. 정보가 늦거나 누락될 때

부가정보값만60분과거로밀고이를최신값처럼수신하는지연표시없는상황,부가정보시간기록을동시에약20%누락하는상황을추가평가했다. 지연실험의관측나이는도착시각기준이며실제측정시각이1시간늦다는별도표시는없다. 따라서정확한측정시각이전달되는지연대응모델의성능으로해석하지않는다. 전력입력/타깃/학습모델은그대로두었다.

7월전체부가정보모델은기본MAE8.441,지연8.426,누락8.746이다. 작은지연변화만으로그입력을사용하지않는다고단정할수없으며부가정보의느린변화나전력이력과의중복정보도가능하다. 누락실험은고정seed한번의스트레스시나리오이며현장누락분포를대표하지않는다.

## 연구 결정과 다음 한 단계

1. **전력값모델은F0 유지**,생산량·날씨·인원은추가기여미확보상태로기록. 설명/조건별오류분석용으로원자료는보존한다.
2. **고부하경보는F1 후보**로검증선정하되F0의낮은오경보도비교유지. 준비시간이후새고부하진입,경보빈도/오경보비용,확률신뢰도평가를다음단계로한다.
3. “믿을입력”은 **이검증조건에서예측기여가있는입력**을뜻하며물리적진실성/센서정상판정을뜻하지않는다. 부가정보연계문제는제공기관의생성메타정보나독립자료없이는확정불가.
4. 높은성능을얻기위해임의로생산량을고치거나같은전력날짜를무조건삭제하지않는다. 앞서수행한중복제거민감도결과와이번입력검증을함께보고한다.

## 산출물/실행

- `python input_reliability.py`:72회학습과48개회귀스트레스예측,집계. 전체재학습없이집계만복구하려면 `python input_reliability.py --summarize-only`(예측gzip필요).
- `python input_reliability_report.py`:도표/문서/산술검증 생성. 의존성requirements-intraday.txt.
- `outputs/input_reliability_v1/`:월별회귀/분류,풀링,날짜별오차,조건별오류,반복태그,7일블록bootstrap,지연/누락,선택/manifest/검증결과.
- `figures/input_reliability_v1/input_ablation_mae.svg`:한글그림. PNG도같은코드로생성. 전력단위미확인으로원자료단위표기,범례없이직접표시,고정동일축.
- 최초실행은학습/예측저장후조건별집계에서중복origins인자로실패했다. 오류수정후저장예측으로집계를복구했고월별→풀링MAE/모든후보동일행검증통과. 재학습했다고표시하지않는다.
- 예측gzip과PNG는재생성가능중간/표현산출물로원격제외. 코드/CSV/JSON/문서/SVG/체크포인트는원격포함. 외부데이터신규사용없음.
'''
(ROOT/'docs/INPUT_RELIABILITY_RESULTS.md').write_text(report)
print(json.dumps(checks,ensure_ascii=False))
