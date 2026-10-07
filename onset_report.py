from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import koreanize_matplotlib
from onset import OUT,ROOT,raw_load,scores
FIG=ROOT/'figures/onset_v1';FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'svg.fonttype':'none','font.size':11,'axes.unicode_minus':False})
selection=json.loads((OUT/'selection.json').read_text());name=selection['model'];q=pd.read_csv(OUT/'predictions.csv',parse_dates=['origin','first_end','date']);e=pd.read_csv(OUT/'episodes.csv',parse_dates=['event_end','event_start']);p=pd.read_csv(OUT/'pooled_metrics.csv');summ=pd.read_csv(OUT/'episode_summary.csv')
boot=[];rng=np.random.default_rng(42)
for period,months in [('AprJun',[4,5,6]),('July',[7])]:
 z=q[q.month.isin(months)&q.model.eq(name)].copy();b=q[q.month.isin(months)&q.model.eq('week')].copy()
 assert z.origin.reset_index(drop=True).equals(b.origin.reset_index(drop=True))
 stats=[]
 for d,a in z.groupby('date'):
  bb=b[b.date.eq(d)];s=scores(a);t=scores(bb);stats.append([s['TP'],s['FP'],s['FN'],s['TN'],t['TP'],t['FP'],t['FN'],t['TN']])
 v=np.array(stats);draws=rng.integers(0,len(v),(2000,int(np.ceil(len(v)/7))));ix=((draws[:,:,None]+np.arange(7))%len(v)).reshape(2000,-1)[:,:len(v)];total=v[ix].sum(axis=1)
 for metric,i,j,bi,bj in [('recall',0,2,4,6),('FPR',1,3,5,7)]:
  delta=total[:,i]/np.maximum(total[:,i]+total[:,j],1)-total[:,bi]/np.maximum(total[:,bi]+total[:,bj],1);lo,hi=np.quantile(delta,[.025,.975]);boot.append(dict(period=period,metric=metric,comparator='week',CI_low=lo,CI_high=hi,days=len(v)))
pd.DataFrame(boot).to_csv(OUT/'week_block_uncertainty.csv',index=False)
# Calibration table and raw reliability graphic: descriptive, small bins reported.
z=q[q.month.eq(7)&q.model.eq(name)].copy();z['bin']=pd.cut(z.prob,np.linspace(0,1,6),include_lowest=True)
bins=z.groupby('bin',observed=False).agg(n=('label','size'),predicted_probability=('prob','mean'),observed_frequency=('label','mean')).reset_index();bins.to_csv(OUT/'probability_reliability.csv',index=False)
fig,ax=plt.subplots(figsize=(6.3,5));ax.plot([0,1],[0,1],color='#888888',linestyle='--',label='예측확률 = 관측비율');ax.plot(bins.predicted_probability,bins.observed_frequency,'o-',color='#246f9c',label='7월 진입 모델')
for _,r in bins.dropna().iterrows():ax.annotate(f"n={int(r['n'])}",(r.predicted_probability,r.observed_frequency),xytext=(5,7),textcoords='offset points',fontsize=9)
ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xlabel='예측된 진입 확률',ylabel='실제로 진입한 비율',title='진입 확률의 신뢰도 · 7월 재사용 진단');ax.grid(alpha=.2);ax.legend(loc='upper left');fig.tight_layout();fig.savefig(FIG/'probability_reliability.svg');fig.savefig(FIG/'probability_reliability.png',dpi=140);plt.close(fig)
y,_,_=raw_load();categories=[('탐지 성공',z.label&z.warn),('준비시간 부족 경보',z.early&z.warn),('진입 미탐지',z.label&~z.warn)];cases=[]
for title,mask in categories:
 if mask.any():cases.append((title,z[mask].sort_values('origin').iloc[0]))
fig,axs=plt.subplots(len(cases),1,figsize=(10,3.2*len(cases)),squeeze=False)
for ax,(title,r) in zip(axs.ravel(),cases):
 t=r.origin;line=y[(y.index>=t-pd.Timedelta(hours=2))&(y.index<=t+pd.Timedelta(hours=2))];ax.plot(line.index,line.values,color='#245e8d',label='실제 전력');ax.axhline(r.threshold,color='#b96522',linestyle='--',label='고부하 기준');ax.axvline(t,color='#555555',linestyle=':',label='경보 발행시점');ax.axvspan(t+pd.Timedelta(minutes=30),t+pd.Timedelta(minutes=120),alpha=.1,color='#29946d',label='준비시간 이후 범위')
 ax.set_title(f'{title} · {t:%m/%d %H:%M} · 진입 확률 {r.prob:.2f}',loc='left',fontsize=12);ax.set_ylabel('전력 (원자료 단위)');ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'));ax.xaxis.set_major_locator(mdates.HourLocator());ax.grid(alpha=.15)
axs[-1,0].set_xlabel('통합 시간축 · 15분 구간 끝시각');handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=9);fig.subplots_adjust(hspace=.45,left=.1,right=.97,top=.94,bottom=.12);fig.savefig(FIG/'entry_cases.svg');fig.savefig(FIG/'entry_cases.png',dpi=140);plt.close(fig)

def table(df):
 def f(v):return f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v)
 return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(f(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])
t=p[p.model.isin([name,'week'])][['period','model','n','positives','TP','FP','FN','recall','precision','FPR','AP','Brier']].copy()
for c in ['recall','precision','FPR']:t[c]*=100
sj=summ[summ.month.eq(7)&summ.model.eq(name)].iloc[0];daily=pd.read_csv(OUT/'daily_metrics.csv');ds=daily[daily.month.eq(7)&daily.model.eq(name)];metrics=p[p.period.eq('July')&p.model.eq(name)].iloc[0]
report='''# 준비시간을 확보한 고부하 진입 모델 — 2026-10-07

## 구현한 질문

**현재는 고부하가 아닐 때, 30분 준비시간을 확보한 뒤 2시간 안에 처음 고부하로 진입하는가?** LightGBMClassifier로 직접 학습했다. 현재전력과 최근 통계/과거전일·전주/달력만 사용하며 미래생산·기온을 넣지 않았다.

예:10:00발행,10:30~10:45구간에서첫초과면정답1. 이구간의기록시각은10:45이지만실제준비시간은30분이다.10:00~10:15 또는10:15~10:30에서먼저초과하면준비시간부족 유형이고이타깃정답은0. 같은고부하가계속되는경우를새진입으로세지않는다.15분이하정확한진입시각은자료로알수없다.

고부하는월시작이전전력q95초과이며설비위험/한전계약기준이라는뜻이아니다. 임계값은월별고정. 적용대상은현재값이관측됐고임계값이하인정시시점이다. 미래실측으로적용대상을선별하지않았다.8타깃관측불가구간은평가불가로남긴다.

## 모델/검증

- F0/F1 후보,4~6월 선택(풀링AP최고1%이내중최소특징),7월은이미검토한재사용진단. 선택은 '''+name+'''. 확정된독립일반화성능이라고부르지않음.
- 각월이전28일을학습에서남김:첫14일logit sigmoid확률보정,뒤14일경보임계값보정. 각타깃이분할경계이전이어야함. 정상타깃보정점수95백분위초과시경보(오경보5%목표). 미래5%보장은아님.
- 전주/전일 동일시각값으로비교하며같은보정기간·확률보정·오경보목표적용. 재학습없음. 8개후보학습으로끝냈으며7월로임계값재조정하지않음.
- 결측/시간오류처리는원본규칙유지,극단전력원본보존,미래입력교란검사PASS. 저장LightGBM text모델재로드예측일치PASS. 미래8타깃의첫초과위치/준비시간검사PASS.

## 정시 판단 단위 성능

재현율/정밀도/FPR은%단위. 같은진입을서로다른정시에서예측할수있으므로다음표의양성은독립사건수가아니다. 보정된월별점수의풀링AP는월간척도차이도포함한다.

'''+table(t)+'''

7월선택모델의경보는총108회중목표진입탐지41회,타깃기준오경보67회다. 그67회중38회는실제로고부하가되지만15/30분뒤여서준비시간이부족한경보다. 따라서모든오경보가불필요한위험경보라는뜻은아니지만작업을30분준비후조정하라는목적에는맞지않는다. 평균일별경보 '''+f'{ds.alarms.mean():.2f}'+'''회,오경보 '''+f'{ds.false_alarms.mean():.2f}'+'''회(평가일 '''+str(len(ds))+''')다. 전주기준보다더많이찾지만오경보도명확히많아운영우위를확정하지않는다.

## 실제 진입 사건과 적용범위

7월실제정상→고부하전환은 '''+str(int(sj.total_entries))+'''건이고,정시발행/현재정상/30분준비/2시간범위 조건에맞춰평가가능한사건은 '''+str(int(sj.evaluable_entries))+'''건('''+f'{sj.coverage*100:.1f}'+'''%)이다. 그중 '''+str(int(sj.detected_entries))+'''건탐지, '''+str(int(sj.missed_entries))+'''건누락,사건별재현율 '''+f'{sj.episode_recall*100:.1f}'+'''%. 탐지사건의준비시간최소 '''+str(int(sj.preparation_min))+'''분,중앙값 '''+str(int(sj.preparation_median))+'''분이다.

**나머지75건을모델이잘맞혔다고도,평가대상에서제외했으니문제없다고도할수없다.** 전체117진입중30분준비를확보해탐지한것은36건(30.8%)이다. 이미고부하인정시/너무빠른진입/관측오류경계/연속고부하내반복초과등으로조건을충족하지못한진입이포함된다. 매정시갱신과준비30분이라는운영설정자체가범위를제한한다. 15분갱신,긴급진입분기,일시적인임계값교차를다루는지속시간정의는다음검토후보이지이번성공결과가아니다.

## 확률/오류/한계

- probability_reliability.csv와그림은7월예측확률과실제빈도를비교하며각bin표본수표시. Brier/빈도표를보존하되유효표본이작고자료반복/검증재사용이있으므로확률신뢰도가확정됐다고하지않는다.
- conditions.csv는발행시점생산기록·기온·최근전력결측조건의오류진단이다. 피처로생산·기온을다시넣어개선했다고하지않는다.
- 7일평가날짜블록bootstrap은정시판단의날짜상관을고려하며선택/자료재사용편향을제거하지못한다.
- 그래프사례는성공/준비시간부족경보/미탐각유형의가장빠른7월시점을선택했다. 한개성공사례로전체성능을대표하지않는다. 실제값의15분끝시각을하나의시간축으로표시했다.

## 현재 결정

모델/예측파일은구현완료,자동작업지연운영은미채택이다. 핵심남은문제는**준비시간부족진입과여유있는진입을구분하는능력**, 보정기간오경보가7월에증가하는점,정시갱신으로대응가능한사건비중이낮은점이다. 다음은안정/긴급진입/준비가능진입을별도로예측하는분기와갱신주기의비교를개발자료에서검증할수있다. 실제설비조정/생산보존/전기요금절감은별도조건부실험이다.

## 실행/산출물

```bash
python onset.py
python onset_report.py
python onset.py --predict-features outputs/onset_v1/example_features.csv
```

requirements-intraday.txt 사용. selected_model.json의features/확률보정계수/경보임계값과선택LightGBM model_7_F1.txt 저장. example_features.csv 입력으로각행의applicable/entry_probability/warn을출력한다. 입력행은발행시점까지의값으로만생성해야한다. 현장연동UI/센서타임스탬프지연확인은미완료이며저장모델은2021년7월자료구성의연구재현용이다.

outputs/onset_v1의월별/풀링/사건별/조건별/일별/보정/분할표와모델text,figures/onset_v1의SVG,코드,인계문서는원격저장한다. 추가외부자료없음. 원본CSV수정없음. 이전회귀모델결과는유지.
'''
(ROOT/'docs/ONSET_RESULTS.md').write_text(report)
print(p[['period','model','recall','precision','FPR','AP','Brier']].round(4).to_string(index=False))
