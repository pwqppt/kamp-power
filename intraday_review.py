"""Source-backed tables/figures, fair calibrated baselines and fixed sensitivity checks."""
from pathlib import Path
import json,warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import koreanize_matplotlib
from intraday import OUT,raw_load,build,make_model,metrics
from intraday_followup import score
ROOT=Path(__file__).resolve().parent;FIG=ROOT/'figures/intraday_v1';FIG.mkdir(parents=True,exist_ok=True)
y,cov,_=raw_load();x,m,groups=build(y,cov)
p=pd.read_csv(OUT/'predictions.csv.gz',parse_dates=['origin','target'])
# Fair past-calibrated baseline warning thresholds. These are warning scores, not probabilities.
rows=[]
for month in [4,5,6,7]:
 cut=pd.Timestamp(2021,month,1);end=cut+pd.offsets.MonthBegin();start=cut-pd.Timedelta(days=28)
 c=float(y.loc[y.index<cut].quantile(.95));cal=(m.origin>=start)&(m.origin+pd.Timedelta(hours=2)<cut);test=(m.origin>=cut)&(m.origin<end)
 for name,col in [('last','lag0'),('week','week')]:
  z=m.copy();z['score']=x[col].fillna(pd.Series(y.ffill().reindex(pd.DatetimeIndex(m.origin)).to_numpy(),index=m.index))
  for level in ['interval','window']:
   ca=z.loc[cal].copy();te=z.loc[test].copy()
   if level=='window':
    ca=ca.groupby('origin').agg(y=('y','max'),score=('score','max'),current=('current','first'));te=te.groupby('origin').agg(y=('y','max'),score=('score','max'),current=('current','first'))
   t=float(np.quantile(ca.loc[ca.y<=c,'score'],.95,method='higher'));a=te.y>c;alert=te.score>t
   pos=a.to_numpy();b=alert.to_numpy();tp=int((pos&b).sum());fp=int((~pos&b).sum());fn=int((pos&~b).sum());tn=int((~pos&~b).sum())
   rows.append({'month':month,'model':name,'level':level,'score_threshold':t,'n':len(te),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'recall':tp/(tp+fn) if tp+fn else np.nan,'precision':tp/(tp+fp) if tp+fp else np.nan,'FPR':fp/(fp+tn) if fp+tn else np.nan})
pd.DataFrame(rows).to_csv(OUT/'calibrated_baseline_metrics.csv',index=False)
# Sensitivity: no imputation of targets, artificial 5%/20% missing hourly histories only.
# Training pattern deduplication uses only complete training days and keeps earliest occurrence.
sens=[];rng=np.random.default_rng(42)
for month in [4,5,6,7]:
 cut=pd.Timestamp(2021,month,1);end=cut+pd.offsets.MonthBegin();train=m.origin+pd.Timedelta(hours=2)<cut;test=(m.origin>=cut)&(m.origin<end);c=float(y.loc[y.index<cut].quantile(.95))
 model=make_model('lgb');model.fit(x.loc[train,groups['F0']],m.loc[train,'y'])
 for rate in [.05,.20]:
  yy=y.copy();cc=cov.copy();drop=cc.index[rng.random(len(cc))<rate];cc.loc[drop]=np.nan
  # A dropped hourly upload removes all four quarter-hour measurements belonging to that row.
  missing_ends=pd.DatetimeIndex(np.concatenate([(drop-pd.Timedelta(minutes=q)).to_numpy() for q in [0,15,30,45]]));yy.loc[yy.index.isin(missing_ends)]=np.nan
  # Corrupt input histories only; preserve all original targets.
  xx,mm,_=build(yy,cc,truth=y)
  assert m[['origin','target']].equals(mm[['origin','target']])
  z=m.loc[test].copy();z['pred']=np.maximum(0,model.predict(xx.loc[test,groups['F0']]));z['threshold']=c
  sens.append({'month':month,'scenario':f'hourly_missing_{rate}',**metrics(z)})
 days={};keep=[]
 for day,g in y.loc[y.index<cut].groupby((y.loc[y.index<cut].index-pd.Timedelta(minutes=1)).normalize()):
  if len(g)!=96 or g.isna().any():continue
  key=tuple(g.to_numpy());
  if key not in days:days[key]=day;keep.append(day)
 dt=train&m.origin.dt.normalize().isin(keep)
 mod=make_model('lgb');mod.fit(x.loc[dt,groups['F0']],m.loc[dt,'y'])
 z=m.loc[test].copy();z['pred']=np.maximum(0,mod.predict(x.loc[test,groups['F0']]));z['threshold']=c
 sens.append({'month':month,'scenario':'deduplicated_training_days','train_origins':int(dt.sum()/8),**metrics(z)})
pd.DataFrame(sens).to_csv(OUT/'sensitivity_metrics.csv',index=False)
# Pooled metrics, exact denominators.
pooled=[]
for period,mask in [('4~6월',p.month<7),('7월',p.month==7)]:
 for model,g in p[mask].groupby('model'):pooled.append({'period':period,'model':model,**metrics(g)})
pd.DataFrame(pooled).to_csv(OUT/'pooled_metrics.csv',index=False)
# Strict plot warnings catch unsupported Hangul glyphs; external legend and uniform palette.
plt.rcParams.update({'font.size':11,'axes.titlesize':14,'axes.labelsize':11,'axes.unicode_minus':False,'svg.fonttype':'path'})
colors={'last':'#7A7A7A','week':'#B78022','lgb_F0':'#2166AC'};labels={'last':'마지막 값 유지','week':'전주 동일 시각','lgb_F0':'LightGBM'}
checks=[]
def save(fig,name):
 with warnings.catch_warnings(record=True) as ww:
  warnings.simplefilter('always');fig.canvas.draw()
  glyph=[str(w.message) for w in ww if 'Glyph' in str(w.message)]
  assert not glyph,glyph
  # Legends must not intersect data axes.
  renderer=fig.canvas.get_renderer()
  for legend in fig.legends:
   for ax in fig.axes:assert not legend.get_window_extent(renderer).overlaps(ax.get_window_extent(renderer))
  fig.savefig(FIG/(name+'.png'),dpi=170);fig.savefig(FIG/(name+'.svg'))
 checks.append({'figure':name,'missing_glyphs':0,'figure_legend_outside_axes':True});plt.close(fig)
mon=pd.read_csv(OUT/'monthly_metrics.csv')
fig,axes=plt.subplots(1,2,figsize=(12,4.8));fig.subplots_adjust(top=.77,bottom=.20,wspace=.28)
for ax,col,title in zip(axes,['MAE','recall'],['15~120분 전력 예측오차','고부하 구간 탐지: 회귀값 임계 판정']):
 for i,name in enumerate(colors):
  z=mon[mon.model==name];ax.bar(np.arange(4)+(i-1)*.24,z[col]*(100 if col=='recall' else 1),width=.23,color=colors[name],label=labels[name])
 ax.set_xticks(range(4),['4월','5월','6월','7월']);ax.set_title(title);ax.set_ylim(bottom=0);ax.set_ylabel('재현율 (%)' if col=='recall' else 'MAE (원자료 단위)');ax.spines[['top','right']].set_visible(False)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.99),ncol=3,frameon=False)
fig.text(.08,.05,'매시간 발행 · 4~6월 선택 / 7월 재평가 · 시간 오류일 제외 · 고부하 = 학습 전력 95백분위 초과',fontsize=10)
save(fig,'01_forecast_and_peak')
h=pd.read_csv(OUT/'horizon_metrics.csv');fig,ax=plt.subplots(figsize=(9,4.8));fig.subplots_adjust(top=.77,bottom=.20)
for name in colors:
 z=h[(h.month==7)&(h.model==name)];ax.plot(z.minutes,z.MAE,'o-',color=colors[name],label=labels[name])
ax.set(xticks=np.arange(15,121,15),xlabel='예측 시점으로부터 경과 시간 (분)',ylabel='MAE (원자료 단위)',title='7월: 예측 거리별 오차');ax.spines[['top','right']].set_visible(False)
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.99),ncol=3,frameon=False)
fig.text(.1,.05,'모든 모델이 같은 발행시각과 관측정보 범위를 사용 · 7월은 독립 미사용 테스트가 아님',fontsize=10)
save(fig,'02_horizon_errors')
# Single timestamp axis: fixed 60-minute horizon, no overlapping forecasts stitched using hindsight.
z=p[(p.month==7)&p.model.eq('lgb_F0')];day=z.groupby(z.origin.dt.normalize()).y.max().idxmax()
fig,ax=plt.subplots(figsize=(11,4.8));fig.subplots_adjust(top=.77,bottom=.25)
g=z[z.origin.dt.normalize().eq(day)&z.horizon.eq(4)].sort_values('target')
ax.plot(g.target,g.y,'k-o',label='실측',markersize=3)
for name in ['week','lgb_F0']:
 q=p[(p.origin.dt.normalize()==day)&p.model.eq(name)&p.horizon.eq(4)].sort_values('target');ax.plot(q.target,q.pred,'--',color=colors[name],label=labels[name])
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0,6,12,18]));ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
ax.set(ylabel='전력값 (원자료 단위)',title='7월 최대부하일: 매시간 발행한 60분 후 예측');ax.spines[['top','right']].set_visible(False)
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.99),ncol=3,frameon=False)
fig.text(.1,.03,'그림 선택: 7월 관측 최대부하일(동률이면 첫날) · 시간축은 날짜와 구간 종료시각을 통합',fontsize=10)
save(fig,'03_peak_day')
(FIG/'visual_checks.json').write_text(json.dumps(checks,indent=2))
print('Review tables, sensitivity and figures complete')
