"""Fixed follow-up: direct high-load classification, delayed input and synthetic jobs."""
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score,brier_score_loss
from intraday import OUT,raw_load,build,make_model,metrics,PARAMS

SPEC={'classification_features':['F0','F2'],'calibration_days':28,'false_positive_rate_target':0.05,'classifier_refit_after_calibration':False,'delay_minutes':60,'job_ready_minutes':30,'job_extra_delay_minutes':[0,30,60],'job_duration_minutes':30,'job_arrival_hours':[6,9,12,15,18],'job_amplitude_fraction_training_q95':[.05,.10],'status':'prespecified before follow-up results'}
(OUT/'followup_spec.json').write_text(json.dumps(SPEC,indent=2))

def score(a,p,b):
 a=np.asarray(a,dtype=bool);p=np.asarray(p);b=np.asarray(b,dtype=bool)
 tp=int((a&b).sum());fp=int((~a&b).sum());fn=int((a&~b).sum());tn=int((~a&~b).sum())
 return {'n':len(a),'positives':int(a.sum()),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'recall':tp/(tp+fn) if tp+fn else np.nan,'precision':tp/(tp+fp) if tp+fp else np.nan,'FPR':fp/(fp+tn) if fp+tn else np.nan,'AP':average_precision_score(a,p) if a.any() else np.nan,'Brier':brier_score_loss(a,p)}

def run():
 y,cov,_=raw_load();x,m,groups=build(y,cov)
 pred=pd.read_csv(OUT/'predictions.csv.gz',parse_dates=['origin','target'])
 rows=[];stored=[];delayrows=[]
 xd,md,gd=build(y,cov,60);assert m[['origin','target']].equals(md[['origin','target']])
 for month in [4,5,6,7]:
  cut=pd.Timestamp(2021,month,1);end=cut+pd.offsets.MonthBegin();calstart=cut-pd.Timedelta(days=28)
  train=(m.origin+pd.Timedelta(hours=2)<cut);fit=m.origin+pd.Timedelta(hours=2)<calstart
  cal=(m.origin>=calstart)&train;test=(m.origin>=cut)&(m.origin<end)
  c=float(y.loc[y.index<cut].quantile(.95));a=m.y>c
  for f in SPEC['classification_features']:
   pars={k:v for k,v in PARAMS['lgb'].items() if k!='objective'}
   clf=LGBMClassifier(**pars);clf.fit(x.loc[fit,groups[f]],a[fit])
   cp=clf.predict_proba(x.loc[cal,groups[f]])[:,1];pr=clf.predict_proba(x.loc[test,groups[f]])[:,1]
   # > threshold ensures quantile ties do not inflate calibration false positives.
   threshold=float(np.quantile(cp[~a[cal]],.95,method='higher'))
   z=m.loc[test].copy();z['prob']=pr;z['model']='classifier_'+f;z['month']=month;z['threshold']=c;z['prob_threshold']=threshold;stored.append(z)
   for policy,t in [('calibrated',threshold),('fixed_0.5',.5)]:
    rows.append({'month':month,'model':'classifier_'+f,'policy':policy,'prob_threshold':t,**score(a[test],pr,pr>t)})
   # Separate direct window classification, not multiplying correlated interval probabilities.
   def window(mask):
    ii=m.loc[mask].groupby('origin').head(1).index
    xx=x.loc[ii,groups[f]].copy();xx=xx.drop(columns=['horizon','quarter'],errors='ignore')
    aa=m.loc[mask].groupby('origin').y.max().to_numpy()>c
    return ii,xx,aa
   fi,fx,fa=window(fit);ci,cx,ca=window(cal);ti,tx,ta=window(test)
   wc=LGBMClassifier(**pars);wc.fit(fx,fa);wp=wc.predict_proba(cx)[:,1];tp=wc.predict_proba(tx)[:,1]
   wt=float(np.quantile(wp[~ca],.95,method='higher'))
   onset=m.loc[ti,'current'].le(c).to_numpy()&m.loc[ti,'current'].notna().to_numpy()
   for subset,mask in [('all',np.ones(len(tp),dtype=bool)),('onset',onset)]:
    rows.append({'month':month,'model':'window_'+f,'policy':subset,'prob_threshold':wt,**score(ta[mask],tp[mask],tp[mask]>wt)})
  # Retrain same selected architecture/features with 60-minute input latency, no retuning.
  mod=make_model('lgb');mod.fit(xd.loc[train,gd['F0']],md.loc[train,'y'])
  z=md.loc[test].copy();z['pred']=np.maximum(0,mod.predict(xd.loc[test,gd['F0']]));z['threshold']=c
  delayrows.append({'month':month,'delay_minutes':60,**metrics(z)})
  print('classification + latency month',month,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'classification_metrics.csv',index=False)
 pd.concat(stored).to_csv(OUT/'classification_predictions.csv.gz',index=False,compression='gzip')
 pd.DataFrame(delayrows).to_csv(OUT/'latency_metrics.csv',index=False)
 # Equal-job simulation: add jobs to observed background; never identify movable load from future actuals.
 jobrows=[];dailyrows=[]
 for month in [4,5,6,7]:
  pp=pred[(pred.month==month)&pred.model.isin(['lgb_F0','last','week'])]
  base=pp[pp.model=='lgb_F0'];threshold=float(base.threshold.iloc[0])
  for frac in SPEC['job_amplitude_fraction_training_q95']:
   amplitude=threshold*frac
   for day,b in base.groupby(base.origin.dt.normalize()):
    origins=[day+pd.Timedelta(hours=h) for h in SPEC['job_arrival_hours']]
    if not all(o in set(b.origin) for o in origins):continue
    ts=pd.date_range(day+pd.Timedelta(minutes=15),day+pd.Timedelta(days=1),freq='15min')
    actual=y.reindex(ts).to_numpy();
    if np.isnan(actual).any():continue
    for policy in ['immediate','last','week','lgb_F0','oracle']:
     load=actual.copy();delays=[]
     for origin in origins:
      g=base[base.origin==origin].sort_values('horizon');true=g.y.to_numpy()
      forecast=true if policy=='oracle' else (pp[(pp.origin==origin)&(pp.model==policy)].sort_values('horizon').pred.to_numpy() if policy!='immediate' else true)
      assert len(forecast)==8
      costs=[]
      for start in [2,4,6]:
       v=forecast.copy();v[start:start+2]+=amplitude;costs.append(v.max())
      choice=0 if policy=='immediate' else int(np.argmin(costs));start=[2,4,6][choice]
      ends=pd.date_range(origin+pd.Timedelta(minutes=15*(start+1)),periods=2,freq='15min')
      ix=ts.get_indexer(ends);assert (ix>=0).all();load[ix]+=amplitude;delays.append(choice*30)
      jobrows.append({'month':month,'date':day,'origin':origin,'fraction':frac,'policy':policy,'extra_delay_minutes':choice*30,'amplitude':amplitude})
     assert np.isclose((load-actual).sum(),len(origins)*2*amplitude)
     dailyrows.append({'month':month,'date':day,'fraction':frac,'policy':policy,'daily_max':load.max(),'background_max':actual.max(),'mean_extra_delay':np.mean(delays),'jobs':len(origins)})
 dd=pd.DataFrame(dailyrows);reference=dd[dd.policy=='immediate'][['date','fraction','daily_max']].rename(columns={'daily_max':'immediate_max'})
 dd=dd.merge(reference,on=['date','fraction']);dd['reduction']=dd.immediate_max-dd.daily_max
 dd.to_csv(OUT/'synthetic_jobs_daily.csv',index=False);pd.DataFrame(jobrows).to_csv(OUT/'synthetic_jobs_ledger.csv',index=False)
 dd.groupby(['month','fraction','policy']).agg(days=('date','size'),mean_reduction=('reduction','mean'),worst_reduction=('reduction','min'),worse_days=('reduction',lambda z:int((z< -1e-8).sum())),period_max=('daily_max','max'),mean_delay=('mean_extra_delay','mean')).to_csv(OUT/'synthetic_jobs_summary.csv')
 # Paired day bootstrap; no i.i.d. quarter-hour claims. Week-block sensitivity.
 rr=[];rng=np.random.default_rng(42)
 for period,mask in [('AprJun',pred.month<7),('July',pred.month==7)]:
  sub=pred[mask&pred.model.isin(['lgb_F0','last','week'])].copy();sub['ae']=abs(sub.y-sub.pred)
  day=sub.groupby([sub.origin.dt.normalize(),'model']).ae.mean().unstack()
  for comparator in ['last','week']:
   delta=(day.lgb_F0-day[comparator]).to_numpy()
   for block in [1,7]:
    estimates=[]
    for _ in range(2000):
     starts=rng.integers(0,len(delta),int(np.ceil(len(delta)/block)));ix=np.concatenate([(s+np.arange(block))%len(delta) for s in starts])[:len(delta)];estimates.append(delta[ix].mean())
    lo,hi=np.quantile(estimates,[.025,.975]);rr.append({'period':period,'comparator':comparator,'block_days':block,'days':len(delta),'MAE_difference':delta.mean(),'CI_low':lo,'CI_high':hi})
 pd.DataFrame(rr).to_csv(OUT/'bootstrap.csv',index=False)
 print('Follow-up complete',flush=True)
if __name__=='__main__':run()
