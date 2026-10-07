"""First high-load entry within 15..120 minutes; no preparation-time filter. docs/FIXED_Q95_PROTOCOL.md."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, Booster
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,brier_score_loss
from intraday import raw_load,build,audit,PARAMS
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/onset_fixed_q95';OUT.mkdir(exist_ok=True)
def scores(q):
 a=q.label.to_numpy(bool);b=q.warn.to_numpy(bool);p=q.prob.to_numpy();tp=int((a&b).sum());fp=int((~a&b).sum());fn=int((a&~b).sum());tn=int((~a&~b).sum())
 return dict(n=len(q),positives=int(a.sum()),TP=tp,FP=fp,FN=fn,TN=tn,recall=tp/(tp+fn) if tp+fn else np.nan,precision=tp/(tp+fp) if tp+fp else np.nan,FPR=fp/(fp+tn) if fp+tn else np.nan,AP=average_precision_score(a,p) if a.any() else np.nan,Brier=brier_score_loss(a,p),entries_within30_minutes=int(q.early.sum()),detected_within30_minutes=int((q.early&q.warn).sum()))
def logit(p):
 p=np.clip(np.asarray(p),1e-6,1-1e-6);return np.log(p/(1-p)).reshape(-1,1)
def apply_cal(p,params):
 if params['method']=='identity':return np.asarray(p)
 return 1/(1+np.exp(-np.clip(params['coef']*logit(p).ravel()+params['intercept'],-50,50)))
def fitcal(p,a):
 if len(np.unique(a))<2:return {'method':'identity','reason':'single-class probability calibration'}
 lr=LogisticRegression(C=1.,random_state=42);lr.fit(logit(p),a);return {'method':'sigmoid_logit','coef':float(lr.coef_[0,0]),'intercept':float(lr.intercept_[0]),'n':len(a),'positives':int(np.sum(a))}
def risk_features(x,m,g,c):
 ids=m.groupby('origin').head(1).index;z=x.loc[ids].copy();z.index=pd.DatetimeIndex(m.loc[ids,'origin']);z['hour']=z.index.hour;z=z.drop(columns=['quarter','horizon'],errors='ignore')
 cols={name:[k for k in g[name] if k not in ['quarter','horizon']] for name in ['F0','F1']}
 shared=[]
 for which in ['day','week']:
  arr=x[which].to_numpy().reshape(-1,8)
  for tag,v in [('early_max',pd.DataFrame(arr[:,:2]).max(axis=1).to_numpy()),('window_max',pd.DataFrame(arr).max(axis=1).to_numpy())]:
   key=which+'_'+tag;z[key]=v;shared.append(key)
 z['threshold_headroom']=(c-z.lag0)/c;shared.append('threshold_headroom')
 for name in cols:cols[name]+=shared
 return z,cols

def run():
 y,cov,info=raw_load();x,m,g=build(y,cov);checks=audit(y,cov,x,m,g);allq=[];monthly=[];exports={};calrows=[]
 for month in [4,5,6,7]:
  cut=pd.Timestamp(2021,month,1);end=cut+pd.offsets.MonthBegin();pstart=cut-pd.Timedelta(days=28);split=cut-pd.Timedelta(days=14);c=float(y[y.index<pd.Timestamp('2021-04-01')].quantile(.95))
  X,groups=risk_features(x,m,g,c);actual=m.y.to_numpy().reshape(-1,8);high=actual>c;first=np.where(high.any(axis=1),high.argmax(axis=1)+1,0)
  meta=m.groupby('origin').head(1).copy().set_index('origin');meta['first_index']=first;meta['label']=(first>=1)&(first<=8);meta['early']=(first>0)&(first<=2);meta['first_end']=meta.index+pd.to_timedelta(first*15,unit='m');meta['date']=meta.index.normalize();meta['threshold']=c;meta['month']=month
  eligible=meta.current.notna()&meta.current.le(c)
  fit=eligible&(meta.index+pd.Timedelta(hours=2)<pstart)
  procal=eligible&(meta.index>=pstart)&(meta.index+pd.Timedelta(hours=2)<split)
  alarmcal=eligible&(meta.index>=split)&(meta.index+pd.Timedelta(hours=2)<cut)
  test=eligible&(meta.index>=cut)&(meta.index+pd.Timedelta(hours=2)<=end)
  assert meta.loc[fit].index.max()+pd.Timedelta(hours=2)<pstart
  calrows.append(dict(month=month,threshold=c,fit_n=int(fit.sum()),fit_positive=int(meta.loc[fit,'label'].sum()),probability_calibration_n=int(procal.sum()),alarm_calibration_n=int(alarmcal.sum()),eligible_test=int(test.sum()),all_valid_test=int(((meta.index>=cut)&(meta.index+pd.Timedelta(hours=2)<=end)).sum())))
  pars={k:v for k,v in PARAMS['lgb'].items() if k!='objective'}
  for name in ['F0','F1','week','day']:
   if name in groups:
    cols=groups[name];model=LGBMClassifier(**pars);model.fit(X.loc[fit,cols],meta.loc[fit,'label'])
    pc=model.predict_proba(X.loc[procal,cols])[:,1];params=fitcal(pc,meta.loc[procal,'label'].to_numpy());pred=apply_cal(model.predict_proba(X.loc[test,cols])[:,1],params);ac=apply_cal(model.predict_proba(X.loc[alarmcal,cols])[:,1],params)
    model.booster_.save_model(str(OUT/f'model_{month}_{name}.txt'));exports[(month,name)]={'features':cols,'probability_calibration':params,'high_load_threshold':c,'training_cutoff':str(pstart),'issue_month':month}
   else:
    # Known historic curve over all eight intervals; no early-entry suppression.
    s=X[name+'_window_max'].fillna(X.lag0).clip(lower=0)
    norm=float(max(c,s[procal].max(),1));params=fitcal((s[procal]/norm).clip(0,1),meta.loc[procal,'label'].to_numpy());pred=apply_cal((s[test]/norm).clip(0,1),params);ac=apply_cal((s[alarmcal]/norm).clip(0,1),params)
   negatives=~meta.loc[alarmcal,'label'].to_numpy(bool);t=float(np.quantile(ac[negatives],.95,method='higher'))
   q=meta.loc[test].copy();q['prob']=pred;q['warn']=pred>t;q['model']=name;q['decision_threshold']=t;allq.append(q.reset_index());monthly.append(dict(month=month,model=name,decision_threshold=t,**scores(q)))
   if name in groups:exports[(month,name)]['decision_threshold']=t
   print(month,name,scores(q),flush=True)
  pd.DataFrame(monthly).to_csv(OUT/'monthly_metrics.csv',index=False)
  if month==6:
   vv=pd.concat(allq);vv=vv[vv.month.lt(7)&vv.model.isin(['F0','F1'])];ap=vv.groupby('model').apply(lambda q:average_precision_score(q.label,q.prob),include_groups=False);near=ap[ap>=ap.max()*.99];selected='F0';selection={'model':selected,'validation_AP':ap.to_dict(),'rule':'F0 frozen from previous experiments; threshold-only study, no new feature selection','frozen_before_July':True,'independent_test':False};(OUT/'selection.json').write_text(json.dumps(selection,indent=2))
 q=pd.concat(allq,ignore_index=True);q.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(calrows).to_csv(OUT/'split_audit.csv',index=False)
 pooled=[];episodes=[];daily=[];conditions=[]
 for period,months in [('AprJun',[4,5,6]),('July',[7])]:
  for name,z in q[q.month.isin(months)].groupby('model'):pooled.append(dict(period=period,model=name,**scores(z)))
 for (month,name),z in q.groupby(['month','model']):
  for day,a in z.groupby('date'):daily.append(dict(month=month,model=name,date=day,alarms=int(a.warn.sum()),false_alarms=int((a.warn&~a.label).sum()),**scores(a)))
  for tag,mask in [('production_zero',z.production.eq(0)),('production_positive',z.production.gt(0)),('temperature_gt30',z.temperature.gt(30)),('recent_power_missing',z.origin.map(X.lag1.isna()) if month==7 else pd.Series(False,index=z.index))]:
   if mask.any():conditions.append(dict(month=month,model=name,condition=tag,**scores(z[mask])))
  c=float(z.threshold.iloc[0]);cut=pd.Timestamp(2021,month,1);end=cut+pd.offsets.MonthBegin();prev=y.shift();cross=y.gt(c)&prev.le(c)&prev.notna();events=y.index[cross&(y.index>cut)&(y.index<=end)]
  reachable=z[z.label].groupby('first_end')
  for e in events:
   if e in reachable.groups:
    a=reachable.get_group(e);warnings=a[a.warn];detected=len(warnings)>0;lead=((e-pd.Timedelta(minutes=15)-warnings.origin.min())/pd.Timedelta(minutes=1)) if detected else np.nan
    episodes.append(dict(month=month,model=name,event_end=e,event_start=e-pd.Timedelta(minutes=15),evaluable=True,detected=detected,available_origins=len(a),lead_to_interval_start_minutes=lead))
   else:episodes.append(dict(month=month,model=name,event_end=e,event_start=e-pd.Timedelta(minutes=15),evaluable=False,detected=False,available_origins=0,lead_to_interval_start_minutes=np.nan))
 pd.DataFrame(pooled).to_csv(OUT/'pooled_metrics.csv',index=False);pd.DataFrame(daily).to_csv(OUT/'daily_metrics.csv',index=False);pd.DataFrame(conditions).to_csv(OUT/'conditions.csv',index=False)
 ep=pd.DataFrame(episodes);ep.to_csv(OUT/'episodes.csv',index=False)
 summary=[]
 for (month,name),a in ep.groupby(['month','model']):
  reach=a[a.evaluable];ok=reach[reach.detected];summary.append(dict(month=month,model=name,total_entries=len(a),evaluable_entries=len(reach),detected_entries=len(ok),missed_entries=len(reach)-len(ok),episode_recall=len(ok)/len(reach) if len(reach) else np.nan,coverage=len(reach)/len(a) if len(a) else np.nan,lead_min=ok.lead_to_interval_start_minutes.min(),lead_median=ok.lead_to_interval_start_minutes.median()))
 pd.DataFrame(summary).to_csv(OUT/'episode_summary.csv',index=False)
 selected=selection['model'];conf=exports[(7,selected)];conf.update(model_file=f'model_7_{selected}.txt',event_definition='current<=JanMar_q95; first future crossing end15..120min; threshold fixed AprJul',selected_on='AprJun only',independent_validation=False);(OUT/'selected_model.json').write_text(json.dumps(conf,ensure_ascii=False,indent=2));X.loc[q[q.month.eq(7)&q.model.eq(selected)].origin.head(10),conf['features']].to_csv(OUT/'example_features.csv',index=False)
 # Verify executable saved text model reproduces selected July probabilities.
 z=q[q.month.eq(7)&q.model.eq(selected)];b=Booster(model_file=str(OUT/conf['model_file']));rp=apply_cal(b.predict(X.loc[z.origin,conf['features']]),conf['probability_calibration']);np.testing.assert_allclose(rp,z.prob,rtol=1e-10,atol=1e-12)
 assert ep[ep.detected].lead_to_interval_start_minutes.ge(0).all()
 manifest={'status':'complete','raw_audit':info,'checks':checks,'first_entry_time_check':'PASS','saved_model_reload':'PASS','selection':selection,'fit_count':8,'classifier_parameters':pars,'feature_sets':groups,'no_external_data':True,'constraints':'data availability delay0 assumed; July reused; no cost savings claim; all first-entry horizons15..120min are positive; no preparation-time assumption'}
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(selection),flush=True)
def current_feature_row(y,origin,conf):
 # Construct one snapshot from prior observations; no future target completeness required.
 origin=pd.Timestamp(origin);past=y.loc[y.index<=origin];c=conf['high_load_threshold']
 row={'hour':origin.hour,'dow':origin.dayofweek,'month':origin.month}
 for j in range(8):row[f'lag{j}']=y.get(origin-pd.Timedelta(minutes=15*j),np.nan)
 for which,days in [('day',1),('week',7)]:
  times=origin+pd.to_timedelta(np.arange(1,9)*15,unit='m')-pd.Timedelta(days=days)
  values=y.reindex(times).to_numpy();row[which]=values[0]
  row[which+'_early_max']=pd.Series(values[:2]).max();row[which+'_window_max']=pd.Series(values).max()
 for hours in [1,3,6,24]:
  vals=past.tail(hours*4)
  for key,v in [('mean',vals.mean()),('max',vals.max()),('std',vals.std())]:row[f'{key}{hours}']=v
  row[f'slope{hours}']=(row['lag0']-y.get(origin-pd.Timedelta(hours=hours),np.nan))/hours
 row['rise15']=row['lag0']-row['lag1'];row['threshold_headroom']=(c-row['lag0'])/c
 return pd.DataFrame([row])[conf['features']]

def predict_features(path,out):
 conf=json.loads((OUT/'selected_model.json').read_text());x=pd.read_csv(path);b=Booster(model_file=str(OUT/conf['model_file']));p=apply_cal(b.predict(x[conf['features']]),conf['probability_calibration']);ready=x.lag0.notna()&x.lag0.le(conf['high_load_threshold']);pd.DataFrame({'applicable':ready,'entry_probability':np.where(ready,p,np.nan),'warn':ready&(p>conf['decision_threshold'])}).to_csv(out,index=False)
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--predict-features');ap.add_argument('--at');ap.add_argument('--output',default=str(OUT/'example_inference.csv'));args=ap.parse_args()
 if args.predict_features:predict_features(args.predict_features,args.output)
 elif args.at:
  conf=json.loads((OUT/'selected_model.json').read_text());y,_,_=raw_load();features=current_feature_row(y,args.at,conf);features.to_csv(OUT/'live_example_features.csv',index=False);predict_features(OUT/'live_example_features.csv',args.output)
 else:run()
