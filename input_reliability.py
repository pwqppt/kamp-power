"""Controlled input-block ablation. See docs/INPUT_RELIABILITY_PROTOCOL.md."""
from pathlib import Path
import json, hashlib, importlib.metadata
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, brier_score_loss
from intraday import raw_load, build, audit, make_model, metrics as base_metrics, PARAMS, COV
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/input_reliability_v1';OUT.mkdir(parents=True,exist_ok=True)

def metrics(p):
 z=base_metrics(p);z['FPR']=z['FP']/(z['n']-z['positive_slots']) if z['n']>z['positive_slots'] else np.nan
 return z

def classification_metrics(a,p,t):
 a=np.asarray(a,dtype=bool);p=np.asarray(p);b=p>t
 tp=int((a&b).sum());fp=int((~a&b).sum());fn=int((a&~b).sum());tn=int((~a&~b).sum())
 return dict(n=len(a),positives=int(a.sum()),TP=tp,FP=fp,FN=fn,TN=tn,recall=tp/(tp+fn) if tp+fn else np.nan,precision=tp/(tp+fp) if tp+fp else np.nan,FPR=fp/(fp+tn) if fp+tn else np.nan,AP=average_precision_score(a,p) if a.any() else np.nan,Brier=brier_score_loss(a,p))

def run():
 y,cov,info=raw_load();x,m,old=build(y,cov);checks=audit(y,cov,x,m,old)
 def block(c):return [c,c+'_change',c+'_age_hours',c+'_missing']
 recent=[c for c in old['F0'] if c not in ['day','week']]
 groups={'recent':recent,'F0':old['F0'],'F1':old['F1'],'production':old['F1']+block('생산량'),'temperature':old['F1']+block('기온'),'humidity':old['F1']+block('습도'),'weather':old['F1']+block('기온')+block('습도'),'personnel':old['F1']+block('공장인원'),'all':old['F2']}
 manifest={'status':'running','raw_audit':info,'checks':checks,'features':groups,'parameters':PARAMS['lgb'],'selection':'AprJun: regression within1% minimum pooledMAE then minimal feature count; classification within1% maximum pooledAP then minimal feature count','July':'reused diagnostic, all candidates; not selection','versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','lightgbm','catboost']},'bootstrap':'paired day errors, circular moving blocks7days,2000 draws','external_data':'none newly used'}
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 # Delayed covariates only; y and labels unchanged. No refitting for stress tests.
 lagcov=cov.copy();lagcov.index=lagcov.index+pd.Timedelta(hours=1)
 xl,ml,_=build(y,lagcov);assert m[['origin','target']].equals(ml[['origin','target']])
 rng=np.random.default_rng(2026);mcov=cov.copy();missing=rng.random(len(mcov))<.2;mcov.loc[missing,:]=np.nan
 xm,mm,_=build(y,mcov);assert m[['origin','target']].equals(mm[['origin','target']])
 delay_fraction=float(missing.mean())
 reg=[];cl=[];rr=[];cr=[];stress=[];importance=[]
 for month in [4,5,6,7]:
  cut=pd.Timestamp(2021,month,1);end=cut+pd.offsets.MonthBegin();calstart=cut-pd.Timedelta(days=28)
  train=m.origin+pd.Timedelta(hours=2)<cut
  fit=m.origin+pd.Timedelta(hours=2)<calstart
  cal=m.origin.ge(calstart)&train
  # Keep target midnight assigned to previous raw date, exclude next-month actuals.
  date=(m.target-pd.Timedelta(minutes=1)).dt.normalize()
  test=m.origin.ge(cut)&m.origin.lt(end)&date.dt.month.eq(month)
  threshold=float(y[y.index<cut].quantile(.95));a=m.y>threshold
  assert m.loc[train,'target'].max()<cut and m.loc[fit,'target'].max()<calstart
  for name,cols in groups.items():
   model=make_model('lgb');model.fit(x.loc[train,cols],m.loc[train,'y'])
   pred=np.maximum(0,model.predict(x.loc[test,cols]));p=m.loc[test].copy();p['pred']=pred;p['model']=name;p['month']=month;p['threshold']=threshold;p['date']=(p.target-pd.Timedelta(minutes=1)).dt.normalize();reg.append(p)
   for scope,mask in [('15_120',p.horizon.ge(1)),('45_120',p.horizon.ge(3))]:rr.append(dict(month=month,model=name,scope=scope,**metrics(p[mask])))
   for c,v in zip(cols,model.feature_importances_):importance.append({'month':month,'model':name,'feature':c,'split_count':int(v)})
   extra=[c for c in cols if c not in old['F1']]
   if extra:
    for kind,xx in [('cov_delay60',xl),('cov_missing20',xm)]:
     xc=x.loc[test,cols].copy();xc.loc[:,extra]=xx.loc[test,extra].to_numpy();z=p.copy();z['pred']=np.maximum(0,model.predict(xc));stress.append(dict(month=month,model=name,stress=kind,**metrics(z)))
   pars={k:v for k,v in PARAMS['lgb'].items() if k!='objective'}
   clf=LGBMClassifier(**pars);clf.fit(x.loc[fit,cols],a[fit]);cp=clf.predict_proba(x.loc[cal,cols])[:,1];pr=clf.predict_proba(x.loc[test,cols])[:,1]
   pp=m.loc[test].copy();pp['prob']=pr;pp['label']=a[test].to_numpy();pp['month']=month;pp['model']=name;pp['date']=p.date
   for scope,hmin in [('15_120',1),('45_120',3)]:
    calib=(~a[cal]).to_numpy()&m.loc[cal,'horizon'].ge(hmin).to_numpy()
    t=float(np.quantile(cp[calib],.95,method='higher'));q=pp[pp.horizon.ge(hmin)].copy();q['prob_threshold']=t;q['warn']=q.prob>t;q['scope']=scope;cl.append(q)
    cr.append(dict(month=month,model=name,scope=scope,prob_threshold=t,**classification_metrics(q.label,q.prob,t)))
   print(f'{month} {name}: MAE {abs(p.y-p.pred).mean():.4f}',flush=True)
  pd.DataFrame(rr).to_csv(OUT/'regression_monthly.csv',index=False);pd.DataFrame(cr).to_csv(OUT/'classification_monthly.csv',index=False);pd.DataFrame(stress).to_csv(OUT/'covariate_stress.csv',index=False)
  if month==6:
   # Freeze selection before looking at July outputs; July fits are diagnostic only.
   pv=pd.concat(reg);vv=pv[pv.month.lt(7)].copy();vv['ae']=abs(vv.y-vv.pred);mae=vv.groupby('model').ae.mean();near=mae[mae<=mae.min()*1.01]
   selected_reg=min(near.index,key=lambda n:(len(groups[n]),float(near[n]),n))
   cv=pd.concat(cl);cv=cv[cv.month.lt(7)&cv.scope.eq('45_120')];ap=cv.groupby('model').apply(lambda q:average_precision_score(q.label,q.prob),include_groups=False);near=ap[ap>=ap.max()*.99]
   selected_cl=min(near.index,key=lambda n:(len(groups[n]),-float(near[n]),n))
   selection={'regression':selected_reg,'classification_45_120':selected_cl,'validation_MAE':mae.to_dict(),'validation_AP':ap.to_dict(),'selection_rule':manifest['selection'],'frozen_before_July':True,'independent_test':False}
   (OUT/'selection.json').write_text(json.dumps(selection,indent=2))
 p=pd.concat(reg,ignore_index=True);c=pd.concat(cl,ignore_index=True)
 p.to_csv(OUT/'predictions.csv.gz',index=False,compression='gzip');c.to_csv(OUT/'classification_predictions.csv.gz',index=False,compression='gzip')
 pd.DataFrame(importance).to_csv(OUT/'feature_split_counts.csv',index=False)
 summarize(p,c,manifest,groups,selection,delay_fraction)

def summarize(p,c,manifest,groups,selection,delay_fraction):
 # Complete FPR reporting for saved monthly/stress summaries from older runs.
 for filename in ['regression_monthly.csv','covariate_stress.csv']:
  table=pd.read_csv(OUT/filename);table['FPR']=table.FP/(table.n-table.positive_slots).replace(0,np.nan);table.to_csv(OUT/filename,index=False)
 # Repeat statuses never used by fitting, calibration or selection.
 member=pd.read_csv(ROOT/'outputs/repeat_audit/daily_membership.csv',parse_dates=['date']).set_index('date')
 p['repeated_anywhere']=p.date.map(member.repeated);p['seen_training_curve']=p.date.map(member.seen_complete_training_curve)
 strata=[]
 for (month,name),q in p.groupby(['month','model']):
  for field in ['repeated_anywhere','seen_training_curve']:
   for value,g in q.groupby(field):strata.append(dict(month=month,model=name,field=field,value=bool(value),target_days=g.date.nunique(),**metrics(g)))
 pd.DataFrame(strata).to_csv(OUT/'retrospective_strata.csv',index=False)
 cond=[]
 for (month,name),q in p.groupby(['month','model']):
  for tag,mask in [('production_zero',q.production.eq(0)),('production_positive',q.production.gt(0)),('temperature_gt30',q.temperature.gt(30)),('temperature_le30',q.temperature.le(30))]:
   if mask.any():cond.append(dict(month=month,model=name,condition=tag,**metrics(q[mask])))
 pd.DataFrame(cond).to_csv(OUT/'known_input_conditions.csv',index=False)
 p['ae']=abs(p.y-p.pred);daily=p.groupby(['month','date','model']).agg(MAE=('ae','mean'),n=('ae','size')).reset_index();daily.to_csv(OUT/'daily_errors.csv',index=False)
 boot=[];rng=np.random.default_rng(42)
 for period,mask in [('AprJun',daily.month.lt(7)),('July',daily.month.eq(7))]:
  wide=daily[mask].pivot(index='date',columns='model',values='MAE')
  draws=rng.integers(0,len(wide),size=(2000,int(np.ceil(len(wide)/7))));ix=(draws[:,:,None]+np.arange(7)[None,None,:])%len(wide);ix=ix.reshape(2000,-1)[:,:len(wide)]
  for base in ['F0','F1']:
   for name in groups:
    if name==base:continue
    delta=(wide[name]-wide[base]).to_numpy();est=delta[ix].mean(axis=1);lo,hi=np.quantile(est,[.025,.975]);boot.append(dict(period=period,model=name,comparator=base,days=len(wide),mean_daily_MAE_difference=delta.mean(),CI_low=lo,CI_high=hi,improved_days=int((delta<0).sum())))
 pd.DataFrame(boot).to_csv(OUT/'paired_week_block_bootstrap.csv',index=False)
 pooled=[]
 for period,mask in [('AprJun',p.month.lt(7)),('July',p.month.eq(7))]:
  for name,q in p[mask].groupby('model'):pooled.append(dict(period=period,task='regression',model=name,scope='15_120',**metrics(q)))
  for name,q in c[c.month.lt(7) if period=='AprJun' else c.month.eq(7)].groupby('model'):
   for scope,z in q.groupby('scope'):
    # Per-month thresholds may differ: stored warning, not one pooled probability cutoff.
    a=z.label.to_numpy(dtype=bool);b=z.warn.to_numpy(dtype=bool);tp=int((a&b).sum());fp=int((~a&b).sum());fn=int((a&~b).sum());tn=int((~a&~b).sum())
    pooled.append(dict(period=period,task='classification',model=name,scope=scope,n=len(z),AP=average_precision_score(a,z.prob),Brier=brier_score_loss(a,z.prob),recall=tp/(tp+fn) if tp+fn else np.nan,precision=tp/(tp+fp) if tp+fp else np.nan,FPR=fp/(fp+tn) if fp+tn else np.nan,TP=tp,FP=fp,FN=fn,TN=tn))
 pd.DataFrame(pooled).to_csv(OUT/'pooled_metrics.csv',index=False)
 manifest.update(status='complete',selection=selection,injected_missing_fraction=delay_fraction,regression_rows=len(p),classifier_rows=len(c),retrospective_strata='future full-day comparison only; not deployable input/gate',missing_handling='LightGBM native NaN, bad dates masked in raw_load; no target clipping beyond nonnegative prediction',causal_inference=False)
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(selection,indent=2),flush=True)
if __name__=='__main__':
 import sys
 if '--summarize-only' in sys.argv:
  manifest=json.loads((OUT/'manifest.json').read_text());selection=json.loads((OUT/'selection.json').read_text())
  p=pd.read_csv(OUT/'predictions.csv.gz',parse_dates=['origin','target','date']);c=pd.read_csv(OUT/'classification_predictions.csv.gz',parse_dates=['origin','target','date'])
  _,cov,_=raw_load();fraction=float((np.random.default_rng(2026).random(len(cov))<.2).mean())
  summarize(p,c,manifest,manifest['features'],selection,fraction)
 else:run()
