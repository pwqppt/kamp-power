"""Preregistered reporting complements; never reselect from observed July results."""
import argparse
import hashlib
import json
import zipfile
from importlib.metadata import distribution
import numpy as np
import pandas as pd
from peak_study import ROOT, DATA, OUT, FOLDS, JULY, load, split, dump, scores, frame, read_forecasts
from experiment import model_for
from context_features import predict
from continue_research import confusion, routing

D=OUT/'research_v4'
D.mkdir(exist_ok=True)

def audit():
    raw=(DATA/'original.csv').read_bytes()
    zpath=ROOT/'work/official/resource-data.zip'
    official={}
    if zpath.exists():
        with zipfile.ZipFile(zpath) as z:
            n=next(n for n in z.namelist() if n.endswith('.csv'))
            b=z.read(n)
        official={'official_csv_sha256':hashlib.sha256(b).hexdigest(),'byte_identical':b==raw}
        assert b==raw
    source=pd.read_csv(DATA/'original.csv')
    long=pd.read_csv(DATA/'quarter_hour.csv',parse_dates=['date','interval_start','interval_end'])
    assert long.interval_start.is_unique
    assert long.interval_start.diff().dropna().eq(pd.Timedelta(minutes=15)).all()
    x,_=load(); rows=[]
    for start,end in [*FOLDS,JULY]:
        tr,v=split(x,start,end)
        days=v[['date','pattern']].drop_duplicates()
        rows.append(dict(fold=start[:7],train_days=tr.date.nunique(),validation_days=len(days),
          validation_days_pattern_seen=int(days.pattern.isin(tr.pattern).sum()),
          unique_train_patterns=tr.pattern.nunique(),unique_validation_patterns=v.pattern.nunique(),
          last_training_label=str(tr.interval_end.max()),first_origin=str(v.origin.min())))
    pd.DataFrame(rows).to_csv(D/'pattern_overlap.csv',index=False)
    dist=distribution('holidays')
    lic=[str(p) for p in dist.files if 'license' in str(p).lower()]
    license_text='\n'.join(dist.locate_file(p).read_text(encoding='utf8') for p in lic)
    dump(D/'data_verification.json',dict(source_sha256=hashlib.sha256(raw).hexdigest(),**official,
       source_rows=len(source),source_columns=source.columns.tolist(),power_rows=len(long),
       missing=source.isna().sum().to_dict(),duplicate_raw_rows=int(source.duplicated().sum()),
       chronological_unique_intervals=True,quarter_hour_production='hourly record repeated; NOT quarter-hour measured production',
       equipment_product_identifiers_available=False,holidays_license_files=lic,
       holidays_license_contains_MIT='MIT' in license_text,source_units_confirmed=False))
    print('AUDIT',official,flush=True)

def benchmark():
    x,_=load(); fs=json.loads((OUT/'feature_sets.json').read_text()); gs=json.loads((OUT/'context_v2/feature_groups.json').read_text())
    configs=[('Ridge_F2','Ridge',fs['F2'],'single'),('ExtraTrees_F2','ExtraTrees',fs['F2'],'single')]
    configs += [(f'LGB_{k}','LGB_medium',fs[k],'single') for k in ['F0','F1','F2','F3']]
    configs += [('LGB_B1','LGB_medium',gs['B1'],'single'),('Mixture_B1','LGB_medium',gs['B1'],'mixture')]
    parts=[]; foldrows=[]
    for start,end in FOLDS:
        tr,v=split(x,start,end)
        for name,model,cols,structure in configs:
            assert not any(c.startswith('actual_') for c in cols)
            if structure=='mixture': p,pr=predict(tr,v,cols,structure)
            else:
                m=model_for(model).fit(tr[cols],tr.y)
                p=np.maximum(m.predict(v[cols]),0);pr=np.full(len(v),np.nan)
            f=frame(v,p,pr,tr,name,start);parts.append(f)
            foldrows.append(dict(candidate=name,fold=start[:7],**scores(f)))
            print(name,start,round(scores(f)['mae'],4),flush=True)
    for name in ['reference','naive']:
        f=read_forecasts(name); f=f[(f.date>='2021-04-01')&(f.date<'2021-07-01')].copy();f['candidate']=name
        parts.append(f)
        foldrows += [dict(candidate=name,fold=fold,**scores(g)) for fold,g in f.groupby('fold')]
    allf=pd.concat(parts,ignore_index=True)
    allf.to_csv(D/'benchmark_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(foldrows).to_csv(D/'benchmark_fold_scores.csv',index=False)
    pd.DataFrame([dict(candidate=n,**scores(g)) for n,g in allf.groupby('candidate')]).to_csv(D/'benchmark_scores.csv',index=False)
    dump(D/'benchmark_decision.json',{'reselection':False,'retained':'reference','adopt_new_peak_method':False,
      'reason':'Reporting ablation only; bounded H1-H3 study closed. All periods previously observed.'})

def diagnose():
    x,_=load(); f=read_forecasts('reference');f=f[f.date>='2021-04-01'].copy()
    f=f.merge(x[['interval_start','dow','recent24_mean','recent7_mean','ctx_zero_fraction_24','power_lag2d','power_lag7d']],on='interval_start',validate='one_to_one')
    h=pd.read_csv(DATA/'hourly_audited.csv',parse_dates=['hour_start'])
    bad=pd.to_datetime(json.loads((OUT/'data_audit.json').read_text())['repaired_dates'])
    h.loc[h.hour_start.dt.normalize().isin(bad),'기온']=np.nan
    delta={}
    for d,g in f.groupby('date'):
        o=g.origin.iloc[0];hist=h[h.hour_start+pd.Timedelta(hours=1)<=o]
        a=hist[(hist.hour_start>=o-pd.Timedelta(hours=24))]['기온'].mean()
        b=hist[(hist.hour_start>=o-pd.Timedelta(hours=48))&(hist.hour_start<o-pd.Timedelta(hours=24))]['기온'].mean()
        delta[d]=a-b
    f['temperature_change_24h']=f.date.map(delta)
    f['recent_temp_change']=pd.cut(f.temperature_change_24h,[-np.inf,-2,2,np.inf],labels=['cooling<-2C','stable+-2C','warming>2C']).astype(str)
    f['weekday']=f.dow.map(dict(enumerate(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])))
    f['recent_power_ratio']=pd.cut(f.recent24_mean/f.recent7_mean.clip(lower=1),[-np.inf,.8,1.2,np.inf],labels=['<0.8','0.8-1.2','>1.2']).astype(str)
    f['recent_production_zero']=pd.cut(f.ctx_zero_fraction_24,[-.01,.25,.75,1.01],labels=['<=25%','25-75%','>75%']).astype(str)
    f['lag_disagreement']=pd.cut((f.power_lag2d-f.power_lag7d).abs(),[-1,20,50,np.inf],labels=['<=20','20-50','>50']).astype(str)
    f['production_record']=np.where(f.actual_production>0,'positive','zero')
    f['weekday_production']=f.weekday+'/'+f.production_record
    rows=[]
    for fold,part in f.groupby('fold'):
        for dim in ['recent_temp_change','weekday','recent_power_ratio','recent_production_zero','lag_disagreement','weekday_production']:
            for condition,g in part.groupby(dim,observed=True):
                e=g.pred-g.y; peak=g.y>=g.threshold
                rows.append(dict(fold=fold,dimension=dim,condition=condition,n=len(g),days=g.date.nunique(),
                 mae=e.abs().mean(),bias=e.mean(),peak_intervals=int(peak.sum()),
                 peak_underprediction=np.maximum(-e[peak],0).mean(),**confusion(g.y,g.pred,g.threshold)))
    pd.DataFrame(rows).to_csv(D/'extended_conditions.csv',index=False)
    f[['date','origin','temperature_change_24h','recent_temp_change','recent_power_ratio','recent_production_zero']].drop_duplicates().to_csv(D/'origin_condition_days.csv',index=False)
    # CI of retained-minus-naive errors; positive is worse. Pattern clusters are diagnostic only.
    rows=[];rng=np.random.default_rng(42)
    for period,g in [('Apr-Jun',f[f.fold<'2021-07']),('July',f[f.fold=='2021-07'])]:
        day=g.groupby(['date','pattern']).apply(lambda z:pd.Series({'mae_delta':(z.pred-z.y).abs().mean()-(z.naive-z.y).abs().mean(),
            'peak_mae_delta':abs(z.pred.max()-z.y.max())-abs(z.naive.max()-z.y.max())}),include_groups=False).reset_index()
        for method,ids in [('day',day.date),('pattern_cluster',day.pattern)]:
            keys=ids.unique();units=[day[ids==k] for k in keys]
            draws=[]
            for _ in range(2000):
                chosen=rng.integers(0,len(keys),len(keys));z=pd.concat([units[i] for i in chosen]);draws.append(z[['mae_delta','peak_mae_delta']].mean().to_numpy())
            draw=np.array(draws)
            for j,metric in enumerate(['mae_delta','peak_mae_delta']):
                rows.append(dict(period=period,method=method,metric=metric,clusters=len(keys),mean=day[metric].mean(),low=np.quantile(draw[:,j],.025),high=np.quantile(draw[:,j],.975)))
    pd.DataFrame(rows).to_csv(D/'uncertainty.csv',index=False)
    print('Extended conditions and two dependence-aware uncertainty summaries saved.',flush=True)

def production():
    f=read_forecasts('reference');f=f[f.date>='2021-04-01'];rows=[]
    for date,g in f.groupby('date'):
        g=g.sort_values('interval_start');r=routing(g.pred.to_numpy(),.10,4)
        production=g.actual_production.to_numpy()/4
        shifted=r@production
        assert np.isclose(production.sum(),shifted.sum(),atol=1e-7)
        rows.append(dict(date=str(date.date()),production_before=production.sum(),production_after=shifted.sum(),
         conservation_error=shifted.sum()-production.sum(),moved_production=np.sum((1-np.diag(r))*production),
         changed_slots=int((abs(shifted-production)>1e-8).sum()),fraction=.10,window_minutes=60))
    pd.DataFrame(rows).to_csv(D/'production_ledger.csv',index=False)
    dump(D/'production_assumptions.json',{'measured_production_preservation':False,'arithmetic_ledger_preservation':True,
      'assumptions':['hourly production divided equally into four artificial 15-minute entries','power and production use same fraction-transfer matrix','divisible, movable output; no setup, capacity, precedence, quality or delivery constraints'],
      'interpretation':'Conditional feasibility illustration only. Real production maintenance cannot be established from this dataset.'})
    print('Conditional production ledger checked, not a field production guarantee.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','benchmark','diagnose','production']);a=p.parse_args();globals()[a.stage]()
