"""Origin-aware day-ahead forecasting. No target-day actuals in deployable models."""
from pathlib import Path
import json, time, warnings, hashlib
import numpy as np
import pandas as pd
import holidays
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
import joblib
from audit import ROOT, DATA, OUT, run as audit_run

SEED=42
TEST_START=pd.Timestamp('2021-08-01')
FEATURE_SETS={}

def build_features():
    hourly,long,audit=audit_run()
    s=long.set_index('interval_start').power.astype(float).copy()
    bad_dates=pd.to_datetime(audit['repaired_dates'])
    s.loc[s.index.normalize().isin(bad_dates)]=np.nan
    hr=hourly.set_index('hour_start')
    kr=holidays.KR(years=[2020,2021])
    records=[]
    for d,g in long.groupby('date'):
        if d<pd.Timestamp('2021-01-22') or d in bad_dates: continue
        origin=d-pd.Timedelta(days=1)+pd.Timedelta(hours=16)
        hist=s.loc[(s.index+pd.Timedelta(minutes=15)<=origin)]
        recent=hist.loc[hist.index>=origin-pd.Timedelta(hours=24)]
        week=hist.loc[hist.index>=origin-pd.Timedelta(days=7)]
        hh=hr.loc[hr.index+pd.Timedelta(hours=1)<=origin]
        h24=hh.loc[hh.index>=origin-pd.Timedelta(hours=24)]
        origin_weather=hh.iloc[-1]
        g=g.sort_values('interval_start')
        for _,r in g.iterrows():
            t=r.interval_start
            slot=int((t-d).total_seconds()/900)
            f=dict(date=d,origin=origin,interval_start=t,interval_end=r.interval_end,
              pattern=r.pattern,y=float(r.power),slot=slot,hour=slot//4,
              dow=d.dayofweek,month=d.month,weekend=int(d.dayofweek>=5),
              holiday=int(d.date() in kr),after_holiday=int((d-pd.Timedelta(days=1)).date() in kr),
              slot_sin=np.sin(2*np.pi*slot/96),slot_cos=np.cos(2*np.pi*slot/96),
              doy_sin=np.sin(2*np.pi*d.dayofyear/365.25),doy_cos=np.cos(2*np.pi*d.dayofyear/365.25),
              horizon_hours=(t-origin).total_seconds()/3600,
              actual_production=r['생산량'],actual_temp=r['기온'],actual_personnel=r['공장인원'])
            for lag in [2,7,14]:
                source=t-pd.Timedelta(days=lag)
                assert source+pd.Timedelta(minutes=15)<=origin
                f[f'power_lag{lag}d']=s.get(source,np.nan)
            f.update(last_power=hist.iloc[-1],recent24_mean=recent.mean(),recent24_max=recent.max(),
              recent24_std=recent.std(),recent7_mean=week.mean(),recent7_max=week.max(),
              recent7_std=week.std(),history_missing=recent.isna().mean(),
              recent_ramp=hist.iloc[-1]-hist.iloc[-5])
            # Weather persistence, NOT an archived weather forecast.
            for source,name in [('기온','temp'),('습도','humidity'),('풍속','wind'),('강수량','rain')]:
                f[f'weather_last_{name}']=origin_weather[source]
                f[f'weather24_{name}']=h24[source].mean()
                f[f'weather_lag2_{name}']=hr.loc[t.floor('h')-pd.Timedelta(days=2),source]
            for source,name in [('생산량','production'),('공장인원','personnel')]:
                f[f'{name}_lag2']=hr.loc[t.floor('h')-pd.Timedelta(days=2),source]
                f[f'{name}_lag7']=hr.loc[t.floor('h')-pd.Timedelta(days=7),source]
                f[f'{name}_recent24']=h24[source].mean()
            f['lag_ramp']=s.get(t-pd.Timedelta(days=2),np.nan)-s.get(t-pd.Timedelta(days=2,minutes=15),np.nan)
            f['morning_monday']=int(d.dayofweek==0 and 7<=f['hour']<10)
            f['temp_x_production']=f['weather_last_temp']*f['production_lag2']
            records.append(f)
    x=pd.DataFrame(records)
    calendar=['slot','hour','dow','month','weekend','holiday','after_holiday','slot_sin','slot_cos','doy_sin','doy_cos','horizon_hours']
    lags=[c for c in x if c.startswith('power_lag')]
    stats=['last_power','recent24_mean','recent24_max','recent24_std','recent7_mean','recent7_max','recent7_std','history_missing','recent_ramp']
    weather=[c for c in x if c.startswith('weather_') or c.startswith('weather24_')]
    prod=[c for c in x if c.startswith('production_') or c.startswith('personnel_')]
    FEATURE_SETS.update(F0=calendar,F1=calendar+lags,F2=calendar+lags+stats,
      F3=calendar+lags+stats+weather,F4=calendar+lags+stats+weather+prod,
      F5=calendar+lags+stats+weather+prod+['lag_ramp','morning_monday','temp_x_production'])
    x.to_pickle(DATA/'features.pkl')
    (OUT/'feature_sets.json').write_text(json.dumps(FEATURE_SETS,indent=2),encoding='utf8')
    assert x.groupby('date').size().eq(96).all()
    assert all('actual_' not in c for fs in FEATURE_SETS.values() for c in fs)
    return x,audit

def model_for(name):
    if name=='Ridge':return make_pipeline(SimpleImputer(add_indicator=True),StandardScaler(),Ridge(alpha=100))
    if name=='ExtraTrees':return make_pipeline(SimpleImputer(add_indicator=True),ExtraTreesRegressor(n_estimators=140,min_samples_leaf=12,max_features=.8,n_jobs=4,random_state=SEED))
    settings={'LGB_small':(15,80,250,.035),'LGB_medium':(31,80,350,.035),'LGB_smooth':(7,150,350,.035)}
    leaves,child,n,lr=settings[name]
    return LGBMRegressor(n_estimators=n,num_leaves=leaves,min_child_samples=child,learning_rate=lr,
        colsample_bytree=.9,reg_lambda=5,n_jobs=4,random_state=SEED,verbosity=-1,deterministic=True,force_col_wise=True)

def metrics(frame):
    y=frame.y.to_numpy(); p=frame.pred.to_numpy()
    d=frame.groupby('date')[['y','pred','threshold']].max()
    actual=d.y>=d.threshold; alarm=d.pred>=d.threshold
    tp=int((actual&alarm).sum())
    return dict(n=len(frame),days=len(d),mae=float(np.abs(y-p).mean()),rmse=float(np.sqrt(np.mean((y-p)**2))),
       wape=float(np.abs(y-p).sum()/max(np.abs(y).sum(),1)),daily_peak_mae=float(np.abs(d.y-d.pred).mean()),
       peak_underprediction=float(np.maximum(d.y-d.pred,0).mean()),peak_day_recall=tp/max(int(actual.sum()),1),
       alarm_precision=tp/max(int(alarm.sum()),1),false_alarm_days=int((~actual&alarm).sum()))

def predict_fold(x,fs,name,start,end,purge=False):
    start=pd.Timestamp(start); end=pd.Timestamp(end)
    # At fold's first forecast origin, only earlier fully completed label-days allowed.
    cutoff=start-pd.Timedelta(days=1)+pd.Timedelta(hours=16)
    train=x.loc[x.interval_end<=cutoff].copy()
    # Require full target-day labels: don't fit a partial day that is later evaluated as a block.
    counts=train.groupby('date').size(); train=train.loc[train.date.isin(counts[counts==96].index)]
    test=x.loc[(x.date>=start)&(x.date<end)].copy()
    if purge:train=train.loc[~train.pattern.isin(test.pattern.unique())]
    cols=FEATURE_SETS[fs]
    m=model_for(name); m.fit(train[cols],train.y)
    out=test[['date','origin','interval_start','interval_end','pattern','slot','hour','y','actual_production','actual_temp','actual_personnel','holiday','dow']].copy()
    out['pred']=np.maximum(m.predict(test[cols]),0)
    out['seen_pattern']=out.pattern.isin(train.pattern.unique())
    out['threshold']=train.y.quantile(.95)
    out['fold']=start.strftime('%Y-%m')
    out['model']=name; out['features']=fs
    return out,m,train

def experiment():
    OUT.mkdir(exist_ok=True)
    x,audit=build_features()
    folds=[('2021-04-01','2021-05-01'),('2021-05-01','2021-06-01'),('2021-06-01','2021-07-01')]
    all_scores=[]; predictions=[]
    def evaluate(fs,name):
        pp=[]
        for start,end in folds:
            st=time.time(); p,_,_=predict_fold(x,fs,name,start,end)
            all_scores.append(dict(features=fs,model=name,fold=start[:7],**metrics(p),seconds=time.time()-st))
            pp.append(p)
        joined=pd.concat(pp); predictions.append(joined)
        score=dict(features=fs,model=name,**metrics(joined))
        print('VALIDATION',score,flush=True)
        return score
    ablations=[evaluate(fs,'LGB_small') for fs in FEATURE_SETS]
    pd.DataFrame(ablations).to_csv(OUT/'ablation_summary.csv',index=False)
    # Within 2% of best validation MAE choose smaller feature set, fewer unverifiable inputs.
    best_mae=min(a['mae'] for a in ablations)
    best_fs=next(a['features'] for a in ablations if a['mae']<=best_mae*1.02)
    candidates=[next(a for a in ablations if a['features']==best_fs)]
    for name in ['Ridge','ExtraTrees','LGB_medium','LGB_smooth']:candidates.append(evaluate(best_fs,name))
    # seasonal naive receives exactly the same target rows and fallback known at origin
    pp=[]
    for start,end in folds:
        p,_,_=predict_fold(x,best_fs,'LGB_small',start,end)
        xx=x.loc[(x.date>=start)&(x.date<end)]
        p['pred']=xx.power_lag7d.fillna(xx.power_lag2d).fillna(xx.last_power).to_numpy()
        p['model']='SeasonalNaive';p['features']='lag7d';pp.append(p)
    naive=pd.concat(pp);predictions.append(naive)
    candidates.append(dict(model='SeasonalNaive',features='lag7d',**metrics(naive)))
    table=pd.DataFrame(candidates)
    # Predeclared compromise: <=5% from best MAE, then smallest daily-peak MAE.
    ml_table=table[table.model!='SeasonalNaive']
    eligible=ml_table[ml_table.mae<=ml_table.mae.min()*1.05]
    chosen=eligible.sort_values(['daily_peak_mae','mae']).iloc[0]
    name=chosen.model
    pd.DataFrame(all_scores).to_csv(OUT/'fold_scores.csv',index=False)
    table.to_csv(OUT/'model_comparison.csv',index=False)
    pd.concat(predictions).to_csv(OUT/'validation_predictions.csv.gz',index=False)
    # July is independent calibration month, not used for model/feature selection.
    cal,_,_=predict_fold(x,best_fs,name,'2021-07-01','2021-08-01')
    residual=cal.y-cal.pred
    lo=float(residual.quantile(.05)); hi=float(residual.quantile(.95))
    # Day-level bound calibrated on daily max residual, distinct from marginal intervals.
    daily=cal.groupby('date')[['y','pred']].max()
    rank=min(int(np.ceil((len(daily)+1)*.90))-1,len(daily)-1)
    peak_margin=float(np.sort((daily.y-daily.pred).to_numpy())[rank])
    spec=dict(features=best_fs,model=name,selected_validation_model=str(chosen.model),
       best_validation_overall_MAE=str(table.sort_values('mae').iloc[0].model),
       selection='Apr-Jun only; features <=2% min MAE prefer fewer; ML candidates <=5% min MAE prefer peak MAE. SeasonalNaive remains explicit competitor even if better.',
       calibration='July held out for residual intervals; no test-driven adjustment.',
       residual_q05=lo,residual_q95=hi,day_peak_upper_margin90=peak_margin,
       test_start='2021-08-01',test_end='2021-09-14',seed=SEED)
    (OUT/'frozen_selection.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
    cal.to_csv(OUT/'calibration_predictions.csv',index=False)
    test,m,train=predict_fold(x,best_fs,name,'2021-08-01','2021-09-15')
    test['lower90']=np.maximum(test.pred+lo,0);test['upper90']=np.maximum(test.pred+hi,0)
    xx=x.loc[x.date>=TEST_START]
    test['naive']=xx.power_lag7d.fillna(xx.power_lag2d).fillna(xx.last_power).to_numpy()
    test['abs_error']=np.abs(test.y-test.pred)
    test.to_csv(OUT/'test_predictions.csv',index=False)
    joblib.dump(dict(model=m,features=FEATURE_SETS[best_fs],selection=spec),OUT/'final_model.joblib')
    scores=[]
    for subset,mask in [('all',np.ones(len(test),bool)),('unseen_patterns',~test.seen_pattern),('seen_patterns',test.seen_pattern)]:
        a=test.loc[mask].copy()
        if len(a):
            scores.append(dict(model=name,subset=subset,**metrics(a)))
            a['pred']=a.naive;scores.append(dict(model='SeasonalNaive',subset=subset,**metrics(a)))
    pd.DataFrame(scores).to_csv(OUT/'test_scores.csv',index=False)
    coverage=float(((test.y>=test.lower90)&(test.y<=test.upper90)).mean())
    dd=test.groupby('date')[['y','pred']].max()
    peak_coverage=float((dd.y<=dd.pred+peak_margin).mean())
    summary=dict(selection=spec,test_scores=scores,pointwise_90_coverage=coverage,
      day_peak_90_upper_coverage=peak_coverage,interval_mean_width=float((test.upper90-test.lower90).mean()),
      test_days_seen_pattern=int(test.groupby('date').seen_pattern.first().sum()),
      train_days=int(train.date.nunique()),latest_training_label=str(train.interval_end.max()))
    (OUT/'experiment_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print('FINAL',json.dumps(summary,ensure_ascii=False),flush=True)
    # Honest auxiliary/oracle experiments: NEVER promoted as deployable performance.
    oracle_cols=FEATURE_SETS[best_fs]+['actual_production','actual_temp','actual_personnel']
    oracle=model_for('LGB_small');oracle.fit(train[oracle_cols],train.y)
    aux=test.copy();aux['pred']=np.maximum(oracle.predict(xx[oracle_cols]),0)
    rows=[dict(scenario='actual_future_inputs_ORACLE_ONLY',**metrics(aux))]
    for factor in [.8,.9,1.1,1.2]:
        pert=xx[oracle_cols].copy();pert['actual_production']*=factor
        aux['pred']=np.maximum(oracle.predict(pert),0)
        rows.append(dict(scenario=f'oracle_production_times_{factor}',**metrics(aux)))
    pd.DataFrame(rows).to_csv(OUT/'oracle_sensitivity_NOT_DEPLOYABLE.csv',index=False)
    # Test-duplicate purge is a separate forensic robustness diagnostic, never selection.
    purged,_,ptrain=predict_fold(x,best_fs,name,'2021-08-01','2021-09-15',purge=True)
    pd.DataFrame([dict(train_days=ptrain.date.nunique(),**metrics(purged))]).to_csv(OUT/'purged_diagnostic.csv',index=False)
    return x,test,m

if __name__=='__main__':experiment()
