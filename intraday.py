"""Hourly-issued pooled direct 15..120min forecasts. See docs/INTRADAY_PROTOCOL.md."""
from pathlib import Path
import json, hashlib, platform, importlib.metadata, argparse
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/intraday_v1'; OUT.mkdir(parents=True,exist_ok=True)
COV=['생산량','기온','습도','공장인원']
PARAMS={'ridge':{'alpha':10.0},'lgb':{'n_estimators':300,'num_leaves':15,'learning_rate':0.05,'min_child_samples':80,'reg_lambda':5,'random_state':42,'n_jobs':2,'verbosity':-1,'objective':'regression'},'cat':{'iterations':300,'depth':5,'learning_rate':0.05,'l2_leaf_reg':5,'loss_function':'RMSE','random_seed':42,'thread_count':2,'verbose':False,'allow_writing_files':False}}

def raw_load():
    raw=pd.read_csv(ROOT/'data/original.csv')
    sha=hashlib.sha256((ROOT/'data/original.csv').read_bytes()).hexdigest()
    assert sha=='8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830'
    d=pd.to_datetime(raw['날짜'].astype(str))
    assert raw.groupby('날짜').size().eq(24).all()
    invalid=~raw['시간'].between(0,23)
    bad_days=d[invalid].unique()
    # Row-order replacement ONLY for invalid dates; all values on those dates masked.
    hour=raw['시간'].copy(); hour[d.isin(bad_days)]=raw.groupby('날짜').cumcount()[d.isin(bad_days)]
    start=d+pd.to_timedelta(hour,unit='h')
    idx=pd.DatetimeIndex(np.repeat(start.to_numpy(),4)+np.tile(np.array([15,30,45,60],dtype='timedelta64[m]'),len(raw)))
    y=pd.Series(raw[['15분','30분','45분','60분']].to_numpy().ravel().astype(float),index=idx)
    assert y.index.is_unique and y.index.is_monotonic_increasing
    y.iloc[np.repeat(d.isin(bad_days).to_numpy(),4)]=np.nan
    cov=raw[COV].copy(); cov.index=pd.DatetimeIndex(start+pd.Timedelta(hours=1)); cov.loc[d.isin(bad_days).to_numpy(),:]=np.nan
    return y,cov,{'sha256':sha,'raw_rows':len(raw),'invalid_hour_rows':int(invalid.sum()),'excluded_dates':[str(pd.Timestamp(x).date()) for x in bad_days]}

def build(y,cov,delay=0,truth=None):
    origins=pd.date_range('2021-01-08','2021-07-31 22:00',freq='h')
    o=pd.DatetimeIndex(np.repeat(origins.to_numpy(),8)); h=np.tile(np.arange(1,9),len(origins)); target=o+pd.to_timedelta(h*15,unit='m'); avail=o-pd.Timedelta(minutes=delay)
    x=pd.DataFrame(index=np.arange(len(o)))
    x['horizon']=h; x['hour']=target.hour; x['quarter']=target.minute//15; x['dow']=target.dayofweek; x['month']=target.month
    for j in range(8): x[f'lag{j}']=y.reindex(avail-pd.Timedelta(minutes=15*j)).to_numpy()
    for name,days in [('day',1),('week',7)]: x[name]=y.reindex(target-pd.Timedelta(days=days)).to_numpy()
    groups={'F0':list(x.columns)}
    for hours in [1,3,6,24]:
        r=y.rolling(hours*4,min_periods=1)
        for name,s in [('mean',r.mean()),('max',r.max()),('std',r.std())]: x[f'{name}{hours}']=s.reindex(avail).to_numpy()
        x[f'slope{hours}']=(y-y.shift(hours*4)).reindex(avail).to_numpy()/hours
    x['rise15']=x.lag0-x.lag1
    groups['F1']=list(x.columns)
    for c in COV:
        x[c]=cov[c].reindex(avail).to_numpy(); x[c+'_change']=cov[c].diff().reindex(avail).to_numpy()
        valid_time=pd.Series(cov.index.where(cov[c].notna()),index=cov.index).ffill()
        x[c+'_age_hours']=(avail-pd.DatetimeIndex(valid_time.reindex(avail)))/pd.Timedelta(hours=1)
        x[c+'_missing']=x[c].isna().astype(int)
    groups['F2']=list(x.columns)
    meta=pd.DataFrame({'origin':o,'target':target,'horizon':h,'y':(y if truth is None else truth).reindex(target).to_numpy(),'current':x.lag0,'production':x['생산량'],'temperature':x['기온']})
    # Preserve all origins, including missing covariates, except any invalid/missing target in window.
    valid=meta.groupby('origin').y.transform(lambda z:z.notna().all()).astype(bool)
    return x.loc[valid].reset_index(drop=True),meta.loc[valid].reset_index(drop=True),groups

def make_model(name):
    if name=='ridge':return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),Ridge(**PARAMS[name]))
    if name=='cat':return make_pipeline(SimpleImputer(strategy='constant',fill_value=-999,add_indicator=True),CatBoostRegressor(**PARAMS[name]))
    return LGBMRegressor(**PARAMS[name])

def metrics(p):
    a=p.y.to_numpy(); b=p.pred.to_numpy(); c=p.threshold.to_numpy(); pos=a>c; alert=b>c
    tp=int((pos&alert).sum()); fp=int((~pos&alert).sum()); fn=int((pos&~alert).sum())
    z=p.groupby('origin').agg(y=('y','max'),pred=('pred','max'),threshold=('threshold','first'),current=('current','first'))
    event=z.y>z.threshold; warn=z.pred>z.threshold; onset=z.current.le(z.threshold)&z.current.notna()
    return {'n':len(p),'origins':len(z),'MAE':float(np.mean(abs(a-b))),'RMSE':float(np.sqrt(np.mean((a-b)**2))),'bias':float(np.mean(b-a)), 'recall':tp/(tp+fn) if tp+fn else np.nan,'precision':tp/(tp+fp) if tp+fp else np.nan,'FP':fp,'FN':fn,'positive_slots':int(pos.sum()),'window_max_MAE':float(abs(z.y-z.pred).mean()),'window_recall':float(warn[event].mean()) if event.any() else np.nan,'onset_events':int((event&onset).sum()),'onset_recall':float(warn[event&onset].mean()) if (event&onset).any() else np.nan}

def audit(y,cov,x,m,groups):
    t=pd.Timestamp('2021-06-15 12:00')
    yy=y.copy(); yy.loc[yy.index>t]=99999
    cc=cov.copy(); cc.loc[cc.index>t]=99999
    xx,mm,_=build(yy,cc)
    ix=m.origin.eq(t); jx=mm.origin.eq(t)
    np.testing.assert_allclose(x.loc[ix].to_numpy(),xx.loc[jx].to_numpy(),equal_nan=True)
    assert m.groupby('origin').size().eq(8).all()
    assert (m.target>m.origin).all()
    return {'future_input_perturbation':'PASS','complete_eight_targets':'PASS','target_after_origin':'PASS','feature_count':{k:len(v) for k,v in groups.items()},'n_origins':int(m.origin.nunique()),'forecast_design':'pooled direct model with horizon feature','cat_missing':'constant -999 plus missing indicators, training fitted','delay_minutes':0}

def run():
    y,cov,info=raw_load(); x,m,groups=build(y,cov)
    checks=audit(y,cov,x,m,groups)
    manifest={'parameters':PARAMS,'checks':checks,'audit':info,'python':platform.python_version(),'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','lightgbm','catboost']},'status':'started'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    results=[]; scores=[]
    for month in [4,5,6,7]:
        cut=pd.Timestamp(2021,month,1); end=cut+pd.offsets.MonthBegin()
        # All 8 labels must be available before train boundary.
        train=(m.origin+pd.Timedelta(hours=2)<cut)
        test=(m.origin>=cut)&(m.origin<end)
        assert m.loc[train,'target'].max()<cut
        threshold=float(y.loc[y.index<cut].quantile(.95))
        if month<7:
            candidates=[(n,f) for n in ['ridge','lgb','cat'] for f in groups]
        else:
            val=pd.DataFrame(scores); means=val.groupby('model').apply(lambda z:np.average(z.MAE,weights=z.n),include_groups=False)
            learned=means[means.index.str.contains('_F')]; near=learned[learned<=learned.min()*1.01]
            selected=sorted(near.index,key=lambda s:(int(s[-1]),float(near[s]),s))[0]
            bestbase=means[~means.index.str.contains('_F')].idxmin()
            selection={'learned_candidate':selected,'best_baseline':bestbase,'validation_MAE':means.to_dict(),'selected_for_forecasting':selected if means[selected]<means[bestbase] else bestbase,'independent_test':False}
            (OUT/'selection.json').write_text(json.dumps(selection,indent=2))
            n,f=selected.rsplit('_',1); candidates=[(n,f)]
        xx=x.loc[test]; last=xx.lag0
        # Prior-only fallback, never a future or whole-series statistic.
        fallback=y.ffill().reindex(pd.DatetimeIndex(m.loc[test,'origin'])).to_numpy()
        baselines={'last':last,'hour_mean':xx[['lag0','lag1','lag2','lag3']].mean(axis=1),'day':xx.day,'week':xx.week}
        for name,s in baselines.items():
            pred=s.fillna(pd.Series(fallback,index=s.index)).to_numpy()
            p=m.loc[test].copy(); p['pred']=pred; p['model']=name; p['month']=month; p['threshold']=threshold; p['fallback_used']=s.isna().to_numpy()
            assert np.isfinite(pred).all(); results.append(p); scores.append({'month':month,'model':name,**metrics(p)})
        for name,f in candidates:
            model=make_model(name); model.fit(x.loc[train,groups[f]],m.loc[train,'y'])
            pred=np.maximum(0,model.predict(xx[groups[f]]))
            p=m.loc[test].copy(); p['pred']=pred; p['model']=name+'_'+f; p['month']=month; p['threshold']=threshold; p['fallback_used']=False
            assert np.isfinite(pred).all(); results.append(p); scores.append({'month':month,'model':name+'_'+f,**metrics(p)})
            print(month,name,f,'MAE',round(scores[-1]['MAE'],4),flush=True)
        pd.DataFrame(scores).to_csv(OUT/'monthly_metrics.csv',index=False)
        pd.concat(results).to_csv(OUT/'predictions.csv.gz',index=False,compression='gzip')
    p=pd.concat(results)
    detail=[]
    for (month,model,h),g in p.groupby(['month','model','horizon']):detail.append({'month':month,'model':model,'minutes':int(h*15),**metrics(g)})
    pd.DataFrame(detail).to_csv(OUT/'horizon_metrics.csv',index=False)
    detail=[]
    for (month,model),g in p.groupby(['month','model']):
        for cond,mask in [('production_zero',g.production.eq(0)),('production_positive',g.production.gt(0)),('temperature_gt30',g.temperature.gt(30)),('temperature_le30',g.temperature.le(30))]:
            if mask.any():detail.append({'month':month,'model':model,'condition':cond,**metrics(g.loc[mask])})
    pd.DataFrame(detail).to_csv(OUT/'condition_metrics.csv',index=False)
    manifest['status']='regression_complete; direct classification/simulation pending'
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print(json.dumps(selection,indent=2),flush=True)
if __name__=='__main__':run()
