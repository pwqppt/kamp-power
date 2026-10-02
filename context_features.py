"""Origin-only contextual features and factorial model experiments; exploratory v2."""
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, brier_score_loss
from experiment import DATA, OUT, model_for, metrics
from continue_research import confusion

D=OUT/'context_v2'

def context(g, hourly, power):
    origin=g.origin.iloc[0];date=g.date.iloc[0]
    h=hourly[hourly.hour_start+pd.Timedelta(hours=1)<=origin]
    s=power[power.index+pd.Timedelta(minutes=15)<=origin]
    assert h.hour_start.max()+pd.Timedelta(hours=1)<=origin
    assert s.index.max()+pd.Timedelta(minutes=15)<=origin
    out=pd.DataFrame(index=g.index); a=[];b=[]
    def scalar(name,value): out[name]=value; a.append(name)
    for hours in [6,24,72]:
        z=h[h.hour_start>=origin-pd.Timedelta(hours=hours)]
        scalar(f'ctx_production_mean_{hours}',z['생산량'].mean())
        scalar(f'ctx_personnel_mean_{hours}',z['공장인원'].mean())
        observed=z['생산량'].dropna()
        scalar(f'ctx_zero_fraction_{hours}',(observed==0).mean())
        scalar(f'ctx_observed_hours_{hours}',len(observed))
    active=h.loc[h['생산량']>0,'hour_start']
    scalar('ctx_hours_since_positive', (origin-(active.iloc[-1]+pd.Timedelta(hours=1))).total_seconds()/3600 if len(active) else np.nan)
    for days,label in [(1,'latest'),(8,'weekago')]:
        start=date-pd.Timedelta(days=days)+pd.Timedelta(hours=7)
        end=start+pd.Timedelta(hours=9)
        z=h[(h.hour_start>=start)&(h.hour_start<end)]
        q=s[(s.index>=start)&(s.index<end)]
        scalar(f'ctx_day_prod_{label}',z['생산량'].mean())
        scalar(f'ctx_day_power_{label}',q.mean())
    for prefix in ['prod','power']:
        scalar(f'ctx_day_{prefix}_ratio',out[f'ctx_day_{prefix}_latest'].iloc[0]/max(out[f'ctx_day_{prefix}_weekago'].iloc[0],1))
    template=[];sameweek=[]
    for t in g.interval_start:
        stamps=[t-pd.Timedelta(days=d) for d in range(2,16)]
        assert max(stamps)+pd.Timedelta(minutes=15)<=origin
        vals=s.reindex(stamps)
        template.append([vals.median(),vals.std(),vals.min(),vals.max(),vals.notna().sum()])
        sameweek.append(s.reindex([t-pd.Timedelta(days=7),t-pd.Timedelta(days=14)]).mean())
    for i,name in enumerate(['median','std','min','max','count']):
        c='ctx_slot14_'+name;out[c]=np.asarray(template)[:,i];b.append(c)
    out['ctx_week_template']=sameweek;b.append('ctx_week_template')
    out['ctx_scaled_template']=out.ctx_week_template*out.ctx_day_power_ratio;b.append('ctx_scaled_template')
    return out,a,b

def build():
    x=pd.read_pickle(DATA/'features.pkl')
    h=pd.read_csv(DATA/'hourly_audited.csv',parse_dates=['hour_start'])
    long=pd.read_csv(DATA/'quarter_hour.csv',parse_dates=['interval_start','date'])
    s=long.set_index('interval_start').power.astype(float)
    bad=pd.to_datetime(json.loads((OUT/'data_audit.json').read_text())['repaired_dates'])
    s.loc[s.index.normalize().isin(bad)]=np.nan
    # Repaired hourly records are not trusted as contextual covariates either.
    h.loc[h.hour_start.dt.normalize().isin(bad),['생산량','공장인원']]=np.nan
    blocks=[]
    for _,g in x.groupby('date',sort=True):
        z,a,b=context(g,h,s);blocks.append(z)
    added=pd.concat(blocks).sort_index();x=x.join(added)
    sets=json.loads((OUT/'feature_sets.json').read_text());base=sets['F2']
    extra=[c for c in sets['F4'] if c not in base]
    x['ctx_positive_lag_agreement']=((x.production_lag2>0)==(x.production_lag7>0)).astype(float)
    x.loc[x[['production_lag2','production_lag7']].isna().any(axis=1),'ctx_positive_lag_agreement']=np.nan
    x['ctx_recent_vs_week_power']=x.recent24_mean/x.recent7_mean.clip(lower=1)
    x['ctx_business_zero']=((x.hour>=7)&(x.hour<17)).astype(int)*x.ctx_zero_fraction_24
    x['ctx_weekend_zero']=x.weekend*x.ctx_zero_fraction_24
    extra+=['ctx_positive_lag_agreement','ctx_recent_vs_week_power','ctx_business_zero','ctx_weekend_zero']
    groups={'B0':base,'B1':base+a,'B2':base+a+b,'B3':base+a+b+extra}
    assert all(not c.startswith('actual_') for cols in groups.values() for c in cols)
    # Actual future measurements perturbation must not change pre-origin context.
    g=x[x.date==pd.Timestamp('2021-06-15')]
    h2=h.copy();h2.loc[h2.hour_start>=g.origin.iloc[0],['생산량','공장인원']]=999999
    s2=s.copy();s2.loc[s2.index>=g.origin.iloc[0]]=999999
    z1,_,_=context(g,h,s);z2,_,_=context(g,h2,s2)
    pd.testing.assert_frame_equal(z1,z2)
    (D/'feature_groups.json').write_text(json.dumps(groups,indent=2))
    (D/'leakage_checks.json').write_text(json.dumps({'future_perturbation_invariant':True,'checked_origin':str(g.origin.iloc[0]),
        'source_end_assertions':'all new source selections <= origin','target_actual_columns_in_inputs':False},indent=2))
    return x,groups

def classifier():
    return LGBMClassifier(n_estimators=350,num_leaves=31,min_child_samples=80,learning_rate=.035,
        colsample_bytree=.9,reg_lambda=5,n_jobs=4,random_state=42,verbosity=-1,deterministic=True,force_col_wise=True)

def split(x,start,end):
    cutoff=pd.Timestamp(start)-pd.Timedelta(days=1)+pd.Timedelta(hours=16)
    train=x[x.interval_end<=cutoff].copy();counts=train.groupby('date').size()
    train=train[train.date.isin(counts[counts==96].index)]
    val=x[(x.date>=start)&(x.date<end)].copy()
    assert train.interval_end.max()<val.origin.min()
    return train,val

def predict(train,val,cols,structure):
    if structure=='single':
        m=model_for('LGB_medium');m.fit(train[cols],train.y)
        return np.maximum(m.predict(val[cols]),0),None
    state=train.actual_production.gt(0).astype(int)
    gate=classifier();gate.fit(train[cols],state)
    probability=gate.predict_proba(val[cols])[:,1]
    preds=[]
    for label in [0,1]:
        mask=state==label;m=model_for('LGB_medium');m.fit(train.loc[mask,cols],train.loc[mask,'y'])
        preds.append(m.predict(val[cols]))
    return np.maximum((1-probability)*preds[0]+probability*preds[1],0),probability

def score(val,p,threshold):
    frame=val[['date','y']].copy();frame['pred']=p;frame['threshold']=threshold
    result=metrics(frame)
    result.update({'interval_'+k:v for k,v in confusion(val.y,p,threshold).items()})
    for label,mask in [('zero_production',val.actual_production==0),('positive_production',val.actual_production>0)]:
        result[label+'_mae']=float(np.abs(val.y.to_numpy()[mask]-p[mask]).mean()) if mask.any() else np.nan
    return result

def run():
    D.mkdir(exist_ok=True);x,groups=build()
    candidates=[(g,s) for g in groups for s in ['single','mixture']]
    scores=[];oof=[];peakrows=[]
    for start,end in [('2021-04-01','2021-05-01'),('2021-05-01','2021-06-01'),('2021-06-01','2021-07-01')]:
        tr,v=split(x,start,end);threshold=float(tr.y.quantile(.95))
        for group,structure in candidates+[('naive','naive')]:
            if group=='naive':p=v.power_lag7d.fillna(v.power_lag2d).fillna(v.last_power).to_numpy();pr=None
            else:p,pr=predict(tr,v,groups[group],structure)
            name=group+'_'+structure
            scores.append(dict(candidate=name,fold=start,**score(v,p,threshold)))
            a=v[['date','interval_start','y','actual_production']].copy();a['pred']=p;a['threshold']=threshold;a['candidate']=name;oof.append(a)
            print('validation',start,name,scores[-1]['mae'],flush=True)
        for group in ['B0','B3']:
            c=classifier();c.fit(tr[groups[group]],tr.y>=threshold);pr=c.predict_proba(v[groups[group]])[:,1]
            y=(v.y>=threshold).astype(int)
            peakrows.append(dict(group=group,fold=start,ap=average_precision_score(y,pr),brier=brier_score_loss(y,pr),
                                 **confusion(y,pr,.5)))
    pd.DataFrame(scores).to_csv(D/'fold_scores.csv',index=False)
    pd.DataFrame(peakrows).to_csv(D/'peak_classifier_validation.csv',index=False)
    joined=pd.concat(oof);summaries=[]
    for name,g in joined.groupby('candidate'):
        # Threshold varies only by past-training fold and stays on each row.
        summaries.append(dict(candidate=name,**metrics(g)))
    table=pd.DataFrame(summaries).sort_values('mae')
    table.to_csv(D/'validation_summary.csv',index=False)
    eligible=table[table.mae<=table.mae.min()*1.02]
    selected=eligible.sort_values(['daily_peak_mae','mae','candidate']).iloc[0].candidate
    (D/'selection.json').write_text(json.dumps({'candidate':selected,'rule':'Apr-Jun pooled MAE within 2% minimum then daily peak MAE; naive eligible',
        'status':'post-hoc exploration; July has been observed in prior research; no new independent test'},indent=2))
    # July only after selection. Fixed reference and ablations reported, no reselection.
    tr,v=split(x,'2021-07-01','2021-08-01');threshold=float(tr.y.quantile(.95));july=[];rows=[]
    for group,structure in candidates+[('naive','naive')]:
        if group=='naive':p=v.power_lag7d.fillna(v.power_lag2d).fillna(v.last_power).to_numpy();pr=None
        else:p,pr=predict(tr,v,groups[group],structure)
        name=group+'_'+structure
        july.append(dict(candidate=name,selected=name==selected,**score(v,p,threshold)))
        a=v[['date','interval_start','y','actual_production']].copy();a['pred']=p;a['candidate']=name
        if pr is not None:a['positive_record_probability']=pr
        rows.append(a)
    pd.DataFrame(july).to_csv(D/'july_scores.csv',index=False)
    pd.concat(rows).to_csv(D/'july_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    peak=[]
    for group in ['B0','B3']:
        c=classifier();c.fit(tr[groups[group]],tr.y>=threshold);pr=c.predict_proba(v[groups[group]])[:,1];y=(v.y>=threshold).astype(int)
        peak.append(dict(group=group,ap=average_precision_score(y,pr),brier=brier_score_loss(y,pr),**confusion(y,pr,.5)))
    pd.DataFrame(peak).to_csv(D/'peak_classifier_july.csv',index=False)
    print('SELECTED',selected,flush=True);print(table.to_string(index=False),flush=True)
    print(pd.DataFrame(july)[['candidate','selected','mae','daily_peak_mae','zero_production_mae','peak_day_recall']].to_string(index=False),flush=True)

if __name__=='__main__':run()
