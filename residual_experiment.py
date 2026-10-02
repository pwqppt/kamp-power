"""Preserve weekly baseline and learn only contextual deviations; post-hoc exploration."""
import json
import numpy as np
import pandas as pd
from experiment import model_for, OUT
from context_features import build, split, score, D as CONTEXT

D=OUT/'residual_v2'

def baseline(frame):
    return frame.power_lag7d.fillna(frame.power_lag2d).fillna(frame.last_power).to_numpy()

def predictions(tr,v,groups):
    base=baseline(v);target=tr.y.to_numpy()-baseline(tr)
    yield 'naive',base
    for group in ['B1','B3']:
        for model in ['LGB_medium','LGB_smooth']:
            m=model_for(model);m.fit(tr[groups[group]],target)
            delta=m.predict(v[groups[group]])
            for alpha in [.25,.5,1.]:
                yield f'{group}_{model}_a{alpha}',np.maximum(0,base+alpha*delta)

def run():
    D.mkdir(exist_ok=True);x,groups=build();folds=[];frames=[];diagnostics=[]
    for start,end in [('2021-04-01','2021-05-01'),('2021-05-01','2021-06-01'),('2021-06-01','2021-07-01')]:
        tr,v=split(x,start,end);threshold=float(tr.y.quantile(.95))
        for name,p in predictions(tr,v,groups):
            folds.append(dict(candidate=name,fold=start,**score(v,p,threshold)))
            a=v[['date','y','actual_production']].copy();a['pred']=p;a['threshold']=threshold;a['candidate']=name;frames.append(a)
        print('completed residual validation',start,flush=True)
    joined=pd.concat(frames);summary=[]
    from experiment import metrics
    for name,g in joined.groupby('candidate'):summary.append(dict(candidate=name,**metrics(g)))
    table=pd.DataFrame(summary).sort_values('mae');table.to_csv(D/'validation_summary.csv',index=False)
    pd.DataFrame(folds).to_csv(D/'fold_scores.csv',index=False)
    eligible=table[table.mae<=table.mae.min()*1.02]
    selected=eligible.sort_values(['daily_peak_mae','mae','candidate']).iloc[0].candidate
    (D/'selection.json').write_text(json.dumps({'candidate':selected,'selection_period':'Apr-Jun','status':'post-hoc; July reused'},indent=2))
    tr,v=split(x,'2021-07-01','2021-08-01');threshold=float(tr.y.quantile(.95));july=[];saved=[]
    for name,p in predictions(tr,v,groups):
        july.append(dict(candidate=name,selected=name==selected,**score(v,p,threshold)))
        if name in [selected,'naive']:
            a=v[['date','interval_start','y','actual_production','pattern']].copy();a['pred']=p;a['candidate']=name;saved.append(a)
    pd.DataFrame(july).to_csv(D/'july_scores.csv',index=False)
    pd.concat(saved).to_csv(D/'selected_july_predictions.csv',index=False)
    # Descriptive diagnosis: actual production/state is not used in prediction choices.
    v=v.copy();v['naive']=baseline(v);v['seen_pattern']=v.pattern.isin(tr.pattern.unique())
    v['zero_production']=v.actual_production.eq(0)
    for dimension in ['zero_production','seen_pattern']:
        for condition,g in v.groupby(dimension):
            diagnostics.append(dict(dimension=dimension,condition=bool(condition),n=len(g),days=g.date.nunique(),
                naive_mae=float((g.y-g.naive).abs().mean()),exact_weekly_value_rate=float(np.isclose(g.y,g.naive,atol=1e-9,rtol=0).mean())))
    pd.DataFrame(diagnostics).to_csv(D/'baseline_strength_diagnosis.csv',index=False)
    # Recompute July score from exported records; invariants before publishing.
    for name,g in pd.concat(saved).groupby('candidate'):
        row=next(r for r in july if r['candidate']==name)
        assert len(g)==29*96 and g.interval_start.nunique()==len(g)
        assert np.isfinite(g.pred).all() and (g.pred>=0).all()
        assert np.isclose((g.y-g.pred).abs().mean(),row['mae'],rtol=0,atol=1e-9)
    print('SELECTED',selected,flush=True);print(table.to_string(index=False),flush=True)
    print(pd.DataFrame(july)[['candidate','selected','mae','daily_peak_mae','zero_production_mae','positive_production_mae']].to_string(index=False),flush=True)
    print(pd.DataFrame(diagnostics).to_string(index=False),flush=True)

if __name__=='__main__':run()
