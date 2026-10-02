"""Exploratory predicted-state routing; never route using actual future production."""
import json
import numpy as np
import pandas as pd
from context_features import build,split,predict,score
from residual_experiment import baseline
from experiment import OUT,metrics

D=OUT/'adaptive_v2'

def variants(v,p,prob):
    b=baseline(v)
    yield 'naive',b
    yield 'mixture',p
    for cutoff in [.5,.75,.9]:
        yield f'predicted_state_{cutoff}',np.where(prob>=cutoff,p,b)

def run():
    D.mkdir(exist_ok=True);x,groups=build();frames=[];scores=[]
    for start,end in [('2021-04-01','2021-05-01'),('2021-05-01','2021-06-01'),('2021-06-01','2021-07-01')]:
        tr,v=split(x,start,end);th=float(tr.y.quantile(.95));p,prob=predict(tr,v,groups['B1'],'mixture')
        for name,forecast in variants(v,p,prob):
            scores.append(dict(candidate=name,fold=start,**score(v,forecast,th)))
            f=v[['date','y']].copy();f['pred']=forecast;f['threshold']=th;f['candidate']=name;frames.append(f)
    joined=pd.concat(frames);table=pd.DataFrame([dict(candidate=n,**metrics(g)) for n,g in joined.groupby('candidate')])
    table.to_csv(D/'validation_summary.csv',index=False);pd.DataFrame(scores).to_csv(D/'fold_scores.csv',index=False)
    eligible=table[table.mae<=table.mae.min()*1.02];selected=eligible.sort_values(['daily_peak_mae','mae','candidate']).iloc[0].candidate
    (D/'selection.json').write_text(json.dumps({'candidate':selected,'source':'Apr-Jun only','status':'post-hoc exploratory reuse of known July'},indent=2))
    tr,v=split(x,'2021-07-01','2021-08-01');th=float(tr.y.quantile(.95));p,prob=predict(tr,v,groups['B1'],'mixture')
    results=[];preds=[]
    for name,forecast in variants(v,p,prob):
        results.append(dict(candidate=name,selected=name==selected,**score(v,forecast,th)))
        if name==selected:
            f=v[['date','interval_start','y','actual_production']].copy();f['pred']=forecast;f['positive_record_probability']=prob;preds.append(f)
    pd.DataFrame(results).to_csv(D/'july_scores.csv',index=False)
    pd.concat(preds).to_csv(D/'selected_july.csv',index=False)
    print(table.to_string(index=False));print(pd.DataFrame(results)[['candidate','selected','mae','daily_peak_mae','zero_production_mae','positive_production_mae']].to_string(index=False))

if __name__=='__main__':run()
