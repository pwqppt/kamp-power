"""Exploratory scope checks using issuance-time conditions; no model reselection."""
import json
import pandas as pd
import numpy as np
from intraday import OUT,raw_load,build,metrics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DEST=ROOT/'outputs/scope_audit';DEST.mkdir(exist_ok=True)
spec={'purpose':'ex ante observable scope audit, exploratory after old results; no confirmation','candidate_scopes':['clean_history','production_positive_clean','approaching_high_load_clean'],'approaching_definition':'training q75 <= current <= training q95, all recent 8 power values present, production record >0','target_horizons':'45..120min (8 quarter slots remain full evaluation comparator)','selection':'no automatic selection by July accuracy; report coverage and peak counts; no prod0 easy-case focus'}
(DEST/'spec.json').write_text(json.dumps(spec,indent=2))
y,cov,info=raw_load();x,m,g=build(y,cov)
p=pd.read_csv(OUT/'predictions.csv.gz',parse_dates=['origin','target'])
rows=[];events=[]
for month in [4,5,6,7]:
 cut=pd.Timestamp(2021,month,1);end=cut+pd.offsets.MonthBegin()
 c=y.loc[y.index<cut].quantile(.95);q=y.loc[y.index<cut].quantile(.75)
 test=m.origin.ge(cut)&m.origin.lt(end);orig=m.loc[test].groupby('origin').head(1)
 ix=orig.index;clean=x.loc[ix,[f'lag{i}' for i in range(8)]].notna().all(axis=1)
 masks={'all':np.ones(len(orig),dtype=bool),'clean_history':clean.to_numpy(),'production_positive_clean':(clean&orig.production.gt(0)).to_numpy(),'approaching_high_load_clean':(clean&orig.production.gt(0)&orig.current.between(q,c)).to_numpy()}
 sub=p[(p.month==month)&p.model.isin(['lgb_F0','week','last'])]
 for scope,mask in masks.items():
  origins=orig.origin[mask]
  for horizon,hm in [('15..120',sub.horizon.ge(1)),('45..120',sub.horizon.ge(3))]:
   for model,df in sub[hm&sub.origin.isin(origins)].groupby('model'):
    rows.append({'month':month,'scope':scope,'horizon':horizon,'model':model,'coverage':len(origins)/len(orig),'q75':q,'q95':c,**metrics(df)})
  dd=sub[(sub.model=='lgb_F0')&sub.origin.isin(origins)&sub.horizon.ge(3)].groupby('origin').agg(maxy=('y','max'),current=('current','first'))
  events.append({'month':month,'scope':scope,'origins':len(dd),'coverage':len(dd)/len(orig),'peak_windows':int((dd.maxy>c).sum()),'new_peak_windows':int((dd.maxy.gt(c)&dd.current.le(c)).sum())})
pd.DataFrame(rows).to_csv(DEST/'scope_metrics.csv',index=False)
pd.DataFrame(events).to_csv(DEST/'coverage_peak_counts.csv',index=False)
print(pd.DataFrame(events).to_string(index=False))
print(pd.DataFrame(rows).query("month==7 and horizon=='45..120'")[['scope','model','coverage','n','MAE','window_max_MAE','window_recall','onset_events']].to_string(index=False))
