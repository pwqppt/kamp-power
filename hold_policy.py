"""Post-hoc exploration: Apr-May cutoffs, June selection, fixed July/test replay."""
import json
import numpy as np
import pandas as pd
from experiment import DATA, OUT, FEATURE_SETS, predict_fold
from continue_research import routing

D=OUT/'hold_policy'

def decide(features, rule):
    if rule['name']=='never': return np.zeros(len(features),dtype=bool)
    mask=np.ones(len(features),dtype=bool)
    if rule['peak_only']:
        mask &= features.predicted_peak.to_numpy() >= features.threshold.to_numpy()
    if rule['limit'] is not None:
        mask &= features.disagreement.to_numpy() <= rule['limit']
    return mask

def evaluate(frame,rule):
    mask=decide(frame[['disagreement','predicted_peak','threshold']],rule)
    gain=np.where(mask,frame.raw_reduction,0.)
    return dict(rule=rule['name'],days=len(frame),executed_days=int(mask.sum()),
       mean_reduction=float(gain.mean()),executed_mean_reduction=float(gain[mask].mean()) if mask.any() else np.nan,
       worsened_days=int((gain < -1e-6).sum()),
       worsened_rate_executed=float((gain[mask]<-1e-6).mean()) if mask.any() else 0.,
       mean_harm=float(np.maximum(-gain,0).mean()),
       missed_positive_gain_sum=float(np.maximum(frame.raw_reduction.to_numpy()[~mask],0).sum()),
       period_max_reduction=float(frame.original_peak.max()-(frame.original_peak.to_numpy()-gain).max()))

def run():
    D.mkdir(exist_ok=True)
    x=pd.read_pickle(DATA/'features.pkl')
    spec=json.loads((OUT/'frozen_selection.json').read_text())
    FEATURE_SETS.update(json.loads((OUT/'feature_sets.json').read_text()))
    pre=[]; checks=[]
    for start,end in [('2021-04-01','2021-05-01'),('2021-05-01','2021-06-01'),('2021-06-01','2021-07-01'),('2021-07-01','2021-08-01')]:
        p,_,tr=predict_fold(x,spec['features'],spec['model'],start,end)
        xx=x[(x.date>=start)&(x.date<end)]
        p['naive']=xx.power_lag7d.fillna(xx.power_lag2d).fillna(xx.last_power).to_numpy()
        assert tr.interval_end.max() < p.origin.min()
        checks.append({'fold':start,'last_label':str(tr.interval_end.max()),'origin':str(p.origin.min()),'days':p.date.nunique()})
        pre.append(p)
    test=pd.read_csv(OUT/'test_predictions.csv',parse_dates=['date','interval_start','origin'])
    allp=pd.concat([*pre,test],ignore_index=True)
    rows=[]
    for date,g in allp.groupby('date',sort=True):
        g=g.sort_values('interval_start');y=g.y.to_numpy()
        disagreement=float(np.mean(np.abs(g.pred-g.naive))/max((g.pred.mean()+g.naive.mean())/2,1))
        for model,col in [('AI','pred'),('SeasonalNaive','naive')]:
            p=g[col].to_numpy();r=routing(p,.1,4);new=r@y
            assert np.isclose(y.sum(),new.sum(),rtol=1e-9,atol=1e-7)
            rows.append(dict(date=date,model=model,disagreement=disagreement,predicted_peak=float(p.max()),
               threshold=float(g.threshold.iloc[0]),original_peak=float(y.max()),raw_reduction=float(y.max()-new.max())))
    daily=pd.DataFrame(rows);daily.to_csv(D/'daily_inputs_outcomes.csv',index=False)
    calibration=daily[(daily.date<'2021-06-01')&(daily.model=='AI')]
    cutoffs=calibration.disagreement.quantile([.25,.5,.75])
    rules=[dict(name='always',peak_only=False,limit=None),dict(name='never',peak_only=False,limit=None),
           dict(name='peak_only',peak_only=True,limit=None)]
    for q,v in cutoffs.items():
        for peak in [False,True]:
            rules.append(dict(name=f'agree_q{int(q*100)}'+('_peak' if peak else ''),peak_only=peak,limit=float(v)))
    selected={};selection=[]
    for model in ['AI','SeasonalNaive']:
        june=daily[(daily.date>='2021-06-01')&(daily.date<'2021-07-01')&(daily.model==model)]
        scores=pd.DataFrame([evaluate(june,r) for r in rules])
        eligible=scores[scores.worsened_rate_executed<=.10]
        best=eligible.sort_values(['mean_reduction','executed_days','rule'],ascending=[False,True,True]).iloc[0]
        selected[model]=next(r for r in rules if r['name']==best.rule)
        scores['model']=model;selection.append(scores)
    pd.concat(selection).to_csv(D/'june_candidate_scores.csv',index=False)
    (D/'frozen_policy.json').write_text(json.dumps({'cutoffs_Apr_May':cutoffs.to_dict(),'selected':selected,
       'selection_period':'June only','allowed_worsening_rate_executed':.10,'fraction':.1,'window_minutes':60,
       'status':'post-hoc exploratory, not fresh independent validation'},indent=2))
    # Once frozen, later outcome values cannot affect rule selection.
    results=[];decisions=[]
    for period,start,end in [('July','2021-07-01','2021-08-01'),('test_replay','2021-08-01','2021-09-15')]:
        for model in ['AI','SeasonalNaive']:
            g=daily[(daily.date>=start)&(daily.date<end)&(daily.model==model)].copy()
            for label,rule in [('always',rules[0]),('never',rules[1]),('selected',selected[model])]:
                results.append(dict(period=period,model=model,policy=label,**evaluate(g,rule)))
            f=g[['disagreement','predicted_peak','threshold']]
            mask=decide(f,selected[model])
            changed=g.copy();changed['original_peak']=-12345;changed['raw_reduction']=12345
            assert np.array_equal(mask,decide(changed[f.columns],selected[model]))
            g['execute']=mask;g['rule']=selected[model]['name'];g['realized_reduction']=np.where(mask,g.raw_reduction,0)
            g['period']=period;decisions.append(g)
    result=pd.DataFrame(results);result.to_csv(D/'evaluation.csv',index=False)
    pd.concat(decisions).to_csv(D/'selected_daily.csv',index=False)
    # Regression check: fixed raw routing matches the previously published replay.
    old=pd.read_csv(OUT/'extension/simulation_daily.csv',parse_dates=['date'])
    old=old[(old.fraction==.1)&(old.window_minutes==60)&old.model.isin(['AI','SeasonalNaive'])]
    joined=daily.merge(old[['date','model','peak_reduction']],on=['date','model'],validate='one_to_one')
    assert len(joined)==90 and np.allclose(joined.raw_reduction,joined.peak_reduction,atol=1e-7,rtol=0)
    (D/'checks.json').write_text(json.dumps({'folds':checks,'later_outcome_independence':True,'replay_matches_previous':True,
       'limitations':'Architecture selected Apr-Jun; protocol proposed after test inspection; July previously used for calibration. No new independent test.'},indent=2))
    print(result.to_string(index=False),flush=True)

if __name__=='__main__':run()
