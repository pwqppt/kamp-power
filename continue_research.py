"""Frozen-model recovery, ex-post conditions, and conditional load-routing simulation."""
import argparse
import json
import platform
from importlib.metadata import version
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import linprog
from experiment import ROOT, OUT, DATA, FEATURE_SETS, build_features, predict_fold, metrics

DEST = OUT / 'extension'
DEST.mkdir(exist_ok=True)


def restore():
    spec = json.loads((OUT / 'frozen_selection.json').read_text())
    x, _ = build_features()
    p, model, train = predict_fold(x, spec['features'], spec['model'], '2021-08-01', '2021-09-15')
    xx = x[x.date >= '2021-08-01']
    p['naive'] = xx.power_lag7d.fillna(xx.power_lag2d).fillna(xx.last_power).to_numpy()
    p['abs_error'] = (p.y - p.pred).abs()
    p['lower90'] = np.maximum(0, p.pred + spec['residual_q05'])
    p['upper90'] = np.maximum(0, p.pred + spec['residual_q95'])
    previous = pd.read_csv(OUT / 'test_scores.csv')
    old = previous[(previous.model == spec['model']) & (previous.subset == 'all')].iloc[0]
    now = metrics(p)
    for key in ['mae', 'rmse', 'daily_peak_mae', 'peak_day_recall']:
        assert np.isclose(now[key], old[key], atol=1e-7), (key, now[key], old[key])
    p.to_csv(OUT / 'test_predictions.csv', index=False)
    joblib.dump({'model': model, 'features': FEATURE_SETS[spec['features']], 'selection': spec}, OUT / 'final_model.joblib')
    record = {'python': platform.python_version(), 'packages': {k: version(k) for k in ['numpy','pandas','scikit-learn','lightgbm','scipy','holidays']},
              'scores_match_previous_atol': 1e-7, 'metrics': now, 'training_days': int(train.date.nunique())}
    (DEST / 'recovery.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


def confusion(y, p, threshold):
    actual, alarm = np.asarray(y) >= threshold, np.asarray(p) >= threshold
    tp, fn = int((actual & alarm).sum()), int((actual & ~alarm).sum())
    fp, tn = int((~actual & alarm).sum()), int((~actual & ~alarm).sum())
    div = lambda a, b: a / b if b else np.nan
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, recall=div(tp,tp+fn), precision=div(tp,tp+fp),
                f1=div(2*tp,2*tp+fn+fp), false_positive_rate=div(fp,fp+tn), peak_rate=float(actual.mean()))


def conditions():
    p = pd.read_csv(OUT / 'test_predictions.csv', parse_dates=['date'])
    x = pd.read_pickle(DATA / 'features.pkl')
    train = x[x.date < '2021-07-31']
    threshold = float(p.threshold.iloc[0])
    assert p.threshold.nunique() == 1
    rows = []
    for name, col in [('AI','pred'),('SeasonalNaive','naive')]:
        for level in ['interval', 'day']:
            g = p if level == 'interval' else p.groupby('date')[[col,'y']].max()
            rows.append(dict(model=name, level=level, n=len(g), threshold=threshold, **confusion(g.y,g[col],threshold)))
    pd.DataFrame(rows).to_csv(DEST / 'peak_classification.csv', index=False)
    cutoffs = {}
    for col in ['actual_production','actual_personnel','actual_temp']:
        cuts = np.unique(train[col].dropna().quantile([1/3,2/3]).values)
        cutoffs[col] = cuts.tolist()
        p[col+'_band'] = pd.cut(p[col],[-np.inf,*cuts,np.inf],labels=False).astype('Int64').astype(str).replace('<NA>','missing')
    p['period'] = pd.cut(p.hour,[-1,6,9,16,21,23],labels=['00-07','07-10','10-17','17-22','22-24']).astype(str)
    p['production_x_period'] = p.actual_production_band + '/' + p.period
    p['production_x_personnel'] = p.actual_production_band + '/' + p.actual_personnel_band
    dims = ['actual_production_band','actual_personnel_band','actual_temp_band','period','production_x_period','production_x_personnel']
    rows = []
    for dim in dims:
        for condition, g in p.groupby(dim, observed=True):
            for name, col in [('AI','pred'),('SeasonalNaive','naive')]:
                rows.append(dict(dimension=dim, condition=condition, model=name, n=len(g), days=g.date.nunique(),
                     mae=float((g.y-g[col]).abs().mean()), bias=float((g[col]-g.y).mean()),
                     mae_difference_from_overall=float((g.y-g[col]).abs().mean()-(p.y-p[col]).abs().mean()),
                     **confusion(g.y,g[col],threshold)))
    pd.DataFrame(rows).to_csv(DEST / 'condition_metrics.csv', index=False)
    # Paired differences, not independent 15-minute observations; descriptive CI only.
    rng = np.random.default_rng(42)
    effects = []
    for dim in dims[:4]:
        for condition, g in p.groupby(dim, observed=True):
            day = g.assign(delta=(g.y-g.pred).abs()-(g.y-g.naive).abs()).groupby('date').agg(total=('delta','sum'),n=('delta','size'))
            a = day.to_numpy(); sampled = a[rng.integers(0,len(a),size=(2000,len(a)))].sum(axis=1)
            boot = sampled[:,0]/sampled[:,1]
            effects.append(dict(dimension=dim,condition=condition,days=len(day),delta_mae=float(a[:,0].sum()/a[:,1].sum()),
                                ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975))))
    pd.DataFrame(effects).to_csv(DEST / 'condition_day_bootstrap.csv',index=False)
    (DEST / 'condition_spec.json').write_text(json.dumps({'band_cutoffs_training_only':cutoffs,'peak_threshold':threshold,
         'analysis':'ex-post, not deployable inputs; no causal claims; hourly covariates repeat four times',
         'intervals':'paired day bootstrap, repeated patterns and serial dependence limit inference; no multiple-testing correction'},indent=2))
    print(pd.DataFrame(rows).query("dimension == 'actual_production_band'").to_string(index=False),flush=True)


def routing(pred, fraction, slots):
    """Minimax forecast load, then minimal movement at the same optimal peak."""
    n = len(pred)
    if fraction == 0:
        return np.eye(n)
    edges = [(i,j) for i in range(n) for j in range(max(0,i-slots), min(n,i+slots+1)) if i != j and pred[i] > 1e-9]
    if not edges:
        return np.eye(n)
    m = len(edges)
    net, outgoing = np.zeros((n,m)), np.zeros((n,m))
    for k,(i,j) in enumerate(edges):
        net[i,k] = -1; net[j,k] = 1; outgoing[i,k] = 1
    A = np.vstack([np.column_stack([net,-np.ones(n)]),np.column_stack([outgoing,np.zeros(n)])])
    b = np.r_[-pred, fraction*pred]
    first = linprog(np.r_[np.zeros(m),1.],A_ub=A,b_ub=b,bounds=(0,None),method='highs')
    assert first.success, first.message
    # Lexicographic tie break: minimize total routed load, deterministic fixed edge order.
    second = linprog(np.r_[np.ones(m),0.],A_ub=A,b_ub=b,bounds=[(0,None)]*m+[(0,first.x[-1]+1e-7)],method='highs')
    assert second.success, second.message
    matrix = np.eye(n)
    for amount,(i,j) in zip(second.x[:-1],edges):
        ratio = amount/pred[i]
        matrix[i,i] -= ratio; matrix[j,i] += ratio
    assert np.allclose(matrix.sum(axis=0),1) and matrix.min() >= -1e-7
    assert np.all(1-np.diag(matrix) <= fraction+1e-7)
    ii,jj=np.where(matrix>1e-9)
    assert np.all(np.abs(ii-jj)<=slots)
    return matrix


def simulate():
    p = pd.read_csv(OUT/'test_predictions.csv',parse_dates=['date','interval_start'])
    rows=[]; examples=[]
    for date,g in p.groupby('date',sort=True):
        g=g.sort_values('interval_start'); y=g.y.to_numpy()
        for fraction in [0,.05,.10,.15]:
            for slots in [2,4]:
                for name,col in [('SeasonalNaive','naive'),('AI','pred'),('Oracle_reference','y')]:
                    pred=g[col].to_numpy(); matrix=routing(pred,fraction,slots)
                    result=matrix@y; planned=matrix@pred
                    assert np.isclose(result.sum(),y.sum(),rtol=1e-9,atol=1e-7)
                    assert result.min()>=-1e-7
                    reduction=float(y.max()-result.max())
                    rows.append(dict(date=str(date.date()),model=name,fraction=fraction,window_minutes=slots*15,
                       original_peak=float(y.max()),adjusted_peak=float(result.max()),peak_reduction=reduction,
                       peak_reduction_pct=100*reduction/y.max() if y.max()>0 else 0,
                       worsened=int(reduction < -1e-6),predicted_reduction=float(pred.max()-planned.max()),
                       moved_load_sum=float(((1-np.diag(matrix))*y).sum()),
                       total_load_sum=float(y.sum()),energy_conservation_error=float(result.sum()-y.sum())))
                    if fraction==.10 and slots==4 and name=='AI':
                        examples.append(pd.DataFrame({'time':g.interval_start,'date':date,'actual':y,'pred':pred,'adjusted':result}))
        print('simulation',date.date(),flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(DEST/'simulation_daily.csv',index=False)
    summaries=[]
    for period, part in [('all_45_days',frame),('August_full_month',frame[frame.date<'2021-09-01'])]:
        for (name,f,w),g in part.groupby(['model','fraction','window_minutes']):
            summaries.append(dict(period=period,model=name,fraction=f,window_minutes=w,days=len(g),
                mean_daily_reduction=float(g.peak_reduction.mean()),median_daily_reduction=float(g.peak_reduction.median()),
                mean_daily_pct=float(g.peak_reduction_pct.mean()),worsened_days=int(g.worsened.sum()),
                period_max_reduction=float(g.original_peak.max()-g.adjusted_peak.max()),
                moved_fraction=float(g.moved_load_sum.sum()/g.total_load_sum.sum())))
    pd.DataFrame(summaries).to_csv(DEST/'simulation_summary.csv',index=False)
    pd.concat(examples).to_csv(DEST/'simulation_example_series.csv',index=False)
    print(pd.DataFrame(summaries).query("period == 'all_45_days' and fraction == 0.1").to_string(index=False),flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['restore','conditions','simulate']); args=ap.parse_args()
    {'restore':restore,'conditions':conditions,'simulate':simulate}[args.stage]()
