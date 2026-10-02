"""Meaningful conservation, oracle, metric and edge-case checks."""
import json
import numpy as np
import pandas as pd
from continue_research import routing, confusion, DEST

def run():
    for y in [np.zeros(8),np.ones(8)*10,np.array([0,0,0,100,0,0,0,0.])]:
        for f in [0,.1]:
            r=routing(y,f,2);z=r@y
            assert np.isclose(z.sum(),y.sum()) and np.all(z>=-1e-7)
            assert z.max()<=y.max()+1e-6
            if f==0: assert np.allclose(z,y)
    z=routing(np.array([0,0,0,100,0,0,0,0.]),.1,2)@np.array([0,0,0,100,0,0,0,0.])
    assert np.isclose(z.max(),90,atol=1e-6)
    assert confusion([0,1,1,0],[0,0,1,1],.5)['f1']==.5
    d=pd.read_csv(DEST/'simulation_daily.csv')
    assert len(d)==45*4*2*3
    assert d.energy_conservation_error.abs().max()<1e-6
    assert d[d.fraction==0].peak_reduction.abs().max()<1e-6
    piv=d.pivot(index=['date','fraction','window_minutes'],columns='model',values='adjusted_peak')
    assert (piv.Oracle_reference<=piv.AI+1e-5).all()
    assert (piv.Oracle_reference<=piv.SeasonalNaive+1e-5).all()
    rng=np.random.default_rng(42);rows=[]
    for (f,w),g in d.groupby(['fraction','window_minutes']):
        a=g.pivot(index='date',columns='model',values='peak_reduction')
        diff=(a.AI-a.SeasonalNaive).to_numpy()
        boot=rng.choice(diff,(2000,len(diff)),replace=True).mean(axis=1)
        rows.append(dict(fraction=f,window_minutes=w,AI_minus_naive_reduction=float(diff.mean()),
                         ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975))))
    pd.DataFrame(rows).to_csv(DEST/'simulation_paired_bootstrap.csv',index=False)
    (DEST/'checks.json').write_text(json.dumps({'passed':True,'simulation_rows':len(d),
        'checks':['zero-load','flat-load','isolated-peak analytical solution','zero-flexibility identity','energy conservation','oracle dominance','confusion example'],
        'limitations':'conditional algebraic routing only; does not validate physical scheduling or causal effects'},indent=2))
    print('PASS: edge cases, conservation, oracle dominance, classification check; 1080 day/scenario/model rows')

if __name__=='__main__':run()
