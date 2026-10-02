"""Conditional, pre-tax tariff/labor comparison for an already fixed load-shift policy."""
import hashlib
import numpy as np
import pandas as pd
import holidays
from peak_study import ROOT,OUT,dump
D=OUT/'research_v4/economics'

def tariff(t):
    summer=t.month in [6,7,8];off=121.5;mid=169.3 if summer else 138.9;peak=234.5 if summer else 156.4
    weekend=t.dayofweek>=5;holiday=t.date() in holidays.KR(years=2021)
    # Weekend rules explicitly treated as scenario assumptions, not certified bill reconstruction.
    if t.hour<8 or t.hour>=22:rate=off;eligible=False
    elif t.dayofweek==6 or holiday:rate=off;eligible=False
    elif t.dayofweek==5:rate=mid;eligible=True
    else:rate=peak if 15<=t.hour<21 else mid;eligible=True
    if not summer and (weekend or holiday) and 11<=t.hour<14:rate*=.5
    return rate,eligible

def run():
    D.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv(OUT/'peak_v3/simulation/example_series.csv.gz',parse_dates=['date','interval_start'])
    f['month']=f.date.dt.strftime('%Y-%m')
    vals=[tariff(t) for t in f.interval_start];f['rate']=[v[0] for v in vals];f['billing_peak_eligible']=[v[1] for v in vals]
    rows=[];labor=[]
    for month,g in f.groupby('month'):
        days=g.date.nunique();complete=days==g.date.iloc[0].days_in_month
        eligible=g[g.billing_peak_eligible]
        raw=float(eligible.actual.max());after=float(eligible.adjusted.max())
        energy=float(((g.actual-g.adjusted)*.25*g.rate).sum())
        moved_days=int(g.assign(changed=(g.actual-g.adjusted).abs()>1e-7).groupby('date').changed.any().sum())
        for residual_demand in [200.,250.]:
            before_billing=max(90.,residual_demand,raw);after_billing=max(90.,residual_demand,after)
            demand=7220*(before_billing-after_billing)
            base=dict(month=month,days=days,complete_month=complete,residual_peak_assumption=residual_demand,
              original_eligible_peak=raw,adjusted_eligible_peak=after,billing_peak_before=before_billing,billing_peak_after=after_billing,
              demand_savings=demand,energy_savings=energy,electric_savings_before_tax=demand+energy,moved_days=moved_days)
            rows.append(base)
            for wage in [13000,16000]:
                for workers in [1,2,4]:
                    for hours in [.25,.5,1.]:
                        for label,multiplier in [('same_hours_shift_to_night',.5),('extra_night_regular',1.5),('extra_night_overtime',2.)]:
                            cost=moved_days*wage*workers*hours*multiplier
                            labor.append(dict(**base,wage=wage,workers=workers,additional_hours_per_moved_day=hours,case=label,
                              incremental_labor_cost=cost,net_savings_before_tax=demand+energy-cost,
                              break_even_total_worker_hours=max(demand+energy,0)/(wage*multiplier)))
    pd.DataFrame(rows).to_csv(D/'electric_scenarios.csv',index=False)
    pd.DataFrame(labor).to_csv(D/'labor_scenarios.csv',index=False)
    f.to_csv(D/'tariff_intervals.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    dump(D/'assumptions.json',dict(policy='fixed 10% +/-60 minutes; NOT economically reselected',
      unit='assumed kW; energy=15min mean power *0.25h',contract='assumed industrial Eul high voltage A choice I, 300kW',
      rates='2026-04-16 onward rates counterfactually applied to 2021 calendar/load',
      basic_rate=7220,contract_floor=90,residual_demand=[200,250],
      unknowns=['actual meter unit','contract and voltage','12-month valid maximum demand','power factor','billing cycle','equipment feasibility','actual staffing'],
      omissions=['VAT','industry fund','power-factor adjustment','setup cost','demand penalties','interest'],
      equal_energy_addons='uniform fuel/climate charges cancel under energy conservation; no bill-total claim',
      July='29 days only, NOT monthly bill estimate',deployment_recommended=False,
      official_pdf_sha256=hashlib.sha256((ROOT/'work/official/tariff-2026.pdf').read_bytes()).hexdigest() if (ROOT/'work/official/tariff-2026.pdf').exists() else 'source in EXTERNAL_DATA_REGISTER'))
    assert np.isclose(((f.actual-f.adjusted)*.25).sum(),0,atol=1e-6)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':run()
