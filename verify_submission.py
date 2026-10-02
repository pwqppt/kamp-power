"""Verify fresh research reproduction and the numerical claims in final deliverables."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;S=ROOT/'outputs/submission'

def run():
    clean=ROOT/'work/reproduce-check';checked=[];floating_differences=[]
    for directory in ['peak_v3','peak_v4','research_v4']:
        for f in sorted((ROOT/'outputs'/directory).rglob('*')):
            if not (f.name.endswith('.csv') or f.name.endswith('.csv.gz')):continue
            rel=f.relative_to(ROOT/'outputs');other=clean/'outputs'/rel
            left,right=pd.read_csv(f),pd.read_csv(other)
            try:pd.testing.assert_frame_equal(left,right,check_exact=True)
            except AssertionError:
                # Parallel floating-point reductions can differ at ~1e-14.
                # Keep a strict absolute tolerance and disclose every differing file.
                pd.testing.assert_frame_equal(left,right,check_exact=False,rtol=0,atol=1e-10)
                numeric=left.select_dtypes(include='number').columns
                delta=float((left[numeric]-right[numeric]).abs().max().max())
                floating_differences.append({'file':rel.as_posix(),'max_absolute_difference':delta})
            checked.append(rel.as_posix())
    for name in ['test_predictions.csv','test_daily_peak_predictions.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(S/name),pd.read_csv(clean/'outputs/submission'/name),check_exact=True)
        checked.append('submission/'+name)
    report={'fresh_zip_extraction':True,'notebook_all_code_cells_executed':True,'same_pinned_environment':True,
        'comparison':'CSV values compared with absolute tolerance 1e-10; exact equality attempted first. Test export required exact equality.',
        'floating_point_differences':floating_differences,
        'files_checked':checked,'count':len(checked),'passed':True,
        'source_sha256':hashlib.sha256((clean/'data/original.csv').read_bytes()).hexdigest(),
        'not_independent_data_validation':True,'artifact_layout_not_part_of_model_reproduction':True}
    (S/'reproduction_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    # Ground the exact display claims; use tolerances only for printed rounding.
    c=pd.read_csv(ROOT/'outputs/peak_v3/comparison.csv');v=c[c.period=='Apr-Jun'].set_index('candidate')
    h=pd.read_csv(ROOT/'outputs/peak_v4/H6/scores.csv');hv=h[h.period=='Apr-Jun'].set_index('model')
    assert round((1-v.loc['reference','mae']/v.loc['naive','mae'])*100,1)==11.4
    assert round((1-hv.loc['peak_head','daily_peak_mae']/hv.loc['naive','daily_peak_mae'])*100,1)==6.1
    assert round(hv.loc['peak_head','recall']*100,1)==52.3
    hj=h[h.period=='July'].set_index('model');assert round(hj.loc['peak_head','daily_peak_mae'],3)==17.483
    assert hj.loc['peak_head','daily_peak_mae']>hj.loc['naive','daily_peak_mae']
    cond=pd.read_csv(ROOT/'outputs/peak_v3/conditions/2021-07/conditions.csv')
    for label,n,peak,fn,fp in [('25-30/positive/10-17',244,149,118,5),('>30/positive/10-17',152,97,54,9)]:
        r=cond[(cond.condition==label)&(cond.model=='pred')].iloc[0]
        assert (r.n,r.peak_intervals,r.fn,r.fp)==(n,peak,fn,fp)
    print('Fresh reproduction:',len(checked),'CSV files;',floating_differences,'; headline/interaction claims verified.',flush=True)

if __name__=='__main__':run()
