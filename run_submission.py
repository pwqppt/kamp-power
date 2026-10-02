"""Reproduce evidence and export already-used historical test predictions, without selection."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from peak_study import ROOT,OUT,DATA,dump,forecast,frame,scores
from context_features import build,split

def test_export():
    x,groups=build();tr,v=split(x,'2021-08-01','2021-09-15')
    p,pr=forecast(tr,v,groups['B1'])
    f=frame(v,p,pr,tr,'retained_reference','2021-08-01')
    dest=OUT/'submission';dest.mkdir(exist_ok=True)
    f[['origin','interval_start','interval_end','pred']].to_csv(dest/'test_predictions.csv',index=False)
    d=f.groupby('date')[['pred','naive','y','threshold']].max();d['peak_head']=(d.pred+d.naive)/2
    d[['peak_head']].to_csv(dest/'test_daily_peak_predictions.csv')
    dump(dest/'test_export.json',{'period':'2021-08-01 to 2021-09-14','rows':len(f),'days':int(f.date.nunique()),
      'model':'frozen retained B1 gate>=0.5 mixture else seasonal naive; separate H6 peak head',
      'training_label_last':str(tr.interval_end.max()),'first_forecast_origin':str(v.origin.min()),
      'selection_using_this_export':False,'independent_test':False,
      'warning':'Historical test has been observed in earlier repository work. This file meets submission format; it is not a new independent generalization claim.',
      'same_day_future_actual_inputs':False,'source_sha256':hashlib.sha256((DATA/'original.csv').read_bytes()).hexdigest()})
    print('Historical test file exported: 45 days / 4320 intervals. No reselection.',flush=True)

def all_stages():
    # Empty-directory execution must not depend on a previous feature cache.
    (OUT/'context_v2').mkdir(parents=True,exist_ok=True)
    stages=[['peak_study.py','prepare'],['peak_study.py','reference'],['peak_study.py','H1'],['peak_study.py','H2'],['peak_study.py','H3'],
       ['peak_study.py','summarize'],['peak_study.py','simulate'],['peak_report.py'],['check_peak_study.py'],
       ['research_v4.py','audit'],['research_v4.py','benchmark'],['research_v4.py','diagnose'],['research_v4.py','production'],
       ['peak_v4.py','H4'],['peak_v4.py','H5'],['peak_head.py'],['economics_v4.py'],['final_evidence.py']]
    for cmd in stages:
        print('RUN',' '.join(cmd),flush=True);subprocess.run([sys.executable,'-X','utf8',*cmd],cwd=ROOT,check=True)
    test_export()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['all','test-export']);a=p.parse_args()
    if a.stage=='all':all_stages()
    else:test_export()
