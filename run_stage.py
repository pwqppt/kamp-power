"""Stage runner: records commands/results without automatically marking research complete."""
import argparse, subprocess, sys, json, hashlib
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parent
def check():
    import pandas as pd
    audit=json.loads((ROOT/'outputs/data_audit.json').read_text())
    assert hashlib.sha256((ROOT/'data/original.csv').read_bytes()).hexdigest()==audit['sha256']
    if not (ROOT/'outputs/test_predictions.csv').exists():
        print('PASS: original SHA256. Full prediction checks require CP02: python run_stage.py 02')
        return
    p=pd.read_csv(ROOT/'outputs/test_predictions.csv',parse_dates=['origin','interval_start','interval_end','date'])
    assert p.groupby('date').size().eq(96).all()
    assert p.interval_start.is_unique and p.interval_start.is_monotonic_increasing
    assert (p.origin<p.interval_start).all()
    assert not p[['y','pred','naive']].isna().any().any()
    assert p.date.min()==pd.Timestamp('2021-08-01') and p.date.max()==pd.Timestamp('2021-09-14')
    print('PASS: original SHA256, 45 x 96 predictions, unique ordered timestamps, forecast origin, finite required values')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['01','02','03','figures','check','restore','conditions','simulate','extension_figures','extension_check','hold_policy']);args=ap.parse_args()
    if args.stage=='check':check();return
    scripts={'01':'audit.py','02':'experiment.py','03':'diagnostics.py','figures':'visualize.py',
             'restore':'continue_research.py','conditions':'continue_research.py','simulate':'continue_research.py',
             'extension_figures':'extension_figures.py','extension_check':'check_extension.py','hold_policy':'hold_policy.py'}
    script=scripts[args.stage]
    command=[sys.executable,str(ROOT/script)]
    if args.stage in ['restore','conditions','simulate']: command.append(args.stage)
    logdir=ROOT/'logs';logdir.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');log=logdir/f'{stamp}_{args.stage}.log'
    with log.open('w',encoding='utf8') as f:
        result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    with (logdir/'runs.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps({'utc':stamp,'stage':args.stage,'command':command,'exit_code':result.returncode,'log':log.name})+'\n')
    print(f'exit={result.returncode}; log={log}');sys.exit(result.returncode)
if __name__=='__main__':main()
