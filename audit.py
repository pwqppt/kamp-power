"""Read-only-source audit; generated outputs are reproducible from the supplied ZIP."""
from pathlib import Path
import io, zipfile, json, hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'

def run():
    DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    archive = ROOT.parent / 'upload' / '5. 자원 최적화 AI 데이터셋.zip'
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            name = next(n for n in z.namelist() if n.endswith('.csv'))
            raw = z.read(name)
        (DATA / 'original.csv').write_bytes(raw)
    else:
        raw = (DATA / 'original.csv').read_bytes()
    df = pd.read_csv(io.BytesIO(raw), encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['날짜'].astype(str), format='%Y%m%d')
    df['source_row'] = np.arange(len(df))
    df['row_hour'] = df.groupby('date', sort=False).cumcount()
    bad = df['시간'].ne(df['row_hour'])
    assert df.groupby('date').size().eq(24).all()
    df.loc[bad].to_csv(OUT/'hour_repair_audit.csv', index=False)
    df['hour_repaired'] = bad
    # Row-order repair is an explicit assumption; repaired dates excluded from labels.
    df['hour'] = df['row_hour']
    df['hour_start'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')
    power_cols = ['15분','30분','45분','60분']
    wide = df.set_index('hour_start')[power_cols]
    hashes = df.groupby('date')[power_cols].apply(
        lambda g: hashlib.sha256(g.to_numpy(dtype='float64').tobytes()).hexdigest())
    df['pattern'] = df['date'].map(hashes)
    long = df.melt(id_vars=[c for c in df if c not in power_cols],
                   value_vars=power_cols, var_name='minute_column', value_name='power')
    long['minute_end'] = long['minute_column'].str.replace('분','').astype(int)
    long['interval_end'] = long['hour_start'] + pd.to_timedelta(long['minute_end'], unit='m')
    long['interval_start'] = long['interval_end'] - pd.Timedelta(minutes=15)
    long = long.sort_values('interval_start').reset_index(drop=True)
    assert long['interval_start'].is_unique
    assert long['interval_start'].diff().dropna().eq(pd.Timedelta(minutes=15)).all()
    assert not long['power'].isna().any()
    df.to_csv(DATA/'hourly_audited.csv', index=False)
    long.to_csv(DATA/'quarter_hour.csv', index=False)
    vc = hashes.value_counts()
    audit = dict(rows=len(df),columns=len(pd.read_csv(io.BytesIO(raw)).columns),
      days=len(hashes),quarter_hours=len(long),start=str(long.interval_start.min()),
      end=str(long.interval_end.max()),sha256=hashlib.sha256(raw).hexdigest(),
      repaired_rows=int(bad.sum()),repaired_dates=df.loc[bad,'date'].dt.strftime('%Y-%m-%d').unique().tolist(),
      unique_day_patterns=len(vc),days_in_repeated_patterns=int(vc[vc>1].sum()),
      repeated_extra_days=int((vc-1).sum()),largest_pattern_group=int(vc.max()),
      missing=df.isna().sum().loc[lambda x:x>0].to_dict(),
      power_min=float(long.power.min()),power_max=float(long.power.max()),
      mean_column_max_difference=float((df[power_cols].mean(axis=1)-df['평균']).abs().max()),
      personnel_range=[float(df['공장인원'].min()),float(df['공장인원'].max())],
      fractional_personnel=int((df['공장인원'].dropna()%1!=0).sum()),
      wage_column_values=df['인건비'].unique().tolist(),
      unit_status='NOT DOCUMENTED: figures use source units; economic model is conditional on kW.',
      time_status='ASSUMPTION: hour is interval-start hour; 15/30/45/60 are interval ends, Asia/Seoul.',
      repair_policy='Infer hour from original row order; exclude entire repaired dates as targets and missing-mask their power history.')
    (OUT/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    return df,long,audit

if __name__=='__main__': run()
