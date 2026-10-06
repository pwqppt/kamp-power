"""Exact daily power duplicates, covariate consistency and forecast stratification."""
from pathlib import Path
import json,hashlib
from itertools import combinations
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs/repeat_audit';OUT.mkdir(exist_ok=True)
r=pd.read_csv(ROOT/'data/original.csv');r['date']=pd.to_datetime(r['날짜'].astype(str));r['row_hour']=r.groupby('date').cumcount()
COLS=['생산량','기온','풍속','습도','강수량','공장인원']
powers={d:tuple(z[['15분','30분','45분','60분']].to_numpy().ravel()) for d,z in r.groupby('date',sort=True)}
groups={};daygroup={}
for day,key in powers.items():
 if key not in groups:groups[key]=[]
 groups[key].append(day)
for i,(key,days) in enumerate(groups.items(),1):
 for day in days:daygroup[day]=i
rows=[];gr=[];pairs=[]
for key,days in groups.items():
 gid=daygroup[days[0]];n=len(days)
 z=r[r.date.isin(days)]
 daily=[]
 for d in days:
  a=r[r.date==d];daily.append({'date':d,'group':gid,'group_size':n,'repeated':n>1,'weekday':d.day_name(),'month':d.month,'production_total':a['생산량'].sum(),'production_positive_hours':int(a['생산량'].gt(0).sum()),'temperature_mean':a['기온'].mean(),'humidity_mean':a['습도'].mean(),'personnel_mean':a['공장인원'].mean(),'power_mean':np.mean(key),'power_max':max(key),'invalid_hour':not a['시간'].between(0,23).all()})
 rows.extend(daily)
 if n<2:continue
 item={'group':gid,'days':n,'dates':','.join(str(d.date()) for d in days),'weekday_count':len({d.dayofweek for d in days}),'month_count':len({d.month for d in days}),'day_span':(days[-1]-days[0]).days,'power_mean':np.mean(key),'power_max':max(key)}
 for c in COLS:
  v=[r.loc[r.date==d,c].to_numpy() for d in days]
  same=all(np.array_equal(v[0],a,equal_nan=True) for a in v[1:]);item[c+'_all_identical']=same
  means=[np.nanmean(a) for a in v];item[c+'_daily_mean_range']=float(max(means)-min(means))
 gr.append(item)
 for a,b in combinations(days,2):
  ra=r[r.date==a];rb=r[r.date==b];pp={'group':gid,'date_a':a,'date_b':b,'gap_days':(b-a).days,'same_weekday':a.dayofweek==b.dayofweek,'same_month':a.month==b.month}
  allsame=True
  for c in COLS:
   va=ra[c].to_numpy();vb=rb[c].to_numpy();eq=np.array_equal(va,vb,equal_nan=True);pp[c+'_identical']=eq;pp[c+'_hourly_MAE']=float(np.nanmean(abs(va-vb)));allsame &= eq
  pp['all_six_covariates_identical']=allsame;pairs.append(pp)
daily=pd.DataFrame(rows).sort_values('date');group=pd.DataFrame(gr);pair=pd.DataFrame(pairs)
daily.to_csv(OUT/'daily_membership.csv',index=False);group.to_csv(OUT/'repeated_groups.csv',index=False);pair.to_csv(OUT/'within_group_pairs.csv',index=False)
monthly=daily.groupby('month').agg(days=('date','size'),repeated_days=('repeated','sum'),unique_curves=('group','nunique'))
monthly['repeated_fraction']=monthly.repeated_days/monthly.days;monthly.to_csv(OUT/'monthly_counts.csv')
weekly=daily.groupby('weekday').agg(days=('date','size'),repeated_days=('repeated','sum'));weekly['repeated_fraction']=weekly.repeated_days/weekly.days;weekly.to_csv(OUT/'weekday_counts.csv')
# Consecutive dates / lag repetition, no assumed repaired-hour order except file ordering.
lag=[]
for k in [1,2,7,14,28]:
 comparable=0;equal=0
 for d,key in powers.items():
  earlier=d-pd.Timedelta(days=k)
  if earlier in powers:comparable+=1;equal+=key==powers[earlier]
 lag.append({'lag_days':k,'comparable_day_pairs':comparable,'equal_pairs':equal,'fraction':equal/comparable})
pd.DataFrame(lag).to_csv(OUT/'calendar_lag_matches.csv',index=False)
# Exact power vector classification against complete day vectors actually covered by training labels.
# Origins spanning midnight are stratified by target date, not just issuance day.
p=pd.read_csv(ROOT/'outputs/intraday_v1/predictions.csv.gz',parse_dates=['origin','target'])
p['target_date']=(p.target-pd.Timedelta(minutes=1)).dt.normalize();status=[]
# Match intraday.build target eligibility without importing/fitting model libraries.
# Monthly training cutoffs are Apr..Jul1: all candidate training dates precede bad-hour dates Jul13/15.
assert r.loc[r.date.lt(pd.Timestamp('2021-07-01')),'시간'].between(0,23).all()
origins=pd.date_range('2021-01-08','2021-06-30 23:00',freq='h')
o=pd.DatetimeIndex(np.repeat(origins.to_numpy(),8))
training_meta=pd.DataFrame({'origin':o,'target':o+pd.to_timedelta(np.tile(np.arange(1,9),len(origins))*15,unit='m')})
training_coverage=[]
for month in [4,5,6,7]:
 cut=pd.Timestamp(2021,month,1)
 train_targets=training_meta.loc[training_meta.origin+pd.Timedelta(hours=2)<cut,'target'].drop_duplicates()
 covered_dates=(train_targets-pd.Timedelta(minutes=1)).dt.normalize().value_counts()
 complete_days=set(covered_dates[covered_dates.eq(96)].index)
 trainkeys={k for d,k in powers.items() if d in complete_days}
 training_coverage.append({'month':month,'complete_training_days':len(complete_days),'unique_training_daily_curves':len(trainkeys)})
 pm=p[p.month==month].copy();pm['seen_training_curve']=pm.target_date.map(lambda d:powers.get(d) in trainkeys)
 # Strict within-month actual days, avoiding next-month target labels at boundary.
 pm=pm[pm.target_date.dt.month.eq(month)]
 for (model,seen),q in pm.groupby(['model','seen_training_curve']):
  error=q.pred-q.y;pos=q.y>q.threshold;warn=q.pred>q.threshold
  tp=int((pos&warn).sum());fp=int((~pos&warn).sum());fn=int((pos&~warn).sum())
  status.append({'month':month,'model':model,'seen_training_curve':seen,'target_days':q.target_date.nunique(),'forecast_pairs':len(q),'unique_target_slots':q.target.nunique(),'MAE':abs(error).mean(),'RMSE':np.sqrt((error**2).mean()),'bias':error.mean(),'TP':tp,'FP':fp,'FN':fn,'recall':tp/(tp+fn) if tp+fn else np.nan})
 # Day membership using raw vectors INCLUDING bad-hour days flags, separate exclusions.
 for day in sorted(d for d in powers if d.month==month):
  daily.loc[daily.date==day,'seen_complete_training_curve']=powers[day] in trainkeys
pd.DataFrame(status).to_csv(OUT/'seen_unseen_forecast_metrics.csv',index=False);daily.to_csv(OUT/'daily_membership.csv',index=False)
# Example groups: largest, consecutive run and within-July.
example_ids={int(group.sort_values('days',ascending=False).iloc[0]['group']),daygroup[pd.Timestamp('2021-04-12')],daygroup[pd.Timestamp('2021-07-28')]}
daily[daily.group.isin(example_ids)].to_csv(OUT/'example_day_covariates.csv',index=False)
summary={'raw_sha256':hashlib.sha256((ROOT/'data/original.csv').read_bytes()).hexdigest(),'days':len(daily),'unique_curves':len(groups),'repeated_groups':len(group),'repeated_days':int(daily.repeated.sum()),'extra_duplicate_days':len(daily)-len(groups),'duplicate_day_pairs':len(pair),'multi_weekday_groups':int((group.weekday_count>1).sum()),'multi_month_groups':int((group.month_count>1).sum()),'all_six_covariates_identical_pairs':int(pair.all_six_covariates_identical.sum()),'covariate_identical_groups':{c:int(group[c+'_all_identical'].sum()) for c in COLS},'covariate_identical_pairs':{c:int(pair[c+'_identical'].sum()) for c in COLS},'calendar_lag_matches':lag,'provenance':'unknown; repetition alone does not prove augmentation','training_daily_curve_coverage':training_coverage,'seen_status':'full 96-value vectors among actual training targets (Jan8 start, strict monthly cutoff); uses future target-day actuals for retrospective diagnosis only, never a model input or live selection rule','time_errors':'raw profiles kept for forensic duplicate counts; bad-hour days already excluded from prediction targets'}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2));print(monthly.to_string());print(daily[daily.group.isin(example_ids)].to_string(index=False));print(pd.DataFrame(status).query("model in ['lgb_F0','week','last']").to_string(index=False))
