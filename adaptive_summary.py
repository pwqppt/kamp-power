"""Paired diagnostic uncertainty and a Korean checkpoint figure."""
import json
import numpy as np
import pandas as pd
from visualize import plt,save,checks,FIG,OUT
D=OUT/'adaptive_v2'
p=pd.read_csv(D/'selected_july.csv',parse_dates=['date','interval_start'])
base=pd.read_csv(OUT/'residual_v2/selected_july_predictions.csv',parse_dates=['date','interval_start'])
base=base[base.candidate=='naive'][['interval_start','pred']].rename(columns={'pred':'naive'})
p=p.merge(base,on='interval_start',validate='one_to_one')
assert len(p)==29*96 and p.interval_start.is_unique
p['delta']=(p.y-p.pred).abs()-(p.y-p.naive).abs()
daily=p.groupby('date').delta.mean().to_numpy();rng=np.random.default_rng(42)
boot=rng.choice(daily,(3000,len(daily)),replace=True).mean(axis=1)
result={'July_MAE_difference_adaptive_minus_naive':float(daily.mean()),'day_bootstrap_95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
        'days':len(daily),'limitations':'Post-hoc exploration; previously viewed July; repeated/serial patterns; no multiplicity adjustment.'}
(D/'bootstrap.json').write_text(json.dumps(result,indent=2))
v=pd.read_csv(D/'validation_summary.csv').set_index('candidate');j=pd.read_csv(D/'july_scores.csv').set_index('candidate')
names=['naive','mixture','predicted_state_0.5'];labels=['전주 기준','컨텍스트 상태혼합','예측상태별 결합 [선택]']
fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
for ax,data,title in [(axes[0],v,'4~6월 검증 (91일)'),(axes[1],j,'7월 재평가 (29일)')]:
    bars=ax.barh(labels,[data.loc[n,'mae'] for n in names],color=['#777777','#bc7730','#227c9d']);ax.invert_yaxis()
    ax.bar_label(bars,fmt='%.2f',padding=5);ax.set_xlim(0,24)
    ax.set(title=title,xlabel='평균 절대오차 (원자료 단위)')
fig.supxlabel('실제 미래 생산량 미사용 | 임계값 0.5는 4~6월 선택 | 사후 탐색이며 새 독립 검증 아님',fontsize=10)
save(fig,'08_context_adaptive_mae')
(FIG/'adaptive_visual_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
assert not any(c['automated_issues'] for c in checks)
print(json.dumps(result))
