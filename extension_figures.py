"""Export Korean research figures from the extension's recorded results."""
import json
import numpy as np
import pandas as pd
from visualize import plt, save, checks, OUT, FIG

D=OUT/'extension'
c=pd.read_csv(D/'condition_metrics.csv')
g=c[c.dimension=='actual_production_band']
fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
colors={'AI':'#227c9d','SeasonalNaive':'#bc7730'}
labels={'AI':'AI','SeasonalNaive':'전주 동일시각'}
for k,(name,color) in enumerate(colors.items()):
    a=g[g.model==name].sort_values('condition')
    b=axes[0].bar(np.arange(3)+(k-.5)*.32,a.mae,width=.30,label=labels[name],color=color)
    axes[0].bar_label(b,fmt='%.1f',padding=3)
axes[0].set_xticks(range(3),['저생산','중생산','고생산'])
axes[0].set(ylabel='평균 절대오차 (원자료 단위)',ylim=(0,58),title='생산조건별 예측오류')
axes[0].legend(loc='lower left',bbox_to_anchor=(0,1.13),ncol=2,frameon=False)
a=g[g.model=='AI'].sort_values('condition')
b=axes[1].bar(range(3),a.peak_rate*100,color='#227c9d')
axes[1].bar_label(b,labels=[f'{v:.1f}%\nn={n:,}' for v,n in zip(a.peak_rate*100,a.n)],padding=4)
axes[1].set_xticks(range(3),['저생산','중생산','고생산'])
axes[1].set(ylabel='피크 구간 비율 (%)',ylim=(0,32),title='생산조건별 피크 발생률')
fig.supxlabel('2021.08.01–09.14 | 학습자료 3분위 구간 · 사후 관측조건 분석 · 인과관계 아님',fontsize=10)
save(fig,'05_production_conditions')

s=pd.read_csv(D/'simulation_summary.csv')
s=s[(s.period=='all_45_days')&(s.fraction==.1)]
fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
for k,(name,color) in enumerate(colors.items()):
    a=s[s.model==name].sort_values('window_minutes')
    bars=axes[0].bar(np.arange(2)+(k-.5)*.32,a.mean_daily_reduction,width=.30,label=labels[name],color=color)
    axes[0].bar_label(bars,fmt='%.2f',padding=4)
    bars=axes[1].bar(np.arange(2)+(k-.5)*.32,a.worsened_days,width=.30,color=color)
    axes[1].bar_label(bars,fmt='%d',padding=4)
axes[0].set(ylabel='평균 일피크 감소 (원자료 단위)',ylim=(-2,6),title='예측 기반 이동의 피크 저감')
axes[1].set(ylabel='일피크가 악화된 일수 / 45일',ylim=(0,25),title='부하 이동의 실패일')
for ax in axes:
    ax.set_xticks(range(2),['±30분','±60분']); ax.axhline(0,color='#333333',lw=.8)
axes[0].legend(loc='lower left',bbox_to_anchor=(0,1.13),ncol=2,frameon=False)
fig.supxlabel('구간별 최대 10% 분할 이동 가정 | 같은 조정기 · 실제 생산 일정 최적화 실적 아님',fontsize=10)
save(fig,'06_conditional_peak_reduction')

p=pd.read_csv(D/'simulation_example_series.csv',parse_dates=['time','date'])
daily=pd.read_csv(D/'simulation_daily.csv')
daily=daily[(daily.model=='AI')&(daily.fraction==.1)&(daily.window_minutes==60)].sort_values(['peak_reduction','date'])
dates=[daily.iloc[0].date,daily.iloc[-1].date]
fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained',sharey=True)
import matplotlib.dates as mdates
for ax,date,kind in zip(axes,dates,['최대 악화일','최대 개선일']):
    a=p[p.date==pd.Timestamp(date)]
    ax.plot(a.time,a.actual,color='#333333',label='이동 전 실제값',lw=1.5)
    ax.plot(a.time,a.adjusted,color='#227c9d',label='가정에 따른 이동 후',lw=1.5,ls='--')
    ax.set(title=f'{kind}: {date}',ylabel='전력값 (원자료 단위)')
    ax.set_xticks([a.time.iloc[0], *pd.date_range(pd.Timestamp(date)+pd.Timedelta(hours=6),periods=3,freq='6h'),a.time.iloc[-1]])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axes[0].legend(loc='lower left',bbox_to_anchor=(0,1.12),ncol=2,frameon=False)
fig.supxlabel('15분 통합 시간축 | 45일 중 개선량의 양 극단을 함께 선정 · 10%/±60분 가정',fontsize=10)
save(fig,'07_peak_success_failure')
(FIG/'extension_visual_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
assert not any(x['automated_issues'] for x in checks)
print('3 PNG/SVG figures generated; automated legend/tick checks passed')
