"""Figures and shared facts for the official report and editable deck."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from peak_report import save_figure
import matplotlib.pyplot as plt
from matplotlib import font_manager
from importlib.util import find_spec
from peak_study import OUT,dump

def run():
    dest=OUT/'submission';dest.mkdir(exist_ok=True);figdir=dest/'figures';figdir.mkdir(exist_ok=True)
    font=Path(find_spec('koreanize_matplotlib').origin).parent/'fonts/NanumGothic.ttf'
    font_manager.fontManager.addfont(str(font));plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    comp=pd.read_csv(OUT/'peak_v3/comparison.csv');head=pd.read_csv(OUT/'peak_v4/H6/scores.csv')
    val=comp[comp.period=='Apr-Jun'].set_index('candidate');hp=head[head.period=='Apr-Jun'].set_index('model')
    fig,ax=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    ax[0].bar(['전주 기준','기존 결합 / 최종'],[val.loc['naive','mae'],val.loc['reference','mae']],color=['#8a96a3','#1d647f'])
    ax[1].bar(['전주 기준','결합 곡선 최대','별도 일최대 출력'],hp.loc[['naive','pred','peak_head'],'daily_peak_mae'],color=['#8a96a3','#c87d3b','#1d647f'])
    for a,title in zip(ax,['15분 평균 MAE: 11.4% 감소','일최대 MAE: 전주 대비 6.1% 감소']):
        a.set_title(title);a.set_ylim(0,25);a.set_ylabel('원자료 단위')
        for bar in a.patches:a.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.5,f'{bar.get_height():.2f}',ha='center')
    fig.suptitle('4~6월 시간순 검증: 평균곡선과 일최대의 역할 분리',fontsize=16)
    save_figure(fig,figdir/'main_result.png')
    cond=pd.read_csv(OUT/'peak_v3/conditions/2021-07/conditions.csv');g=cond[(cond.dimension=='temperature_5C')&(cond.model=='pred')].set_index('condition').loc[['<=20','20-25','25-30','>30']]
    fig,ax=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    labels=['20°C 이하','20~25°C','25~30°C','30°C 초과'];ax[0].bar(labels,g.mae,color='#1d647f');ax[0].set_title('실측 기온 구간별 MAE');ax[0].set_ylabel('원자료 단위');ax[0].set_ylim(0,21)
    for i,(_,r) in enumerate(g.iterrows()):ax[0].text(i,r.mae+.5,f'{r.mae:.2f}\nn={int(r.n)}',ha='center',fontsize=10)
    ext=pd.read_csv(OUT/'research_v4/extended_conditions.csv');w=ext[(ext.fold=='2021-07')&(ext.dimension=='weekday')].set_index('condition').loc[['Mon','Tue','Wed','Thu','Fri','Sat','Sun']]
    ax[1].bar(['월','화','수','목','금','토','일'],w.mae,color='#c87d3b');ax[1].set_title('요일별 MAE: 표본 3~5일');ax[1].set_ylim(0,24);ax[1].set_ylabel('원자료 단위')
    for i,(_,r) in enumerate(w.iterrows()):ax[1].text(i,r.mae+.5,f'{r.mae:.1f}',ha='center',fontsize=10)
    fig.suptitle('7월 사후 오류 진단 · 실측 기온은 예측 입력에 미사용',fontsize=15)
    save_figure(fig,figdir/'conditions.png')
    d=pd.read_csv(OUT/'peak_v3/simulation/daily.csv');g=d[(d.model=='retained_model')&(d.fraction==.1)&(d.window_minutes==60)]
    fig,ax=plt.subplots(figsize=(11,3.6),layout='constrained');dates=pd.to_datetime(g.date)
    ax.bar(dates,g.peak_reduction,color=np.where(g.peak_reduction>=0,'#1d647f','#b34c3b'));ax.axhline(0,color='#333333',linewidth=.7)
    ax.set_ylabel('일최대 감소량 / 원자료 단위');ax.set_title('10% 이동 가능 / ±60분: 감소일과 악화일 전체 120일')
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'));ax.xaxis.set_major_locator(mdates.MonthLocator());ax.set_ylim(-70,30);ax.set_yticks([-60,-40,-20,0,20])
    save_figure(fig,figdir/'simulation_days.png')
    facts={'validation':val.loc[['naive','reference']].reset_index().to_dict('records'),
      'peak_head':head.to_dict('records'),'benchmark':pd.read_csv(OUT/'research_v4/benchmark_scores.csv').to_dict('records'),
      'simulation':pd.read_csv(OUT/'peak_v3/simulation/summary.csv').query("fraction == 0.1 and window_minutes == 60 and model == 'retained_model'").to_dict('records'),
      'economics':pd.read_csv(OUT/'research_v4/economics/electric_scenarios.csv').to_dict('records')}
    dump(dest/'facts.json',facts)
if __name__=='__main__':run()
