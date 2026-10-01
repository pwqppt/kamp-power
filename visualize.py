"""Source-backed Korean figures; PNG/SVG, explicit units and selection rules."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
font_path=ROOT/'assets/NanumGothic-Regular.ttf'
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
else:
    # koreanize-matplotlib bundles a Korean font for clean clones where assets are omitted.
    import koreanize_matplotlib
plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False,'font.size':11,'axes.titlesize':15,'axes.titlepad':18,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})
checks=[]
def save(fig,name):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();issues=[]
    for ax in fig.axes:
        leg=ax.get_legend()
        if leg is not None and leg.get_window_extent(renderer).overlaps(ax.get_window_extent(renderer)):
            issues.append('legend overlaps plotting rectangle')
        labels=[t for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
        boxes=[t.get_window_extent(renderer) for t in labels]
        if any(a.overlaps(b) for a,b in zip(boxes,boxes[1:])):issues.append('adjacent x tick labels overlap')
    for fmt in ['png','svg']:fig.savefig(FIG/f'{name}.{fmt}',dpi=180,bbox_inches='tight',facecolor='white')
    checks.append({'figure':name,'automated_issues':issues,'font':'NanumGothic','units':'source unit, not verified kW'})
    plt.close(fig)
def bar(labels,values,title,xlabel,name):
    fig,ax=plt.subplots(figsize=(10,5.8),layout='constrained')
    bars=ax.barh(labels,values,color='#227c9d',height=.58);ax.invert_yaxis()
    ax.bar_label(bars,fmt='%.2f',padding=7);ax.set_xlim(0,max(values)*1.18)
    ax.set(title=title,xlabel=xlabel);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);save(fig,name)
def run():
    a=pd.read_csv(OUT/'ablation_summary.csv')
    bar(['F0 달력','F1 + 같은 시각 과거 전력','F2 + 최근 전력 통계 [선택]','F3 + 과거 날씨','F4 + 과거 생산·인원','F5 + 상호작용'],a.mae,'특징 추가가 항상 개선을 만들지는 않음 | 4~6월 검증','평균 절대오차 (원자료 단위, 낮을수록 좋음)','01_feature_comparison')
    p=pd.read_csv(OUT/'test_predictions.csv',parse_dates=['interval_start','date'])
    # Fixed first 7 test days avoids selecting a visually flattering example.
    g=p[p.date<p.date.min()+pd.Timedelta(days=7)]
    fig,ax=plt.subplots(figsize=(13,5.5),layout='constrained')
    ax.plot(g.interval_start,g.y,label='실제값',color='#222222',lw=1.25)
    ax.plot(g.interval_start,g.pred,label='AI 예측',color='#0086a8',lw=1.1)
    ax.fill_between(g.interval_start,g.lower90,g.upper90,color='#0086a8',alpha=.12,label='명목 90% 구간')
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=3,frameon=False)
    ax.set_title('최초 테스트 7일 | 전체 테스트 구간 포함률 84.7%',pad=50)
    ax.xaxis.set_major_locator(mdates.DayLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set(xlabel='2021년 날짜·시간 (15분 단위로 통합)',ylabel='전력값 (원자료 단위)');ax.grid(alpha=.18);save(fig,'02_forecast_first_week')
    t=pd.read_csv(OUT/'treatment_sensitivity.csv')
    labels=['기본: 결측 자체 처리','중앙값 대체','0 대체','결측 학습행 제외','입력 극단값 절단','타깃 피크 절단 [비권장 비교]','중복 학습일 제거']
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    for ax,col,title in zip(axes,['mae','peak_day_recall'],['평균 절대오차 ↓','피크일 재현율 (%) ↑']):
        values=t[col]*(100 if col=='peak_day_recall' else 1)
        b=ax.barh(labels,values,color=['#227c9d']*5+['#cc6849','#227c9d']);ax.invert_yaxis();ax.bar_label(b,fmt='%.1f',padding=5)
        ax.set_xlim(0,max(values)*1.2);ax.set_title(title);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    axes[1].set_yticklabels([]);fig.suptitle('전처리 민감도 | 피크를 잘라내면 탐지를 놓칠 수 있음',fontsize=16)
    save(fig,'03_treatment_sensitivity')
    c=pd.read_csv(OUT/'condition_errors.csv');c=c[c.dimension=='period']
    order=['00–07시','07–10시','10–17시','17–22시','22–24시'];c=c.set_index('condition').loc[order]
    bar(order,c.mae,'시간대별 예측 실패 분석 | 테스트 45일','평균 절대오차 (원자료 단위)','04_error_by_hour')
    (FIG/'visual_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
    assert not any(x['automated_issues'] for x in checks),checks
    print('Created 4 PNG + 4 SVG; automatic legend/tick checks passed. Manual review also required.')
if __name__=='__main__':run()
