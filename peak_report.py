"""Complete the four competition deliverables without economic conversion."""
import base64
import html
import json
from importlib.util import find_spec
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from peak_study import ROOT, D, read_forecasts, dump, update_status


def markdown_table(frame):
    def cell(value):
        return f'{value:.3f}' if isinstance(value, (float,np.floating)) else str(value)
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+ '\n'.join(
        '| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))


def save_figure(fig, path):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    from matplotlib.text import Text
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text(): continue
        box=text.get_window_extent(renderer)
        assert box.x0>=-2 and box.y0>=-2 and box.x1<=fig.bbox.width+2 and box.y1<=fig.bbox.height+2, text.get_text()
    fig.savefig(path,dpi=150)
    plt.close(fig)


def run():
    # Read the packaged font without importing its legacy distutils-dependent initializer.
    font=Path(find_spec('koreanize_matplotlib').origin).parent/'fonts/NanumGothic.ttf'
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus']=False
    comparison = pd.read_csv(D/'comparison.csv')
    sim = pd.read_csv(D/'simulation/summary.csv')
    daily_sim = pd.read_csv(D/'simulation/daily.csv')
    averages = []; hourly_parts = []
    for name in ['naive','reference','H1','H2','H3']:
        f = read_forecasts(name)
        for period, g in [('Apr-Jun',f[f.date.between('2021-04-01','2021-06-30')]),('July',f[f.fold=='2021-07'])]:
            hourly = g.groupby(g.interval_start.dt.floor('h'))[['y','pred']].mean()
            daily = g.groupby('date')[['y','pred']].mean()
            averages.append(dict(candidate=name,period=period,interval_mae=(g.y-g.pred).abs().mean(),
                hourly_mean_mae=(hourly.y-hourly.pred).abs().mean(),daily_mean_mae=(daily.y-daily.pred).abs().mean()))
            if name in ['reference','naive']:
                hourly['candidate']=name; hourly['period']=period; hourly_parts.append(hourly.reset_index())
    averages = pd.DataFrame(averages); averages.to_csv(D/'mean_power_scores.csv',index=False)
    pd.concat(hourly_parts).to_csv(D/'hourly_mean_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    boot_rows = []
    rng = np.random.default_rng(42)
    for period, part in [('Apr-Jun',daily_sim[daily_sim.fold<'2021-07']),('July',daily_sim[daily_sim.fold=='2021-07'])]:
        for fraction in [.05,.10,.15]:
            for window in [30,60]:
                p=part[(part.fraction==fraction)&(part.window_minutes==window)].pivot(index='date',columns='model',values='peak_reduction')
                delta=(p.retained_model-p.seasonal_naive).to_numpy()
                draws=delta[rng.integers(0,len(delta),size=(2000,len(delta)))].mean(axis=1)
                boot_rows.append(dict(period=period,fraction=fraction,window_minutes=window,days=len(delta),
                    mean_reduction_difference=delta.mean(),ci_low=np.quantile(draws,.025),ci_high=np.quantile(draws,.975)))
    pd.DataFrame(boot_rows).to_csv(D/'simulation/paired_bootstrap.csv',index=False)

    figures=D/'figures'; figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#8b95a5','#245d87','#c27436','#994f71','#568275']
    labels=['전주 기준','기존 결합','H1 가중','H2 비대칭','H3 보정']
    val=comparison[comparison.period=='Apr-Jun'].set_index('candidate').loc[['naive','reference','H1','H2','H3']]
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for ax,col,title in zip(axes,['mae','daily_peak_mae','peak_day_recall'],['평균 MAE','일최대 MAE','피크일 재현율 (%)']):
        vals=val[col].to_numpy()*(100 if 'recall' in col else 1)
        ax.bar(labels,vals,color=colors)
        ax.set_title(title); ax.tick_params(axis='x',rotation=25)
        ax.set_ylim(0,vals.max()*1.22)
        for i,y in enumerate(vals):ax.text(i,y+vals.max()*.02,f'{y:.2f}',ha='center',fontsize=9)
        ax.set_ylabel('%' if 'recall' in col else '원자료 단위')
    fig.suptitle('4~6월 시간순 검증 · 91일 / 8,736구간\n세 가설 모두 채택 조건 미충족',fontsize=15)
    save_figure(fig,figures/'validation.png')

    f=read_forecasts('reference'); july=f[f.fold=='2021-07']
    peaks=july.groupby('date')[['y','pred','naive']].max().reindex(pd.date_range('2021-07-01','2021-07-31'))
    fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained')
    for col,label,color in [('y','실측','#1f2937'),('pred','기존 결합','#245d87'),('naive','전주 기준','#c27436')]:
        axes[0].plot(peaks.index,peaks[col],label=label,color=color,marker='o',markersize=3)
    axes[0].set_title('7월 일최대 크기 · 29일 (7/13·7/15 제외)');axes[0].set_ylabel('원자료 단위'); axes[0].legend(ncol=3)
    axes[0].set_ylim(0,260);axes[0].set_yticks(np.arange(0,251,50));axes[0].set_xlim(peaks.index.min(),peaks.index.max())
    series=pd.read_csv(D/'simulation/example_series.csv.gz',parse_dates=['date','interval_start'])
    worst=daily_sim[(daily_sim.model=='retained_model')&(daily_sim.fraction==.1)&(daily_sim.window_minutes==60)].sort_values(['peak_reduction','date']).iloc[0]
    case=series[series.date==worst.date]
    for col,label,color in [('actual','미조정 실측','#1f2937'),('pred','예측','#245d87'),('adjusted','이동 후 실측','#b34c3b')]:
        axes[1].plot(case.interval_start,case[col],label=label,color=color)
    axes[1].set_title(f"고정 10% / ±60분 시나리오의 최대 악화일: {worst.date}\n일최대 {worst.original_peak:.1f} → {worst.adjusted_peak:.1f}")
    axes[1].set_ylabel('원자료 단위');axes[1].legend(ncol=3)
    axes[1].set_xlim(case.interval_start.min(),case.interval_start.max());axes[1].set_ylim(0,270);axes[1].set_yticks(np.arange(0,251,50))
    import matplotlib.dates as mdates
    axes[0].xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1,6,11,16,21,26,31]));axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    axes[1].xaxis.set_major_locator(mdates.HourLocator(byhour=list(range(0,24,3))));axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    save_figure(fig,figures/'peak_failures.png')

    vtable=val[['mae','daily_peak_mae','peak_day_recall','interval_recall']].reset_index()
    jtable=comparison[comparison.period=='July'][['candidate','mae','daily_peak_mae','peak_day_recall']]
    stable=sim[(sim.period.isin(['Apr-Jun','July'])) & (sim.fraction==.1) & (sim.window_minutes==60)][
        ['period','model','mean_daily_reduction','worsened_days','days','period_max_reduction']]
    avgtable=averages[averages.candidate.isin(['naive','reference'])]
    report='''# 피크 개선 v3: 세 가설 종료와 연구 본체 결과

2026-10-02. **새 방법은 모두 미채택**이다. 설정 탐색을 멈췄고 기존 결합모델을 탐색 대조군으로 유지한다. 7월 결과로 설정이나 선택을 바꾸지 않았다. 평균 예측의 장점과 피크·운영 실패를 함께 보고한다. main 및 출발 실험 브랜치의 모델은 교체하지 않았다.

## 1. 평균전력 예측

전날 16시의 가용 정보로 다음날 96구간을 예측한다. 아래 MAE는 원자료 단위이고, 각 피크 임계값은 fold 학습 전력의 95백분위다. 재현율은 0~1 비율이다.

### 4~6월 rolling time validation

'''+markdown_table(vtable)+'''

H1과 H2는 피크일 재현율이 높아졌지만 평균 MAE·일최대 MAE·구간 재현율 조건을 충족하지 못했다. H3는 과거 예측 위험구간에서 잔차 평균이 음수여서 네 fold 모두 상향 보정 0이었다. 기존과 같아 개선 조건을 충족하지 못했다. 7월만 보고 더 강한 가중치·보정값을 시도하지 않았다.

### 7월 사후 재평가

'''+markdown_table(jtable)+'''

H1의 7월 피크일 재현율 85%는 탈락을 뒤집는 근거가 아니다. 전주 기준은 두 기간 모두 일최대 오차와 피크 탐지에 더 강하다. 평균 MAE 우위를 전반적인 피크 예측 우위로 해석하지 않는다.

### 시간평균과 일평균 전력

'''+markdown_table(avgtable)+'''

시간평균 타깃은 같은 시간의 15분 실측 네 개 평균으로 정의했다. 원자료의 반올림된 ‘평균’ 열과 최대 0.5 차이가 있는 기존 감사 결과를 유지한다. 15분 MAE, 시간평균 MAE, 일평균 MAE는 서로 다른 지표다.

현재 환경의 기존 함수와 새 비교 함수는 출력이 정확히 같지만, 과거 저장값과 완전 재현되지는 않았다. 과거 4~6월 MAE 16.811990 대신 현재 공통 대조군 16.910845를 사용했다. 세 후보 모두 같은 Python 3.13/Windows 환경에서 비교했다. 원인 확정 없이 이 차이를 모델 개선으로 표현하지 않는다. 환경·월별 차이는 `reproduction.json`, `historical_drift.csv`에 있다.

## 2. 오차 집중조건

기존 저장 7월 예측에서 30도 초과 구간 MAE 15.989, 편향 -10.930이며 20~25도 MAE 8.838보다 크다. 생산 양수 구간은 MAE 13.355, 생산 0 구간은 5.854다. 이것은 평균 오차의 비교다.

반면 **극단적인 절대오차**는 다른 조건에 모인다. 현재 공통 대조군의 과거 3~6월 OOF 절대오차 90백분위 60.881을 7월에 고정 적용하면 큰 오류는 40구간이다. 생산 0 구간이 32개, 00~07시가 31개다. 생산 0의 큰 오류율 3.14%는 양수 0.45%보다 높다. 평균오차가 작은 조건에서도 드문 큰 실패가 발생한다. 이 한계는 피크 과소예측과 구분한다.

오류의 전체 조건표: `large_error_conditions.csv`, `condition_comparison.csv`. 월별 분모·일수·편향·미탐·오경보를 포함한다. 기온×생산×시간대의 모든 관측 조합도 공개한다. 시간별 기록이 네 번 반복되므로 15분 행을 독립 표본으로 해석하지 않는다.

## 3. 피크 발생조건

7월 실측 피크 442구간 중 437개(98.9%)가 생산 양수 구간이다. 10~17시에 297개, 07~10시에 116개, 17~22시에 29개가 있다. 생산 0의 피크율은 5/1,020=0.49%, 생산 양수는 437/1,764=24.77%다. 30도 초과 피크율은 121/256=47.27%다.

이는 사후 연관관계다. 미래 실측 기온·생산·인원은 예측 입력으로 넣지 않았으며, 냉방이나 설비 가동의 인과효과·실제 미래 생산계획을 확보했다고 주장하지 않는다. 과거 기온 3분위는 7월 전부를 최상위로 분류하므로 계절 이동과 온도 자체의 효과를 분리할 수 없다.

저장된 과거 결합모델은 17~22시 피크 29개를 모두 놓쳤다. 일최대 평균 편향 -12.943보다 실제 최대시각 편향 -21.171이 크다. 크기와 시각 문제를 함께 남겼다. 상세: [7월 진단](PEAK_V3_DIAGNOSIS.md).

## 4. 피크 저감 시뮬레이션

4~6월 91일과 7월 29일에 0/5/10/15% 이동, ±30/60분, 기존 결합·전주 기준·oracle를 모두 적용했다. 총 2,880개 날짜×시나리오 결과다. 0%는 무조정 대조군이다. 운영 예측으로 이동 비율행렬을 먼저 고정하고 실측 부하를 그 비율대로 옮겨 평가했다. oracle만 실제값을 미리 안다.

아래 10%·±60분은 전체 표 중 고정 예시이며 결과로 선택한 운영정책이 아니다. 양수는 피크 감소, 음수는 증가다.

'''+markdown_table(stable)+'''

기존 결합은 4~6월 평균 일최대를 4.154 낮추지만 16/91일 악화했고 전체기간 최대는 오히려 21.273 증가했다. 최대 악화일 4/2에는 실제 일최대가 181→243.273으로 62.273 증가했다. 예측에서는 14.045 감소를 예상했던 사례다. 7월 평균 감소 6.357과 4/29일 악화만 강조하면 이 위험을 놓친다.

따라서 **자동 이동정책을 채택하지 않는다**. 모든 실패 사례를 `simulation/all_worsened_cases.csv`에 보존했다. 일별 평균과 기간 전체 최대값을 분리 보고하며, 특정 이동량이나 시간창을 사후 최적값으로 고르지 않는다. paired day bootstrap은 반복된 역사자료의 기술적 불확실성으로만 제공한다.

원자료 단위, 설비별 이동 가능 부하, 선후공정·재기동·납기·수신 용량은 미확인이다. 비음수·열합 1·이동 범위·최대 반출률·일 총량 보존을 검사했다. 전력 합 보존은 생산량 보존 증명이 아니다. **경제성·요금 환산은 수행하지 않았다.**

## 검증과 재현

- 고정 requirements 설치. 원본 SHA256 일치.
- 6/15·7/20의 예측시점 이후 원자료 전력·기온·생산·인원·기타 날씨를 바꾸고 입력 특징 불변 검사.
- 4~7월 보정값의 미래 OOF 라벨 교란 불변, 학습·보정 종료시각 검사.
- 비대칭 손실의 gradient/Hessian 수치 미분 대조.
- 채택 조건 재계산, 예측행 고유성·96구간·유한값 검사. 시뮬레이션 전 조합의 이동제약·총량 검사.
- 8~9월을 재평가하지 않았다. 알려진 4~7월의 반복 사용, 반복 패턴, 다중 가설 때문에 독립 검증이 아니며 bootstrap을 확정 우위 증거로 쓰지 않는다.

Windows 재현 명령(새 clone의 저장소 루트):

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -X utf8 peak_study.py prepare
.venv/Scripts/python.exe -X utf8 peak_study.py reference
.venv/Scripts/python.exe -X utf8 peak_study.py H1
.venv/Scripts/python.exe -X utf8 peak_study.py H2
.venv/Scripts/python.exe -X utf8 peak_study.py H3
.venv/Scripts/python.exe -X utf8 check_peak_study.py
.venv/Scripts/python.exe -X utf8 peak_study.py summarize
.venv/Scripts/python.exe -X utf8 peak_study.py simulate
.venv/Scripts/python.exe -X utf8 peak_report.py
.venv/Scripts/python.exe -X utf8 run_stage.py check
```

`run_stage.py check`의 기존 검사만으로 새 예측 검증을 대신하지 않는다. 원래 01~03 전체 탐색은 재실행할 필요가 없다. 새 clone에서 이전 Python/OS와 완전 일치까지 보장하지 않는다.

## 브랜치와 다음 작업

진단 `experiments/peak-diagnosis-v3` → H1 `experiments/peak-v3-h1-weighted` → H2 `experiments/peak-v3-h2-asymmetric` → H3 `experiments/peak-v3-h3-calibration` → 본체 종합 `experiments/peak-v3-summary` 순의 누적 이력이다. 각 가설 종료 지점이 별도 브랜치로 남는다.

연구 본체 네 항목의 분석 산출물은 완료했지만 모델 개선·자동 운영의 채택은 실패했다. 다음은 새 독립 기간 데이터와 당시 발행된 기온 예보, 생산계획의 입수시각·설비 이동제약 확인이다. 같은 7월에서 추가 하이퍼파라미터 탐색은 하지 않는다. 경제성은 검증되지 않은 자동 운영성과를 금전 절감으로 바꾸는 근거로 사용하지 않는다.
'''
    (ROOT/'docs/PEAK_V3_RESULTS.md').write_text(report,encoding='utf-8')
    # A self-contained review artifact is convenient outside the repository.
    output=ROOT.parent/'outputs'; output.mkdir(exist_ok=True)
    def image_data(name):return 'data:image/png;base64,'+base64.b64encode((figures/name).read_bytes()).decode()
    sections=[('4~6월 고정 검증',vtable),('7월 사후 평가',jtable),('평균전력 집계',avgtable),('조건부 이동 예시: 10% · ±60분',stable)]
    document='<!doctype html><html lang="ko"><meta charset="utf-8"><title>피크 예측 검증 결과</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font-family:Arial,\"Malgun Gothic\",sans-serif;color:#172b3a;line-height:1.65}h1,h2{line-height:1.3}table{border-collapse:collapse;font-size:13px;width:100%;margin:24px 0}th,td{padding:8px;border-bottom:1px solid #ccd4db;text-align:right}th{background:#eef3f7}img{width:100%}pre{white-space:pre-wrap;font:inherit;background:#f6f8fa;padding:24px}</style><h1>피크 예측 검증: 세 가설 모두 미채택</h1><p>2026-10-02 · 4~6월 rolling validation · 7월 사후 진단 · 미래 실측 기온·생산량 입력 제외</p><p>기존 결합모델의 평균오차 이득을 유지하면서 피크 지표를 개선한 후보는 없었습니다. 추가 탐색은 중단했습니다. 조건부 이동도 평균 저감과 일부 날짜의 큰 악화가 함께 나타나 자동정책은 채택하지 않았습니다. 요금 분석은 수행하지 않았습니다.</p>'
    for title,t in sections: document+='<h2>'+html.escape(title)+'</h2>'+t.to_html(index=False,float_format=lambda n:f'{n:.3f}')
    document+='<img alt="4~6월 세 가설 비교" src="'+image_data('validation.png')+'"><img alt="7월 피크 및 최악의 반동피크" src="'+image_data('peak_failures.png')+'"><h2>전체 결과와 한계</h2><pre>'+html.escape(report)+'</pre></html>'
    (output/'peak-study-results.html').write_text(document,encoding='utf-8')
    dump(D/'figures/checks.json',{'files':['validation.png','peak_failures.png'],
        'data_rows_validation':91*96,'july_days':29,'all_text_within_canvas':True,'visual_review':'pending'})
    update_status('세 가설 모두 미채택, 추가 탐색 종료. 평균전력·오차 집중조건·피크 발생조건·조건부 피크 저감의 네 항목 분석 완료. 2,880개 날짜×시나리오 검증; 10%/±60분 기존 결합은 4~6월 16/91일 악화, 전체기간 최대 21.273 증가. 자동 이동정책 미채택. 요금 분석 미실행. 종합: docs/PEAK_V3_RESULTS.md. 다음은 새로운 독립 데이터/예보 발행시각/생산계획·이동제약 확보; 같은 7월 추가 탐색 금지.')
    print('Report, mean-power metrics, paired simulation bootstrap, and figures generated.')


if __name__=='__main__':run()
