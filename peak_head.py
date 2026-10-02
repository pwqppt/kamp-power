"""Preregistered separate daily peak head; point forecasts remain unchanged."""
import numpy as np
import pandas as pd
from peak_study import OUT,ROOT,read_forecasts,dump,update_status
from continue_research import confusion
D=OUT/'peak_v4/H6'

def measure(g,col):
    e=g[col]-g.y
    return dict(days=len(g),daily_peak_mae=float(e.abs().mean()),peak_underprediction=float(np.maximum(-e,0).mean()),**confusion(g.y,g[col],g.threshold))

def run():
    D.mkdir(parents=True,exist_ok=True)
    f=read_forecasts('reference');f=f[f.date>='2021-04-01']
    d=f.groupby(['date','fold'])[['y','pred','naive','threshold']].max().reset_index()
    d['peak_head']=(d.pred+d.naive)/2
    v=d[d.fold<'2021-07'];a=measure(v,'peak_head');b=measure(v,'pred')
    checks={'unchanged_interval_forecasts':True,'daily_peak_mae_improved':a['daily_peak_mae']<b['daily_peak_mae'],
      'peak_day_recall_improved':a['recall']>b['recall'],'underprediction_not_worse':a['peak_underprediction']<=b['peak_underprediction']}
    dump(D/'validation_decision.json',{'eligible':all(checks.values()),'checks':checks,'output_type':'separate daily peak, NOT maximum of point forecast',
      'peak_head_validation':a,'reference_validation':b,'naive_validation':measure(v,'naive'),'independent_validation':False})
    # July metrics only after frozen decision.
    rows=[]
    for period,g in [('Apr-Jun',v),('July',d[d.fold=='2021-07']),*d.groupby('fold')]:
        for col in ['pred','naive','peak_head']:rows.append(dict(period=period,model=col,**measure(g,col)))
    pd.DataFrame(rows).to_csv(D/'scores.csv',index=False);d.to_csv(D/'daily_predictions.csv',index=False)
    rng=np.random.default_rng(42);rows=[]
    for period,g in [('Apr-Jun',v),('July',d[d.fold=='2021-07'])]:
        for ref in ['pred','naive']:
            e=(g.peak_head-g.y).abs().to_numpy()-(g[ref]-g.y).abs().to_numpy()
            draw=e[rng.integers(0,len(e),size=(2000,len(e)))].mean(axis=1)
            rows.append(dict(period=period,reference=ref,mean=e.mean(),low=np.quantile(draw,.025),high=np.quantile(draw,.975)))
    pd.DataFrame(rows).to_csv(D/'bootstrap.csv',index=False)
    msg=f"H6 별도 일최대 출력 {'탐색 채택' if all(checks.values()) else '미채택'}: 4~6월 일최대 MAE {a['daily_peak_mae']:.6f}, 재현율 {a['recall']:.2%}. 15분 곡선 불변; 96구간 모델 개선으로 표현하지 않는다."
    update_status(msg+' 재현 python -X utf8 peak_head.py. 다음: 채택 출력·조건부 운영·공식 보고서 확정.')
    (ROOT/'docs/H6_RESULTS.md').write_text('# H6 결과\n\n'+msg+'\n\n설계: PEAK_HEAD_PROTOCOL.md. 전체 점수·불확실성·월별 실패: outputs/peak_v4/H6. 독립 검증이 아니며 전주 기준 대비 지표가 모두 우월한지 반드시 구분한다.\n',encoding='utf8')
    print(msg);print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':run()
