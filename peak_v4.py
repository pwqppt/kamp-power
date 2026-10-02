"""Two final preregistered hypotheses: modal-state median and whole-day analog."""
import argparse
import json
import numpy as np
import pandas as pd
from peak_study import ROOT,OUT,FOLDS,JULY,load,split,frame,scores,read_forecasts,gate_decision,dump,update_status
from context_features import classifier
from experiment import model_for
from residual_experiment import baseline

D=OUT/'peak_v4'

def predict_h4(tr,v,cols):
    state=tr.actual_production.gt(0).astype(int)
    gate=classifier().fit(tr[cols],state);prob=gate.predict_proba(v[cols])[:,1]
    m=model_for('LGB_medium').set_params(objective='regression_l1')
    m.fit(tr.loc[state==1,cols],tr.loc[state==1,'y'])
    pred=np.where(prob>=.5,np.maximum(m.predict(v[cols]),0),baseline(v))
    return pred,prob,[]

def predict_h5(tr,v,cols):
    names=['dow_sin','dow_cos','holiday','recent24_mean','recent7_mean','recent24_max','ctx_production_mean_24','ctx_zero_fraction_24','ctx_day_power_ratio']
    def daily(x):
        t=x.sort_values('interval_start').groupby('date').first().copy()
        t['dow_sin']=np.sin(2*np.pi*t.dow/7);t['dow_cos']=np.cos(2*np.pi*t.dow/7)
        return t
    a=daily(tr).sort_index(ascending=False);b=daily(v)
    med=a[names].median();scale=(a[names].quantile(.75)-a[names].quantile(.25)).replace(0,1)
    aa=((a[names].fillna(med)-med)/scale).to_numpy();bb=((b[names].fillna(med)-med)/scale).to_numpy()
    selected={};audit=[]
    for i,(date,row) in enumerate(b.iterrows()):
        distance=np.sqrt(((aa-bb[i])**2).sum(axis=1));idx=int(distance.argmin());source=a.index[idx]
        ratio=float(np.clip(row.recent24_mean/max(a.iloc[idx].recent24_mean,1),.5,1.5))
        curve=tr[tr.date==source].sort_values('interval_start').y.to_numpy()*ratio
        assert len(curve)==96 and tr[tr.date==source].interval_end.max()<row.origin
        selected[date]=curve
        audit.append(dict(date=str(date.date()),source_day=str(source.date()),origin=str(row.origin),distance=float(distance[idx]),scale=ratio))
    pred=np.concatenate([selected[date] for date,_ in v.groupby('date',sort=True)])
    assert v.interval_start.is_monotonic_increasing
    return pred,np.full(len(v),np.nan),audit

def run(name):
    dest=D/name;dest.mkdir(parents=True,exist_ok=True)
    x,cols=load();pieces=[];audits=[]
    fn=predict_h4 if name=='H4' else predict_h5
    for start,end in FOLDS:
        tr,v=split(x,start,end);p,pr,a=fn(tr,v,cols)
        f=frame(v,p,pr,tr,name,start);pieces.append(f);audits+=a
        print(name,start,scores(f),flush=True)
    val=pd.concat(pieces,ignore_index=True)
    ref=read_forecasts('reference');rv=ref[ref.date.between('2021-04-01','2021-06-30')]
    decision=gate_decision(val,rv)
    naive=read_forecasts('naive');nv=naive[naive.date.between('2021-04-01','2021-06-30')]
    decision['versus_naive']=gate_decision(val,nv)
    dump(dest/'validation_decision.json',decision)
    val.to_csv(dest/'validation_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    # Decision persisted before looking at July outcomes.
    tr,v=split(x,*JULY);p,pr,a=fn(tr,v,cols);j=frame(v,p,pr,tr,name,JULY[0]);audits+=a
    allf=pd.concat([val,j],ignore_index=True)
    allf.to_csv(dest/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame([dict(period=period,**scores(g)) for period,g in [('Apr-Jun',val),('July',j)]]).to_csv(dest/'scores.csv',index=False)
    pd.DataFrame([dict(fold=k,**scores(g)) for k,g in allf.groupby('fold')]).to_csv(dest/'fold_scores.csv',index=False)
    if audits:pd.DataFrame(audits).to_csv(dest/'analog_sources.csv',index=False)
    # Origin perturbation: actual validation labels/covariates cannot affect inference.
    changed=v.copy();changed[['y','actual_temp','actual_production','actual_personnel']]=999999
    pp,_,_=fn(tr,changed,cols);assert np.array_equal(p,pp)
    paired=val[['interval_start','date','y','pred']].merge(rv[['interval_start','pred']],on='interval_start',suffixes=('','_ref'),validate='one_to_one')
    day=paired.groupby('date').apply(lambda g:pd.Series({'mae_delta':(g.pred-g.y).abs().mean()-(g.pred_ref-g.y).abs().mean(),
      'peak_mae_delta':abs(g.pred.max()-g.y.max())-abs(g.pred_ref.max()-g.y.max())}),include_groups=False)
    rng=np.random.default_rng(42);draw=day.to_numpy()[rng.integers(0,len(day),size=(2000,len(day)))].mean(axis=1)
    dump(dest/'bootstrap.json',{k:{'mean':float(day[k].mean()),'low':float(np.quantile(draw[:,i],.025)),'high':float(np.quantile(draw[:,i],.975))} for i,k in enumerate(day.columns)})
    dump(dest/'checks.json',{'future_actual_perturbation_invariant':True,'complete_days':True,'decision_saved_before_July':True,'new_independent_validation':False})
    msg=f"{name}: {'탐색 채택 조건 통과' if decision['eligible'] else '미채택'}. 4~6월 MAE {scores(val)['mae']:.6f}, 일최대 MAE {scores(val)['daily_peak_mae']:.6f}, 피크일 재현율 {scores(val)['peak_day_recall']:.2%}."
    (ROOT/'docs'/f'{name}_RESULTS.md').write_text('# '+name+' 결과\n\n'+msg+'\n\n설계: PEAK_V4_PROTOCOL.md. 전체 표와 미래값 교란검사: outputs/peak_v4/'+name+'.\n\n미충족: '+', '.join(decision['failed_checks'])+'\n\n반대 검토: 반복·재사용된 과거 검증, 상태오판 또는 유사일 불일치. 독립 실증이 아니며 통계적 구간이 0을 포함하면 우위 확정 금지. 다음: 남은 고정 가설 실행 후 보고서 결론 재검토.\n',encoding='utf8')
    update_status(msg+' 사용자의 추가 지시에 따라 최대 5개 가설까지 검증 중. 보고서 주결론 확정 보류. 재현 python -X utf8 peak_v4.py '+name)
    print(msg,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name',choices=['H4','H5']);run(p.parse_args().name)
