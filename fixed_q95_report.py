"""Fixed JanMar q95 analysis: retrained onset and unchanged regression predictions."""
from pathlib import Path
import json,importlib.util
import numpy as np
import pandas as pd
from lightgbm import Booster
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from onset_fixed_q95 import raw_load,build,OUT,ROOT,current_feature_row,risk_features,apply_cal
from intraday import make_model,metrics
REG=OUT/"regression";REG.mkdir(parents=True,exist_ok=True)
def table(df):
    fmt=lambda v:f"{v:.3f}" if isinstance(v,(float,np.floating)) else str(v)
    return "\n".join(["| "+" | ".join(df.columns)+" |","| "+" | ".join(["---"]*len(df.columns))+" |"]+["| "+" | ".join(fmt(v) for v in r)+" |" for r in df.itertuples(index=False,name=None)])
def run():
    y,cov,info=raw_load();x,m,g=build(y,cov)
    cut0=pd.Timestamp("2021-04-01")
    c=float(y[y.index<cut0].quantile(.95))
    oldreg=pd.read_csv(ROOT/"outputs/intraday_v1/monthly_metrics.csv")
    cp=pd.read_csv(OUT/"predictions.csv",parse_dates=["origin"])
    assert cp.threshold.eq(c).all()
    config=json.loads((OUT/"selected_model.json").read_text())
    X,_=risk_features(x,m,g,c);z=cp[cp.month.eq(7)&cp.model.eq("F0")]
    origin=z.origin.iloc[0]
    row=current_feature_row(y,origin,config)
    np.testing.assert_allclose(row,X.loc[[origin],config["features"]],equal_nan=True)
    yy=y.copy();yy.loc[yy.index>origin]=99999
    np.testing.assert_allclose(row,current_feature_row(yy,origin,config),equal_nan=True)
    b=Booster(model_file=str(OUT/config["model_file"]))
    np.testing.assert_allclose(apply_cal(b.predict(X.loc[z.origin,config["features"]]),config["probability_calibration"]),z.prob,rtol=1e-10,atol=1e-12)
    results,rows,targets,windows,conditions=[],[],[],[],[]
    for month in [4,5,6,7]:
        cut=pd.Timestamp(2021,month,1);end=cut+pd.offsets.MonthBegin()
        train=m.origin+pd.Timedelta(hours=2)<cut
        test=(m.origin>=cut)&(m.origin<end)
        xx=x.loc[test]
        fallback=y.ffill().reindex(pd.DatetimeIndex(m.loc[test,"origin"])).to_numpy()
        reg=make_model("lgb");reg.fit(x.loc[train,g["F0"]],m.loc[train,"y"])
        reg.booster_.save_model(str(REG/f"model_{month}_lgb_F0.txt"))
        predictions={"lgb_F0":np.maximum(0,reg.predict(xx[g["F0"]])),
                     "last":xx.lag0,"week":xx.week,"day":xx.day,
                     "hour_mean":xx[["lag0","lag1","lag2","lag3"]].mean(axis=1)}
        for name,pred in predictions.items():
            if isinstance(pred,pd.Series):pred=pred.fillna(pd.Series(fallback,index=pred.index)).to_numpy()
            p=m.loc[test].copy();p["pred"]=pred;p["model"]=name;p["month"]=month;p["threshold"]=c
            s=metrics(p)
            original=oldreg[oldreg.month.eq(month)&oldreg.model.eq(name)].iloc[0]
            for k in ["MAE","RMSE"]:
                np.testing.assert_allclose(s[k],original[k],rtol=1e-7,atol=1e-7)
            rows.append(dict(month=month,model=name,**s));results.append(p)
            for h,a in p.groupby("horizon"):targets.append(dict(month=month,model=name,minutes=int(h*15),**metrics(a)))
            for tag,mask in [("production_zero",p.production.eq(0)),("production_positive",p.production.gt(0)),("temperature_gt30",p.temperature.gt(30))]:
                if mask.any():conditions.append(dict(month=month,model=name,condition=tag,**metrics(p[mask])))
            for o,a in p.groupby("origin"):
                av=a.sort_values("horizon")
                actualmax=float(av.y.max());predmax=float(av.pred.max())
                predtime=av.loc[av.pred.idxmax(),"target"]
                truetimes=av.loc[av.y.eq(actualmax),"target"]
                timeerror=float(abs(truetimes-predtime).min()/pd.Timedelta(minutes=1))
                windows.append(dict(month=month,model=name,origin=o,actual_max=actualmax,predicted_max=predmax,max_abs_error=abs(actualmax-predmax),max_underprediction=max(actualmax-predmax,0),peak_time_error_minutes=timeerror,flat_prediction=av.pred.nunique()==1))
    p=pd.concat(results,ignore_index=True)
    p.to_csv(REG/"predictions.csv.gz",index=False,compression="gzip")
    pd.DataFrame(rows).to_csv(REG/"monthly_metrics.csv",index=False)
    pd.DataFrame(targets).to_csv(REG/"horizon_metrics.csv",index=False)
    pd.DataFrame(conditions).to_csv(REG/"conditions.csv",index=False)
    win=pd.DataFrame(windows);win.to_csv(REG/"window_peaks.csv",index=False)
    ws=win.groupby(["month","model"]).agg(n=("origin","size"),window_max_MAE=("max_abs_error","mean"),window_max_underprediction=("max_underprediction","mean"),peak_time_MAE_minutes=("peak_time_error_minutes","mean"),flat_prediction_rate=("flat_prediction","mean")).reset_index()
    ws.to_csv(REG/"window_peak_summary.csv",index=False)
    daily=[]
    for (month,name),a in p[p.horizon.le(4)].groupby(["month","model"]):
        a=a.copy();a["day"]=(a.target-pd.Timedelta(minutes=1)).dt.normalize()
        for day,v in a.groupby("day"):
            if len(v)!=96:continue
            assert v.target.nunique()==96
            am=float(v.y.max());pm=float(v.pred.max())
            pt=v.loc[v.pred.idxmax(),"target"]
            tt=v.loc[v.y.eq(am),"target"]
            daily.append(dict(month=month,model=name,day=day,actual_max=am,predicted_max=pm,max_abs_error=abs(am-pm),peak_time_error_minutes=float(abs(tt-pt).min()/pd.Timedelta(minutes=1))))
    daily=pd.DataFrame(daily);daily.to_csv(REG/"daily_rolling_peaks.csv",index=False)
    ds=daily.groupby(["month","model"]).agg(days=("day","size"),daily_max_MAE=("max_abs_error","mean"),peak_time_MAE_minutes=("peak_time_error_minutes","mean")).reset_index()
    ds.to_csv(REG/"daily_rolling_peak_summary.csv",index=False)
    pooled=[]
    for period,months in [("AprJun",[4,5,6]),("July",[7])]:
        for name,a in p[p.month.isin(months)].groupby("model"):pooled.append(dict(period=period,model=name,**metrics(a)))
    pooled=pd.DataFrame(pooled);pooled.to_csv(REG/"pooled_metrics.csv",index=False)
    frequency=[]
    for month in [1,2,3,4,5,6,7]:
        cut=pd.Timestamp(2021,month,1);end=cut+pd.offsets.MonthBegin()
        vals=y[(y.index>cut)&(y.index<=end)].dropna()
        frequency.append(dict(month=month,valid_slots=len(vals),high_slots=int(vals.gt(c).sum()),high_fraction=float(vals.gt(c).mean()),monthly_prior_q95=float(y[y.index<cut].quantile(.95)) if month>1 else np.nan,fixed_threshold=c))
    freq=pd.DataFrame(frequency);freq.to_csv(OUT/"threshold_frequency.csv",index=False)
    spec=importlib.util.find_spec("koreanize_matplotlib")
    fonts=sorted(Path(spec.origin).parent.rglob("*.ttf"));font=next((v for v in fonts if v.stem=="NanumGothic"),fonts[0])
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family":font_manager.FontProperties(fname=str(font)).get_name(),"axes.unicode_minus":False,"svg.fonttype":"none","font.size":10})
    figs=ROOT/"figures/fixed_q95";figs.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.5))
    axs[0].plot(freq.month,freq.fixed_threshold,"o-",label="1~3월 기준 고정")
    aa=freq[freq.month.ge(4)]
    axs[0].plot(aa.month,aa.monthly_prior_q95,"s--",label="이전 월별 기준")
    axs[0].set(xlabel="월",ylabel="전력 · 원자료 단위",title="고부하 기준 비교")
    axs[0].set_xticks(range(1,8));axs[0].grid(alpha=.2)
    axs[0].legend(loc="upper center",bbox_to_anchor=(.5,-.2),ncol=2,fontsize=9)
    axs[1].bar(freq.month,100*freq.high_fraction,color="#2c799e")
    axs[1].set(xlabel="월",ylabel="고부하 구간 비율 (%)",title=f"고정 기준 초과 비율 · 기준 {c:g}")
    axs[1].set_xticks(range(1,8));axs[1].grid(axis="y",alpha=.2)
    fig.tight_layout(rect=(0,.12,1,1));fig.savefig(figs/"fixed_threshold.png",dpi=150);fig.savefig(figs/"fixed_threshold.svg");plt.close(fig)
    h=pd.DataFrame(targets)
    fig,ax=plt.subplots(figsize=(8,4.8))
    for name,color in [("lgb_F0","#2477ab"),("last","#d58023"),("week","#727272")]:
        a=h[h.month.eq(7)&h.model.eq(name)]
        ax.plot(a.minutes,a.MAE,"o-",label={"lgb_F0":"LightGBM","last":"현재값 유지","week":"전주"}[name],color=color)
    ax.set(xlabel="앞으로 예측할 시간 (분)",ylabel="MAE · 원자료 단위",title="전력값 예측 오차 · 7월 재사용 진단")
    ax.grid(alpha=.2);fig.legend(loc="lower center",ncol=3);fig.tight_layout(rect=(0,.1,1,1))
    fig.savefig(figs/"july_horizon_MAE.png",dpi=150);fig.savefig(figs/"july_horizon_MAE.svg");plt.close(fig)
    classifier=pd.read_csv(OUT/"pooled_metrics.csv")
    sel=classifier[classifier.model.isin(["F0","week"])][["period","model","n","positives","TP","FP","FN","recall","precision","FPR","AP"]].copy()
    sel[["recall","precision","FPR"]]*=100
    regshow=pooled[pooled.model.isin(["lgb_F0","last","week"])][["period","model","n","MAE","RMSE","window_max_MAE","recall","precision","FP","FN"]].copy()
    regshow[["recall","precision"]]*=100
    checks={"status":"complete","fixed_threshold":c,"threshold_reference_start":"2021-01-01","threshold_reference_end_exclusive":"2021-04-01 00:00:00","reference_n":int(y[y.index<cut0].notna().sum()),"raw_sha256":info["sha256"],"threshold_equal_across_all_classifier_rows":"PASS","single_snapshot_future_perturbation":"PASS","classifier_model_reload":"PASS","regression_MAE_RMSE_reproduction":"PASS rtol=atol=1e-7","regression_model":"lgb_F0","classification_model":"F0 frozen","July_reused":True,"daily_peak_is_rolling_reconstruction_not_day_ahead":True,"no_preparation_time":True,"new_external_data":False}
    (OUT/"verification.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2))
    doc="# 固定 기준 재분석\n\n".replace("固定","고정")
    doc+=f"전력값 기준은1~3월 자료(구간 끝시각4/1 00:00 미만)의95백분위수 {c:g}, 원자료 단위다. 4~7월 모든 모델과 정답에 동일하게 적용한다. 기준을 뒤에 다시 고르지 않았으며 대회 지정값/계약전력/설비 위험 기준이 아니다. 기준을 정한1~3월에서 예측 성능을 평가했다고 주장하지 않는다.\n\n"
    doc+="## 전력값 예측\n\nLightGBM 회귀와 현재값유지/전주/전일/최근1시간평균을 그대로 재현했다. MAE/RMSE는 이전값과 일치하며 고부하 판정만 새 기준을 사용한다. recall은15분 구간 초과 탐지이며 진입 분류와 다르다.\n\n"+table(regshow)+"\n\n"
    doc+="## 고부하 진입 분류\n\n현재값이 기준 이하일 때 향후15~120분 첫 초과를 예측한다. 바뀐 정답으로 F0/F1 분류기를 재학습했지만 특징 구성 선택은 기존F0로 고정했다. 확률보정/경보보정은 월 시작 전 분리된14일씩이며 과거 정상오경보5% 목표다. 고부하 전력 기준이 같아도 확률 경보 기준은 월별 과거 자료로 보정된다. 이전 월별 고부하 타깃과 재현율을 직접 성능 개선율로 비교하지 않는다.\n\n"+table(sel)+"\n\n"
    doc+="## 최대값과 발생시각\n\n향후2시간의 최대값 오차와 발생시각 오차를 window_peak_summary.csv로 분리했다. 동률은 예측의 첫 최대시각과 실제 최대시각들 중 가장 가까운 시각의 차이로 계산한다. 현재값유지는 평평한 예측이므로 최대시각을 구분하지 못하며 flat_prediction_rate와 함께 해석한다.\n\n"
    doc+="daily_rolling_peak_summary.csv는 매시간 발행한향후15~60분 예측을 이어 완전한96구간인 날만 집계한 일최대 진단이다. 하루 시작 전에 일최대값/시각을 예측했다는 성과가 아니다.\n\n"
    doc+="## 오류 조건·적용범위\n\n진입 conditions.csv/episodes.csv/episode_summary.csv와 전력회귀 regression/conditions.csv를 별도 저장했다. 생산량·기온은 관측 조건별 진단이며 인과 원인이나 선택 모델 입력을 뜻하지 않는다. 같은 사건의 중복 정시 예측과 실제 사건 수를 구분하고 적용불가능한 사건도 전체 분모에 남겼다.\n\n"
    doc+="## 검증과 한계\n\n원본해시·고정 기준·미래 입력교란·분류모델재로드·기존회귀오차 재현 검증 통과. 7월 및 개발 자료는 재사용 진단, 반복곡선/자료생성/현장기록 공개시차 미확인. 기준 선택을 이미 관찰한자료에 맞춰 바꾼 후속 탐색 실험이며 독립 검증 성과로 표현하지 않는다. 이 분석용 고부하 기준으로 실제 원화절감/설비 위험을 확정하지 않는다. 기존 모델/결과/보고서초안은 보존했고 과거 PPT를 이번 결과로 갱신하지 않았다. 그림 직접 렌더 검토는 생성 후 확인한다.\n\n"
    doc+="재현: python onset_fixed_q95.py → python fixed_q95_report.py. 구간별회귀/최대값/진입 결과는 outputs/onset_fixed_q95.\n"
    (ROOT/"docs/FIXED_Q95_RESULTS.md").write_text(doc)
    note=f"## 최신 실행 완료:1~3월q95 고정 기준 재분석\n고정 전력기준{c:g}(원자료단위),4~7월동일. 고부하진입 재학습/기존전력회귀 재현 및2시간최대값/당일롤링 일최대 진단 완료. 상세 docs/FIXED_Q95_RESULTS.md, outputs/onset_fixed_q95/verification.json. 7월재사용·타깃변경·일최대는하루전예측아님을명시. 다음은결과·한글그림검토/체크포인트 정리.\n\n"
    for filename in ["PROJECT_STATUS.md","HANDOFF.md"]:
        path=ROOT/filename;path.write_text(note+path.read_text())
    print(json.dumps(checks,ensure_ascii=False),flush=True);print(regshow.to_string(index=False),flush=True);print(sel.to_string(index=False),flush=True);print(ws.to_string(index=False),flush=True)
if __name__=="__main__":run()
