"""Monthly-frozen versus causal seven-day alarm threshold updates."""
from pathlib import Path
import json, importlib.util
import numpy as np
import pandas as pd
from lightgbm import Booster
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from onset_v2 import raw_load, build, risk_features, fitcal, apply_cal, scores
ROOT=Path(__file__).resolve().parent
BASE=ROOT/"outputs/onset_v2"
OUT=ROOT/"outputs/onset_v2_weekly"
OUT.mkdir(parents=True,exist_ok=True)
ALPHAS=[.01,.05]
MODELS=["F0","week"]
MIN_NEGATIVES=30

def threshold(z,update,alpha,fallback):
    a=z[(z.index>=update-pd.Timedelta(days=14))&(z.index+pd.Timedelta(hours=2)<update)]
    negative=a.loc[~a.label,"prob"]
    t=float(np.quantile(negative,1-alpha,method="higher")) if len(negative)>=MIN_NEGATIVES else fallback
    return t,a,len(negative)<MIN_NEGATIVES

def table(df):
    fmt=lambda v:f"{v:.3f}" if isinstance(v,(float,np.floating)) else str(v)
    return "\n".join(["| "+" | ".join(df.columns)+" |","| "+" | ".join(["---"]*len(df.columns))+" |"]+["| "+" | ".join(fmt(v) for v in row)+" |" for row in df.itertuples(index=False,name=None)])

def run():
    y,cov,info=raw_load()
    x,m,g=build(y,cov)
    actual=m.y.to_numpy().reshape(-1,8)
    old=pd.read_csv(BASE/"predictions.csv",parse_dates=["origin"])
    allq,logs,monthly=[],[],[]
    for month in [4,5,6,7]:
        cut=pd.Timestamp(2021,month,1);end=cut+pd.offsets.MonthBegin()
        pstart=cut-pd.Timedelta(days=28);split=cut-pd.Timedelta(days=14)
        c=float(y[y.index<cut].quantile(.95))
        X,groups=risk_features(x,m,g,c)
        high=actual>c;first=np.where(high.any(axis=1),high.argmax(axis=1)+1,0)
        meta=m.groupby("origin").head(1).copy().set_index("origin")
        meta["label"]=first>0;meta["early"]=(first>0)&(first<=2)
        meta["first_index"]=first;meta["first_end"]=meta.index+pd.to_timedelta(first*15,unit="m")
        meta["date"]=meta.index.normalize();meta["month"]=month
        eligible=meta.current.notna()&meta.current.le(c)
        procal=eligible&(meta.index>=pstart)&(meta.index+pd.Timedelta(hours=2)<split)
        relevant=eligible&(meta.index>=split)&(meta.index+pd.Timedelta(hours=2)<=end)
        test=relevant&(meta.index>=cut)
        for name in MODELS:
            if name=="F0":
                b=Booster(model_file=str(BASE/f"model_{month}_F0.txt"))
                params=fitcal(b.predict(X.loc[procal,groups["F0"]]),meta.loc[procal,"label"].to_numpy())
                prob=apply_cal(b.predict(X.loc[relevant,groups["F0"]]),params)
            else:
                s=X.week_window_max.fillna(X.lag0).clip(lower=0)
                norm=float(max(c,s[procal].max(),1))
                params=fitcal((s[procal]/norm).clip(0,1),meta.loc[procal,"label"].to_numpy())
                prob=apply_cal((s[relevant]/norm).clip(0,1),params)
            z=meta.loc[relevant].copy();z["prob"]=prob;z["model"]=name
            before=old[old.month.eq(month)&old.model.eq(name)]
            target=z.loc[test.reindex(z.index)]
            assert np.array_equal(before.origin.to_numpy(),target.index.to_numpy())
            np.testing.assert_allclose(before.prob,target.prob,rtol=1e-10,atol=1e-12)
            for alpha in ALPHAS:
                base_t,base_a,_=threshold(z,cut,alpha,np.nan)
                assert np.isfinite(base_t)
                if alpha==.05:np.testing.assert_allclose(base_t,before.decision_threshold.iloc[0])
                for policy in ["monthly","weekly"]:
                    q=target.copy();q["decision_threshold"]=base_t
                    q["update_at"]=cut;q["policy"]=policy;q["alpha"]=alpha
                    updates=[cut] if policy=="monthly" else list(pd.date_range(cut,end-pd.Timedelta(seconds=1),freq="7D"))
                    for i,update in enumerate(updates):
                        block_end=updates[i+1] if i+1<len(updates) else end
                        t,a,fallback=threshold(z,update,alpha,base_t)
                        assert (a.index+pd.Timedelta(hours=2)<update).all()
                        poisoned=z.copy()
                        poison=poisoned.index+pd.Timedelta(hours=2)>=update
                        poisoned.loc[poison,"prob"]=99999.
                        poisoned.loc[poison,"label"]=~poisoned.loc[poison,"label"]
                        tt,_,_=threshold(poisoned,update,alpha,base_t)
                        np.testing.assert_allclose(t,tt)
                        idx=(q.index>=update)&(q.index<block_end)
                        q.loc[idx,"decision_threshold"]=t;q.loc[idx,"update_at"]=update
                        logs.append(dict(month=month,model=name,policy=policy,alpha=alpha,update_at=update,valid_until=block_end,threshold=t,calibration_n=len(a),negatives=int((~a.label).sum()),fallback=fallback,latest_calibration_target=a.index.max()+pd.Timedelta(hours=2),calibration_FPR=float((a.loc[~a.label,"prob"]>t).mean())))
                    q["warn"]=q.prob>q.decision_threshold
                    assert (q.update_at<=q.index).all()
                    if policy=="monthly" and alpha==.05:assert np.array_equal(q.warn,before.warn)
                    allq.append(q.reset_index())
                    monthly.append(dict(period=f"month{month}",model=name,policy=policy,alpha=alpha,**scores(q)))
    q=pd.concat(allq,ignore_index=True)
    rows=monthly
    for period,months in [("AprJun",[4,5,6]),("July",[7])]:
        for (name,policy,alpha),z in q[q.month.isin(months)].groupby(["model","policy","alpha"]):
            rows.append(dict(period=period,model=name,policy=policy,alpha=alpha,**scores(z)))
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/"metrics.csv",index=False)
    q.to_csv(OUT/"predictions.csv",index=False)
    log=pd.DataFrame(logs);log.to_csv(OUT/"update_audit.csv",index=False)
    assert (log.latest_calibration_target<log.update_at).all()
    comparisons=[]
    for (period,name,alpha),a in metrics.groupby(["period","model","alpha"]):
        w=a[a.policy.eq("weekly")].iloc[0];f=a[a.policy.eq("monthly")].iloc[0]
        comparisons.append(dict(period=period,model=name,alpha=alpha,recall_difference=w.recall-f.recall,FPR_difference=w.FPR-f.FPR,FP_difference=int(w.FP-f.FP),FN_difference=int(w.FN-f.FN),weak_Pareto=bool(w.recall>=f.recall and w.FPR<=f.FPR),strict_gain=bool(w.recall>f.recall or w.FPR<f.FPR)))
    comp=pd.DataFrame(comparisons);comp.to_csv(OUT/"policy_comparison.csv",index=False)
    # Incident counts remain distinct from hourly, overlapping prediction rows.
    eps=[]
    for (month,name,policy,alpha),z in q.groupby(["month","model","policy","alpha"]):
        cut=pd.Timestamp(2021,month,1);end=cut+pd.offsets.MonthBegin()
        c=float(y[y.index<cut].quantile(.95));previous=y.shift()
        events=y.index[y.gt(c)&previous.le(c)&previous.notna()&(y.index>cut)&(y.index<=end)]
        reachable=z[z.label].groupby("first_end")
        evaluable=detected=0
        for e in events:
            if e in reachable.groups:
                evaluable+=1;detected+=int(reachable.get_group(e).warn.any())
        eps.append(dict(month=month,model=name,policy=policy,alpha=alpha,total_entries=len(events),evaluable_entries=evaluable,detected_entries=detected,missed_entries=evaluable-detected,coverage=evaluable/len(events) if len(events) else np.nan,episode_recall=detected/evaluable if evaluable else np.nan))
    pd.DataFrame(eps).to_csv(OUT/"episode_summary.csv",index=False)
    failures=q[q.model.eq("F0")&((q.label&~q.warn)|(~q.label&q.warn))].copy()
    failures["error_type"]=np.where(failures.label,"miss","false_alarm")
    failures.to_csv(OUT/"error_cases.csv",index=False)
    # Fixed development success rule; July never selects the policy.
    verdict={}
    for alpha in ALPHAS:
        a=comp[comp.model.eq("F0")&comp.alpha.eq(alpha)]
        p=a[a.period.eq("AprJun")].iloc[0]
        monthly=a[a.period.isin(["month4","month5","month6"])]
        verdict[str(alpha)]=dict(pooled_Pareto=bool(p.weak_Pareto and p.strict_gain),no_month_recall_drop_over5pp=bool(monthly.recall_difference.ge(-.05).all()),passes=bool(p.weak_Pareto and p.strict_gain and monthly.recall_difference.ge(-.05).all()))
    verification={"status":"complete","raw_sha256":info["sha256"],"saved_probability_reproduction":"PASS","monthly005_warning_reproduction":"PASS","causal_label_maturity":"PASS","future_label_score_poisoning":"PASS","model_and_probability_calibration_frozen":True,"high_load_threshold_fixed_per_month":True,"no_preparation_time":True,"July_reused":True,"alpha_targets":ALPHAS,"development_success_rule":verdict,"policy_selected":False,"new_external_data":False}
    (OUT/"verification.json").write_text(json.dumps(verification,ensure_ascii=False,indent=2))
    spec=importlib.util.find_spec("koreanize_matplotlib")
    fonts=sorted(Path(spec.origin).parent.rglob("*.ttf"))
    font=next((p for p in fonts if p.stem=="NanumGothic"),fonts[0])
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family":font_manager.FontProperties(fname=str(font)).get_name(),"axes.unicode_minus":False,"svg.fonttype":"none","font.size":10})
    figs=ROOT/"figures/onset_v2_weekly";figs.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(11,4.8))
    for ax,metric,title in zip(axs,["recall","FPR"],["高","誤"]):
        for policy,color in [("monthly","#2477ab"),("weekly","#d58023")]:
            for alpha,style in [(.01,"--"),(.05,"-")]:
                a=metrics[metrics.model.eq("F0")&metrics.policy.eq(policy)&metrics.alpha.eq(alpha)&metrics.period.str.startswith("month")].copy()
                a["month"]=a.period.str.replace("month","",regex=False).astype(int);a=a.sort_values("month")
                ax.plot(a.month,100*a[metric],style+"o",color=color,label=("월초 고정" if policy=="monthly" else "매주 갱신")+f" · 과거 목표 {alpha*100:g}%")
        ax.set_xticks([4,5,6,7],["4월","5월","6월","7월"])
        ax.set_title("고부하 진입 탐지율" if metric=="recall" else "정상 오경보율")
        ax.set_ylabel("%");ax.grid(alpha=.2)
        if metric=="recall":ax.set_ylim(-3,103)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=2,fontsize=9)
    fig.tight_layout(rect=(0,.15,1,1))
    fig.savefig(figs/"monthly_weekly_comparison.png",dpi=150);fig.savefig(figs/"monthly_weekly_comparison.svg");plt.close(fig)
    show=metrics[["period","model","policy","alpha","recall","precision","FPR","TP","FP","FN"]].copy()
    show[["alpha","recall","precision","FPR"]]*=100
    doc="# 고부하 진입 경보: 월 고정과 매주 갱신\n\n"
    doc+="모델/확률 보정/월별 고부하 기준을 고정하고 경보 기준만 비교했다. 매월1일 및 이후7일 간격에 지난14일 기록 중 향후2시간 정답이 갱신 시각 전에 확정된 정상현재 구간을 사용한다. 현재월의 과거 정답은 이후 기준 갱신에 사용할 수 있지만 미래 정답은 사용할 수 없다. 7월은 기존에 확인한 자료의 탐색 진단이며 독립 테스트가 아니다.\n\n"
    doc+="사전 고정 목표1%와5%, 비교 LightGBM F0 및 전주 기준. 과거 오경보 목표는 미래 보장이 아니다. 음성30개 미만이면 월초 기준으로 대체한다. 준비시간 요구/물리적 작업 이동/경제성 가정은 추가하지 않았다.\n\n"
    doc+="## 성능\n\nalpha·recall·precision·FPR은 % 단위.\n\n"+table(show)+"\n\n"
    doc+="## 사전 성공 기준\n\n4~6월 합산에서 탐지율 감소 없이 오경보율을 줄이거나 오경보율 증가 없이 탐지율을 높이고, 모든 개발월의 탐지율 하락이5%p 이하여야 한다. 결과: "+json.dumps(verdict,ensure_ascii=False)+"\n\n"
    doc+="## 검증·한계\n\n기존 확률 및 월고정5% 경보 재현, 정답 시간 경계, 미래 점수/정답 교란 검증 통과. 두 방식의 연속 점수가 동일하므로 점수 분리 능력 자체를 개선한 실험이 아니다. 월별 결과/실패 기록/사건별 탐지/적용 범위를 모두 저장했다. 보고의 정시 예측과 실제 사건 수를 혼동하지 않는다. 실험 도중 최종 정책이나 기준을 재선택하지 않았다. 7월이 좋더라도4월 약점이 해결된 것으로 표현하지 않는다. 반복 전력곡선/센서 기록 공개 시차 미확인은 유지한다.\n\n"
    doc+="그래프 직접 검토는 별도 수행한다. 재현: python onset_v2.py 다음 python onset_v2_weekly.py. 출력 outputs/onset_v2_weekly.\n"
    (ROOT/"docs/ONSET_V2_WEEKLY_RESULTS.md").write_text(doc)
    note="## 최신 실행 완료: 월 고정/매주 경보 기준 갱신\n정답이 갱신시점 전에 확정된 과거14일만 사용, 모델과 확률보정 고정, 목표1/5% 및 전주 비교. docs/ONSET_V2_WEEKLY_RESULTS.md/outputs/onset_v2_weekly/verification.json 참조. 기존 확률·월고정경보 재현 및 미래교란 검증 통과. 정책 미선택. 다음은 결과 판단·그림 직접 검토·체크포인트 정리.\n\n"
    for filename in ["PROJECT_STATUS.md","HANDOFF.md"]:
        p=ROOT/filename;p.write_text(note+p.read_text())
    print(show.to_string(index=False),flush=True)
    print(json.dumps(verification,ensure_ascii=False),flush=True)

if __name__=="__main__":
    run()
