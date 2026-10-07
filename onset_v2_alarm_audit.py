"""Frozen onset_v2 alarm tradeoff audit; no new model selection on July."""
from pathlib import Path
import json, hashlib, importlib.util
import numpy as np
import pandas as pd
from lightgbm import Booster
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from onset_v2 import raw_load, build, risk_features, fitcal, apply_cal, scores
ROOT = Path(__file__).resolve().parent
BASE = ROOT/"outputs/onset_v2"
OUT = ROOT/"outputs/onset_v2_alarm_audit"
OUT.mkdir(parents=True, exist_ok=True)
ALPHAS = [0.01, 0.025, 0.05, 0.10]
MODELS = ["F0", "week", "day", "current_level"]

def table(df):
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    return "\n".join(["| "+" | ".join(df.columns)+" |", "| "+" | ".join(["---"]*len(df.columns))+" |"]+["| "+" | ".join(fmt(v) for v in row)+" |" for row in df.itertuples(index=False, name=None)])

def run():
    y,cov,info = raw_load()
    x,m,g = build(y,cov)
    actual = m.y.to_numpy().reshape(-1,8)
    selection = json.loads((BASE/"selection.json").read_text())
    assert selection["model"] == "F0", "Audit scope was preregistered for frozen F0"
    old = pd.read_csv(BASE/"predictions.csv", parse_dates=["origin"])
    thresholds, allq, comparisons, diagnostics = [], [], [], []
    for month in [4,5,6,7]:
        cut = pd.Timestamp(2021,month,1)
        end = cut+pd.offsets.MonthBegin()
        pstart = cut-pd.Timedelta(days=28)
        split = cut-pd.Timedelta(days=14)
        c = float(y[y.index<cut].quantile(.95))
        X, groups = risk_features(x,m,g,c)
        high = actual>c
        first = np.where(high.any(axis=1),high.argmax(axis=1)+1,0)
        meta = m.groupby("origin").head(1).copy().set_index("origin")
        meta["label"] = first>0
        meta["early"] = (first>0)&(first<=2)
        meta["first_end"] = meta.index+pd.to_timedelta(first*15,unit="m")
        meta["first_index"] = first
        meta["date"] = meta.index.normalize()
        meta["month"] = month
        eligible = meta.current.notna()&meta.current.le(c)
        procal = eligible&(meta.index>=pstart)&(meta.index+pd.Timedelta(hours=2)<split)
        alarmcal = eligible&(meta.index>=split)&(meta.index+pd.Timedelta(hours=2)<cut)
        test = eligible&(meta.index>=cut)&(meta.index+pd.Timedelta(hours=2)<=end)
        assert meta.loc[alarmcal].index.max()+pd.Timedelta(hours=2)<cut
        negative = ~meta.loc[alarmcal,"label"].to_numpy(bool)
        for name in MODELS:
            if name == "F0":
                booster = Booster(model_file=str(BASE/f"model_{month}_F0.txt"))
                rawp = booster.predict(X.loc[procal,groups["F0"]])
                params = fitcal(rawp,meta.loc[procal,"label"].to_numpy())
                pred = apply_cal(booster.predict(X.loc[test,groups["F0"]]),params)
                ac = apply_cal(booster.predict(X.loc[alarmcal,groups["F0"]]),params)
            else:
                s = X.lag0 if name=="current_level" else X[name+"_window_max"].fillna(X.lag0).clip(lower=0)
                norm = float(max(c,s[procal].max(),1))
                params = fitcal((s[procal]/norm).clip(0,1),meta.loc[procal,"label"].to_numpy())
                pred = apply_cal((s[test]/norm).clip(0,1),params)
                ac = apply_cal((s[alarmcal]/norm).clip(0,1),params)
            z = meta.loc[test].copy()
            z["prob"] = pred
            z["model"] = name
            if name!="current_level":
                before = old[old.month.eq(month)&old.model.eq(name)]
                assert np.array_equal(before.origin.to_numpy(),z.index.to_numpy())
                np.testing.assert_allclose(before.prob,pred,rtol=1e-10,atol=1e-12)
            fpr,tpr,_ = roc_curve(z.label,z.prob,drop_intermediate=False)
            for budget in ALPHAS:
                diagnostics.append(dict(month=month,model=name,observed_FPR_budget=budget,maximum_recall=float(tpr[fpr<=budget+1e-12].max()),use="retrospective only; no deployable threshold selected"))
            for alpha in ALPHAS:
                t = float(np.quantile(ac[negative],1-alpha,method="higher"))
                q = z.copy()
                q["warn"] = pred>t
                q["alpha"] = alpha
                q["decision_threshold"] = t
                q["headroom"] = (c-X.loc[q.index,"lag0"])/c
                q["recent_rise"] = X.loc[q.index,"lag0"]-X.loc[q.index,"lag1"]
                thresholds.append(dict(month=month,model=name,alpha=alpha,threshold=t,calibration_n=int(alarmcal.sum()),calibration_negatives=int(negative.sum()),calibration_FPR=float((ac[negative]>t).mean()),latest_calibration_target=str(meta.loc[alarmcal].index.max()+pd.Timedelta(hours=2)),test_start=str(cut)))
                allq.append(q.reset_index())
                comparisons.append(dict(period=f"month{month}",month=month,model=name,alpha=alpha,**scores(q)))
    q = pd.concat(allq,ignore_index=True)
    for period,months in [("AprJun",[4,5,6]),("July",[7])]:
        for (name,alpha),z in q[q.month.isin(months)].groupby(["model","alpha"]):
            comparisons.append(dict(period=period,month=0,model=name,alpha=alpha,**scores(z)))
    metrics = pd.DataFrame(comparisons)
    metrics.to_csv(OUT/"metrics.csv",index=False)
    pd.DataFrame(thresholds).to_csv(OUT/"thresholds.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/"retrospective_ROC.csv",index=False)
    q.to_csv(OUT/"predictions.csv",index=False)
    conds, errors = [], []
    for (month,name,alpha),z in q.groupby(["month","model","alpha"]):
        for tag,mask in [
            ("headroom_le10pct",z.headroom.le(.10)),
            ("headroom_gt10pct",z.headroom.gt(.10)),
            ("recent_rise_positive",z.recent_rise.gt(0)),
            ("recent_rise_nonpositive",z.recent_rise.le(0)),
            ("production_positive",z.production.gt(0)),
            ("production_zero",z.production.eq(0)),
            ("temperature_gt30",z.temperature.gt(30))]:
            a = z[mask]
            if len(a):
                conds.append(dict(month=month,model=name,alpha=alpha,condition=tag,small_sample=len(a)<30,**scores(a)))
        if name=="F0":
            a = z[(z.label&~z.warn)|(~z.label&z.warn)].copy()
            a["error_type"] = np.where(a.label,"miss","false_alarm")
            errors.append(a)
    pd.DataFrame(conds).to_csv(OUT/"conditions.csv",index=False)
    pd.concat(errors).to_csv(OUT/"F0_error_cases.csv",index=False)
    # Paired calendar-day circular seven-day block diagnostics, not independence proof.
    rng=np.random.default_rng(42)
    ci=[]
    for period,months in [("AprJun",[4,5,6]),("July",[7])]:
        for alpha in ALPHAS:
            a=q[q.month.isin(months)&q.model.eq("F0")&q.alpha.eq(alpha)]
            b=q[q.month.isin(months)&q.model.eq("week")&q.alpha.eq(alpha)]
            assert np.array_equal(a.origin.to_numpy(),b.origin.to_numpy())
            assert np.array_equal(a.label.to_numpy(),b.label.to_numpy())
            rows=[]
            for date,group in a.groupby("date"):
                s1,s2=scores(group),scores(b[b.date.eq(date)])
                rows.append([s1[k] for k in ["TP","FP","FN","TN"]]+[s2[k] for k in ["TP","FP","FN","TN"]])
            v=np.array(rows)
            starts=rng.integers(0,len(v),(1000,int(np.ceil(len(v)/7))))
            indices=((starts[:,:,None]+np.arange(7))%len(v)).reshape(1000,-1)[:,:len(v)]
            totals=v[indices].sum(axis=1)
            for key,i,j,ii,jj in [("recall",0,2,4,6),("FPR",1,3,5,7)]:
                diff=totals[:,i]/np.maximum(totals[:,i]+totals[:,j],1)-totals[:,ii]/np.maximum(totals[:,ii]+totals[:,jj],1)
                lo,hi=np.quantile(diff,[.025,.975])
                ci.append(dict(period=period,alpha=alpha,metric=key,CI_low=lo,CI_high=hi,days=len(v)))
    pd.DataFrame(ci).to_csv(OUT/"paired_week_block_CI.csv",index=False)
    # Figures show operational points chosen using earlier calibration only.
    spec=importlib.util.find_spec("koreanize_matplotlib")
    fonts=sorted(Path(spec.origin).parent.rglob("*.ttf"))
    font=next((p for p in fonts if p.stem=="NanumGothic"),fonts[0])
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family":font_manager.FontProperties(fname=str(font)).get_name(),"axes.unicode_minus":False,"svg.fonttype":"none","font.size":10})
    figs=ROOT/"figures/onset_v2_alarm_audit"
    figs.mkdir(parents=True,exist_ok=True)
    names={"F0":"LightGBM","week":"전주 기준","day":"전일 기준","current_level":"현재 전력만"}
    fig,axs=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    for ax,period in zip(axs,["AprJun","July"]):
        for name in MODELS:
            a=metrics[metrics.period.eq(period)&metrics.model.eq(name)].sort_values("alpha")
            ax.plot(100*a.FPR,100*a.recall,"o-",label=names[name])
        ax.set_title("4~6월" if period=="AprJun" else "7월 · 재사용 진단")
        ax.set_xlabel("실제 정상 오경보율 (%)")
        ax.set_ylim(-3,103)
        ax.grid(alpha=.2)
    axs[0].set_ylabel("실제 진입 탐지율 (%)")
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=4)
    fig.tight_layout(rect=(0,.10,1,1))
    fig.savefig(figs/"operational_tradeoffs.png",dpi=150)
    fig.savefig(figs/"operational_tradeoffs.svg")
    plt.close(fig)
    frozen=q[q.model.eq("F0")&q.alpha.eq(.05)]
    assert np.array_equal(frozen.warn.to_numpy(),old[old.model.eq("F0")].warn.to_numpy())
    for (_,alpha),z in q.groupby(["month","alpha"]):
        ref=z[z.model.eq("F0")]
        for name in MODELS:
            other=z[z.model.eq(name)]
            assert np.array_equal(ref.origin.to_numpy(),other.origin.to_numpy())
    summary={"status":"complete","raw_sha256":info["sha256"],"alphas":ALPHAS,"models":MODELS,"July_reused":True,"thresholds_chosen_before_test":True,"v2_probability_reproduction":"PASS","v2_alpha005_warning_reproduction":"PASS","same_candidate_rows":"PASS","future_FPR_guarantee":False,"operating_model_selected":False,"no_preparation_time":True,"new_external_data":False}
    (OUT/"verification.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    display=metrics[metrics.period.isin(["AprJun","July"])][["period","model","alpha","recall","precision","FPR","TP","FP","FN","AP"]].copy()
    display[["alpha","recall","precision","FPR"]]*=100
    doc="# 高負荷: 경보 기준 민감도와 단순 기준 비교\n\n"
    doc="## 고부하 진입 경보: 기준 민감도 검증\n\n"
    doc+="준비시간 조건 없는 v2를 보존하고 LightGBM F0를 고정했다. 정상 오경보 목표 1/2.5/5/10%와 비교 모델 4종을 실행 전에 지정했다. 목표는 과거 보정 구간의 목표일 뿐, 미래 실제 오경보율 보장이 아니다. 모든 후보는 동일한 정상 현재값/완전한 미래 정답을 가진 정시에 평가했다.\n\n"
    doc+="확률 보정: 월 시작 전28~14일, 경보 기준 보정: 전14일. 각 구간의 2시간 정답이 경계를 넘지 않도록 제외했다. 현재 전력만 기준도 동일한 보정 절차를 따른다. 기존 F0/전주/전일 확률 및 F0 5% 기준 경보의 재현을 확인했다.\n\n"
    doc+="### 실제 운용 기준 결과\n\nalpha·recall·precision·FPR은 % 단위. 7월은 독립 평가가 아닌 재사용 진단이다.\n\n"+table(display)+"\n\n"
    doc+="### 같은 실제 오경보율의 진단\n\nretrospective_ROC.csv는 평가 정답으로 가능한 탐지율 상한을 읽은 사후 진단이다. 배포할 경보 기준을 선정하거나 실제 운용 성과로 인용하지 않는다. 과거 목표가 같아도 월별 실제 오경보율이 다르면 정확히 같은 오경보 부담의 비교라고 부르지 않는다.\n\n"
    doc+="### 실패 조건\n\nconditions.csv에는 고부하 기준과의 거리10% 이내/밖, 최근 전력 상승/비상승, 관측 생산량0/>0, 기온30초과를 고정한 조건으로 집계했다. 표본수와 양성수를 함께 보고한다. 생산/기온은 진단 표지이며 F0 입력이라는 의미나 인과관계가 아니다. F0_error_cases.csv는 모든 실패 기록이며 좋은 사례를 골라 범위를 바꾸지 않았다.\n\n"
    doc+="### 적용과 한계\n\n운영 기준은 이번 단계에서 선택하지 않았다. 관측 자료 재사용/반복 곡선/독립성 미확인은 유지된다. 7일 블록 신뢰구간은 설명용이며 원자료 편향을 제거하지 않는다. 진입 분류만으로 전력 회귀/피크 저감 요구를 대체하지 않는다. 준비시간·설비 이동·절감액은 추가하지 않았다. 이미지 직접 검토는 생성 후 수행하며 완료로 미리 기록하지 않는다.\n\n"
    doc+="재현: python onset_v2.py 다음 python onset_v2_alarm_audit.py. 원본 불변. 모델 선택·기존 selected_model.json을 변경하지 않는다.\n"
    (ROOT/"docs/ONSET_V2_ALARM_AUDIT_RESULTS.md").write_text(doc)
    note="## 최신 단계 완료: v2 경보 기준 민감도 검증\n고정 LightGBM/전주/전일/현재전력 기준을 과거 오경보 목표1/2.5/5/10%로 비교했다. 실제 월별 오경보율은 별도 측정했고 v2 확률/5%경보 재현 검증 통과. 결과 docs/ONSET_V2_ALARM_AUDIT_RESULTS.md, outputs/onset_v2_alarm_audit. 운영 기준 선택 없음, 7월 재사용 진단 유지. 다음은 수치·실패조건 검토와 생성 그림 직접 확인.\n\n"
    for filename in ["PROJECT_STATUS.md","HANDOFF.md"]:
        path=ROOT/filename
        path.write_text(note+path.read_text())
    print(display.to_string(index=False),flush=True)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=="__main__":
    run()
