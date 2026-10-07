"""Report/verification for onset_v2.py; no preparation-time success criterion."""
from pathlib import Path
import json, hashlib, platform, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import importlib.util
from matplotlib import font_manager
_font_spec = importlib.util.find_spec("koreanize_matplotlib")
_font_files = sorted(Path(_font_spec.origin).parent.rglob("*.ttf"))
assert _font_files, "Korean font resource missing"
_font_path = next((p for p in _font_files if p.stem == "NanumGothic"), _font_files[0])
font_manager.fontManager.addfont(str(_font_path))
plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(_font_path)).get_name()
plt.rcParams["axes.unicode_minus"] = False
from onset_v2 import OUT, ROOT, scores, raw_load, build, risk_features, current_feature_row, apply_cal
from lightgbm import Booster

def table(df):
    def f(v):
        return f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    return "\n".join(lines + ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)])

def run():
    config = json.loads((OUT / "selected_model.json").read_text())
    selection = json.loads((OUT / "selection.json").read_text())
    name = selection["model"]
    q = pd.read_csv(OUT / "predictions.csv", parse_dates=["origin", "date", "first_end"])
    p = pd.read_csv(OUT / "pooled_metrics.csv")
    ep = pd.read_csv(OUT / "episodes.csv", parse_dates=["event_end", "event_start"])
    summaries = pd.read_csv(OUT / "episode_summary.csv")
    daily = pd.read_csv(OUT / "daily_metrics.csv")
    y, cov, audit_info = raw_load()
    x, m, g = build(y, cov)
    checks = {}
    counts = q.groupby(["month", "model"]).size().unstack()
    assert counts.nunique(axis=1).eq(1).all()
    assert q.label.equals(q.first_index.between(1, 8))
    assert q.loc[q.early, "label"].all()
    assert q.current.notna().all() and q.current.le(q.threshold).all()
    assert ((q.first_end - q.origin) == pd.to_timedelta(q.first_index * 15, unit="m")).all()
    assert ep.loc[ep.detected, "lead_to_interval_start_minutes"].ge(0).all()
    checks.update(equal_candidate_rows="PASS", all_15_to_120_entries_positive="PASS", eligibility_prior_only="PASS", first_entry_time="PASS")
    selected_july = q[q.month.eq(7) & q.model.eq(name)]
    current = selected_july.origin.iloc[0]
    X, _ = risk_features(x, m, g, config["high_load_threshold"])
    row = current_feature_row(y, current, config)
    np.testing.assert_allclose(row, X.loc[[current], config["features"]], equal_nan=True)
    changed = y.copy()
    changed.loc[changed.index > current] = 99999
    np.testing.assert_allclose(row, current_feature_row(changed, current, config), equal_nan=True)
    model = Booster(model_file=str(OUT / config["model_file"]))
    probs = apply_cal(model.predict(X.loc[selected_july.origin, config["features"]]), config["probability_calibration"])
    np.testing.assert_allclose(probs, selected_july.prob, rtol=1e-10, atol=1e-12)
    checks.update(single_snapshot_equivalence="PASS", future_perturbation="PASS", saved_model_reload="PASS")
    checks["raw_sha256"] = audit_info["sha256"]
    checks["python"] = platform.python_version()
    checks["versions"] = {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "lightgbm", "scikit-learn"]}
    # Separate raw-origin metrics, real episode metrics, and coverage.
    j = p[p.period.eq("July") & p.model.eq(name)].iloc[0]
    b = p[p.period.eq("July") & p.model.eq("week")].iloc[0]
    e = summaries[summaries.month.eq(7) & summaries.model.eq(name)].iloc[0]
    d = daily[daily.month.eq(7) & daily.model.eq(name)]
    result = {"status": "complete", "selected_model": name, "July_reused_diagnostic": True,
              "July": j.to_dict(), "week_July": b.to_dict(), "July_episode_summary": e.to_dict(),
              "July_days": len(d), "daily_alarms_mean": float(d.alarms.mean()),
              "daily_false_alarms_mean": float(d.false_alarms.mean()), "no_preparation_time_filter": True}
    (OUT / "result_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (OUT / "verification.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    # Reliability and uncertainty are descriptive diagnostics on reused data.
    z = selected_july.copy()
    z["probability_bin"] = pd.cut(z.prob, np.linspace(0, 1, 6), include_lowest=True)
    bins = z.groupby("probability_bin", observed=False).agg(n=("label", "size"), probability=("prob", "mean"), frequency=("label", "mean")).reset_index()
    bins.to_csv(OUT / "probability_reliability.csv", index=False)
    rng = np.random.default_rng(42)
    bootstrap = []
    for period, months in [("AprJun", [4, 5, 6]), ("July", [7])]:
        selected = q[q.month.isin(months) & q.model.eq(name)]
        week = q[q.month.isin(months) & q.model.eq("week")]
        assert selected.origin.reset_index(drop=True).equals(week.origin.reset_index(drop=True))
        rows = []
        for date, group in selected.groupby("date"):
            a, bb = scores(group), scores(week[week.date.eq(date)])
            rows.append([a[k] for k in ["TP", "FP", "FN", "TN"]] + [bb[k] for k in ["TP", "FP", "FN", "TN"]])
        v = np.array(rows)
        starts = rng.integers(0, len(v), (2000, int(np.ceil(len(v)/7))))
        indices = ((starts[:, :, None] + np.arange(7)) % len(v)).reshape(2000, -1)[:, :len(v)]
        totals = v[indices].sum(axis=1)
        for metric, a, aa, bb, bbb in [("recall", 0, 2, 4, 6), ("FPR", 1, 3, 5, 7)]:
            difference = totals[:, a]/np.maximum(totals[:, a]+totals[:, aa], 1) - totals[:, bb]/np.maximum(totals[:, bb]+totals[:, bbb], 1)
            lo, hi = np.quantile(difference, [.025, .975])
            bootstrap.append({"period": period, "metric": metric, "comparator": "week", "CI_low": lo, "CI_high": hi, "days": len(v)})
    pd.DataFrame(bootstrap).to_csv(OUT / "week_block_uncertainty.csv", index=False)
    figures = ROOT / "figures/onset_v2"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 11, "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([0,1], [0,1], "--", color="#888888", label="예측확률 = 관측비율")
    ax.plot(bins.probability, bins.frequency, "o-", color="#246f9c", label="고부하 진입 모델")
    for _, r in bins.dropna().iterrows():
        ax.annotate(f"n={int(r['n'])}", (r.probability, r.frequency), xytext=(3, 6), textcoords="offset points", fontsize=9)
    ax.set(xlim=(-.03,1.06), ylim=(-.03,1.04), xlabel="예측된 진입 확률", ylabel="실제로 진입한 비율", title="고부하 진입 확률 · 7월 재사용 진단")
    ax.grid(alpha=.2); ax.legend(loc="upper left"); fig.tight_layout()
    fig.savefig(figures/"probability_reliability.svg"); fig.savefig(figures/"probability_reliability.png", dpi=140); plt.close(fig)
    cases = []
    for title, mask in [("탐지 성공", z.label & z.warn), ("오경보", ~z.label & z.warn), ("미탐지", z.label & ~z.warn)]:
        if mask.any():
            cases.append((title, z[mask].sort_values("origin").iloc[0]))
    fig, axes = plt.subplots(len(cases), 1, figsize=(10, 3.2*len(cases)), squeeze=False)
    for ax, (title, r) in zip(axes.ravel(), cases):
        t = r.origin
        load = y[(y.index>=t-pd.Timedelta(hours=2)) & (y.index<=t+pd.Timedelta(hours=2))]
        ax.plot(load.index, load, color="#245e8d", label="실제 전력")
        ax.axhline(r.threshold, color="#b96522", linestyle="--", label="자료상 고부하 기준")
        ax.axvline(t, color="#555555", linestyle=":", label="예측 발행시점")
        ax.axvspan(t, t+pd.Timedelta(hours=2), alpha=.1, color="#29946d", label="향후 2시간")
        ax.set_title(f"{title} · {t:%m/%d %H:%M} · 확률 {r.prob:.2f}", loc="left", fontsize=12)
        ax.set_ylabel("전력 (원자료 단위)"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")); ax.xaxis.set_major_locator(mdates.HourLocator()); ax.grid(alpha=.15)
    axes[-1,0].set_xlabel("단일 시간축 · 15분 구간 끝시각")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.subplots_adjust(hspace=.45, left=.1, right=.97, top=.94, bottom=.12)
    fig.savefig(figures/"entry_cases.svg"); fig.savefig(figures/"entry_cases.png", dpi=140); plt.close(fig)
    (figures/"README.md").write_text("수치 검증과 그림 생성 완료. 한글 폰트 지정. 직접 이미지 렌더 검토는 별도 수행해야 하며 완료로 간주하지 않는다.\n")
    t = p[p.model.isin([name,"week"])][["period","model","n","positives","TP","FP","FN","recall","precision","FPR","AP","Brier"]].copy()
    for c in ["recall","precision","FPR"]:
        t[c] *= 100
    doc = "# 고부하 진입 v2 결과 — 준비시간 조건 없음\n\n"
    doc += "현재가 고부하가 아닐 때 향후15~120분 내 처음 고부하가 되는지 예측한다. 15·30분 뒤 진입도 양성이다. 고부하는 월 시작 이전 전력q95 초과로 정의하며 설비 위험·계약기준이라는 의미는 없다.\n\n"
    doc += f"4~6월 AP 기반 선택: {name}. 7월은 이미 관찰한 재사용 진단이며 독립 테스트가 아니다. 준비시간이 있었던 v1과 정답이 달라 성능차를 모델 개선으로 해석하지 않는다.\n\n"
    doc += "## 정시 발행 단위 평가\n\n재현율·정밀도·FPR은 % 단위.\n\n" + table(t) + "\n\n"
    doc += f"7월 선택모델은 {int(j.TP)}회 탐지, {int(j.FN)}회 미탐지, {int(j.FP)}회 오경보. 일평균경보 {d.alarms.mean():.2f}회, 오경보 {(d.false_alarms.mean()):.2f}회({len(d)}일). 재현율만으로 우위를 판단하지 않는다.\n\n"
    doc += "## 실제 사건과 적용범위\n\n"
    doc += f"7월 실제 정상→고부하 전환 {int(e.total_entries)}건 중 현재정상인 정시의 2시간 첫진입으로 평가가능한 사건 {int(e.evaluable_entries)}건({100*e.coverage:.1f}%). 그중 {int(e.detected_entries)}건 탐지/{int(e.missed_entries)}건 누락, 사건별재현율 {100*e.episode_recall:.1f}%. 전체전환 중 탐지비중 {100*e.detected_entries/e.total_entries:.1f}%. 나머지 사건도 전체성과에서 숨기지 않는다.\n\n"
    doc += "첫초과 구간 끝시각과 발행시각의 차이를 기록했다. 구간 시작시각 차이는 설명용으로만 남기며 설비 준비시간이나 성공조건으로 사용하지 않는다. 반복 임계값교차·정시발행·이미높은현재값·관측불가 때문에 실제 사건 적용범위는 제한될 수 있다.\n\n"
    doc += "## 검증과 한계\n\n원본해시/타깃15~120분/후보동일행/저장모델재로드/단일시점특징일치/미래교란검증 통과. 원본결측·시간오류제외 유지. 확률보정과 임계값보정은 이전28일의 서로 다른14일에 수행했고 재학습하지 않았다. 과거 정상오경보5%는 미래 보장이 아니다. 신뢰도표·7일블록 bootstrap·조건별오류는 진단자료이며 반복데이터/자료재사용의 편향을 제거하지 않는다.\n\n"
    doc += "전력값 회귀모델은 별도 유지한다. 실제 설비 이동/생산보존/원화절감을 검증했다고 표현하지 않는다. 메타정보·원자료 생성과정은 미확인. 그래프는 각 성공/오경보/미탐의 가장 이른7월사례를 사용했고 이미지 직접검토는 아직 미완료.\n\n"
    doc += "## 재현\n\npython onset_v2.py → python onset_v2_report.py. 추론: python onset_v2.py --at \"2021-07-01 07:00\". requirements-intraday.txt. selected_model.json의 모델text/특징/확률보정/경보임계값을 함께 사용한다.\n"
    (ROOT/"docs/ONSET_V2_RESULTS.md").write_text(doc)
    note = f"## 최신 실행 완료: 준비시간 없는 고부하 진입 v2\n\n선택 {name}; 7월재사용 진단 재현율{100*j.recall:.1f}%, FPR{100*j.FPR:.2f}%, 정밀도{100*j.precision:.1f}%. 사건 {int(e.detected_entries)}/{int(e.evaluable_entries)}탐지, 전체진입{int(e.total_entries)}. 상세 docs/ONSET_V2_RESULTS.md/outputs/onset_v2/verification.json. 15·30분진입은양성이고준비시간필터없음. 이미지직접검토미완료. 다음은결과검토·그림렌더확인·실패조건보강이며기존v1성능을새타깃에인용하지않음.\n\n"
    for filename in ["PROJECT_STATUS.md","HANDOFF.md"]:
        path = ROOT/filename
        path.write_text(note+path.read_text())
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    run()
