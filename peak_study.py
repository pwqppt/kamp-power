"""Bounded peak study: origin-safe forecasts, chronological decisions, full failures."""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from experiment import ROOT, DATA, OUT, build_features, metrics, model_for
from context_features import build, split, classifier
from residual_experiment import baseline
from continue_research import confusion

D = OUT / 'peak_v3'
W = ROOT / 'work' / 'peak_v3'
FOLDS = [('2021-04-01', '2021-05-01'), ('2021-05-01', '2021-06-01'),
         ('2021-06-01', '2021-07-01')]
JULY = ('2021-07-01', '2021-08-01')


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load():
    return pd.read_pickle(W / 'features.pkl'), json.loads((W / 'columns.json').read_text())


def forecast(tr, v, cols, kind='reference'):
    """Future actuals are absent from the inference frame; state labels are training only."""
    state = tr.actual_production.gt(0).astype(int)
    gate = classifier().fit(tr[cols], state)
    inputs = v[cols].copy()
    assert not any(c.startswith('actual_') for c in inputs)
    prob = gate.predict_proba(inputs)[:, 1]
    q95 = float(tr.y.quantile(.95))
    predictions = []
    for label in [0, 1]:
        mask = state == label
        y = tr.loc[mask, 'y']
        m = model_for('LGB_medium')
        fit_args = {}
        offset = 0.
        if kind == 'H1':
            fit_args['sample_weight'] = np.where(y >= q95, 2., 1.)
        elif kind == 'H2':
            def asymmetric(y_true, y_pred):
                weight = np.where((y_true >= q95) & (y_pred < y_true), 2., 1.)
                return weight * (y_pred - y_true), weight
            m.set_params(objective=asymmetric)
            offset = float(y.mean())
            fit_args['init_score'] = np.full(len(y), offset)
        m.fit(tr.loc[mask, cols], y, **fit_args)
        predictions.append(m.predict(inputs) + offset)
    mixed = np.maximum((1 - prob) * predictions[0] + prob * predictions[1], 0)
    return np.where(prob >= .5, mixed, baseline(v)), prob


def frame(v, p, prob, tr, name, start):
    f = v[['date', 'origin', 'interval_start', 'interval_end', 'hour', 'y',
           'actual_temp', 'actual_production', 'actual_personnel', 'pattern']].copy()
    f['pred'] = p
    f['prob'] = prob
    f['naive'] = baseline(v)
    f['threshold'] = float(tr.y.quantile(.95))
    f['risk_threshold'] = float(tr.y.quantile(.90))
    f['fold'] = start[:7]
    f['candidate'] = name
    f['latest_training_label'] = tr.interval_end.max()
    assert (f.latest_training_label < f.origin).all()
    assert f.groupby('date').size().eq(96).all()
    assert np.isfinite(f[['y', 'pred', 'naive']]).all().all()
    return f


def read_forecasts(name):
    return pd.read_csv(D / name / 'predictions.csv.gz', parse_dates=[
        'date', 'origin', 'interval_start', 'interval_end', 'latest_training_label'])


def scores(f):
    result = metrics(f)
    day_mean = f.groupby('date')[['y', 'pred']].mean()
    result['daily_mean_mae'] = float((day_mean.y-day_mean.pred).abs().mean())
    result.update({'interval_' + k: v for k, v in confusion(f.y, f.pred, f.threshold).items()})
    d = f.groupby('date')[['y', 'pred', 'threshold']].max()
    result.update({'day_' + k: v for k, v in confusion(d.y, d.pred, d.threshold).items()})
    return result


def prepare():
    D.mkdir(exist_ok=True)
    W.mkdir(parents=True, exist_ok=True)
    if not (DATA / 'features.pkl').exists():
        build_features()
    x, groups = build()
    # Do not evaluate the already-used August/September test in this study.
    x = x[x.date < '2021-08-01'].copy()
    x.to_pickle(W / 'features.pkl')
    dump(W / 'columns.json', groups['B1'])
    old = pd.read_csv(OUT / 'adaptive_v2' / 'selected_july.csv', parse_dates=['date', 'interval_start'])
    tr, v = split(x, *JULY)
    f = v.merge(old[['interval_start', 'pred', 'positive_record_probability']], on='interval_start', validate='one_to_one')
    f = frame(f, f.pred.to_numpy(), f.positive_record_probability.to_numpy(), tr, 'reference', JULY[0])
    diagnose(f, tr, D / 'diagnosis')
    dump(D / 'environment.json', {
        'source_sha256': hashlib.sha256((DATA / 'original.csv').read_bytes()).hexdigest(),
        'columns': groups['B1'], 'test_evaluated': False,
        'origin': 'D-1 16:00; measurements available at interval end; zero publication delay assumed'})
    print('Prepared July diagnosis and origin-only B1 features.', flush=True)


def diagnose(f, tr, dest):
    dest.mkdir(parents=True, exist_ok=True)
    f = f.copy()
    cuts = np.unique(tr.actual_temp.dropna().quantile([1/3, 2/3]).to_numpy())
    f['temperature'] = pd.cut(f.actual_temp, [-np.inf, *cuts, np.inf]).astype(str)
    f['temperature_5C'] = pd.cut(f.actual_temp, [-np.inf, 20, 25, 30, np.inf],
        labels=['<=20', '20-25', '25-30', '>30']).astype(str)
    f['production'] = np.select([f.actual_production.isna(), f.actual_production.eq(0)],
                                ['missing', 'zero'], default='positive')
    f['period'] = pd.cut(f.hour, [-1, 6, 9, 16, 21, 23],
                         labels=['00-07', '07-10', '10-17', '17-22', '22-24']).astype(str)
    f['temperature_production_period'] = f.temperature_5C + '/' + f.production + '/' + f.period
    f['production_period'] = f.production + '/' + f.period
    rows = []
    for dim in ['temperature', 'temperature_5C', 'production', 'period', 'production_period', 'temperature_production_period']:
        for condition, g in f.groupby(dim, observed=True):
            actual_peak = g.y >= g.threshold
            for model in ['pred', 'naive']:
                error = g[model] - g.y
                rows.append(dict(dimension=dim, condition=condition, model=model, n=len(g), days=g.date.nunique(),
                    mae=error.abs().mean(), bias=error.mean(), underprediction=np.maximum(-error, 0).mean(),
                    peak_intervals=int(actual_peak.sum()), peak_rate=actual_peak.mean(),
                    peak_underprediction=np.maximum(-error[actual_peak], 0).mean(),
                    **{k: v for k, v in confusion(g.y, g[model], g.threshold).items() if k != 'peak_rate'}))
    pd.DataFrame(rows).to_csv(dest / 'conditions.csv', index=False)
    daily = []
    for date, g in f.groupby('date'):
        a = g.loc[g.y.idxmax()]
        for model in ['pred', 'naive']:
            p = g.loc[g[model].idxmax()]
            daily.append(dict(date=str(date.date()), model=model, actual_peak=a.y, predicted_peak=p[model],
                peak_size_error=p[model]-a.y, error_at_actual_peak=a[model]-a.y,
                timing_error_minutes=abs((p.interval_start-a.interval_start).total_seconds())/60,
                actual_peak_time=str(a.interval_start), predicted_peak_time=str(p.interval_start),
                actual_temperature=a.actual_temp, actual_production=a.actual_production,
                temperature=a.temperature, production=a.production, period=a.period,
                true_peak_day=bool(a.y>=a.threshold), detected=bool(p[model]>=a.threshold)))
    pd.DataFrame(daily).to_csv(dest / 'daily_peaks.csv', index=False)
    dump(dest / 'spec.json', {'temperature_cuts_training_only': cuts.tolist(),
        'threshold': float(f.threshold.iloc[0]), 'date_count': int(f.date.nunique()),
        'interpretation': 'Actual temperature/production are ex-post labels only, never forecast inputs. Associations are not cooling or production causal effects. First maximum breaks timing ties.',
        'excluded_dates': ['2021-07-13', '2021-07-15']})


def save_forecasts(f, dest):
    dest.mkdir(parents=True, exist_ok=True)
    f.to_csv(dest / 'predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    pd.DataFrame([dict(fold=n, **scores(g)) for n, g in f.groupby('fold')]).to_csv(dest / 'fold_scores.csv', index=False)
    pd.DataFrame([dict(period=n, **scores(g)) for n, g in [
        ('Apr-Jun', f[(f.date >= '2021-04-01') & (f.date < '2021-07-01')]),
        ('July', f[f.date >= '2021-07-01'])]]).to_csv(dest / 'scores.csv', index=False)


def references():
    x, cols = load()
    frames = []
    for start, end in [('2021-03-01', '2021-04-01'), *FOLDS, JULY]:
        tr, v = split(x, start, end)
        p, prob = forecast(tr, v, cols)
        f = frame(v, p, prob, tr, 'reference', start)
        frames.append(f)
        print('reference', start, scores(f)['mae'], flush=True)
    f = pd.concat(frames, ignore_index=True)
    old = pd.read_csv(OUT / 'adaptive_v2' / 'fold_scores.csv')
    drift = []
    for fold, g in f[f.date.between('2021-04-01', '2021-06-30')].groupby('fold'):
        expected = old[(old.candidate=='predicted_state_0.5') & (old.fold==fold+'-01')].iloc[0]
        for key in ['mae', 'daily_peak_mae', 'peak_day_recall']:
            now = scores(g)[key]
            drift.append(dict(fold=fold, metric=key, historical=float(expected[key]), current=now, delta=now-float(expected[key])))
    old_july = pd.read_csv(OUT / 'adaptive_v2' / 'selected_july.csv')
    # Cross-platform historical mismatch must be disclosed, never silently tolerated.
    from context_features import predict
    tr, v = split(x, *JULY)
    original_p, prob = predict(tr, v, cols, 'mixture')
    original_p = np.where(prob>=.5, original_p, baseline(v))
    assert np.array_equal(original_p, f[f.fold=='2021-07'].pred.to_numpy())
    delta_july = float(np.max(np.abs(f[f.fold=='2021-07'].pred.to_numpy()-old_july.pred.to_numpy())))
    save_forecasts(f, D / 'reference')
    naive = f.copy(); naive['pred'] = naive.naive; naive['candidate'] = 'naive'
    save_forecasts(naive, D / 'naive')
    pd.DataFrame(drift).to_csv(D / 'historical_drift.csv', index=False)
    dump(D / 'reproduction.json', {'historical_exact_reproduction': delta_july<=1e-7,
         'historical_July_prediction_max_abs_difference': delta_july,
         'current_original_function_vs_wrapper_exact_match': True,
         'environment': {'python': platform.python_version(), 'platform': platform.platform(),
             'packages': {n: version(n) for n in ['numpy', 'scipy', 'pandas', 'lightgbm', 'scikit-learn', 'holidays']}},
         'policy': 'Same current-environment original-model reference for all candidates. Preserve historical outputs. Difference not attributed to a specific cause.',
         'August_September_evaluated': False})


def gate_decision(candidate, reference):
    a, b = scores(candidate), scores(reference)
    checks = dict(mean_mae_preserved=a['mae']<=b['mae']+1e-9,
        daily_peak_mae_improved=a['daily_peak_mae']<b['daily_peak_mae']-1e-9,
        peak_day_recall_improved=a['peak_day_recall']>b['peak_day_recall']+1e-9,
        interval_recall_not_worse=a['interval_recall']>=b['interval_recall']-1e-9)
    return dict(eligible=all(checks.values()), checks=checks,
        failed_checks=[k for k, v in checks.items() if not v],
        candidate_validation=a, reference_validation=b,
        selection_data='Apr-Jun only; exploratory reused history, not independent confirmation')


def bootstrap(candidate, reference):
    joined = candidate[['interval_start', 'date', 'y', 'pred']].merge(
        reference[['interval_start', 'pred']], on='interval_start', suffixes=('', '_ref'), validate='one_to_one')
    joined['error_delta'] = (joined.y-joined.pred).abs() - (joined.y-joined.pred_ref).abs()
    days = joined.groupby('date')
    daily_max = days[['y', 'pred', 'pred_ref']].max()
    deltas = pd.DataFrame({'mae': days.error_delta.mean(),
        'daily_peak_mae': (daily_max.y-daily_max.pred).abs()-(daily_max.y-daily_max.pred_ref).abs()})
    rng = np.random.default_rng(42)
    draws = deltas.to_numpy()[rng.integers(0, len(deltas), size=(2000, len(deltas)))].mean(axis=1)
    return {c: {'delta': float(deltas[c].mean()), 'low': float(np.quantile(draws[:, i], .025)),
                'high': float(np.quantile(draws[:, i], .975))} for i, c in enumerate(deltas)}


def calibration(tr, v, ref):
    prior = ref[ref.interval_end <= v.origin.min()].copy()
    counts = prior.groupby('date').size()
    prior = prior[prior.date.isin(counts[counts==96].index)]
    assert prior.interval_end.max() < v.origin.min()
    risk = prior[prior.pred >= prior.risk_threshold]
    raw = float((risk.y-risk.pred).mean()) if len(risk) else 0.
    cap = float(tr.y.quantile(.95)) * .1
    delta = float(np.clip(.5*raw, 0, cap)) if len(risk)>=96 and risk.date.nunique()>=7 else 0.
    return delta, dict(risk_n=len(risk), risk_days=risk.date.nunique(),
        latest_calibration_label=str(prior.interval_end.max()), origin=str(v.origin.min()),
        raw_mean_residual=raw, correction=delta, cap=cap)


def hypothesis(name):
    x, cols = load()
    ref = read_forecasts('reference')
    dest = D / name; dest.mkdir(exist_ok=True)
    parts = []; calibrations = []
    def one(start, end):
        tr, v = split(x, start, end)
        if name=='H3':
            r = ref[ref.fold==start[:7]]
            assert np.array_equal(v.interval_start.to_numpy(), r.interval_start.to_numpy())
            delta, record = calibration(tr, v, ref)
            calibrations.append(dict(fold=start[:7], **record))
            p = r.pred.to_numpy() + (r.pred.to_numpy()>=float(tr.y.quantile(.90))) * delta
            prob = r.prob.to_numpy()
        else:
            p, prob = forecast(tr, v, cols, name)
        f = frame(v, p, prob, tr, name, start)
        print(name, start, scores(f)['mae'], scores(f)['daily_peak_mae'], flush=True)
        return f
    for start, end in FOLDS:
        parts.append(one(start, end))
    val = pd.concat(parts, ignore_index=True)
    r = ref[(ref.date>='2021-04-01') & (ref.date<'2021-07-01')]
    decision = gate_decision(val, r)
    # This file is written before any candidate July forecast is calculated.
    dump(dest / 'validation_decision.json', decision)
    val.to_csv(dest / 'validation_predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    parts.append(one(*JULY))
    f = pd.concat(parts, ignore_index=True)
    save_forecasts(f, dest)
    if calibrations:
        dump(dest / 'calibration.json', calibrations)
    dump(dest / 'bootstrap.json', {'Apr-Jun': bootstrap(val, r),
        'July': bootstrap(parts[-1], ref[ref.fold=='2021-07'])})
    write_hypothesis_report(name, decision, f)


def write_hypothesis_report(name, decision, f):
    a = decision['candidate_validation']; b = decision['reference_validation']
    july = scores(f[f.fold=='2021-07'])
    result = '탐색 채택 조건 통과' if decision['eligible'] else '미채택'
    next_step = {'H1': 'H2: 고정 비대칭 손실', 'H2': 'H3: 과거 OOF 잔차 보정',
                 'H3': '평균전력·오류조건·피크조건 정리 후 고정 제약 피크 저감 시뮬레이션'}[name]
    report = f'''# {name} 결과: {result}

설정과 채택 조건: [PEAK_V3_PROTOCOL.md](PEAK_V3_PROTOCOL.md). 설정 재탐색 없음.

| 지표 | 4~6월 기존 결합 | 4~6월 {name} | 7월 {name} 사후 평가 |
|---|---:|---:|---:|
| 평균 MAE | {b['mae']:.6f} | {a['mae']:.6f} | {july['mae']:.6f} |
| 일최대 MAE | {b['daily_peak_mae']:.6f} | {a['daily_peak_mae']:.6f} | {july['daily_peak_mae']:.6f} |
| 피크일 재현율 | {b['peak_day_recall']:.6f} | {a['peak_day_recall']:.6f} | {july['peak_day_recall']:.6f} |
| 구간 피크 재현율 | {b['interval_recall']:.6f} | {a['interval_recall']:.6f} | {july['interval_recall']:.6f} |

미충족 조건: {', '.join(decision['failed_checks']) or '없음'}.
월별 악화와 모든 오경보·미탐 수는 `outputs/peak_v3/{name}/fold_scores.csv`에 보존했다.
전체 예측과 불확실성 구간은 같은 폴더의 `predictions.csv.gz`, `bootstrap.json`에 있다.
7월 결과로 판정을 바꾸지 않았다. 반복 사용된 과거 데이터이므로 독립 검증이나 배포 승인이 아니다.

다음 작업: {next_step}. 실행: `.venv/Scripts/python.exe peak_study.py { {'H1':'H2','H2':'H3','H3':'summarize'}[name] }`.
'''
    (ROOT / 'docs' / f'PEAK_V3_{name}_RESULTS.md').write_text(report, encoding='utf-8')
    update_status(f'{name}: {result}. 4~6월 MAE {a["mae"]:.6f}, 일최대 MAE {a["daily_peak_mae"]:.6f}, 피크일 재현율 {a["peak_day_recall"]:.2%}. 미충족: {", ".join(decision["failed_checks"]) or "없음"}. 다음: {next_step}.')


def update_status(message):
    for file in ['HANDOFF.md', 'PROJECT_STATUS.md']:
        path = ROOT / file
        old = path.read_text(encoding='utf-8')
        title, rest = old.split('\n', 1)
        path.write_text(title+'\n\n## 피크 개선 v3 최신 체크포인트 — 2026-10-02\n\n'+message+
            '\n\n프로토콜: docs/PEAK_V3_PROTOCOL.md. 경제성보다 연구 본체 우선. 아래 과거 다음단계보다 이 기록이 우선한다.\n'+rest, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'reference', 'H1', 'H2', 'H3'])
    args = parser.parse_args()
    if args.stage=='prepare': prepare()
    elif args.stage=='reference': references()
    else: hypothesis(args.stage)
