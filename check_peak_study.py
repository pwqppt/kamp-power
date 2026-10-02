"""Meaningful origin perturbation, chronological calibration, and adoption checks."""
from unittest.mock import patch
import json
import numpy as np
import pandas as pd
import experiment
from context_features import context
from peak_study import DATA, OUT, D, W, load, split, read_forecasts, calibration, gate_decision, dump, peak_gradient


def future_source_check():
    x, cols = load()
    hourly = pd.read_csv(DATA/'hourly_audited.csv', parse_dates=['date', 'hour_start'])
    long = pd.read_csv(DATA/'quarter_hour.csv', parse_dates=['date', 'interval_start', 'interval_end'])
    audit = json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
    bad = pd.to_datetime(audit['repaired_dates'])
    checked = []
    # Generate only the target date while keeping the complete historical source series.
    for day in ['2021-06-15', '2021-07-20']:
        date = pd.Timestamp(day); g = x[x.date==date]; origin = g.origin.iloc[0]
        h = hourly.copy(); q = long.copy()
        h.loc[h.hour_start+pd.Timedelta(hours=1)>origin,
              ['생산량', '공장인원', '기온', '습도', '풍속', '강수량']] = 999999.
        q.loc[q.interval_end>origin,
              ['power', '생산량', '공장인원', '기온', '습도', '풍속', '강수량']] = 999999.

        class OneTargetDay(pd.DataFrame):
            @property
            def _constructor(self):
                return pd.DataFrame

            def groupby(self, by=None, *args, **kwargs):
                grouped = super().groupby(by, *args, **kwargs)
                return [(date, grouped.get_group(date))] if by=='date' else grouped

        scratch = W/'source_check'/day; scratch.mkdir(parents=True, exist_ok=True)
        with patch.object(experiment, 'audit_run', return_value=(h, OneTargetDay(q), audit)), \
             patch.object(experiment, 'DATA', scratch), patch.object(experiment, 'OUT', scratch):
            rebuilt, _ = experiment.build_features()
        s = q.set_index('interval_start').power.astype(float)
        s.loc[s.index.normalize().isin(bad)] = np.nan
        h.loc[h.hour_start.dt.normalize().isin(bad), ['생산량', '공장인원']] = np.nan
        added, _, _ = context(rebuilt, h, s)
        rebuilt = rebuilt.join(added)
        pd.testing.assert_frame_equal(g[cols].reset_index(drop=True), rebuilt[cols].reset_index(drop=True),
                                      check_dtype=False, atol=1e-12, rtol=1e-12)
        assert (rebuilt.actual_temp==999999).all()  # intervention actually reached future records
        checked.append({'day': day, 'origin': str(origin), 'changed_future_actuals': True,
                        'all_B1_inputs_unchanged': True, 'feature_count': len(cols)})
    return checked


def run():
    tests = {'source_perturbation': future_source_check()}
    y = np.array([10., 200., 200., 10.]); p = np.array([5., 150., 210., 15.])
    def loss(z):
        weight = np.where((y>=176.) & (z<y), 2., 1.)
        return .5*weight*(z-y)**2
    grad, hess = peak_gradient(y, p, 176.)
    eps = 1e-3
    np.testing.assert_allclose(grad, (loss(p+eps)-loss(p-eps))/(2*eps), atol=1e-7)
    np.testing.assert_allclose(hess, (loss(p+eps)-2*loss(p)+loss(p-eps))/eps**2, atol=1e-5)
    tests['asymmetric_gradient_hessian_finite_difference'] = True
    x, cols = load(); ref = read_forecasts('reference')
    calibration_checks = []
    for month in ['04', '05', '06', '07']:
        tr, v = split(x, f'2021-{month}-01', str(pd.Timestamp(f'2021-{month}-01')+pd.offsets.MonthBegin(1)))
        a, rec = calibration(tr, v, ref)
        corrupt = ref.copy()
        corrupt.loc[corrupt.interval_end>v.origin.min(), ['y','pred','risk_threshold']] = 999999.
        b, _ = calibration(tr, v, corrupt)
        assert a==b
        assert pd.Timestamp(rec['latest_calibration_label']) < v.origin.min()
        calibration_checks.append(rec)
    tests['future_calibration_labels_invariant'] = calibration_checks
    val = ref[ref.date.between('2021-04-01', '2021-06-30')].copy()
    assert not gate_decision(val, val)['eligible']
    bad = val.copy(); bad['pred'] = 999999.
    assert not gate_decision(bad, val)['eligible']
    for name in ['H1', 'H2', 'H3']:
        path = D/name/'predictions.csv.gz'
        if not path.exists(): continue
        f = read_forecasts(name)
        assert f.interval_start.is_unique
        assert f.groupby('date').size().eq(96).all()
        assert (f.latest_training_label<f.origin).all()
        assert np.isfinite(f[['y','pred']]).all().all()
        saved = json.loads((D/name/'validation_decision.json').read_text(encoding='utf-8'))
        current = gate_decision(f[f.fold<'2021-07'], val)
        assert current['checks']==saved['checks']
    tests['gate_rejects_no_improvement_and_large_errors'] = True
    tests['candidate_metrics_and_time_order_checked'] = [n for n in ['H1','H2','H3'] if (D/n/'predictions.csv.gz').exists()]
    dump(D/'checks.json', tests)
    print('PASS: future-source perturbation; future calibration labels; strict adoption; available candidate rows.')


if __name__=='__main__': run()
