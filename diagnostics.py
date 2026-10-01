"""Held-out diagnostics and data-treatment sensitivity; no model retuning on test."""
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier,export_text
from sklearn.metrics import balanced_accuracy_score
from sklearn.inspection import permutation_importance
import joblib
from experiment import ROOT,DATA,OUT,model_for,metrics,FEATURE_SETS,predict_fold

def run():
    x=pd.read_pickle(DATA/'features.pkl')
    spec=json.loads((OUT/'frozen_selection.json').read_text())
    FEATURE_SETS.update(json.loads((OUT/'feature_sets.json').read_text()))
    bundle=joblib.load(OUT/'final_model.joblib');cols=bundle['features'];model=bundle['model']
    test=pd.read_csv(OUT/'test_predictions.csv',parse_dates=['date','origin','interval_start','interval_end'])
    cal=pd.read_csv(OUT/'calibration_predictions.csv',parse_dates=['date','origin','interval_start','interval_end'])
    xx=x[x.date>='2021-08-01'].copy()
    # The main model is fixed; tests here are sensitivity diagnostics, not selection.
    cutoff=pd.Timestamp('2021-07-31 16:00')
    train=x[x.interval_end<=cutoff].copy(); counts=train.groupby('date').size()
    train=train[train.date.isin(counts[counts==96].index)]
    rows=[]
    for label in ['native_missing','median_imputed','zero_imputed','drop_missing_train','winsorize_features_01_99','clip_labels_99_NEGATIVE_CONTROL','deduplicate_label_days']:
        tr=train.copy();a=tr[cols].copy();b=xx[cols].copy();y=tr.y.copy()
        if label in ['median_imputed','zero_imputed']:
            imp=SimpleImputer(strategy='median' if label=='median_imputed' else 'constant',fill_value=0,add_indicator=True)
            a=imp.fit_transform(a);b=imp.transform(b)
        elif label=='drop_missing_train':
            keep=a.notna().all(axis=1);a=a.loc[keep];y=y.loc[keep]
        elif label=='winsorize_features_01_99':
            ql=a.quantile(.01);qh=a.quantile(.99);a=a.clip(ql,qh,axis=1);b=b.clip(ql,qh,axis=1)
        elif label=='clip_labels_99_NEGATIVE_CONTROL':y=y.clip(upper=y.quantile(.99))
        elif label=='deduplicate_label_days':
            first=tr[['date','pattern']].drop_duplicates().drop_duplicates('pattern').date
            keep=tr.date.isin(first);a=a.loc[keep];y=y.loc[keep]
        m=model_for(spec['model']);m.fit(a,y)
        p=test.copy();p['pred']=np.maximum(m.predict(b),0)
        rows.append(dict(treatment=label,train_rows=len(y),**metrics(p)))
    pd.DataFrame(rows).to_csv(OUT/'treatment_sensitivity.csv',index=False)
    # Additional historical business features contain the actual raw missingness.
    auxrows=[];auxcols=FEATURE_SETS['F4']
    for method in ['native_missing','median_imputed','zero_imputed']:
        a=train[auxcols];b=xx[auxcols]
        if method!='native_missing':
            im=SimpleImputer(strategy='median' if method=='median_imputed' else 'constant',fill_value=0,add_indicator=True)
            a=im.fit_transform(a);b=im.transform(b)
        m=model_for('LGB_small');m.fit(a,train.y);p=test.copy();p['pred']=np.maximum(m.predict(b),0)
        auxrows.append(dict(treatment=method,features='F4',**metrics(p)))
    pd.DataFrame(auxrows).to_csv(OUT/'raw_missingness_sensitivity.csv',index=False)
    # Missingness stress test for recent power and lag channels; model is not refit.
    rng=np.random.default_rng(42);stress=[]
    for frac in [0,.01,.05,.10]:
        b=xx[cols].copy();lagcols=[c for c in cols if c.startswith('power_lag') or c.startswith('recent') or c=='last_power']
        for c in lagcols:b.loc[rng.random(len(b))<frac,c]=np.nan
        p=test.copy();p['pred']=np.maximum(model.predict(b),0)
        stress.append(dict(missing_fraction=frac,**metrics(p)))
    pd.DataFrame(stress).to_csv(OUT/'missingness_stress.csv',index=False)
    # Error segmentation: target-day actual production/weather permitted for EX POST analysis only.
    q=train.actual_production.quantile([.33,.67]).values
    test['production_band']=pd.cut(test.actual_production,[-np.inf,*np.unique(q),np.inf],duplicates='drop').astype(str)
    test['temp_band']=pd.cut(test.actual_temp,[-np.inf,0,15,25,np.inf]).astype(str)
    test['period']=pd.cut(test.hour,[-1,6,9,16,21,23],labels=['00–07시','07–10시','10–17시','17–22시','22–24시']).astype(str)
    test['month']=test.date.dt.strftime('%Y-%m')
    errtables=[]
    for dim in ['period','month','dow','holiday','production_band','temp_band']:
        for value,g in test.groupby(dim,observed=True):
            errtables.append(dict(dimension=dim,condition=str(value),n=len(g),days=g.date.nunique(),mae=float(g.abs_error.mean()),
                bias=float((g.pred-g.y).mean()),peak_rate=float((g.y>=g.threshold).mean()),
                coverage90=float(((g.y>=g.lower90)&(g.y<=g.upper90)).mean())))
    pd.DataFrame(errtables).to_csv(OUT/'condition_errors.csv',index=False)
    # Explanation rules fit on pretest OOF data; report held-out leaf counts, never causal.
    validation=pd.read_csv(OUT/'validation_predictions.csv.gz',parse_dates=['date'])
    validation=validation[(validation.model==spec['model'])&(validation.features==spec['features'])]
    pre=pd.concat([validation,cal],ignore_index=True)
    explain_cols=['hour','dow','holiday','actual_production','actual_temp']
    imp=SimpleImputer();a=imp.fit_transform(pre[explain_cols]);b=imp.transform(test[explain_cols])
    rules=[]
    for task,target in [('peak',(pre.y>=pre.threshold).astype(int)),('large_error',((pre.y-pre.pred).abs()>=30).astype(int))]:
        tree=DecisionTreeClassifier(max_depth=3,min_samples_leaf=300,random_state=42,class_weight='balanced')
        tree.fit(a,target)
        text=export_text(tree,feature_names=explain_cols)
        (OUT/f'{task}_rules.txt').write_text(text,encoding='utf8')
        leaf=tree.apply(b);actual=(test.y>=test.threshold).astype(int) if task=='peak' else (test.abs_error>=30).astype(int)
        for l in np.unique(leaf):
            mask=leaf==l
            rules.append(dict(task=task,leaf=int(l),n=int(mask.sum()),days=int(test.loc[mask,'date'].nunique()),rate=float(actual.loc[mask].mean())))
    pd.DataFrame(rules).to_csv(OUT/'heldout_rule_rates.csv',index=False)
    # Permutation: shuffle whole daily feature blocks to respect within-day dependence.
    base=np.abs(test.y-test.pred).mean();rng=np.random.default_rng(17);pr=[]
    dayids=xx.date.unique();n_days=len(dayids)
    for c in cols:
        deltas=[]
        for rep in range(3):
            b=xx[cols].copy();v=b[c].to_numpy().reshape(n_days,96)
            b[c]=v[rng.permutation(n_days)].ravel()
            deltas.append(float(np.abs(model.predict(b)-xx.y).mean()-base))
        pr.append(dict(feature=c,delta_mae=np.mean(deltas),std=np.std(deltas)))
    pd.DataFrame(pr).sort_values('delta_mae',ascending=False).to_csv(OUT/'block_permutation_importance.csv',index=False)
    # LightGBM native TreeSHAP (exact contribution sums checked), avoids extra SHAP runtime.
    sample=xx[cols].iloc[::4]
    if hasattr(model,'booster_'):
        contrib=model.booster_.predict(sample,pred_contrib=True)
        assert np.allclose(contrib.sum(axis=1),model.predict(sample),atol=1e-6)
        pd.DataFrame({'feature':cols,'mean_abs_shap':np.abs(contrib[:,:-1]).mean(axis=0)}).sort_values('mean_abs_shap',ascending=False).to_csv(OUT/'shap_importance.csv',index=False)
    # Paired day-block bootstrap for ML-vs-naive MAE difference.
    daily=test.assign(naive_error=(test.y-test.naive).abs()).groupby('date')[['abs_error','naive_error']].mean()
    dif=(daily.abs_error-daily.naive_error).values;rng=np.random.default_rng(42)
    means=rng.choice(dif,size=(3000,len(dif)),replace=True).mean(axis=1)
    result=dict(mae_difference_ML_minus_naive=float(dif.mean()),day_bootstrap_95=[float(np.quantile(means,.025)),float(np.quantile(means,.975))],
       interpretation='Day-block resampling; repeated patterns/serial dependence limit inference; no causal claim.')
    (OUT/'bootstrap.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    test.to_csv(OUT/'test_predictions_diagnostics.csv',index=False)
    print('DIAGNOSTICS DONE',json.dumps(result),flush=True)

if __name__=='__main__':run()
