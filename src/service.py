"""Пакетный прогноз и мониторинг: python -m src.service [file.csv]."""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from src.config import ROOT, FEATURES, TARGET
from src.data.validate import validate
from src.modeling import save, metrics, psi

def load_bundle(): return joblib.load(ROOT/'models/model_v1.joblib')

def score(df,bundle):
    clean,report=validate(df.drop(columns=[TARGET],errors='ignore'),bundle['parameters'],False)
    if report['status']!='ok' or clean.empty:
        raise ValueError(json.dumps(report,ensure_ascii=False))
    prob=bundle['pipeline'].predict_proba(clean[FEATURES])[:,1]
    out=pd.DataFrame({'row_id':clean.index,'probability':prob,'high_risk':(prob>=bundle['threshold']).astype(int),'model_version':bundle['version']})
    return clean,out,report

def process(path,bundle=None,outdir=None):
    start=time.perf_counter(); path=Path(path); bundle=bundle or load_bundle()
    outdir=Path(outdir or ROOT/'reports/batches'); outdir.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    model_hash=hashlib.sha256((ROOT/'models/model_v1.joblib').read_bytes()).hexdigest()
    key=hashlib.sha256((digest+model_hash).encode()).hexdigest()[:20]
    folder=outdir/key
    if (folder/'complete.json').exists():
        return json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    df=pd.read_csv(path)
    try: clean,pred,dq=score(df,bundle)
    except ValueError as e:
        save({'file':path.name,'error':str(e)},outdir/(key+'_rejected.json')); raise
    folder.mkdir(exist_ok=True)
    pred.to_csv(folder/'predictions.csv',index=False)
    p=pred.probability.to_numpy()
    result={'file':path.name,'key':key,'model_version':bundle['version'],'source_sha256':digest,'model_sha256':model_hash,'n':len(clean),'mean_loan':float(clean.loan_amnt.mean()),'high_risk_share':float(pred.high_risk.mean()),'expected_loss_proxy':float(np.sum(p*clean.loan_amnt.to_numpy()*.45)),'dq':dq,'psi_income':psi(bundle['reference'].person_income,clean.person_income)}
    seg=clean[['loan_grade','loan_intent']].copy(); seg['probability']=p
    for col in seg.columns[:2]:
        seg.groupby(col).probability.agg(['count','mean']).to_csv(folder/f'by_{col}.csv')
    # Метки используются только после получения прогнозов, как имитация зрелых исходов.
    if TARGET in df:
        initial=pd.read_csv(ROOT/'data/initial/initial.csv')
        known=set(pd.util.hash_pandas_object(initial[FEATURES],index=False))
        hashes=pd.util.hash_pandas_object(clean[FEATURES],index=False)
        mask=~hashes.isin(known)
        result['evaluation_excluded_overlap']=int((~mask).sum())
        if df.loc[clean.index[mask],TARGET].nunique()==2:
            result['quality']=metrics(df.loc[clean.index[mask],TARGET],p[mask],bundle['threshold'])
    result['alerts']=[]
    if result['psi_income']>.25: result['alerts'].append('PSI > 0.25')
    if result.get('quality',{}).get('roc_auc',1)<.8: result['alerts'].append('ROC-AUC < 0.80')
    if result.get('quality',{}).get('recall',1)<.75: result['alerts'].append('Recall < 0.75')
    result['seconds']=time.perf_counter()-start
    save(result,folder/'complete.json')
    return result

def main():
    paths=[Path(sys.argv[1])] if len(sys.argv)>1 else sorted((ROOT/'data/incoming').glob('batch_*.csv'))
    b=load_bundle(); results=[process(f,b) for f in paths]
    save(results,ROOT/'reports/batch_results.json')
    print([(r['file'],r['n'],r['quality']['roc_auc']) for r in results])

if __name__=='__main__': main()
