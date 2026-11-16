"""Полный эксперимент. Запуск: python -m src.modeling."""
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, brier_score_loss
from sklearn.inspection import permutation_importance
from src.config import ROOT, FEATURES, NUMERIC, CATEGORICAL, TARGET, load_params
from src.data.validate import validate
from src.features.build_features import add_features, make_preprocessor, NEW_NUMERIC, NEW_CATEGORICAL

P = load_params()
REP = ROOT / 'reports/model'
def save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x,'item') else str(x)), encoding='utf-8')

def make_model(estimator, extra=True):
    nums=NUMERIC + (NEW_NUMERIC if extra else [])
    cats=([c for c in CATEGORICAL if c not in ['loan_grade','cb_person_default_on_file']]+NEW_CATEGORICAL) if extra else CATEGORICAL
    steps=[]
    if extra: steps.append(('features',FunctionTransformer(add_features,kw_args={'params':P})))
    steps.extend([('preprocess',make_preprocessor(nums,cats)),('model',estimator)])
    return Pipeline(steps)

def metrics(y,p,t=.5):
    pred=np.asarray(p)>=t
    return {'roc_auc':float(roc_auc_score(y,p)), 'ap':float(average_precision_score(y,p)),
            'precision':float(precision_score(y,pred,zero_division=0)), 'recall':float(recall_score(y,pred)),
            'f1':float(f1_score(y,pred)), 'accuracy':float(accuracy_score(y,pred)),
            'brier':float(brier_score_loss(y,p)), 'confusion':confusion_matrix(y,pred).tolist()}

def psi(a,b):
    edges=np.unique(np.quantile(a,np.linspace(0,1,11)))
    edges[0],edges[-1]=-np.inf,np.inf
    x=np.histogram(a,edges)[0]+.5; z=np.histogram(b,edges)[0]+.5
    x=x/x.sum(); z=z/z.sum()
    return float(np.sum((x-z)*np.log(x/z)))

def main():
    warnings.filterwarnings('ignore',category=FutureWarning)
    REP.mkdir(parents=True,exist_ok=True)
    initial=pd.read_csv(ROOT/P['data']['initial_file'])
    clean,dq=validate(initial,P,True)
    # Одинаковые входы не должны одновременно оказаться в обучающей и тестовой выборках.
    clean=clean.drop_duplicates(subset=FEATURES)
    train,test=train_test_split(clean,test_size=.2,stratify=clean[TARGET],random_state=42)
    train.to_csv(REP/'train.csv',index=False); test.to_csv(REP/'test.csv',index=False)
    X,y=train[FEATURES],train[TARGET]
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    candidates=[('Dummy',DummyClassifier(strategy='prior'),True)]
    for penalty in ['l1','l2']:
        for c in [.1,1,10]:
            candidates.append((f'LR_{penalty}_C{c}',LogisticRegression(C=c,solver='liblinear',penalty=penalty,max_iter=2000,random_state=42),True))
    for depth in [3,5,8,12]: candidates.append((f'Tree_{depth}',DecisionTreeClassifier(max_depth=depth,min_samples_leaf=20,random_state=42),True))
    candidates += [('NB',GaussianNB(),True),('RF_8',RandomForestClassifier(n_estimators=120,max_depth=8,min_samples_leaf=5,n_jobs=2,random_state=42),True),('RF_None',RandomForestClassifier(n_estimators=120,min_samples_leaf=5,n_jobs=2,random_state=42),True)]
    for leaves in [15,31]:
        candidates.append((f'HGB_{leaves}',HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=leaves,learning_rate=.1,random_state=42),True))
    candidates.append(('LR_without_features',LogisticRegression(C=1,solver='liblinear',max_iter=2000,random_state=42),False))
    rows=[]; fitted={}
    for name,est,extra in candidates:
        start=time.perf_counter(); model=make_model(est,extra)
        result=cross_validate(model,X,y,cv=cv,scoring=['roc_auc','average_precision','recall'],n_jobs=1)
        model.fit(X,y); fitted[name]=model
        row={'name':name,'cv_auc':float(result['test_roc_auc'].mean()),'cv_std':float(result['test_roc_auc'].std()),'cv_ap':float(result['test_average_precision'].mean()),'cv_recall_05':float(result['test_recall'].mean()),'seconds':time.perf_counter()-start,'fold_auc':result['test_roc_auc'].tolist()}
        if hasattr(model[-1],'coef_'): row['nonzero']=int((abs(model[-1].coef_)>1e-8).sum())
        rows.append(row); save(rows,REP/'experiments.json'); print(name,round(row['cv_auc'],4),flush=True)
    # Предпочтение более простой модели при разнице ROC-AUC не более 0.005.
    best=max(r['cv_auc'] for r in rows if r['name']!='LR_without_features')
    eligible=[r for r in rows if r['cv_auc']>=best-.005 and r['name']!='LR_without_features']
    def rank(r):
        n=r['name']; return (0 if n.startswith('LR') else 1 if n.startswith('Tree') else 2 if n=='NB' else 3 if n.startswith('RF') else 4,-r['cv_auc'])
    chosen=min(eligible,key=rank)['name']; model=fitted[chosen]
    oof=cross_val_predict(clone(model),X,y,cv=cv,method='predict_proba',n_jobs=1)[:,1]
    thresholds=[]
    for t in np.arange(.05,.81,.01):
        m=metrics(y,oof,t); tn,fp=m['confusion'][0]; fn,tp=m['confusion'][1]
        thresholds.append({'threshold':float(t),'cost':int(5*fn+fp),'recall':m['recall'],'precision':m['precision']})
    threshold=min(thresholds,key=lambda x:(x['cost'], -x['recall']))['threshold']
    pred=model.predict_proba(test[FEATURES])[:,1]
    base=fitted['Dummy'].predict_proba(test[FEATURES])[:,1]
    result={'selected':chosen,'threshold':threshold,'train_n':len(train),'test_n':len(test),'clean_n':len(clean),'dq':dq,'test':metrics(test[TARGET],pred,threshold),'test_05':metrics(test[TARGET],pred,.5),'baseline':metrics(test[TARGET],base,.5)}
    rng=np.random.default_rng(42); boot=[]; diff=[]
    for _ in range(500):
        ix=rng.integers(0,len(test),len(test)); yy=test[TARGET].to_numpy()[ix]
        if len(np.unique(yy))<2: continue
        a=roc_auc_score(yy,pred[ix]); boot.append(a); diff.append(a-roc_auc_score(yy,base[ix]))
    result['auc_ci95']=np.quantile(boot,[.025,.975]).tolist(); result['auc_difference_ci95']=np.quantile(diff,[.025,.975]).tolist()
    result['acceptance']={'auc_at_least_080':result['test']['roc_auc']>=.8,'recall_at_least_075':result['test']['recall']>=.75,'auc_gain_at_least_010':result['test']['roc_auc']-result['baseline']['roc_auc']>=.1}
    save(thresholds,REP/'thresholds.json'); save(result,REP/'result.json')
    pd.DataFrame({'actual':test[TARGET],'probability':pred}).to_csv(REP/'test_predictions.csv',index=False)
    imp=permutation_importance(model,test[FEATURES],test[TARGET],scoring='roc_auc',n_repeats=5,random_state=42,n_jobs=1)
    pd.DataFrame({'feature':FEATURES,'importance':imp.importances_mean,'std':imp.importances_std}).sort_values('importance',ascending=False).to_csv(REP/'importance.csv',index=False)
    lr=fitted['LR_l2_C1']; names=lr['preprocess'].get_feature_names_out()
    pd.DataFrame({'feature':names,'coefficient':lr[-1].coef_[0]}).to_csv(REP/'coefficients.csv',index=False)
    slices=[]
    for col in ['loan_grade','person_home_ownership']:
        for value, group in test.groupby(col):
            ix=test.index.get_indexer(group.index); m=metrics(group[TARGET],pred[ix],threshold) if group[TARGET].nunique()==2 else {}
            slices.append({'column':col,'value':value,'n':len(group),'defaults':int(group[TARGET].sum()),**m})
    save(slices,REP/'slices.json')
    perturbed=test[FEATURES].copy(); perturbed['person_income']*=1.05
    pp=model.predict_proba(perturbed)[:,1]
    result['stability']={'mean_abs_delta':float(abs(pp-pred).mean()),'changed_class_share':float(((pp>=threshold)!=(pred>=threshold)).mean())}
    # Локальная чувствительность к исходным признакам (не причинный эффект).
    local=[]
    for index in [int(np.argmax(pred)),int(np.argmin(abs(pred-threshold)))]:
        row=test[FEATURES].iloc[[index]].copy(); reference={}
        for c in FEATURES: reference[c]=train[c].median() if c in NUMERIC else train[c].mode().iloc[0]
        changes=[]
        for c in FEATURES:
            altered=row.copy(); altered[c]=reference[c]
            changes.append({'feature':c,'delta':float(pred[index]-model.predict_proba(altered)[0,1])})
        local.append({'row':row.to_dict('records')[0],'actual':int(test[TARGET].iloc[index]),'probability':float(pred[index]),'changes':changes})
    save(local,REP/'local.json'); save(result,REP/'result.json')
    bundle={'pipeline':model,'threshold':threshold,'version':'v1','reference':train[FEATURES],'parameters':P}
    (ROOT/'models').mkdir(exist_ok=True); joblib.dump(bundle,ROOT/'models/model_v1.joblib')
    save({'version':'v1','selected':chosen,'threshold':threshold,'protocol':'60% initial; 80/20; CV5; cost FN=5 FP=1; seed=42','acceptance':result['acceptance']},ROOT/'models/model_v1.json')
    print('SELECTED',chosen,threshold,result['test'],flush=True)

if __name__=='__main__': main()
