"""Замеры, контрольное переобучение, графики и панель."""
import json,time,platform,importlib.metadata
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from src.config import ROOT,FEATURES,TARGET
from src.service import load_bundle,score
from src.modeling import save,metrics
from src.data.validate import validate

def main():
    b=load_bundle(); rep=ROOT/'reports'; fig=rep/'final_figures'; fig.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    result=json.loads((rep/'model/result.json').read_text(encoding='utf-8'))
    batches=json.loads((rep/'batch_results.json').read_text(encoding='utf-8'))
    test=pd.read_csv(rep/'model/test.csv'); train=pd.read_csv(rep/'model/train.csv')
    # Порция 1 имитирует новые зрелые данные; сравнение кандидата на отложенной части порции 2.
    mature,_=validate(pd.read_csv(ROOT/'data/incoming/batch_01.csv'),b['parameters'],True)
    development=pd.concat([train,mature]).drop_duplicates(subset=FEATURES)
    candidate=clone(b['pipeline']).fit(development[FEATURES],development[TARGET])
    control,_=validate(pd.read_csv(ROOT/'data/incoming/batch_02.csv'),b['parameters'],True)
    known=set(pd.util.hash_pandas_object(pd.concat([development,test])[FEATURES],index=False))
    control=control[~pd.util.hash_pandas_object(control[FEATURES],index=False).isin(known)]
    old=metrics(control[TARGET],b['pipeline'].predict_proba(control[FEATURES])[:,1],b['threshold'])
    new=metrics(control[TARGET],candidate.predict_proba(control[FEATURES])[:,1],b['threshold'])
    promote=new['roc_auc']>=old['roc_auc']+.005 and new['recall']>=.75 and new['brier']<=old['brier']
    save({'development_n':len(development),'control_n':len(control),'old':old,'candidate':new,'eligible_for_promotion':promote,'active_version':'v1','decision':'Сохранена v1; кандидат не проходит правило улучшения' if not promote else 'Кандидат проходит правило; v1 оставлена для воспроизводимости основного эксперимента'},rep/'retraining.json')
    joblib.dump({**b,'pipeline':candidate,'version':'v2_candidate'},ROOT/'models/model_v2_candidate.joblib')
    # 10000 разных входных записей; пропуски заполняются внутри конвейера.
    benchmark=pd.read_csv(ROOT/'data/raw/credit_risk_dataset.csv').drop_duplicates(subset=FEATURES).iloc[:10000]
    start=time.perf_counter(); score(benchmark,b); elapsed=time.perf_counter()-start
    from fastapi.testclient import TestClient
    from src.api import app
    client=TestClient(app); payload=test[FEATURES].iloc[0].replace({np.nan:None}).to_dict()
    client.post('/predict',json=payload)
    latencies=[]
    for _ in range(50):
        start=time.perf_counter(); response=client.post('/predict',json=payload); assert response.status_code==200
        latencies.append(1000*(time.perf_counter()-start))
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy','matplotlib','seaborn','joblib','pyyaml','pytest','fastapi','uvicorn','httpx','jupyter']}
    save({'batch_10000_seconds':elapsed,'api_median_ms':float(np.median(latencies)),'api_p95_ms':float(np.quantile(latencies,.95)),'api_mode':'TestClient; 50 warm requests; без сетевой задержки','python':platform.python_version(),'platform':platform.platform(),'versions':versions,'example_request':payload,'example_response':response.json()},rep/'verification.json')
    (ROOT/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n',encoding='utf-8')
    y=test[TARGET]; p=b['pipeline'].predict_proba(test[FEATURES])[:,1]
    fig1,axes=plt.subplots(1,3,figsize=(12,3.4))
    fpr,tpr,_=roc_curve(y,p); axes[0].plot(fpr,tpr); axes[0].plot([0,1],[0,1],'--',color='gray'); axes[0].set(xlabel='Доля ложных срабатываний',ylabel='Полнота',title='ROC-кривая')
    precision,recall,_=precision_recall_curve(y,p); axes[1].plot(recall,precision); axes[1].set(xlabel='Полнота',ylabel='Точность',title='Precision–Recall')
    actual,predicted=calibration_curve(y,p,n_bins=8,strategy='quantile'); axes[2].plot(predicted,actual,'o-'); axes[2].plot([0,1],[0,1],'--',color='gray'); axes[2].set(xlabel='Прогнозная вероятность',ylabel='Фактическая доля',title='Калибровочная кривая')
    fig1.tight_layout(); fig1.savefig(fig/'diagnostics.png',dpi=180); plt.close(fig1)
    cm=np.array(result['test']['confusion']); f,ax=plt.subplots(figsize=(5,3.5)); ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>1500 else 'black',fontsize=17)
    ax.set(xticks=[0,1],yticks=[0,1],xlabel='Прогноз',ylabel='Фактический класс'); f.tight_layout(); f.savefig(fig/'confusion.png',dpi=180);plt.close(f)
    imp=pd.read_csv(rep/'model/importance.csv').sort_values('importance'); f,ax=plt.subplots(figsize=(8,4.5)); ax.barh(imp.feature,imp.importance);ax.set_xlabel('Снижение ROC-AUC после перестановки');f.tight_layout();f.savefig(fig/'importance.png',dpi=180);plt.close(f)
    ex=pd.DataFrame(json.loads((rep/'model/experiments.json').read_text(encoding='utf-8')))
    f,ax=plt.subplots(figsize=(8,4));ax.barh(ex.name,ex.cv_auc);ax.set(xlim=(.45,1),xlabel='Средний ROC-AUC на пяти блоках');f.tight_layout();f.savefig(fig/'comparison.png',dpi=180);plt.close(f)
    t=pd.DataFrame(json.loads((rep/'model/thresholds.json').read_text(encoding='utf-8')))
    f,axes=plt.subplots(1,2,figsize=(9,3.5));axes[0].plot(t.threshold,t.cost);axes[0].set(xlabel='Порог',ylabel='5 × FN + FP'); axes[1].plot(t.threshold,t.recall,label='Полнота');axes[1].plot(t.threshold,t.precision,label='Точность');axes[1].legend();axes[1].set(xlabel='Порог',ylabel='Значение');f.tight_layout();f.savefig(fig/'threshold.png',dpi=180);plt.close(f)
    # Снимки одной и той же панели после каждой очередной порции.
    for end in range(1,5):
        current=batches[:end]; x=np.arange(1,end+1)
        f,axes=plt.subplots(2,2,figsize=(9,6))
        for ax,key,title in zip(axes.flat,['n','high_risk_share','expected_loss_proxy','psi_income'],['Количество заявок','Доля высокого риска','Условные ожидаемые потери, ден. ед.','PSI дохода']):
            ax.plot(x,[r[key] for r in current],'o-'); ax.set(title=title,xlabel='Номер порции',xticks=x);ax.grid(alpha=.2)
        f.tight_layout(); f.savefig(fig/f'dashboard_{end}.png',dpi=160);plt.close(f)
    cells=[{'cell_type':'markdown','metadata':{},'source':['# Панель кредитного риска\nДанные обновляются командой python -m src.service. Все четыре снимка сохранены в reports/final_figures.']}]
    cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':["from pathlib import Path\nimport json\nimport pandas as pd\nimport matplotlib.pyplot as plt\nroot = Path.cwd() if (Path.cwd()/'reports').exists() else Path.cwd().parent\nrows = json.loads((root/'reports/batch_results.json').read_text(encoding='utf-8'))\ndf = pd.DataFrame(rows)\ndisplay(df[['file','n','mean_loan','high_risk_share','expected_loss_proxy','psi_income']])\ndf.set_index('file')[['high_risk_share','psi_income']].plot(marker='o', subplots=True, figsize=(9,6));\n"]})
    save({'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5},ROOT/'notebooks/03_credit_dashboard.ipynb')
if __name__=='__main__': main()
