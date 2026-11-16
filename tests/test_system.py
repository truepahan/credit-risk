import numpy as np
import pandas as pd
import pytest
from src.config import ROOT, FEATURES
from src.service import load_bundle, score, process

@pytest.fixture(scope='module')
def b(): return load_bundle()
@pytest.fixture
def sample(): return pd.read_csv(ROOT/'reports/model/test.csv').iloc[:10][FEATURES]
def test_target_independence(b,sample):
    _,a,_=score(sample,b); sample['loan_status']=1
    _,c,_=score(sample,b)
    assert np.array_equal(a.probability,c.probability)
def test_schema_rejected(b,sample):
    with pytest.raises(ValueError): score(sample.drop(columns='person_age'),b)
def test_manual_review_rejected(b,sample):
    sample.loc[:,'person_age']=150
    with pytest.raises(ValueError): score(sample,b)
def test_repeat_and_reload(b,sample):
    assert np.array_equal(score(sample,b)[1].probability,score(sample,load_bundle())[1].probability)
def test_idempotency(b,sample,tmp_path):
    path=tmp_path/'batch.csv'; sample.to_csv(path,index=False)
    a=process(path,b,tmp_path/'out'); files={str(f):f.read_bytes() for f in (tmp_path/'out').rglob('*') if f.is_file()}
    c=process(path,b,tmp_path/'out')
    assert a==c
    assert files=={str(f):f.read_bytes() for f in (tmp_path/'out').rglob('*') if f.is_file()}
def test_api(b,sample):
    from fastapi.testclient import TestClient
    from src.api import app
    client=TestClient(app)
    data=sample.iloc[0].replace({np.nan:None}).to_dict()
    r=client.post('/predict',json=data)
    assert r.status_code==200 and 0<=r.json()['probability']<=1
    assert client.post('/predict',json={}).status_code==422
    data['person_income']=-1
    assert client.post('/predict',json=data).status_code==422
