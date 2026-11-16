"""Локальный API: uvicorn src.api:app --host 127.0.0.1 --port 8000."""
from functools import lru_cache
from typing import Literal
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from src.service import load_bundle, score

app=FastAPI(title='Оценка кредитного риска',version='1.0')
class Application(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    person_age:int=Field(ge=18,le=100)
    person_income:float=Field(gt=0)
    person_home_ownership:Literal['RENT','MORTGAGE','OWN','OTHER']
    person_emp_length:float|None=Field(default=None,ge=0,le=60)
    loan_intent:Literal['EDUCATION','MEDICAL','VENTURE','PERSONAL','DEBTCONSOLIDATION','HOMEIMPROVEMENT']
    loan_grade:Literal['A','B','C','D','E','F','G']
    loan_amnt:float=Field(gt=0)
    loan_int_rate:float|None=Field(default=None,ge=5,le=25)
    loan_percent_income:float=Field(ge=0)
    cb_person_default_on_file:Literal['Y','N']
    cb_person_cred_hist_length:int=Field(ge=0)

@lru_cache(maxsize=1)
def bundle(): return load_bundle()

@app.get('/health')
def health(): return {'status':'ok','version':bundle()['version']}

@app.post('/predict')
def predict(application:Application):
    try: _,result,report=score(pd.DataFrame([application.model_dump()]),bundle())
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc))
    return {**result.iloc[0].to_dict(),'threshold':bundle()['threshold'],'recommendation':'Дополнительная проверка' if int(result.high_risk.iloc[0]) else 'Стандартное рассмотрение','notice':'Окончательное решение принимает специалист'}
