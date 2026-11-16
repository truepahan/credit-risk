import pandas as pd

from src.config import load_params
from src.data.validate import validate

P = load_params()


def row(**kw):
    base = dict(person_age=30, person_income=50000, person_home_ownership="RENT",
                person_emp_length=5.0, loan_intent="EDUCATION", loan_grade="B",
                loan_amnt=10000, loan_int_rate=11.0, loan_status=0,
                loan_percent_income=0.2, cb_person_default_on_file="N",
                cb_person_cred_hist_length=4)
    base.update(kw)
    return base


def test_valid_row_passes():
    clean, rep = validate(pd.DataFrame([row()]), P)
    assert len(clean) == 1 and rep["status"] == "ok"


def test_impossible_age_and_emp_rejected():
    df = pd.DataFrame([row(), row(person_age=144), row(person_emp_length=123.0)])
    clean, rep = validate(df, P)
    assert len(clean) == 1
    assert rep["violations"]["DQ-03 возраст вне диапазона"] == 1


def test_duplicates_removed():
    clean, _ = validate(pd.DataFrame([row(), row()]), P)
    assert len(clean) == 1


def test_missing_column_rejects_batch():
    df = pd.DataFrame([row()]).drop(columns=["loan_amnt"])
    _, rep = validate(df, P)
    assert rep["status"] == "rejected"


def test_missing_rate_is_allowed():
    clean, _ = validate(pd.DataFrame([row(loan_int_rate=None)]), P)
    assert len(clean) == 1
