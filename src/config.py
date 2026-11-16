"""Загрузка параметров и описание схемы набора Credit Risk Dataset."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

NUMERIC = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]
TARGET = "loan_status"
FEATURES = NUMERIC + CATEGORICAL
ALL_COLUMNS = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_status",
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length",
]

ALLOWED_CATEGORIES = {
    "person_home_ownership": {"RENT", "MORTGAGE", "OWN", "OTHER"},
    "loan_intent": {"EDUCATION", "MEDICAL", "VENTURE", "PERSONAL",
                    "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"},
    "loan_grade": {"A", "B", "C", "D", "E", "F", "G"},
    "cb_person_default_on_file": {"Y", "N"},
}


def load_params(path: Path | str = ROOT / "params.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
