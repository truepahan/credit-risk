"""Формирование признаков (план признаков — табл. 2.5 отчета ПР2).

add_features() — детерминированные новые признаки, не требуют обучения.
make_preprocessor() — преобразования с параметрами (медианы, масштаб, словари
категорий); параметры оцениваются один раз на обучающих данных (fit) и
сохраняются, к новым порциям применяется только transform.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

GRADES = ["A", "B", "C", "D", "E", "F", "G"]

NEW_NUMERIC = ["log_income", "emp_share_of_working_age", "credit_hist_to_age",
               "grade_num", "int_rate_missing", "high_burden", "prev_default"]
NEW_CATEGORICAL = ["age_group"]


def add_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    f = params["features"]
    out = df.copy()
    # 1. Логарифм дохода: распределение дохода сильно скошено вправо
    out["log_income"] = np.log1p(out["person_income"])
    # 2. Доля трудовой жизни, проведенная на текущем месте работы (стабильность)
    working_years = (out["person_age"] - 14).clip(lower=1)
    out["emp_share_of_working_age"] = out["person_emp_length"] / working_years
    # 3. Длина кредитной истории относительно возраста
    out["credit_hist_to_age"] = out["cb_person_cred_hist_length"] / out["person_age"]
    # 4. Рейтинг займа как порядковое число A=1 … G=7
    out["grade_num"] = out["loan_grade"].map({g: i + 1 for i, g in enumerate(GRADES)})
    # 5. Флаг пропуска ставки (до заполнения медианой)
    out["int_rate_missing"] = out["loan_int_rate"].isna().astype(int)
    # 6. Высокая долговая нагрузка: платеж > 30 % дохода
    out["high_burden"] = (out["loan_percent_income"] > f["high_burden_threshold"]).astype(int)
    # 7. Дефолт в прошлом как 0/1
    out["prev_default"] = (out["cb_person_default_on_file"] == "Y").astype(int)
    # 8. Возрастная группа: зависимость риска от возраста нелинейна
    out["age_group"] = pd.cut(out["person_age"], bins=f["age_bins"],
                              labels=f["age_labels"]).astype("string")
    return out


def make_preprocessor(numeric: list[str], nominal: list[str]) -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    nom = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, numeric), ("nom", nom, nominal)],
                             verbose_feature_names_out=False)
