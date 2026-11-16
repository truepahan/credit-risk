"""Проверка качества данных (требования DQ-01 … DQ-10 из отчета ПР2).

validate(df, params, has_target) -> (clean_df, report)
  * DQ-01 — нарушение структуры: порция не обрабатывается (report["status"] = "rejected");
  * DQ-02 … DQ-06, DQ-09 — некорректные записи исключаются с указанием причины;
  * DQ-07, DQ-08 — предупреждения, записи остаются;
  * DQ-10 — слишком много исключенных записей: порция уходит на ручную проверку.
"""
import pandas as pd

from src.config import ALLOWED_CATEGORIES, CATEGORICAL, FEATURES, NUMERIC, TARGET


def _row_rules(df: pd.DataFrame, q: dict, has_target: bool) -> dict[str, pd.Series]:
    """Возвращает маски нарушений: True — запись нарушает правило."""
    age, emp = df["person_age"], df["person_emp_length"]
    rules = {
        "DQ-02 полный дубликат": df.duplicated(keep="first"),
        "DQ-03 возраст вне диапазона": ~age.between(q["age_min"], q["age_max"]),
        "DQ-04 недопустимый стаж": emp.notna() & ((emp < 0) | (emp > q["emp_length_max"])
                                                   | (emp > age - 14)),
        "DQ-05 доход или сумма <= 0": (df["person_income"] <= 0) | (df["loan_amnt"] <= 0),
        "DQ-06 несогласованная доля платежа": (
            (df["loan_percent_income"] - df["loan_amnt"] / df["person_income"]).abs()
            > q["percent_income_tolerance"]),
    }
    # обязательные поля (кроме тех, для которых пропуск допустим и заполняется)
    required = [c for c in FEATURES if c not in ("person_emp_length", "loan_int_rate")]
    rules["DQ-01 пропуск в обязательном поле"] = df[required].isna().any(axis=1)
    if has_target:
        rules["DQ-09 некорректная метка"] = ~df[TARGET].isin([0, 1])
    return rules


def validate(df: pd.DataFrame, params: dict, has_target: bool = True):
    q = params["quality"]
    report = {"rows_in": int(len(df)), "status": "ok", "violations": {}, "warnings": []}

    # DQ-01: структура и типы
    expected = FEATURES + ([TARGET] if has_target else [])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        report["status"] = "rejected"
        report["warnings"].append(f"DQ-01 нет столбцов: {missing}")
        return df.iloc[0:0], report
    df = df[expected].copy()
    for c in NUMERIC + ([TARGET] if has_target else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL:
        df[c] = df[c].astype("string").str.strip().str.upper()

    # DQ-02 … DQ-06, DQ-09: построчные правила
    bad = pd.Series(False, index=df.index)
    for name, mask in _row_rules(df, q, has_target).items():
        mask = mask.fillna(False)
        report["violations"][name] = int(mask.sum())
        bad |= mask

    # DQ-07: пропуски и диапазон ставки
    rate = df["loan_int_rate"]
    share = float(rate.isna().mean())
    report["int_rate_missing_share"] = round(share, 4)
    if share > q["int_rate_max_missing_share"]:
        report["warnings"].append(f"DQ-07 доля пропусков ставки {share:.1%} выше порога")
    out_of_range = rate.notna() & ~rate.between(q["int_rate_min"], q["int_rate_max"])
    if out_of_range.any():
        report["warnings"].append(f"DQ-07 ставка вне диапазона: {int(out_of_range.sum())} записей")

    # DQ-08: неизвестные категории — журнал, записи остаются
    for c, allowed in ALLOWED_CATEGORIES.items():
        unknown = sorted(set(df[c].dropna()) - allowed)
        if unknown:
            report["warnings"].append(f"DQ-08 {c}: неизвестные значения {unknown}")

    clean = df[~bad]
    report["rows_rejected"] = int(bad.sum())
    report["rows_out"] = int(len(clean))
    rejected_share = bad.mean() if len(df) else 0.0
    report["rejected_share"] = round(float(rejected_share), 4)

    # DQ-10: доля исключенных записей
    if rejected_share > q["max_rejected_share"]:
        report["status"] = "manual_review"
    return clean, report
