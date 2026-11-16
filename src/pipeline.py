"""Пайплайн обработки данных: загрузка -> проверка -> очистка -> признаки -> сохранение.

  python -m src.pipeline train            # исходная часть: оценка параметров и сохранение
  python -m src.pipeline batch <file>     # одна новая порция теми же шагами
  python -m src.pipeline all              # train + все порции из data/incoming

Шаги загрузки, проверки, очистки и формирования признаков общие.
Только при обучении: разделение train/test и fit препроцессора (медианы,
масштаб, словари категорий). Для порций используется сохраненный препроцессор.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CATEGORICAL, NUMERIC, ROOT, TARGET, load_params
from src.data.validate import validate
from src.features.build_features import NEW_CATEGORICAL, NEW_NUMERIC, add_features, make_preprocessor

NUMERIC_IN = NUMERIC + NEW_NUMERIC
NOMINAL_IN = [c for c in CATEGORICAL if c not in ("loan_grade", "cb_person_default_on_file")] \
    + NEW_CATEGORICAL


def _paths(p):
    proc = ROOT / p["data"]["processed_dir"]
    rep = ROOT / p["data"]["reports_dir"]
    proc.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    return proc, rep


def _save_report(report: dict, path: Path) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_frame(pre, X):
    return pd.DataFrame(pre.transform(X), columns=pre.get_feature_names_out(), index=X.index)


def run_train(p: dict) -> None:
    proc, rep = _paths(p)
    df = pd.read_csv(ROOT / p["data"]["initial_file"])                  # загрузка
    clean, report = validate(df, p, has_target=True)                    # проверка + очистка
    _save_report(report, rep / "initial_quality.json")
    if report["status"] == "rejected":
        sys.exit("Исходная часть не прошла проверку структуры")
    feats = add_features(clean, p)                                      # новые признаки
    train, test = train_test_split(feats, test_size=0.2, stratify=feats[TARGET],
                                   random_state=p["seed"])              # только обучение
    pre = make_preprocessor(NUMERIC_IN, NOMINAL_IN).fit(train)          # только обучение
    joblib.dump(pre, proc / "preprocessor.joblib")
    for name, part in (("train", train), ("test", test)):
        out = _to_frame(pre, part)
        out[TARGET] = part[TARGET].values
        out.to_csv(proc / f"{name}_features.csv", index=False)
    print(f"initial: {report['rows_in']} -> {report['rows_out']} строк; "
          f"train {len(train)}, test {len(test)}")


def run_batch(p: dict, file: Path) -> dict:
    proc, rep = _paths(p)
    pre = joblib.load(proc / "preprocessor.joblib")
    df = pd.read_csv(file)                                              # загрузка
    label = df[TARGET] if TARGET in df.columns else None                # метка известна позже
    clean, report = validate(df.drop(columns=[TARGET], errors="ignore"), p, has_target=False)
    report["file"] = str(Path(file).name)
    if report["status"] != "rejected":
        feats = add_features(clean, p)
        out = _to_frame(pre, feats)                                     # применение ранее обученного преобразования
        if label is not None:
            out[TARGET] = label.loc[out.index].values                   # для мониторинга
        out.to_csv(proc / f"{Path(file).stem}_features.csv", index=False)
        # показатели без модели (табл. 2.2): объем и средняя сумма заявок
        report["kpi"] = {"applications": int(len(clean)),
                         "mean_loan_amnt": round(float(clean["loan_amnt"].mean()), 2)}
    _save_report(report, rep / f"{Path(file).stem}_quality.json")
    print(f"{report['file']}: {report['status']}, {report['rows_in']} -> {report.get('rows_out')}")
    return report


def main(argv: list[str]) -> None:
    p = load_params()
    cmd = argv[1] if len(argv) > 1 else "all"
    if cmd == "train":
        run_train(p)
    elif cmd == "batch":
        run_batch(p, Path(argv[2]))
    elif cmd == "all":
        run_train(p)
        for f in sorted((ROOT / p["data"]["incoming_dir"]).glob("batch_*.csv")):
            run_batch(p, f)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
