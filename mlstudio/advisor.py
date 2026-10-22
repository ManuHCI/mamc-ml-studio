"""Which kind of problem is this, and which model should I try - and WHY?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_check import column_kind, is_num
from .models import MODELS, TASK_MODELS, model_name


def detect_task(y: pd.Series) -> tuple[str, str]:
    """Return (task, reason in plain words)."""
    yy = y.dropna()
    nun = yy.nunique()
    if nun < 2:
        return "invalid", "The outcome column has only one value - there is nothing to predict."
    if not is_num(yy):
        if nun == 2:
            return "classification", f"The outcome has two categories ({', '.join(map(str, yy.unique()[:2]))}), so this is a " \
                                     "**binary classification** problem: the model will predict which group a patient belongs to."
        return "classification", f"The outcome has {nun} categories, so this is a **multi-class classification** problem."
    vals = yy.astype(float)
    integer = np.allclose(vals, vals.round())
    if nun == 2:
        return "classification", "The outcome has only two values (e.g. 0/1), so it is treated as two groups: **binary classification**."
    if integer and nun <= 5:
        return "classification", f"The outcome has only {nun} whole-number values, so it is treated as categories " \
                                 "(**classification**). If these are real counts, choose 'Predict a number' instead."
    return "regression", "The outcome is a measured number with many different values, so this is a **regression** problem: " \
                         "the model will predict a number."


def usable_features(df: pd.DataFrame, target: str) -> list[str]:
    feats = []
    for c in df.columns:
        if c == target:
            continue
        if column_kind(df[c]) in ("numeric", "binary", "categorical"):
            feats.append(c)
    return feats


def data_profile(df: pd.DataFrame, features: list[str], target: str, task: str) -> dict:
    d = df[df[target].notna()]
    n = len(d)
    num = [c for c in features if is_num(d[c]) or column_kind(d[c]) == "numeric"]
    cat = [c for c in features if c not in num]
    p_terms = len(num) + sum(max(d[c].dropna().nunique() - 1, 1) for c in cat)
    prof = dict(n=n, n_features=len(features), n_numeric=len(num), n_categorical=len(cat), p_terms=p_terms,
                missing_pct=float(d[features].isna().mean().mean() * 100) if features else 0.0)
    if task == "classification":
        vc = d[target].astype(str).value_counts()
        prof.update(n_classes=len(vc), minority=int(vc.min()), minority_label=str(vc.idxmin()),
                    minority_share=float(vc.min() / vc.sum()), imbalance=float(vc.min() / vc.max()),
                    epv=float(vc.min() / max(p_terms, 1)))
    else:
        prof.update(epv=float(n / max(p_terms, 1)))
    return prof


PRIORITY_BASE = {
    "explain": {"logreg": 5.0, "linreg": 5.0, "ridge": 4.6, "lasso": 4.7, "tree": 4.5, "nb": 3.2, "rf": 3.0, "gb": 2.6,
                "knn": 2.4, "svm": 1.8, "mlp": 1.0},
    "accuracy": {"gb": 5.0, "rf": 4.8, "svm": 3.9, "logreg": 3.6, "ridge": 3.6, "lasso": 3.4, "linreg": 3.3, "mlp": 3.0,
                 "knn": 3.0, "tree": 2.5, "nb": 2.4},
    "balanced": {"logreg": 4.7, "ridge": 4.5, "linreg": 4.4, "rf": 4.7, "gb": 4.4, "lasso": 4.2, "tree": 3.6, "svm": 3.3,
                 "knn": 3.0, "nb": 3.0, "mlp": 2.5},
}


def recommend(prof: dict, task: str, priority: str = "balanced") -> list[dict]:
    """Score every model for this dataset. Returns a list sorted best-first:
    {key, name, score, stars, reasons: [..], cautions: [..], recommended: bool}"""
    n = prof["n"]
    p = prof["p_terms"]
    out = []
    for key in TASK_MODELS[task]:
        m = MODELS[key]
        s = PRIORITY_BASE[priority][key]
        why, care = [], []

        # ---- sample size
        if n < 150:
            if key in ("logreg", "linreg", "ridge", "lasso"):
                s += 0.8
                why.append(f"Only {n} patients: simple models like this one are the safest with small data.")
            if key == "nb":
                s += 0.5
                why.append("Needs very little data.")
            if key in ("mlp", "gb"):
                s -= 1.8 if key == "mlp" else 0.9
                care.append(f"{n} patients is too few for this flexible model - it will probably memorise them (overfit).")
            if key == "rf":
                s -= 0.3
        elif n < 500:
            if key == "mlp":
                s -= 1.0
                care.append(f"Neural networks usually need thousands of patients; you have {n}.")
            if key in ("logreg", "linreg", "ridge"):
                s += 0.3
        else:
            if key in ("rf", "gb"):
                s += 0.4
                why.append(f"{n} patients is enough for an ensemble of trees to learn complex patterns.")
            if key == "mlp" and n >= 2000:
                s += 1.0
                why.append(f"With {n} patients a neural network has enough examples to learn from.")
        if n > 5000 and key == "svm":
            s -= 1.0
            care.append("Becomes slow with more than ~5,000 patients.")
        if n > 20000 and key == "knn":
            s -= 0.6
            care.append("Prediction becomes slow with very large datasets.")

        # ---- number of predictors vs patients
        epv = prof.get("epv", 99)
        if task == "classification" and epv < 10:
            if key in ("logreg",):
                s += 0.4
                why.append(f"Few events per predictor (EPV {epv:.1f}); its regularisation (C) keeps it stable.")
            if key in ("mlp", "gb", "tree"):
                s -= 0.5
                care.append(f"Low events-per-predictor ({epv:.1f}) - high risk of overfitting.")
        if p > n / 5:
            if key in ("knn",):
                s -= 1.0
                care.append("Many predictors compared with patients: distances become meaningless ('curse of dimensionality').")
            if key in ("logreg", "ridge", "lasso", "svm"):
                s += 0.4
                why.append("Handles many predictors well thanks to regularisation.")

        # ---- types of predictors
        if prof["n_categorical"] > prof["n_numeric"]:
            if key in ("tree", "rf", "gb"):
                s += 0.3
                why.append("Handles yes/no and category predictors naturally.")
            if key in ("knn", "nb"):
                s -= 0.5
                care.append("Works best with numeric predictors; most of yours are categories.")
        elif prof["n_numeric"] >= 3 and key in ("svm", "knn"):
            why.append("Most predictors are measured numbers, which suits distance-based methods.")

        # ---- class balance
        if task == "classification" and prof.get("imbalance", 1) < 0.25:
            if m["balance"]:
                why.append("Can give extra weight to the smaller outcome group ('balance the groups').")
            else:
                s -= 0.4
                care.append("Cannot re-weight the smaller outcome group; it may ignore it.")
        if task == "classification" and prof.get("n_classes", 2) > 2 and key in ("nb", "rf", "gb", "knn", "logreg", "tree"):
            why.append("Handles more than two outcome groups directly.")

        # ---- generic strengths by priority
        if priority == "explain" and key in ("logreg", "linreg", "ridge", "lasso"):
            why.append("You can report each factor's effect (odds ratio / coefficient) in your thesis.")
        if priority == "explain" and key == "tree":
            why.append("Produces a flowchart you can print and follow by hand.")
        if priority == "accuracy" and key in ("gb", "rf"):
            why.append("Usually among the most accurate methods for spreadsheet-type medical data.")
        if priority == "explain" and key in ("svm", "mlp", "knn"):
            care.append("Hard to explain how it reaches a decision (black box).")
        if key in ("logreg", "linreg"):
            why.append("Serves as the essential baseline: any complex model must beat it to be worth using.")
        if not why:
            why.append(m["one_line"])
        out.append(dict(key=key, name=model_name(key, task), score=round(s, 2), reasons=why, cautions=care))

    out.sort(key=lambda r: -r["score"])
    top = out[0]["score"]
    for i, r in enumerate(out):
        r["stars"] = int(np.clip(round(5 - (top - r["score"]) * 1.4), 1, 5))
        r["recommended"] = i < 3
    # always make sure the simple baseline is among the recommended
    base = "logreg" if task == "classification" else "linreg"
    if not any(r["key"] == base and r["recommended"] for r in out):
        for r in out:
            if r["key"] == base:
                r["recommended"] = True
                r["reasons"].insert(0, "Added as the baseline model to compare against.")
    return out


def advice_summary(prof: dict, task: str) -> list[str]:
    lines = []
    n = prof["n"]
    lines.append(f"**{n} patients** with an outcome, **{prof['n_features']} predictors** "
                 f"({prof['n_numeric']} numbers, {prof['n_categorical']} categories) -> about {prof['p_terms']} model terms.")
    if task == "classification":
        lines.append(f"Smallest outcome group: **{prof['minority_label']}** with {prof['minority']} patients "
                     f"({prof['minority_share'] * 100:.0f}%). Events per predictor (EPV) about **{prof['epv']:.1f}**.")
        if prof["epv"] < 10:
            lines.append("EPV below 10: keep the model simple, use fewer predictors, or collect more patients.")
        if prof.get("imbalance", 1) < 0.25:
            lines.append("The groups are unbalanced - switch on **Balance the outcome groups** and judge models by sensitivity, "
                         "specificity and AUC rather than accuracy.")
    if prof["missing_pct"] > 0:
        lines.append(f"About {prof['missing_pct']:.1f}% of predictor cells are blank; they will be filled in automatically inside the model.")
    if n < 100:
        lines.append("Fewer than 100 patients: results will vary a lot. Treat any model as exploratory.")
    return lines
