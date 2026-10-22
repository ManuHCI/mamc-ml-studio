"""Preparing data, splitting, training, tuning and evaluating models."""
from __future__ import annotations

import inspect
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, median_absolute_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .data_check import is_num
from .models import MODELS, make_estimator, model_name, supports_sample_weight

POSITIVE_WORDS = ["yes", "positive", "present", "1", "true", "malignant", "dead", "died", "death", "case", "disease",
                  "diseased", "abnormal", "high", "severe", "complication", "infected", "relapse", "readmitted"]

_CV_PARAMS_KW = "params" if "params" in inspect.signature(cross_validate).parameters else "fit_params"


def _ohe():
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        return OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse=False)


# ---------------------------------------------------------------------------
# Preparing X and y
# ---------------------------------------------------------------------------
def _label(v) -> str:
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def class_labels(y: pd.Series) -> list[str]:
    return sorted({_label(v) for v in y.dropna()})


def guess_positive(labels: list[str]) -> str:
    low = [lab.lower() for lab in labels]
    for w in POSITIVE_WORDS:
        if w in low:
            return labels[low.index(w)]
    return labels[-1]


def clean_X(X: pd.DataFrame) -> pd.DataFrame:
    """Numbers as float, everything else as python strings with NaN for missing (safe for scikit-learn).
    Columns that are mostly numbers typed as text ('45 yrs') are converted to numbers."""
    from .data_check import _mostly_numeric_text, to_number
    X = X.copy()
    for c in X.columns:
        if not is_num(X[c]) and _mostly_numeric_text(X[c]):
            X[c] = to_number(X[c])
        if is_num(X[c]):
            X[c] = X[c].astype(float)
        else:
            X[c] = pd.Series([_label(v) if not pd.isna(v) else np.nan for v in X[c]], index=X.index, dtype=object)
    return X


def prepare_xy(df: pd.DataFrame, features: list[str], target: str, task: str, positive: str | None = None):
    d = df[df[target].notna()]
    X = clean_X(d[features])
    if task == "classification":
        ys = d[target].map(_label)
        labels = sorted(ys.unique().tolist())
        if len(labels) == 2:
            pos = positive if positive in labels else guess_positive(labels)
            neg = [lab for lab in labels if lab != pos][0]
            y = (ys == pos).astype(int).to_numpy()
            labels = [neg, pos]
        else:
            y = np.asarray(ys, dtype=object)
    else:
        y = d[target].astype(float).to_numpy()
        labels = None
    return X, y, labels


def make_preprocessor(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    num = [c for c in X.columns if is_num(X[c])]
    cat = [c for c in X.columns if c not in num]
    parts = []
    if num:
        steps = [("fill_blanks", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scale", StandardScaler()))
        parts.append(("numbers", Pipeline(steps), num))
    if cat:
        parts.append(("categories", Pipeline([("fill_blanks", SimpleImputer(strategy="most_frequent")),
                                              ("one_hot", _ohe())]), cat))
    return ColumnTransformer(parts, sparse_threshold=0, verbose_feature_names_out=False)


def make_pipeline(key: str, task: str, X: pd.DataFrame, params: dict | None = None, seed: int = 42, balance: bool = False):
    est = make_estimator(key, task, params, seed, balance)
    return Pipeline([("prep", make_preprocessor(X, MODELS[key]["scale"])), ("model", est)])


def split_data(X, y, task, test_size=0.2, seed=42, stratify=True):
    strat = y if (task == "classification" and stratify) else None
    if strat is not None:
        _, counts = np.unique(strat, return_counts=True)
        if counts.min() < 2:
            strat = None
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def scorers(task: str, n_classes: int = 2) -> dict:
    if task == "regression":
        return {"R2": "r2", "MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error"}
    if n_classes == 2:
        return {"AUC": "roc_auc", "Balanced accuracy": "balanced_accuracy", "Accuracy": "accuracy",
                "Sensitivity": "recall", "F1": "f1"}
    return {"AUC": "roc_auc_ovr", "Balanced accuracy": "balanced_accuracy", "Accuracy": "accuracy", "F1 (macro)": "f1_macro"}


METRIC_HELP = {
    "AUC": "Area under the ROC curve: the chance that the model gives a higher risk to a random patient WITH the outcome "
           "than to a random patient WITHOUT it. 0.5 = coin toss, 0.7-0.8 = acceptable, 0.8-0.9 = good, > 0.9 = excellent.",
    "Balanced accuracy": "The average of sensitivity and specificity (or of the recall of every group). Not fooled by unbalanced groups.",
    "Accuracy": "Percentage of all patients classified correctly. Misleading when one group is much larger.",
    "Sensitivity": "Of the patients who truly have the outcome, how many does the model catch? (recall, true-positive rate)",
    "F1": "Balance between sensitivity and PPV (precision). Useful when the positive group is small.",
    "F1 (macro)": "F1 calculated for every group and averaged, so every group counts equally.",
    "R2": "Share of the differences between patients that the model explains (0 = no better than guessing the average, 1 = perfect).",
    "MAE": "Mean absolute error: on average, how far the prediction is from the truth, in the outcome's own units.",
    "RMSE": "Root mean squared error: like MAE but punishes big misses more. Same units as the outcome.",
}

LOWER_IS_BETTER = {"MAE", "RMSE"}


def _flip(name, v):
    return -v if name in LOWER_IS_BETTER else v


# ---------------------------------------------------------------------------
# Training (with optional tuning)
# ---------------------------------------------------------------------------
def train_model(key: str, task: str, X_train, y_train, *, mode: str = "default", params: dict | None = None,
                tune_params: list[str] | None = None, grid_override: dict | None = None, n_iter: int = 20,
                cv_folds: int = 5, main_metric: str | None = None, seed: int = 42, balance: bool = False,
                n_classes: int = 2) -> dict:
    """mode: 'default' | 'manual' | 'grid' | 'random'."""
    from .models import param_grid, random_distributions
    t0 = time.time()
    sc = scorers(task, n_classes)
    main = main_metric or list(sc)[0]
    pipe = make_pipeline(key, task, X_train, params if mode == "manual" else None, seed, balance)
    cv = (StratifiedKFold(cv_folds, shuffle=True, random_state=seed) if task == "classification"
          else KFold(cv_folds, shuffle=True, random_state=seed))
    fit_params = {}
    if balance and task == "classification" and MODELS[key]["balance"] == "sample_weight" \
            and supports_sample_weight(pipe.named_steps["model"]):
        fit_params["model__sample_weight"] = compute_sample_weight("balanced", y_train)

    tuning = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        if mode in ("grid", "random"):
            if mode == "grid":
                full = grid_override or param_grid(key, len(y_train))
                if tune_params:
                    full = {k: v for k, v in full.items() if k in tune_params}
                space = {f"model__{k}": v for k, v in full.items()}
                search = GridSearchCV(pipe, space, scoring=sc, refit=main, cv=cv, return_train_score=True,
                                      error_score=np.nan)
            else:
                dist = random_distributions(key, len(y_train))
                if tune_params:
                    dist = {k: v for k, v in dist.items() if k in tune_params}
                space = {f"model__{k}": v for k, v in dist.items()}
                search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring=sc, refit=main, cv=cv,
                                            return_train_score=True, random_state=seed, error_score=np.nan)
            if not space:
                mode = "default"
            else:
                search.fit(X_train, y_train, **fit_params)
                fitted = search.best_estimator_
                best = dict(_defaults(key), **{k.replace("model__", ""): _py(v) for k, v in search.best_params_.items()})
                res = pd.DataFrame(search.cv_results_)
                tuning = _tuning_table(res, list(sc), main)
                i = search.best_index_
                cv_summary = {m: dict(mean=_flip(m, res[f"mean_test_{m}"][i]), sd=res[f"std_test_{m}"][i],
                                      train=_flip(m, res[f"mean_train_{m}"][i]),
                                      folds=[_flip(m, res[f"split{k}_test_{m}"][i]) for k in range(cv_folds)])
                              for m in sc}
        if mode in ("default", "manual"):
            cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=sc, return_train_score=True,
                                   error_score=np.nan, **{_CV_PARAMS_KW: fit_params})
            cv_summary = {m: dict(mean=_flip(m, np.nanmean(cvres[f"test_{m}"])), sd=float(np.nanstd(cvres[f"test_{m}"])),
                                  train=_flip(m, np.nanmean(cvres[f"train_{m}"])),
                                  folds=[_flip(m, v) for v in cvres[f"test_{m}"]]) for m in sc}
            fitted = pipe.fit(X_train, y_train, **fit_params)
            best = dict(_defaults(key), **params) if (mode == "manual" and params) else _defaults(key)
    return dict(key=key, name=model_name(key, task), pipe=fitted, mode=mode, params=best, cv=cv_summary,
                tuning=tuning, main=main, fit_time=time.time() - t0, balance=balance, cv_folds=cv_folds)


def _py(v):
    """numpy scalar -> plain python value (cleaner display and code export)."""
    if isinstance(v, np.generic):
        return v.item()
    return v


def _defaults(key):
    from .models import default_params
    return default_params(key)


def _tuning_table(res: pd.DataFrame, metrics: list[str], main: str) -> pd.DataFrame:
    pcols = [c for c in res.columns if c.startswith("param_")]
    out = pd.DataFrame({c.replace("param_model__", ""): res[c].astype(object) for c in pcols})
    for m in metrics:
        out[f"CV {m}"] = res[f"mean_test_{m}"].map(lambda v: _flip(m, v))
    out[f"Train {main}"] = res[f"mean_train_{main}"].map(lambda v: _flip(main, v))
    out[f"CV {main} SD"] = res[f"std_test_{main}"]
    out["Rank"] = res[f"rank_test_{main}"]
    return out.sort_values("Rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def binary_metrics(y, proba, threshold=0.5) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    out = dict(TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn), Sensitivity=sens, Specificity=spec, PPV=ppv, NPV=npv,
               Accuracy=(tp + tn) / len(y), **{"Balanced accuracy": np.nanmean([sens, spec])},
               F1=(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else np.nan)
    out["LR+"] = sens / (1 - spec) if (np.isfinite(spec) and spec < 1) else np.inf
    out["LR-"] = (1 - sens) / spec if (np.isfinite(spec) and spec > 0) else np.nan
    if len(np.unique(y)) == 2:
        out["AUC"] = roc_auc_score(y, proba)
        out["Brier score"] = brier_score_loss(y, proba)
    else:
        out["AUC"] = np.nan
        out["Brier score"] = np.nan
    return out


def multiclass_metrics(y, proba, classes) -> dict:
    y = np.asarray(y, dtype=object)
    pred = np.asarray(classes, dtype=object)[np.argmax(proba, axis=1)]
    out = {"Accuracy": accuracy_score(y, pred), "Balanced accuracy": balanced_accuracy_score(y, pred),
           "F1 (macro)": f1_score(y, pred, average="macro", zero_division=0)}
    try:
        out["AUC"] = roc_auc_score(y, proba, multi_class="ovr", labels=list(classes))
    except ValueError:
        out["AUC"] = np.nan
    rows = []
    for c in classes:
        yt = y == c
        yp = pred == c
        tp = int((yt & yp).sum())
        fn = int((yt & ~yp).sum())
        fp = int((~yt & yp).sum())
        tn = int((~yt & ~yp).sum())
        rows.append({"Group": c, "Patients in test set": int(yt.sum()),
                     "Sensitivity (recall)": tp / (tp + fn) if tp + fn else np.nan,
                     "Specificity": tn / (tn + fp) if tn + fp else np.nan,
                     "PPV (precision)": tp / (tp + fp) if tp + fp else np.nan,
                     "F1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan})
    out["per_class"] = pd.DataFrame(rows)
    out["confusion"] = confusion_matrix(y, pred, labels=list(classes))
    return out


def regression_metrics(y, pred) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    out = {"R2": r2_score(y, pred), "MAE": mean_absolute_error(y, pred),
           "RMSE": float(np.sqrt(mean_squared_error(y, pred))), "Median absolute error": median_absolute_error(y, pred)}
    if np.all(y > 0):
        out["MAPE %"] = float(np.mean(np.abs((y - pred) / y)) * 100)
    return out


def predict_scores(pipe, X, task):
    if task == "regression":
        return pipe.predict(X)
    proba = pipe.predict_proba(X)
    return proba


def evaluate(result: dict, X_train, y_train, X_test, y_test, task: str, labels, n_boot: int = 400, seed: int = 42):
    pipe = result["pipe"]
    if task == "regression":
        p_test = pipe.predict(X_test)
        p_train = pipe.predict(X_train)
        result["test"] = regression_metrics(y_test, p_test)
        result["train"] = regression_metrics(y_train, p_train)
        result["pred_test"] = p_test
        result["ci"] = bootstrap_ci(lambda yt, pp: regression_metrics(yt, pp), y_test, p_test, ["R2", "MAE", "RMSE"], n_boot, seed)
    elif len(labels) == 2:
        p_test = pipe.predict_proba(X_test)[:, 1]
        p_train = pipe.predict_proba(X_train)[:, 1]
        result["test"] = binary_metrics(y_test, p_test)
        result["train"] = binary_metrics(y_train, p_train)
        result["proba_test"] = p_test
        result["ci"] = bootstrap_ci(lambda yt, pp: binary_metrics(yt, pp), y_test, p_test,
                                    ["AUC", "Sensitivity", "Specificity", "PPV", "NPV", "Accuracy", "Balanced accuracy"], n_boot, seed)
    else:
        classes = list(pipe.classes_)
        p_test = pipe.predict_proba(X_test)
        p_train = pipe.predict_proba(X_train)
        result["classes"] = classes
        result["test"] = multiclass_metrics(y_test, p_test, classes)
        result["train"] = multiclass_metrics(y_train, p_train, classes)
        result["proba_test"] = p_test
        result["ci"] = bootstrap_ci(lambda yt, pp: multiclass_metrics(yt, pp, classes), y_test, p_test,
                                    ["Accuracy", "Balanced accuracy", "AUC"], n_boot, seed)
    result["verdict"] = fit_verdict(result, task, labels)
    return result


def bootstrap_ci(fn, y, scores, names, n_boot=400, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores = np.asarray(scores)
    n = len(y)
    vals = {k: [] for k in names}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            try:
                m = fn(y[idx], scores[idx])
            except Exception:
                continue
            for k in names:
                v = m.get(k, np.nan)
                if v is not None and np.isfinite(v):
                    vals[k].append(v)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) > 20 else (np.nan, np.nan)
            for k, v in vals.items()}


def quality_word(metric: str, v: float) -> str:
    if v is None or not np.isfinite(v):
        return "not available"
    if metric == "AUC":
        return ("excellent" if v >= 0.9 else "good" if v >= 0.8 else "acceptable" if v >= 0.7 else
                "poor" if v >= 0.6 else "no better than chance")
    if metric == "R2":
        return ("strong" if v >= 0.75 else "moderate" if v >= 0.5 else "weak" if v >= 0.25 else "very weak")
    return ("excellent" if v >= 0.9 else "good" if v >= 0.8 else "fair" if v >= 0.7 else "poor")


def fit_verdict(result: dict, task: str, labels) -> dict:
    """Is the model over-fitting, under-fitting or about right? Plain words.
    Compares the score on the training patients with the score on unseen patients (cross-validation + test)."""
    if task == "regression":
        m, limit_gap, weak = "R2", 0.15, 0.25
    else:
        m = "AUC" if np.isfinite(result["test"].get("AUC", np.nan)) else "Balanced accuracy"
        limit_gap, weak = 0.08, 0.65
    tr = result["train"][m]
    te = result["test"][m]
    cvm = result["cv"][m]["mean"] if m in result["cv"] else te
    unseen = float(np.nanmean([cvm, te]))
    gap = tr - unseen
    shown = "R²" if m == "R2" else m
    ens = result["key"] in ("rf", "gb")
    if unseen < weak:
        code, label = "underfit", "Weak model (underfitting)"
        text = (f"On patients it has not seen, {shown} is only {unseen:.2f} ({quality_word(m, unseen)}). The predictors may "
                "not carry enough information, or the model is too simple. Try a more flexible model, add relevant "
                "predictors, or check the data.")
    elif gap > limit_gap:
        code, label = "overfit", "Overfitting (memorising)"
        text = (f"{shown} is {tr:.2f} on the training patients but {unseen:.2f} on unseen patients "
                f"(cross-validation {cvm:.2f}, test {te:.2f}). The model has partly memorised its training patients. ")
        if ens:
            text += ("Random forests and boosting nearly always score close to perfect on their own training data, so judge "
                     f"them by the unseen-patient score, which is {quality_word(m, unseen)} here. ")
        text += "A simpler setting (smaller depth, more regularisation) or fewer predictors may generalise better - try tuning."
    else:
        code, label = "good", "Generalises well"
        text = (f"{shown}: training {tr:.2f}, cross-validation {cvm:.2f}, test {te:.2f}. The model performs about as well "
                f"on patients it has never seen ({quality_word(m, unseen)}) - it learned general patterns, not individual "
                "patients.")
    return dict(code=code, label=label, text=text, metric=m, train=tr, cv=cvm, test=te)


# ---------------------------------------------------------------------------
# Explaining the model
# ---------------------------------------------------------------------------
def permutation_importance_table(result, X_test, y_test, task, labels, n_repeats=10, seed=42) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance
    sc = scorers(task, len(labels) if labels else 2)
    metric = "AUC" if task == "classification" else "R2"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = permutation_importance(result["pipe"], X_test, y_test, scoring=sc[metric], n_repeats=n_repeats,
                                   random_state=seed)
    df = pd.DataFrame({"Factor": X_test.columns, "Importance": r.importances_mean, "SD": r.importances_std})
    df["metric"] = metric
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


def encoded_names(pipe) -> list[str]:
    try:
        return list(pipe.named_steps["prep"].get_feature_names_out())
    except Exception:
        n = pipe.named_steps["model"].n_features_in_
        return [f"x{i}" for i in range(n)]


def encoded_meanings(pipe) -> dict:
    """For each model input column: what does a 1-unit change mean? ('Yes' vs 'No', per 1 SD ...)."""
    out = {}
    prep = pipe.named_steps["prep"]
    for name, trans, cols in prep.transformers_:
        if name == "numbers":
            scaled = "scale" in trans.named_steps
            for c in cols:
                out[c] = "per 1 SD increase" if scaled else "per 1 unit increase"
        elif name == "categories":
            oh = trans.named_steps["one_hot"]
            drop = getattr(oh, "drop_idx_", None)
            for i, c in enumerate(cols):
                cats = [str(v) for v in oh.categories_[i]]
                if drop is not None and drop[i] is not None and len(cats) == 2:
                    kept = [v for j, v in enumerate(cats) if j != drop[i]][0]
                    ref = cats[drop[i]]
                    out[f"{c}_{kept}"] = f"'{kept}' compared with '{ref}'"
                else:
                    for v in cats:
                        out[f"{c}_{v}"] = f"'{v}' compared with the other {c} groups"
    return out


def model_weights(result, task, labels) -> pd.DataFrame | None:
    """Odds ratios (logistic) / coefficients (linear) / built-in importances (trees)."""
    pipe = result["pipe"]
    model = pipe.named_steps["model"]
    names = encoded_names(pipe)
    meanings = encoded_meanings(pipe)
    num_cols = []
    for name, _, cols in pipe.named_steps["prep"].transformers_:
        if name == "numbers":
            num_cols = list(cols)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if task == "classification":
            if coef.shape[0] == 1:
                df = pd.DataFrame({"Term": names, "Coefficient": coef[0]})
                df["Odds ratio"] = np.exp(df["Coefficient"])
                df["Meaning"] = [meanings.get(t, "") for t in names]
                df["Direction"] = np.where(df["Coefficient"] > 0, f"raises chance of '{labels[1]}'",
                                           np.where(df["Coefficient"] < 0, f"lowers chance of '{labels[1]}'", "no effect"))
                return df.reindex(df["Coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)
            df = pd.DataFrame(coef.T, columns=[f"Coef: {c}" for c in model.classes_])
            df.insert(0, "Term", names)
            return df
        df = pd.DataFrame({"Term": names, "Coefficient": coef.ravel()})
        df["Meaning"] = ["change in outcome " + meanings.get(t, "") for t in names]
        return df.reindex(df["Coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    if hasattr(model, "feature_importances_"):
        df = pd.DataFrame({"Term": names, "Built-in importance": model.feature_importances_})
        return df.sort_values("Built-in importance", ascending=False).reset_index(drop=True)
    return None


def partial_dependence(pipe, X: pd.DataFrame, feature: str, task: str, labels, grid_points: int = 25, max_rows: int = 300):
    Xs = X.sample(min(len(X), max_rows), random_state=0) if len(X) > max_rows else X.copy()
    col = X[feature]
    if is_num(col):
        lo, hi = np.nanpercentile(col.astype(float), [2, 98])
        grid = np.unique(np.linspace(lo, hi, grid_points))
    else:
        grid = col.dropna().value_counts().index[:15].tolist()
    ys = []
    for v in grid:
        Xm = Xs.copy()
        Xm[feature] = v
        if task == "regression":
            ys.append(float(np.mean(pipe.predict(Xm))))
        elif labels and len(labels) == 2:
            ys.append(float(np.mean(pipe.predict_proba(Xm)[:, 1])))
        else:
            ys.append(np.mean(pipe.predict_proba(Xm), axis=0))
    return grid, ys


def what_if_contributions(pipe, X_train: pd.DataFrame, patient: pd.DataFrame, task: str, labels, class_index: int = 1):
    """How much does each of this patient's values move the prediction compared with a 'typical' value?"""
    def score(Xp):
        if task == "regression":
            return float(pipe.predict(Xp)[0])
        return float(pipe.predict_proba(Xp)[0, class_index])
    base = score(patient)
    rows = []
    for c in X_train.columns:
        typical = X_train[c].median() if is_num(X_train[c]) else X_train[c].mode().iloc[0]
        Xm = patient.copy()
        Xm[c] = typical
        rows.append({"Factor": c, "Patient value": patient[c].iloc[0], "Typical value": typical,
                     "Effect": base - score(Xm)})
    df = pd.DataFrame(rows)
    return base, df.reindex(df["Effect"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def threshold_table(y, proba, n=101) -> pd.DataFrame:
    rows = []
    for t in np.linspace(0, 1, n):
        m = binary_metrics(y, proba, t)
        rows.append({"Threshold": t, "Sensitivity": m["Sensitivity"], "Specificity": m["Specificity"],
                     "PPV": m["PPV"], "NPV": m["NPV"]})
    return pd.DataFrame(rows)


def youden_threshold(y, proba) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, proba)
    j = tpr - fpr
    t = thr[int(np.argmax(j))]
    return float(min(max(t, 0.0), 1.0))


def learning_curve_data(result, X_train, y_train, task, labels, cv_folds=5, seed=42):
    from sklearn.model_selection import learning_curve
    sc = scorers(task, len(labels) if labels else 2)
    metric = result["main"]
    cv = (StratifiedKFold(cv_folds, shuffle=True, random_state=seed) if task == "classification"
          else KFold(cv_folds, shuffle=True, random_state=seed))
    from sklearn.base import clone
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sizes, tr, te = learning_curve(clone(result["pipe"]), X_train, y_train, cv=cv, scoring=sc[metric],
                                       train_sizes=np.linspace(0.15, 1.0, 6), shuffle=True, random_state=seed,
                                       error_score=np.nan)
    return sizes, _flip(metric, tr), _flip(metric, te), metric


def decision_surface(key, task, X2: pd.DataFrame, y, params=None, seed=42, steps=120):
    """Train a model on just 2 numeric columns and predict on a grid (for 'what does it do to my data')."""
    pipe = make_pipeline(key, task, X2, params, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X2, y)
    a, b = X2.columns
    xa = X2[a].astype(float)
    xb = X2[b].astype(float)
    pa = (xa.max() - xa.min()) * 0.05 or 1
    pb = (xb.max() - xb.min()) * 0.05 or 1
    ga = np.linspace(xa.min() - pa, xa.max() + pa, steps)
    gb = np.linspace(xb.min() - pb, xb.max() + pb, steps)
    A, B = np.meshgrid(ga, gb)
    grid = pd.DataFrame({a: A.ravel(), b: B.ravel()})
    if task == "regression":
        Z = pipe.predict(grid).reshape(A.shape)
        classes = None
    else:
        proba = pipe.predict_proba(grid)
        classes = list(pipe.classes_)
        Z = proba[:, 1].reshape(A.shape) if len(classes) == 2 else np.argmax(proba, axis=1).reshape(A.shape)
    train_score = pipe.score(X2, y)
    return ga, gb, Z, classes, train_score, pipe
