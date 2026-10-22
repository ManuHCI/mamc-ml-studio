"""Write a stand-alone Python script that reproduces the analysis done in the app.

The script uses only pandas, numpy, scikit-learn, scipy and matplotlib, so a student can open it, read it,
run it, and modify it - the natural next step after the point-and-click app.
"""
from __future__ import annotations

import datetime as _dt
import json

from . import APP_NAME, APP_VERSION
from .models import MODELS, model_name, param_grid

HEADER = '''"""
Machine-learning analysis exported from {app} v{ver} on {date}
Dataset : {data}
Outcome : {target}  ({task}{pos})

HOW TO RUN
  1. Install Python 3.10+ and then:  pip install pandas numpy scikit-learn scipy matplotlib openpyxl
  2. Keep this file in the same folder as '{cleaned}' (and '{original}').
  3. Run:  python {script}
Results are printed and the charts are saved as PNG files next to this script.
Small differences from the app are possible with other scikit-learn versions.
"""
import inspect
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform  # noqa: F401  (used by random search)
from sklearn.calibration import CalibratedClassifierCV  # noqa: F401
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,  # noqa: F401
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge  # noqa: F401
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,  # noqa: F401
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold,  # noqa: F401
                                     cross_validate, train_test_split)
from sklearn.naive_bayes import GaussianNB  # noqa: F401
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # noqa: F401
from sklearn.neural_network import MLPClassifier, MLPRegressor  # noqa: F401
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR  # noqa: F401
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: F401
from sklearn.utils.class_weight import compute_sample_weight  # noqa: F401

warnings.filterwarnings("ignore")
'''

HELPERS = '''

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
import re

MISSING_TEXT = {"", "na", "n/a", "n.a.", "nan", "none", "nil", "null", "-", "--", "---", "?", ".",
                "not done", "nd", "missing", "not available", "#n/a", "#value!", "#div/0!"}


def to_number(series):
    """'45 yrs' -> 45, '27,5' -> 27.5, '1,234' -> 1234, 'NA' -> NaN."""
    def parse(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            return float(v)
        s = str(v).strip()
        if s.lower() in MISSING_TEXT:
            return np.nan
        if re.match(r"^\\s*[-+]?\\d{1,3}(,\\d{3})+(\\.\\d+)?\\s*$", s):
            return float(s.replace(",", ""))
        m = re.match(r"^\\s*([-+]?\\d+(?:[.,]\\d+)?)\\s*([a-zA-Z%/\\u00b5\\u00b0\\.]*\\s*[a-zA-Z/0-9]*)\\s*$", s)
        return float(m.group(1).replace(",", ".")) if m else np.nan
    return pd.Series([parse(v) for v in series], index=series.index, dtype="float64")


def as_text(v):
    if isinstance(v, float) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def prepare_X(X):
    """Numbers as float, categories as text (blank stays blank)."""
    X = X.copy()
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(float)
        else:
            X[c] = pd.Series([as_text(v) if not pd.isna(v) else np.nan for v in X[c]], index=X.index, dtype=object)
    return X


def make_preprocessor(X, scale):
    """Fill blanks (median / most common), scale numbers (optional), one-hot encode categories.
    Everything is learned from the training patients only, because it sits inside the Pipeline."""
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    parts = []
    if num:
        steps = [("fill_blanks", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
        parts.append(("numbers", Pipeline(steps), num))
    if cat:
        kw = "sparse_output" if "sparse_output" in inspect.signature(OneHotEncoder).parameters else "sparse"
        ohe = OneHotEncoder(handle_unknown="ignore", drop="if_binary", **{kw: False})
        parts.append(("categories", Pipeline([("fill_blanks", SimpleImputer(strategy="most_frequent")),
                                              ("one_hot", ohe)]), cat))
    return ColumnTransformer(parts, sparse_threshold=0, verbose_feature_names_out=False)


from sklearn.base import BaseEstimator, ClassifierMixin


class LogisticModel(ClassifierMixin, BaseEstimator):
    """Logistic regression with a simple L1/L2 switch that works on old and new scikit-learn versions."""

    def __init__(self, C=1.0, penalty="l2", class_weight=None, max_iter=5000, random_state=None):
        self.C, self.penalty, self.class_weight = C, penalty, class_weight
        self.max_iter, self.random_state = max_iter, random_state

    def fit(self, X, y, sample_weight=None):
        kw = dict(C=self.C, class_weight=self.class_weight, max_iter=self.max_iter, random_state=self.random_state,
                  solver="saga" if self.penalty == "l1" else "lbfgs")
        if inspect.signature(LogisticRegression).parameters["penalty"].default == "deprecated":
            kw["l1_ratio"] = 1.0 if self.penalty == "l1" else 0.0
        else:
            kw["penalty"] = self.penalty
        self.model_ = LogisticRegression(**kw).fit(X, y, sample_weight=sample_weight)
        self.classes_, self.coef_, self.intercept_ = self.model_.classes_, self.model_.coef_, self.model_.intercept_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
'''


def _pyval(v):
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, tuple):
        return repr(tuple(v))
    if isinstance(v, float):
        return repr(float(v))
    return repr(v)


def _estimator_code(key: str, task: str, params: dict, seed: int, balance: bool) -> tuple[str, str]:
    """Return (constructor code, prefix for grid parameter names)."""
    cls = task == "classification"
    cw = "'balanced'" if (balance and cls and MODELS[key]["balance"] == "class_weight") else "None"
    p = {k: _pyval(v) for k, v in params.items()}
    if key == "logreg":
        return f"LogisticModel(C={p['C']}, penalty={p['penalty']}, class_weight={cw}, random_state={seed})", "model__"
    if key == "linreg":
        return "LinearRegression()", "model__"
    if key == "ridge":
        return f"Ridge(alpha={p['alpha']}, random_state={seed})", "model__"
    if key == "lasso":
        return f"Lasso(alpha={p['alpha']}, max_iter=20000, random_state={seed})", "model__"
    if key == "tree":
        if cls:
            return (f"DecisionTreeClassifier(max_depth={p['max_depth']}, min_samples_leaf={p['min_samples_leaf']}, "
                    f"criterion={p['criterion']}, class_weight={cw}, random_state={seed})"), "model__"
        return (f"DecisionTreeRegressor(max_depth={p['max_depth']}, min_samples_leaf={p['min_samples_leaf']}, "
                f"random_state={seed})"), "model__"
    if key == "rf":
        name = "RandomForestClassifier" if cls else "RandomForestRegressor"
        extra = f", class_weight={cw}" if cls else ""
        return (f"{name}(n_estimators={p['n_estimators']}, max_depth={p['max_depth']}, max_features={p['max_features']}, "
                f"min_samples_leaf={p['min_samples_leaf']}{extra}, random_state={seed}, n_jobs=-1)"), "model__"
    if key == "gb":
        name = "HistGradientBoostingClassifier" if cls else "HistGradientBoostingRegressor"
        extra = f", class_weight={cw}" if cls else ""
        return (f"{name}(max_iter={p['max_iter']}, learning_rate={p['learning_rate']}, max_depth={p['max_depth']}, "
                f"min_samples_leaf={p['min_samples_leaf']}, l2_regularization={p['l2_regularization']}, "
                f"early_stopping=False{extra}, random_state={seed})"), "model__"
    if key == "svm":
        if cls:
            return (f"CalibratedClassifierCV(SVC(C={p['C']}, kernel={p['kernel']}, gamma={p['gamma']}, class_weight={cw}, "
                    f"random_state={seed}), method='sigmoid', cv=5, ensemble=False)"), "model__estimator__"
        return f"SVR(C={p['C']}, kernel={p['kernel']}, gamma={p['gamma']})", "model__"
    if key == "knn":
        name = "KNeighborsClassifier" if cls else "KNeighborsRegressor"
        return f"{name}(n_neighbors={p['n_neighbors']}, weights={p['weights']}, p={p['p']})", "model__"
    if key == "nb":
        return f"GaussianNB(var_smoothing={p['var_smoothing']})", "model__"
    if key == "mlp":
        name = "MLPClassifier" if cls else "MLPRegressor"
        return (f"{name}(hidden_layer_sizes={p['hidden_layer_sizes']}, alpha={p['alpha']}, "
                f"learning_rate_init={p['learning_rate_init']}, max_iter={p['max_iter']}, random_state={seed})"), "model__"
    raise KeyError(key)


def _dist_code(key: str, names: list, n_rows: int, prefix: str) -> str:
    """Python code for the random-search distributions (mirrors models.random_distributions)."""
    parts = []
    for p in MODELS[key]["params"]:
        nm = p["name"]
        if nm not in names:
            continue
        k = f"'{prefix}{nm}'"
        if p["kind"] == "logfloat":
            lo, hi = p["range"]
            parts.append(f"{k}: loguniform({lo!r}, {hi!r})")
        elif p["kind"] == "int":
            lo, hi = p["range"]
            if nm == "n_neighbors" and n_rows:
                hi = int(max(min(hi, n_rows * 0.6), 3))
            if p["default"] is None or (p["grid"] and None in p["grid"]):
                parts.append(f"{k}: list(range({int(lo)}, {int(hi) + 1})) + [None]")
            else:
                parts.append(f"{k}: randint({int(lo)}, {int(hi) + 1})")
        elif p["kind"] == "float":
            lo, hi = p["range"]
            parts.append(f"{k}: uniform({lo!r}, {hi - lo!r})")
        elif p["kind"] == "choice":
            parts.append(f"{k}: {list(p['options'])!r}")
    return "{" + ", ".join(parts) + "}"


def build_script(ss, original_file: str = "original_data.csv", cleaned_file: str = "cleaned_data.csv",
                 script_name: str = "ml_analysis.py") -> str:
    task, target, feats = ss["task"], ss["target"], list(ss["features"])
    sp = ss["split"]
    labels = sp["labels"]
    binary = task == "classification" and labels and len(labels) == 2
    seed, folds, metric = sp["seed"], sp["folds"], sp["metric"]
    pos = f", positive = '{labels[1]}'" if binary else ""
    out = [HEADER.format(app=APP_NAME, ver=APP_VERSION, date=_dt.date.today().isoformat(), data=ss.get("data_name"),
                         target=target, task=task, pos=pos, cleaned=cleaned_file, original=original_file, script=script_name)]
    out.append(HELPERS)

    # ---------------- data
    log_code = [e["code"] for e in ss.get("clean_log", []) if e.get("code")]
    out.append(f'''

# ---------------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------------
START_FROM_ORIGINAL = False   # True = start from the original file and repeat the cleaning steps below
if START_FROM_ORIGINAL:
    df = pd.read_csv({original_file!r})
    # ---- 2. The same cleaning steps that were applied in the app ----''')
    if log_code:
        out.append("\n".join("    " + c for c in log_code))
    else:
        out.append("    pass  # no cleaning steps were applied")
    out.append(f'''else:
    df = pd.read_csv({cleaned_file!r})   # the data exactly as they were after cleaning in the app

TARGET = {target!r}
FEATURES = {json.dumps(feats)}
df = df[df[TARGET].notna()].reset_index(drop=True)
X = prepare_X(df[FEATURES])''')
    if binary:
        out.append(f'''POSITIVE = {labels[1]!r}          # the 'disease present' group
y = (df[TARGET].map(as_text) == POSITIVE).astype(int).to_numpy()
LABELS = [{labels[0]!r}, POSITIVE]
print("Patients:", len(y), "| with", POSITIVE, ":", int(y.sum()))''')
    elif task == "classification":
        out.append('''y = np.asarray(df[TARGET].map(as_text), dtype=object)
print("Patients:", len(y)); print(pd.Series(y).value_counts())''')
    else:
        out.append('''y = df[TARGET].astype(float).to_numpy()
print("Patients:", len(y), "| mean outcome:", round(y.mean(), 3))''')

    strat = "y" if (task == "classification" and sp.get("stratify", True)) else "None"
    out.append(f'''

# ---------------------------------------------------------------------------
# 3. Split: {100 - sp['test_pct']}% training, {sp['test_pct']}% test (locked away until the end)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={sp['test_pct'] / 100!r}, random_state={seed},
                                                    stratify={strat})
cv = {"StratifiedKFold" if task == "classification" else "KFold"}(n_splits={folds}, shuffle=True, random_state={seed})''')

    # scoring
    sc_map = {"AUC": "roc_auc" if binary else "roc_auc_ovr", "Balanced accuracy": "balanced_accuracy", "Accuracy": "accuracy",
              "Sensitivity": "recall", "F1": "f1", "F1 (macro)": "f1_macro", "R2": "r2", "MAE": "neg_mean_absolute_error",
              "RMSE": "neg_root_mean_squared_error"}
    out.append(f'SCORING = {sc_map.get(metric, "roc_auc")!r}   # used to compare settings ({metric})')

    # ---------------- models
    out.append('''

# ---------------------------------------------------------------------------
# 4. Models (with the settings / tuning used in the app)
# ---------------------------------------------------------------------------
models = {}''')
    for k, r in ss["results"].items():
        bal = r.get("balance", False)
        mode = r["mode"]
        base_params = dict(r["params"])
        code, prefix = _estimator_code(k, task, base_params, seed, bal)
        scale = MODELS[k]["scale"]
        nm = model_name(k, task)
        out.append(f"\n# {nm}: " + {"default": "starting settings", "manual": "settings chosen by the user",
                                      "grid": "settings tuned by grid search (best values shown below)",
                                      "random": "settings tuned by random search (best values shown below)"}[mode])
        out.append(f"pipe = Pipeline([('prep', make_preprocessor(X_train, scale={scale})), ('model', {code})])")
        sw = ""
        if bal and task == "classification" and MODELS[k]["balance"] == "sample_weight":
            sw = ", model__sample_weight=compute_sample_weight('balanced', y_train)"
        if mode in ("grid", "random") and r.get("tuning") is not None:
            tuned = [c for c in r["tuning"].columns if not c.startswith(("CV ", "Train ", "Rank"))]
            pre = prefix or "model__"
            if mode == "grid":
                g = param_grid(k, len(sp["y_train"]))
                grid = "{" + ", ".join(f"'{pre}{t}': {g[t]!r}" for t in tuned if t in g) + "}"
                out.append(f"search = GridSearchCV(pipe, {grid}, scoring=SCORING, cv=cv)")
            else:
                dist = _dist_code(k, tuned, len(sp["y_train"]), pre)
                n_iter = len(r["tuning"])
                out.append(f"search = RandomizedSearchCV(pipe, {dist}, n_iter={n_iter}, scoring=SCORING, cv=cv, "
                           f"random_state={seed})")
            out.append(f"search.fit(X_train, y_train{sw})")
            out.append(f"print({nm!r}, 'best settings:', search.best_params_)")
            out.append(f"models[{nm!r}] = search.best_estimator_")
        else:
            if sw:
                out.append(f"models[{nm!r}] = pipe.fit(X_train, y_train{sw})")
            else:
                out.append(f"models[{nm!r}] = pipe.fit(X_train, y_train)")

    # ---------------- evaluation
    if binary:
        out.append('''

# ---------------------------------------------------------------------------
# 5. Final exam on the test set
# ---------------------------------------------------------------------------
rows = []
plt.figure(figsize=(6, 6))
for name, model in models.items():
    cvres = cross_validate(model, X_train, y_train, cv=cv, scoring="roc_auc")
    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    rows.append({"Model": name, "CV AUC": cvres["test_score"].mean(), "Test AUC": roc_auc_score(y_test, p),
                 "Sensitivity": tp / (tp + fn), "Specificity": tn / (tn + fp),
                 "PPV": tp / (tp + fp) if tp + fp else np.nan, "NPV": tn / (tn + fn) if tn + fn else np.nan,
                 "Accuracy": (tp + tn) / len(y_test)})
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC {roc_auc_score(y_test, p):.3f})")
plt.plot([0, 1], [0, 1], ":", color="grey")
plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity"); plt.title("ROC curves (test set)"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig("roc_curves.png", dpi=200); plt.close()
results = pd.DataFrame(rows)
print(results.round(3).to_string(index=False))''')
    elif task == "classification":
        out.append('''

# ---------------------------------------------------------------------------
# 5. Final exam on the test set
# ---------------------------------------------------------------------------
rows = []
for name, model in models.items():
    proba = model.predict_proba(X_test)
    pred = model.classes_[proba.argmax(axis=1)]
    rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                 "Balanced accuracy": balanced_accuracy_score(y_test, pred),
                 "F1 (macro)": f1_score(y_test, pred, average="macro"),
                 "AUC (one-vs-rest)": roc_auc_score(y_test, proba, multi_class="ovr", labels=list(model.classes_))})
    print(name); print(pd.DataFrame(confusion_matrix(y_test, pred, labels=model.classes_),
                                    index=[f"true {c}" for c in model.classes_], columns=[f"pred {c}" for c in model.classes_]))
results = pd.DataFrame(rows)
print(results.round(3).to_string(index=False))''')
    else:
        out.append('''

# ---------------------------------------------------------------------------
# 5. Final exam on the test set
# ---------------------------------------------------------------------------
rows = []
for name, model in models.items():
    pred = model.predict(X_test)
    rows.append({"Model": name, "R2": r2_score(y_test, pred), "MAE": mean_absolute_error(y_test, pred),
                 "RMSE": float(np.sqrt(mean_squared_error(y_test, pred)))})
    plt.figure(figsize=(5, 5)); plt.scatter(y_test, pred, alpha=.7)
    lo, hi = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], ":", color="grey"); plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(name)
    plt.tight_layout(); plt.savefig(f"pred_vs_actual_{name.replace(' ', '_')}.png", dpi=200); plt.close()
results = pd.DataFrame(rows)
print(results.round(3).to_string(index=False))''')

    score = "roc_auc" if binary else ("roc_auc_ovr" if task == "classification" else "r2")
    out.append(f'''

# ---------------------------------------------------------------------------
# 6. Which factors matter? (permutation importance on the test set)
# ---------------------------------------------------------------------------
for name, model in models.items():
    imp = permutation_importance(model, X_test, y_test, scoring={score!r}, n_repeats=10, random_state={seed})
    tab = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
    plt.figure(figsize=(6, 0.35 * len(tab) + 1.2)); tab.plot.barh(color="#2a78d6")
    plt.title(f"Permutation importance - {{name}}"); plt.xlabel("Drop in score when the factor is scrambled")
    plt.tight_layout(); plt.savefig(f"importance_{{name.replace(' ', '_')}}.png", dpi=200); plt.close()
    print("\\n", name, "- most important factors:"); print(tab.sort_values(ascending=False).head(10).round(4).to_string())
''')
    return "\n".join(out)
