"""Engine tests: every demo dataset, every model, every tuning mode, the exported Python script and the Word report.

Run:  python tests/run_engine_tests.py          (about 5-10 minutes)
"""
import io
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mlstudio import clustering, sample_size  # noqa: E402
from mlstudio.advisor import data_profile, detect_task, recommend, usable_features  # noqa: E402
from mlstudio.codegen import build_script  # noqa: E402
from mlstudio.data_check import auto_clean, health_check, read_table  # noqa: E402
from mlstudio.demo_data import DEMO_DATASETS, load_demo  # noqa: E402
from mlstudio.models import MODELS, TASK_MODELS  # noqa: E402
from mlstudio.report import build_docx  # noqa: E402
from mlstudio.trainer import (evaluate, learning_curve_data, model_weights, partial_dependence,  # noqa: E402
                              permutation_importance_table, prepare_xy, split_data, train_model, what_if_contributions)

PASS, FAIL = [], []


def ok(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))


# ----------------------------------------------------------------------------- sample size
r = sample_size.riley_binary(0.174, 24, 0.288)
ok("Riley BMJ 2020 worked example gives n = 662", r["n"] == 662, r["n"])

# ----------------------------------------------------------------------------- reading files
df = load_demo("messy")
buf = io.BytesIO()
df.to_excel(buf, index=False)
x = read_table(buf.getvalue(), "t.xlsx")
ok("Excel round-trip keeps all patients", x.dropna(how="all").shape == df.dropna(how="all").shape, x.shape)
x = read_table(df.to_csv(index=False, sep=";").encode("utf-8"), "t.csv")
ok("semicolon CSV detected", x.shape == df.shape, x.shape)

# ----------------------------------------------------------------------------- data check
m = load_demo("messy")
keys = {i.key for i in health_check(m, "Diabetes_5yr", "classification")}
for k in ["blank_rows", "dup_rows", "num_text:Age", "cat_spell:Sex", "miss_text:Fasting_glucose", "id_col:Patient_ID",
          "leak:On_metformin", "const_col:Hospital", "empty_col:Remarks", "miss_target:Diabetes_5yr",
          "implaus:Systolic BP (mmHg)"]:
    ok(f"health check finds {k}", k in keys)
clean, log = auto_clean(m, "Diabetes_5yr", "classification")
ok("auto-clean leaves no high-severity issue",
   not [i for i in health_check(clean, "Diabetes_5yr", "classification") if i.severity == "high"])
ok("auto-clean removes the leaky column", "On_metformin" not in clean.columns)
ok("auto-clean converts Age to numbers", pd.api.types.is_numeric_dtype(clean["Age"]))
ok("auto-clean merges Sex spellings", set(clean["Sex"].dropna()) == {"Male", "Female"}, set(clean["Sex"].dropna()))
for k, info in DEMO_DATASETS.items():
    d = load_demo(k)
    t = info["target"]
    try:
        health_check(d, t, info["task"] if info["task"] != "clustering" else None)
        ok(f"health check runs on {k}", True)
    except Exception as e:  # noqa: BLE001
        ok(f"health check runs on {k}", False, repr(e))

# ----------------------------------------------------------------------------- every model, every task
CASES = [("diabetes", "classification"), ("anaemia", "classification"), ("stay", "regression"), ("breast", "classification")]
sessions = {}
for ds, task in CASES:
    d = load_demo(ds)
    t = DEMO_DATASETS[ds]["target"]
    d, lg = auto_clean(d, t, task)
    feats = usable_features(d, t)
    prof = data_profile(d, feats, t, task)
    rec = recommend(prof, task, "balanced")
    ok(f"{ds}: advisor recommends 3+ models incl. baseline",
       sum(r["recommended"] for r in rec) >= 3 and any(r["recommended"] and r["key"] in ("logreg", "linreg") for r in rec))
    X, y, labels = prepare_xy(d, feats, t, task, DEMO_DATASETS[ds]["positive"])
    Xtr, Xte, ytr, yte = split_data(X, y, task, 0.2, 42, True)
    nc = len(labels) if labels else 2
    results = {}
    for key in TASK_MODELS[task]:
        t0 = time.time()
        try:
            res = train_model(key, task, Xtr, ytr, mode="default", cv_folds=3, balance=True, n_classes=nc)
            evaluate(res, Xtr, ytr, Xte, yte, task, labels, n_boot=60)
            main = "AUC" if task == "classification" else "R2"
            v = res["test"][main]
            good = np.isfinite(v) and (v > 0.6 if task == "classification" else v > 0.2)
            ok(f"{ds}: {key} trains (test {main} {v:.3f})", good, f"{time.time() - t0:.1f}s")
            results[key] = res
        except Exception as e:  # noqa: BLE001
            ok(f"{ds}: {key} trains", False, repr(e)[:300])
    # tuning modes on two models
    for key, mode in [("tree", "grid"), ("knn", "random"), ("logreg" if task == "classification" else "ridge", "grid"),
                      ("svm", "random")]:
        try:
            res = train_model(key, task, Xtr, ytr, mode=mode, n_iter=6, cv_folds=3, n_classes=nc,
                              tune_params=[p["name"] for p in MODELS[key]["params"] if not p["advanced"]][:2])
            evaluate(res, Xtr, ytr, Xte, yte, task, labels, n_boot=60)
            ok(f"{ds}: {key} {mode} search", res["tuning"] is not None and len(res["tuning"]) > 1, str(res["params"]))
            results[key] = res
        except Exception as e:  # noqa: BLE001
            ok(f"{ds}: {key} {mode} search", False, repr(e)[:300])
    # explanations
    try:
        k0 = "rf"
        perm = permutation_importance_table(results[k0], Xte, yte, task, labels, n_repeats=3)
        grid, ys = partial_dependence(results[k0]["pipe"], Xtr, perm["Factor"].iloc[0], task, labels)
        base, wi = what_if_contributions(results[k0]["pipe"], Xtr, Xte.iloc[[0]], task, labels)
        lc = learning_curve_data(results["tree"], Xtr, ytr, task, labels, 3)
        w = model_weights(results["logreg" if task == "classification" else "linreg"], task, labels)
        ok(f"{ds}: importance / PDP / what-if / learning curve / weights", len(perm) == len(feats) and len(grid) > 1
           and len(wi) == len(feats) and len(lc[0]) > 2 and w is not None)
    except Exception as e:  # noqa: BLE001
        ok(f"{ds}: explanations", False, repr(e)[:300])
    sessions[ds] = dict(task=task, target=t, features=feats, positive=labels[1] if labels and len(labels) == 2 else None,
                        split=dict(X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte, labels=labels, test_pct=20, seed=42,
                                   stratify=True, folds=3, metric="AUC" if task == "classification" else "R2", balance=True),
                        results={k: results[k] for k in list(results)[:4] + ["tree", "knn", "svm"] if k in results},
                        clean_log=lg, data_name=DEMO_DATASETS[ds]["title"], df=d, df_raw=load_demo(ds))

# ----------------------------------------------------------------------------- report + exported script
for ds, ss in sessions.items():
    try:
        b = build_docx(ss)
        from docx import Document
        doc = Document(io.BytesIO(b))
        ok(f"{ds}: Word report builds ({len(b) // 1024} kB, {len(doc.inline_shapes)} images)", len(doc.inline_shapes) >= 2)
    except Exception as e:  # noqa: BLE001
        ok(f"{ds}: Word report builds", False, repr(e)[:300])
    try:
        script = build_script(ss)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "ml_analysis.py").write_text(script, encoding="utf-8")
            ss["df"].to_csv(Path(tmp, "cleaned_data.csv"), index=False)
            ss["df_raw"].to_csv(Path(tmp, "original_data.csv"), index=False)
            pr = subprocess.run([sys.executable, "ml_analysis.py"], cwd=tmp, capture_output=True, text=True, timeout=900)
            pngs = [f for f in os.listdir(tmp) if f.endswith(".png")]
            ok(f"{ds}: exported Python script runs ({len(pngs)} charts)", pr.returncode == 0, pr.stderr[-800:])
            if pr.returncode == 0 and ss["task"] == "classification" and ss["positive"]:
                # the script's test AUC should equal the app's for the logistic regression
                mm = re.search(r"Logistic Regression\s+[\d.]+\s+([\d.]+)", pr.stdout)
                app_auc = ss["results"]["logreg"]["test"]["AUC"]
                ok(f"{ds}: script reproduces the app's logistic-regression AUC", mm and abs(float(mm.group(1)) - app_auc) < 0.002,
                   f"script {mm.group(1) if mm else None} vs app {app_auc:.3f}")
            if ds == "diabetes":
                # starting from the original file must give the same cleaned data
                s2 = script.replace("START_FROM_ORIGINAL = False", "START_FROM_ORIGINAL = True")
                Path(tmp, "ml_analysis.py").write_text(s2, encoding="utf-8")
                pr2 = subprocess.run([sys.executable, "ml_analysis.py"], cwd=tmp, capture_output=True, text=True, timeout=900)
                ok(f"{ds}: script also runs from the original file", pr2.returncode == 0, pr2.stderr[-500:])
    except Exception as e:  # noqa: BLE001
        ok(f"{ds}: exported Python script runs", False, repr(e)[:300])

# messy -> original-file path must reproduce the cleaning exactly
m = load_demo("messy")
clean, lg = auto_clean(m, "Diabetes_5yr", "classification")
code = "\n".join(e["code"] for e in lg if e.get("code"))
with tempfile.TemporaryDirectory() as tmp:
    m.to_csv(Path(tmp, "o.csv"), index=False)
    helpers = build_script.__globals__["HELPERS"]
    prog = "import numpy as np, pandas as pd\n" + helpers + "\ndf = pd.read_csv('o.csv')\n" + code + \
           "\ndf.to_csv('c.csv', index=False)\n"
    Path(tmp, "p.py").write_text(prog, encoding="utf-8")
    pr = subprocess.run([sys.executable, "p.py"], cwd=tmp, capture_output=True, text=True)
    if pr.returncode == 0:
        c2 = pd.read_csv(Path(tmp, "c.csv"))
        c1 = pd.read_csv(io.StringIO(clean.to_csv(index=False)))
        same = c1.shape == c2.shape and c1.fillna(-999).astype(str).equals(c2.fillna(-999).astype(str))
        ok("cleaning code reproduces the app's cleaned data exactly", same, f"{c1.shape} vs {c2.shape}")
    else:
        ok("cleaning code runs", False, pr.stderr[-600:])

# ----------------------------------------------------------------------------- clustering
g = load_demo("groups")
feats = [c for c in g.columns if c != "Subject"]
Xall, Z = clustering.prepare(g, feats)
ks, inert, sils = clustering.scan_k(Z)
ok("clustering finds k = 3 in the metabolic demo", ks[int(np.nanargmax(sils))] == 3, str(dict(zip(ks, np.round(sils, 2)))))
for meth in ["kmeans", "hier", "dbscan"]:
    lab, sil = clustering.run(meth, Z, k=3, eps=1.2, min_samples=5)
    ok(f"clustering {meth} runs", len(lab) == len(g), f"silhouette {sil:.2f}")
means, z = clustering.profile(g, Xall, lab)
ok("cluster profile table", len(means) >= 1)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", *FAIL, sep="\n  ")
    sys.exit(1)
