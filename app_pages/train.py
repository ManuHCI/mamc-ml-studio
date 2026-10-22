import math
import time

import numpy as np
import pandas as pd
import streamlit as st

from mlstudio import RANDOM_SEED, plots, ui
from mlstudio.models import MODELS, TASK_MODELS, describe_params, fmt_value, model_name, param_grid
from mlstudio.trainer import METRIC_HELP, evaluate, prepare_xy, scorers, split_data, train_model
from mlstudio.widgets import param_widget

ui.page_head("5", "Train & tune", "Split the patients, let the models learn, and find their best settings.")
ui.need_setup()
ss = st.session_state
df, target, task, feats = ss.df, ss.target, ss.task, ss.features

ui.demo_banner("Keep the 80 : 20 split and 5-fold cross-validation. For **Random Forest**, choose **Find the best "
               "settings automatically (grid search)** to see tuning in action. Then press **Train**. It takes a few seconds.",
               None)

X, y, labels = prepare_xy(df, feats, target, task, ss.positive)
n = len(y)
n_classes = len(labels) if labels else 2

# ---------------------------------------------------------------- 1. split
st.subheader("Step 1 - Split the patients into a training set and a test set")
c1, c2, c3 = st.columns([2, 1, 1])
test_pct = c1.slider("Patients kept aside for the final test (%)", 10, 40, int(ss.get("test_pct", 20)), 5,
                     help="20-30% is usual. The test patients are NEVER shown to the model while it learns.")
seed = c2.number_input("Random seed", 0, 99999, int(ss.get("seed", RANDOM_SEED)),
                       help="The 'shuffle number'. The same seed always gives the same split, so results can be reproduced.")
strat = c3.checkbox("Stratified split", value=True, disabled=task != "classification",
                    help="Keeps the same proportion of each outcome group in the training and test sets.")
ss.test_pct, ss.seed = test_pct, seed
n_test = int(math.ceil(n * test_pct / 100))
ui.show(plots.split_bar(n - n_test, n_test), key="splitbar")
ui.simple("Think of an exam. The model studies the **training** patients (with their answers). The **test** patients are "
          "locked in a cupboard and used only once, at the very end, to see how the model does on patients it has never "
          "seen. If we tested on the same patients it studied, a model that simply memorised them would look perfect.",
          "Why split?")

# ---------------------------------------------------------------- 2. pipeline
st.subheader("Step 2 - What happens to the data inside every model")
num = [c for c in feats if pd.api.types.is_numeric_dtype(X[c])]
cat = [c for c in feats if c not in num]
ui.flow([("1. Fill blanks", "median / most common value"), ("2. Scale numbers", "to mean 0, SD 1 *"),
         ("3. Categories -> 0/1", f"{len(cat)} column(s)"), ("4. Algorithm", "learns the pattern"),
         ("5. Prediction", "probability or number")])
st.caption(f"{len(num)} number predictor(s), {len(cat)} category predictor(s). * Scaling is used only by models that "
           "need it (logistic/linear regression, SVM, KNN, neural network); trees do not care about scale.")
ui.simple("All three preparation steps are **learned from the training patients only** (e.g. the median used to fill a "
          "blank glucose comes from training patients). Then exactly the same recipe is applied to the test patients. "
          "Doing it this way prevents information from the test set 'leaking' into training.", "Important")

# ---------------------------------------------------------------- 3. CV
st.subheader("Step 3 - Cross-validation inside the training set")
c1, c2 = st.columns([1, 1.6])
with c1:
    folds = st.slider("Number of rounds (k-fold)", 3, 10, int(ss.get("folds", 5)),
                      help="5 or 10 is usual. More rounds = more reliable estimate but slower.")
    ss.folds = folds
    sc = scorers(task, n_classes)
    metric_list = list(sc)
    default_metric = ss.get("main_metric") if ss.get("main_metric") in metric_list else metric_list[0]
    main_metric = st.selectbox("Score used to judge and tune the models", metric_list, index=metric_list.index(default_metric),
                               help="AUC is the usual choice for yes/no outcomes; R² or MAE for numbers.")
    ss.main_metric = main_metric
    st.caption(METRIC_HELP.get(main_metric, ""))
with c2:
    ui.show(plots.kfold_diagram(folds), key="kfold")
ui.simple(f"Before the final exam, the model takes **{folds} practice tests**. The training patients are cut into {folds} "
          f"parts. In each round the model learns from {folds - 1} parts and is checked on the remaining part. The average "
          "of the rounds tells us how well the model is likely to do on new patients, and is used to compare settings "
          "(tuning) - without ever touching the test set.", "Cross-validation in simple words")

# ---------------------------------------------------------------- 4. settings
st.subheader("Step 4 - Settings (hyperparameters) of each model")
ui.simple("Settings are chosen by YOU before learning starts (e.g. how deep a tree may grow). They are like the dose of a "
          "drug: too little and the model is too simple (underfitting), too much and it memorises (overfitting). "
          "**Tuning** = trying many settings and keeping the one with the best cross-validation score - a dose-finding study.",
          "What is a hyperparameter?")

chosen = [k for k in ss.get("chosen_models", []) if k in TASK_MODELS[task]]
if not chosen:
    from mlstudio.advisor import data_profile, recommend
    chosen = [r["key"] for r in recommend(data_profile(df, feats, target, task), task, ss.get("priority", "balanced"))
              if r["recommended"]]
models = st.multiselect("Models to train", TASK_MODELS[task], default=chosen,
                        format_func=lambda k: model_name(k, task))
balance = False
if task == "classification":
    vc = pd.Series(y).value_counts()
    imb = vc.min() / vc.max()
    balance = st.checkbox("Balance the outcome groups (give extra weight to the smaller group)", value=bool(imb < 0.5),
                          help="Recommended when one group is much smaller. Not available for K-Nearest Neighbours.")

MODES = {"default": "Use the recommended starting values", "manual": "Set them myself",
         "grid": "Find the best settings automatically (grid search)",
         "random": "Find the best settings automatically (random search)"}
plan = {}
total_fits = 0
for k in models:
    m = MODELS[k]
    with st.expander(f"{model_name(k, task)}", expanded=len(models) <= 2):
        if not m["params"]:
            st.caption("This model has no settings to tune - it is trained directly.")
            plan[k] = dict(mode="default")
            total_fits += folds + 1
            continue
        mode = st.radio("Settings", list(MODES), format_func=MODES.get, key=f"mode_{k}", horizontal=False,
                        index=list(MODES).index(ss.get(f"mode_{k}", "default")) if ss.get(f"mode_{k}") in MODES else 0)
        entry = dict(mode=mode)
        if mode == "default":
            st.markdown("Starting values: " + "; ".join(f"{p['label']} = **{fmt_value(p, p['default'])}**" for p in m["params"]))
            total_fits += folds + 1
        elif mode == "manual":
            params = {}
            for p in m["params"]:
                params[p["name"]] = param_widget(p, f"hp_{k}_{p['name']}")
                st.caption(f"Low: {p['low']}  |  High: {p['high']}" if p["low"] else "")
            entry["params"] = params
            total_fits += folds + 1
        elif mode == "grid":
            grid = param_grid(k, n)
            names = list(grid)
            pmap = {p["name"]: p for p in m["params"]}
            default_tune = [p["name"] for p in m["params"] if not p["advanced"] and p["name"] in grid][:2]
            tune = st.multiselect("Settings to tune", names, default=default_tune, key=f"tune_{k}",
                                  format_func=lambda x: pmap[x]["label"])
            combos = 1
            for t in tune:
                st.markdown(f"- **{pmap[t]['label']}**: try " + ", ".join(fmt_value(pmap[t], v) for v in grid[t]))
                combos *= len(grid[t])
            fits = combos * folds + 1
            st.caption(f"{combos} combination(s) x {folds} rounds = **{fits} model fits**.")
            entry["tune"] = tune
            total_fits += fits
        else:
            pmap = {p["name"]: p for p in m["params"]}
            names = [p["name"] for p in m["params"] if not p["advanced"]]
            tune = st.multiselect("Settings to tune", names, default=names, key=f"rtune_{k}",
                                  format_func=lambda x: pmap[x]["label"])
            n_iter = st.slider("Number of random combinations to try", 5, 60, 15, key=f"niter_{k}")
            for t in tune:
                p = pmap[t]
                rng = f"{fmt_value(p, p['range'][0])} to {fmt_value(p, p['range'][1])}" if p.get("range") else \
                    ", ".join(fmt_value(p, o) for o in p["options"])
                st.markdown(f"- **{p['label']}**: anywhere in {rng}")
            st.caption(f"{n_iter} combinations x {folds} rounds = **{n_iter * folds + 1} model fits**.")
            entry.update(tune=tune, n_iter=n_iter)
            total_fits += n_iter * folds + 1
        plan[k] = entry
        slow = (k == "svm" and n > 3000) or (k == "mlp" and n > 5000) or (k in ("rf", "gb") and entry["mode"] in ("grid", "random") and n > 3000)
        if slow:
            ui.box("warn", "This model may take several minutes on a dataset of this size.")

if total_fits > 400:
    ui.box("warn", f"About {total_fits} model fits are planned. This may take a few minutes - reduce the settings to tune "
                   "or the number of rounds if your computer is slow.")

# ---------------------------------------------------------------- train
st.subheader("Step 5 - Train")
sig = (ss.data_version, target, tuple(feats), ss.positive, task, test_pct, seed, strat)
if ss.results and ss.split_sig != sig:
    ui.box("note", "The data, outcome, predictors or split have changed since the models below were trained. Training "
                   "again will start a fresh comparison.")
go = st.button(f"Train {len(models)} model(s)", type="primary", disabled=not models, icon=":material/play_arrow:")

if go:
    Xtr, Xte, ytr, yte = split_data(X, y, task, test_pct / 100, seed, strat)
    if ss.split_sig != sig:
        ss.results = {}
    ss.split = dict(X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte, labels=labels, test_pct=test_pct, seed=seed,
                    stratify=strat, folds=folds, metric=main_metric, balance=balance)
    ss.split_sig = sig
    prog = st.progress(0.0, text="Starting...")
    with st.status("Training ...", expanded=True) as status:
        for i, k in enumerate(models):
            e = plan[k]
            name = model_name(k, task)
            st.write(f"**{name}**")
            what = {"default": f"{folds}-fold cross-validation with the starting settings",
                    "manual": f"{folds}-fold cross-validation with your settings",
                    "grid": f"grid search over the chosen settings, each checked by {folds}-fold cross-validation",
                    "random": f"random search ({e.get('n_iter', 15)} tries), each checked by {folds}-fold cross-validation"}[e["mode"]]
            st.write(f"- learning from {len(ytr)} training patients: {what} ...")
            prog.progress(i / len(models), text=f"{name}: learning ...")
            t0 = time.time()
            try:
                r = train_model(k, task, Xtr, ytr, mode=e["mode"], params=e.get("params"), tune_params=e.get("tune"),
                                n_iter=e.get("n_iter", 15), cv_folds=folds, main_metric=main_metric, seed=seed,
                                balance=balance, n_classes=n_classes)
                st.write(f"- final model fitted on all {len(ytr)} training patients; now the final exam on "
                         f"{len(yte)} locked-away test patients ...")
                evaluate(r, Xtr, ytr, Xte, yte, task, labels, seed=seed)
                ss.results[k] = r
                st.write(f"- done in {time.time() - t0:.1f} s: test {r['verdict']['metric']} = "
                         f"{r['verdict']['test']:.3f} ({r['verdict']['label']})")
            except Exception as ex:  # noqa: BLE001
                st.error(f"{name} failed: {ex}")
        prog.progress(1.0, text="Finished")
        status.update(label=f"Finished training {len(ss.results)} model(s)", state="complete", expanded=False)

# ---------------------------------------------------------------- summary
if ss.results and ss.split_sig == sig:
    res = ss.results
    sp = ss.split
    m = sp["metric"]
    st.subheader("Training summary")
    rows = []
    for k, r in res.items():
        tm = r["test"].get(m, np.nan)
        rows.append({"Model": r["name"], "Settings": "starting values" if r["mode"] == "default" else
                     {"manual": "set by you", "grid": "grid search", "random": "random search"}[r["mode"]],
                     f"Training {m}": r["cv"][m]["train"], f"Cross-validation {m} (mean)": r["cv"][m]["mean"],
                     "CV SD": r["cv"][m]["sd"], f"Test {m}": tm, "Verdict": r["verdict"]["label"],
                     "Time (s)": round(r["fit_time"], 1)})
    tab = pd.DataFrame(rows)
    st.dataframe(tab, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in tab.columns
                                if c.startswith(("Training", "Cross", "CV", "Test"))})
    ui.simple(f"**Training {m}**: how well the model does on the patients it studied (always flattering). "
              f"**Cross-validation**: the average of the {sp['folds']} practice tests. **Test**: the final exam on "
              f"{len(sp['y_test'])} patients never seen before. A big gap between training and the other two = overfitting.",
              "How to read this table")
    ui.show(plots.cv_folds(res, m), key="cvfolds")

    tuned = {k: r for k, r in res.items() if r["tuning"] is not None}
    if tuned:
        st.subheader("Tuning results")
        k = st.selectbox("Model", list(tuned), format_func=lambda x: tuned[x]["name"])
        r = tuned[k]
        tab = r["tuning"]
        pcols = [c for c in tab.columns if not c.startswith(("CV ", "Train ", "Rank"))]
        st.markdown("**Best settings found:** " + "; ".join(describe_params(k, r["params"])))
        plabel = {p["name"]: p["label"] for p in MODELS[k]["params"]}
        if r["mode"] == "grid" and pcols:
            c1, c2 = st.columns(2)
            p1 = c1.selectbox("Show the effect of", pcols, key="tp1", format_func=lambda c: plabel.get(c, c))
            others = [c for c in pcols if c != p1]
            p2 = c2.selectbox("Separate lines for", ["(average over the others)"] + others, key="tp2",
                              format_func=lambda c: plabel.get(c, c))
            ui.show(plots.tuning_plot(tab, p1, m, None if p2.startswith("(") else p2, label=plabel.get(p1)), key="tuneplot")
            ui.simple("The blue line is the score on unseen patients (cross-validation); the dashed orange line is the score "
                      "on the training patients. Where the orange line keeps rising but the blue line falls, the model has "
                      "started to memorise - that is overfitting. The best setting is where the blue line is highest.",
                      "How to read this chart")
        elif pcols:
            p1 = st.selectbox("Show the trials for", pcols, key="tp3", format_func=lambda c: plabel.get(c, c))
            ui.show(plots.tuning_scatter(tab, p1, m), key="tunescatter")
        with st.expander("All combinations tried (table)"):
            st.dataframe(tab, hide_index=True, width="stretch")

    with st.expander("Show what was learned in the preparation steps (from the training patients only)"):
        k0 = list(res)[0]
        prep = res[k0]["pipe"].named_steps["prep"]
        for name, trans, cols in prep.transformers_:
            if name == "numbers":
                imp = trans.named_steps["fill_blanks"]
                d = pd.DataFrame({"Predictor": cols, "Median used to fill blanks": imp.statistics_})
                if "scale" in trans.named_steps:
                    sc_ = trans.named_steps["scale"]
                    d["Mean (for scaling)"] = sc_.mean_
                    d["SD (for scaling)"] = sc_.scale_
                st.dataframe(d, hide_index=True, width="stretch")
            if name == "categories":
                imp = trans.named_steps["fill_blanks"]
                oh = trans.named_steps["one_hot"]
                st.dataframe(pd.DataFrame({"Predictor": cols, "Most common value (fills blanks)": imp.statistics_,
                                           "Categories found": [", ".join(map(str, c)) for c in oh.categories_]}),
                             hide_index=True, width="stretch")
    ui.next_step("app_pages/results.py", "Next: 6. Results")
ui.footer()
