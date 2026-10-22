import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from mlstudio import plots, ui
from mlstudio.models import MODELS, describe_params
from mlstudio.trainer import (METRIC_HELP, binary_metrics, bootstrap_ci, learning_curve_data, model_weights,
                              partial_dependence, permutation_importance_table, quality_word, threshold_table,
                              youden_threshold)

ui.page_head("6", "Results", "How good are the models on patients they have never seen - explained in clinical terms.")
ui.need_results()
ss = st.session_state
res, sp, task = ss.results, ss.split, ss.task
labels = sp["labels"]
binary = task == "classification" and labels and len(labels) == 2
yte = sp["y_test"]

ui.demo_banner("Compare the models in the table and the ROC chart. Then, under **One model in detail**, move the "
               "**cut-off** slider: a lower cut-off catches more future diabetics (higher sensitivity) but raises more "
               "false alarms (lower specificity) - exactly like choosing a cut-off for a screening test.",
               "app_pages/predict.py", "Demo: continue to 7. Predict a new patient")


def ci_text(r, k):
    lo, hi = r.get("ci", {}).get(k, (np.nan, np.nan))
    return f"95% CI {lo:.3f} - {hi:.3f}" if np.isfinite(lo) else ""


# ================================================================ comparison
st.subheader("Model comparison (test set)")
if binary:
    keys = ["AUC", "Sensitivity", "Specificity", "PPV", "NPV", "Balanced accuracy", "Accuracy", "Brier score"]
elif task == "classification":
    keys = ["AUC", "Balanced accuracy", "Accuracy", "F1 (macro)"]
else:
    keys = ["R2", "MAE", "RMSE", "Median absolute error"]
rows = []
for k, r in res.items():
    row = {"Model": r["name"]}
    for m in keys:
        row[m] = r["test"].get(m, np.nan)
    row["Verdict"] = r["verdict"]["label"]
    rows.append(row)
comp = pd.DataFrame(rows)
main = "AUC" if task == "classification" and comp["AUC"].notna().all() else ("Balanced accuracy" if task == "classification" else "R2")
st.dataframe(comp.rename(columns={"R2": "R\u00b2"}), hide_index=True, width="stretch",
             column_config={("R\u00b2" if m == "R2" else m): st.column_config.NumberColumn(format="%.3f") for m in keys})
if binary:
    st.caption("Sensitivity, specificity, PPV and NPV are at the default cut-off of 0.5; change it below. Brier score: "
               "average squared error of the predicted risks (lower = better).")

if len(res) > 1:
    ci_map = {r["name"]: r["ci"].get(main, (np.nan, np.nan)) for r in res.values()}
    ui.show(plots.compare_bars(comp, main, ci_map), key="cmp")
    best = comp.sort_values(main, ascending=main in ("MAE", "RMSE")).iloc[0]
    bk = [k for k, r in res.items() if r["name"] == best["Model"]][0]
    blo, bhi = res[bk]["ci"].get(main, (np.nan, np.nan))
    overlap = [r["name"] for k, r in res.items() if k != bk and np.isfinite(blo) and r["test"].get(main, np.nan) >= blo]
    shown_main = "R\u00b2" if main == "R2" else main
    msg = f"**{best['Model']}** has the best test {shown_main} ({best[main]:.3f})."
    if overlap:
        msg += (f" However, {', '.join(overlap)} {'is' if len(overlap) == 1 else 'are'} inside its 95% confidence interval, "
                "so the difference may be due to chance in this test set. When models perform similarly, prefer the "
                "simpler, more explainable one.")
    ui.box("note", msg, "Which model is best?")

if binary:
    c1, c2 = st.columns(2)
    curves = []
    cal = []
    pr = []
    for r in res.values():
        fpr, tpr, _ = roc_curve(yte, r["proba_test"])
        curves.append((r["name"], fpr, tpr, r["test"]["AUC"]))
        pv, rv, _ = precision_recall_curve(yte, r["proba_test"])
        pr.append((r["name"], rv, pv, average_precision_score(yte, r["proba_test"])))
        nb = int(min(10, max(3, len(yte) // 15)))
        fp_, mp_ = calibration_curve(yte, r["proba_test"], n_bins=nb, strategy="quantile")
        cal.append((r["name"], mp_, fp_, None))
    with c1:
        ui.show(plots.roc(curves), key="roc_all")
    with c2:
        ui.show(plots.calibration(cal), key="cal_all")
    ui.simple("**ROC curve**: each point is one possible cut-off; the curve shows the trade-off between catching the "
              "disease (sensitivity) and false alarms (1 - specificity). The closer it hugs the top-left corner, the better; "
              "the area under it is the **AUC**. **Calibration**: if the model says '30% risk', about 30% of such patients "
              "should really have the outcome - points near the diagonal mean the risks can be believed.",
              "How to read these charts")

# ================================================================ detail
st.divider()
st.subheader("One model in detail")
key = st.selectbox("Model", list(res), format_func=lambda k: res[k]["name"])
r = res[key]
v = r["verdict"]
st.markdown(f"<div class='verdict {v['code']}'><b>{ui.esc(v['label'])}.</b> {ui.esc(v['text'])}</div>", unsafe_allow_html=True)
st.caption("Settings used: " + ("; ".join(describe_params(key, r["params"])) or "none") +
           (" (found by tuning)" if r["mode"] in ("grid", "random") else ""))

tabs = st.tabs([":material/fact_check: Performance", ":material/insights: Which factors matter",
                ":material/show_chart: Over/under-fitting", ":material/account_tree: Inside the model"])

# ---------------------------------------------------------------- performance
with tabs[0]:
    if binary:
        proba = r["proba_test"]
        pos, neg = labels[1], labels[0]
        c1, c2, c3 = st.columns([2, 1, 1])
        tk = f"thr_{key}"
        if tk not in st.session_state:
            st.session_state[tk] = 0.5
        if c2.button("Best balance (Youden)", help="The cut-off where sensitivity + specificity is highest."):
            st.session_state[tk] = float(min(max(round(youden_threshold(yte, proba), 2), 0.01), 0.99))
        if c3.button("Reset to 0.5"):
            st.session_state[tk] = 0.5
        thr = c1.slider(f"Cut-off: call a patient '{pos}' when the predicted probability is at least", 0.01, 0.99,
                        step=0.01, key=tk)
        ss.thresholds = dict(ss.get("thresholds", {}), **{key: thr})
        m = binary_metrics(yte, proba, thr)
        ci = r["ci"] if abs(thr - 0.5) < 1e-9 else bootstrap_ci(lambda a, b: binary_metrics(a, b, thr), yte, proba,
                                                                 ["Sensitivity", "Specificity", "PPV", "NPV", "Accuracy"], 300)
        ci["AUC"] = r["ci"]["AUC"]
        cols = st.columns(3)
        tiles = [
            ("Sensitivity", f"Of 100 patients who truly are '{pos}', the model catches {m['Sensitivity'] * 100:.0f}."),
            ("Specificity", f"Of 100 patients who truly are '{neg}', it correctly clears {m['Specificity'] * 100:.0f}."),
            ("AUC", f"Discrimination is {quality_word('AUC', m['AUC'])}."),
            ("PPV", f"When it says '{pos}', it is right {m['PPV'] * 100:.0f}% of the time." if np.isfinite(m["PPV"]) else "No positive predictions."),
            ("NPV", f"When it says '{neg}', it is right {m['NPV'] * 100:.0f}% of the time." if np.isfinite(m["NPV"]) else ""),
            ("Accuracy", f"{m['Accuracy'] * 100:.0f}% of all test patients classified correctly."),
        ]
        for i, (name, say) in enumerate(tiles):
            with cols[i % 3]:
                val = m[name]
                lo, hi = ci.get(name, (np.nan, np.nan))
                ui.tile(name, f"{val:.3f}" if name == "AUC" else ui.pct(val), f"95% CI {lo:.2f} - {hi:.2f}" if np.isfinite(lo) else "", say)
                st.write("")
        st.caption("PPV and NPV depend on how common the outcome is in your data "
                   f"({100 * np.mean(yte):.0f}% in the test set). In a population with a different prevalence they will differ. "
                   f"LR+ = {m['LR+']:.2f}, LR- = {m['LR-']:.2f}.")
        c1, c2 = st.columns([1, 1.1])
        with c1:
            ui.show(plots.confusion(np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]]), [neg, pos]), key="cm_bin")
            ui.simple(f"This is the familiar 2 x 2 table of a diagnostic test. **TP** = '{pos}' correctly flagged, **FN** = "
                      f"'{pos}' missed, **FP** = false alarm, **TN** = correctly cleared.", "The confusion matrix")
        with c2:
            fpr, tpr, _ = roc_curve(yte, proba)
            ui.show(plots.roc([(r["name"], fpr, tpr, m["AUC"])], point=(1 - m["Specificity"], m["Sensitivity"])), key="roc_one")
        ui.show(plots.threshold_plot(threshold_table(yte, proba), thr), key="thr_plot")
        ui.simple("For **screening** (do not miss disease) choose a lower cut-off: sensitivity rises, specificity falls. For "
                  "**confirming** a diagnosis (avoid false alarms) choose a higher cut-off. The model stays the same - only "
                  "the decision rule changes.", "Choosing a cut-off")
        with st.expander("Precision-recall curve (useful when the positive group is small)"):
            pv, rv, _ = precision_recall_curve(yte, proba)
            ui.show(plots.pr_curve([(r["name"], rv, pv, average_precision_score(yte, proba))], float(np.mean(yte))), key="pr_one")
    elif task == "classification":
        t = r["test"]
        cols = st.columns(4)
        for col, name in zip(cols, ["Accuracy", "Balanced accuracy", "F1 (macro)", "AUC"]):
            with col:
                ui.tile(name, f"{t[name]:.3f}", ci_text(r, name), "")
        classes = r["classes"]
        ui.show(plots.confusion(t["confusion"], classes, binary=False), key="cm_multi")
        ui.simple("Rows = the true group, columns = what the model said. The diagonal (top-left to bottom-right) holds the "
                  "correct answers; anything off the diagonal shows which groups the model confuses with each other.",
                  "The confusion matrix")
        st.dataframe(t["per_class"], hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                    ["Sensitivity (recall)", "Specificity", "PPV (precision)", "F1"]})
        st.caption("Each group is judged as 'this group vs all the others'.")
    else:
        t = r["test"]
        cols = st.columns(4)
        for col, name, say in zip(cols, ["R2", "MAE", "RMSE", "Median absolute error"],
                                  [f"Explains {max(t['R2'], 0) * 100:.0f}% of the differences between patients ({quality_word('R2', t['R2'])}).",
                                   f"On average the prediction is off by {t['MAE']:.2f} (outcome units).",
                                   "Like MAE but big misses count more.", "Half of the predictions are closer than this."]):
            with col:
                ui.tile("R²" if name == "R2" else name, f"{t[name]:.3f}", ci_text(r, name), say)
        c1, c2 = st.columns(2)
        with c1:
            ui.show(plots.pred_vs_actual(yte, r["pred_test"], r["name"]), key="pva")
        with c2:
            ui.show(plots.residuals(yte, r["pred_test"]), key="resid")
        ui.simple("In a good model the dots hug the diagonal line, and the errors scatter evenly around zero without a "
                  "pattern. A funnel or curve in the error plot means the model misses something systematic.",
                  "How to read these charts")

# ---------------------------------------------------------------- importance
with tabs[1]:
    if "perm" not in r:
        with st.spinner("Shuffling each factor to measure its importance..."):
            r["perm"] = permutation_importance_table(r, sp["X_test"], yte, task, labels, seed=sp["seed"])
    perm = r["perm"]
    ui.show(plots.importance(perm, "Importance", "Factor", "SD",
                             title=f"Permutation importance: drop in {perm['metric'].iloc[0]} when the factor is scrambled",
                             xt=f"Decrease in {perm['metric'].iloc[0]}"), key="perm")
    ui.simple("To find out how much the model relies on a factor, we **scramble** that column in the test set (as if the "
              "values belonged to random other patients) and see how much the score drops. A big drop = the model "
              "depends on that factor. Near zero (or negative) = the model does not really use it. This works for every "
              "type of model. Importance is not the same as cause.", "How is importance measured?")
    w = model_weights(r, task, labels)
    if w is not None:
        if "Odds ratio" in w:
            st.markdown("**Odds ratios from the logistic regression**")
            ui.show(plots.odds_ratio_plot(w, labels[1]), key="or")
            st.dataframe(w[["Term", "Odds ratio", "Meaning", "Direction"]], hide_index=True, width="stretch",
                         column_config={"Odds ratio": st.column_config.NumberColumn(format="%.2f")})
            ui.box("note", "Numbers were scaled, so odds ratios are **per 1 standard deviation** increase. These come from a "
                           "regularised model built for prediction - for p-values and 95% CIs of adjusted odds ratios, run a "
                           "classical binary logistic regression (e.g. in MAMC BioStat).")
        elif "Coefficient" in w:
            st.markdown("**Coefficients**")
            st.dataframe(w, hide_index=True, width="stretch", column_config={"Coefficient": st.column_config.NumberColumn(format="%.3f")})
        elif "Built-in importance" in w:
            with st.expander("Built-in (impurity) importance of the tree model"):
                st.dataframe(w, hide_index=True, width="stretch")
                st.caption("How much each 0/1 or number column improved the splits in the trees. It can favour columns "
                           "with many different values; prefer the permutation importance above.")
    st.markdown("**How does the prediction change with one factor?** (partial dependence)")
    X_tr = sp["X_train"]
    f = st.selectbox("Factor", list(perm["Factor"]), key=f"pdp_{key}")
    grid, ys = partial_dependence(r["pipe"], X_tr, f, task, labels)
    if task == "regression":
        ui.show(plots.pdp(grid, ys, f, "Average predicted value"), key="pdp")
    elif binary:
        ui.show(plots.pdp(grid, ys, f, f"Average predicted P({labels[1]})"), key="pdp")
    else:
        ui.show(plots.pdp(grid, ys, f, "Average predicted probability", classes=r["classes"]), key="pdp")
    ui.simple(f"Every patient's {f} is set to each value along the x-axis while everything else stays as it was, and the "
              "predictions are averaged. The curve shows the model's average response to that one factor.", "How it is made")

# ---------------------------------------------------------------- fitting
with tabs[2]:
    vv = r["verdict"]
    c = st.columns(3)
    with c[0]:
        ui.tile(f"Training {vv['metric']}", f"{vv['train']:.3f}", "", "Patients the model studied")
    with c[1]:
        ui.tile(f"Cross-validation {vv['metric']}", f"{vv['cv']:.3f}", f"SD {r['cv'][vv['metric']]['sd']:.3f}" if vv['metric'] in r['cv'] else "",
                "Practice tests")
    with c[2]:
        ui.tile(f"Test {vv['metric']}", f"{vv['test']:.3f}", ci_text(r, vv["metric"]), "Final exam - never seen")
    ui.simple("**Overfitting** = memorising: excellent on the training patients, clearly worse on new ones (like a student "
              "who learned the answers of last year's paper by heart). **Underfitting** = too simple: poor everywhere. "
              "A good model scores about the same on all three.", "Overfitting and underfitting")
    if st.button("Draw the learning curve", icon=":material/show_chart:"):
        with st.spinner("Re-training the model on growing portions of the training set..."):
            r["lc"] = learning_curve_data(r, sp["X_train"], sp["y_train"], task, labels, sp["folds"], sp["seed"])
    if "lc" in r:
        sizes, tr, te, mm = r["lc"]
        ui.show(plots.learning_curve(sizes, tr, te, mm), key="lc")
        ui.simple("If the two lines are still far apart at the right end, the model is overfitting and **more patients** "
                  "would probably help. If both lines are low and close together, the model is too simple - more patients "
                  "will not help; try a more flexible model or better predictors.", "How to read the learning curve")

# ---------------------------------------------------------------- inside
with tabs[3]:
    model = r["pipe"].named_steps["model"]
    if key == "tree":
        from sklearn.tree import export_text, plot_tree
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(r["pipe"].named_steps["prep"].get_feature_names_out())
        maxd = max(1, model.get_depth())
        depth = st.slider("Show the tree down to depth", 1, maxd, min(2, maxd),
                          help="Deeper levels become small; use the text rules below to read the whole tree.") if maxd > 1 else 1
        fig, ax = plt.subplots(figsize=(min(5 + 4.2 * 2 ** (depth - 1), 30), 1.6 + 1.9 * depth), dpi=110)
        cls_names = [str(c) for c in (labels if binary else getattr(model, "classes_", []))] if task == "classification" else None
        plot_tree(model, max_depth=depth, feature_names=names, class_names=cls_names, filled=True, rounded=True,
                  impurity=False, proportion=True, fontsize=11, ax=ax)
        st.pyplot(fig, width="stretch")
        ui.simple("Start at the top box and follow the questions: go **left if the answer is true**, right if false. The "
                  "colour shows which outcome dominates in each box; 'value' shows the proportion of patients in each "
                  "group. Numbers were not scaled, so cut-offs are in the original units. A question such as "
                  "'Family_history_Yes <= 0.5' simply means 'Family history is not Yes' (categories are coded 1 = yes, 0 = no).",
                  "How to read the tree")
        with st.expander("The same tree as text rules"):
            st.code(export_text(model, feature_names=names, max_depth=6, decimals=2))
    elif key == "gb" and hasattr(model, "staged_predict_proba") and task == "classification":
        from sklearn.metrics import roc_auc_score
        Xt_tr = r["pipe"].named_steps["prep"].transform(sp["X_train"])
        Xt_te = r["pipe"].named_steps["prep"].transform(sp["X_test"])
        if binary:
            tr = [roc_auc_score(sp["y_train"], p[:, 1]) for p in model.staged_predict_proba(Xt_tr)]
            te = [roc_auc_score(yte, p[:, 1]) for p in model.staged_predict_proba(Xt_te)]
            ui.show(plots.staged_curve(tr, te, "AUC"), key="staged")
            ui.simple("Each new small tree corrects the remaining errors. The training score keeps climbing; the test score "
                      "levels off - after that point, more trees only add memorisation.", "Boosting, tree by tree")
        else:
            st.info("The tree-by-tree chart is shown for two-group outcomes.")
    elif key == "gb" and task == "regression":
        from sklearn.metrics import r2_score
        Xt_tr = r["pipe"].named_steps["prep"].transform(sp["X_train"])
        Xt_te = r["pipe"].named_steps["prep"].transform(sp["X_test"])
        tr = [r2_score(sp["y_train"], p) for p in model.staged_predict(Xt_tr)]
        te = [r2_score(yte, p) for p in model.staged_predict(Xt_te)]
        ui.show(plots.staged_curve(tr, te, "R²"), key="staged_r")
    elif key == "mlp" and hasattr(model, "loss_curve_"):
        ui.show(plots.loss_curve(model.loss_curve_), key="loss")
        ui.simple(f"The network studied the training patients {len(model.loss_curve_)} times (epochs). After each pass the "
                  "weights were nudged to reduce the error (back-propagation).", "How a neural network learns")
    elif key == "rf":
        st.markdown(f"This forest contains **{len(model.estimators_)} trees**. Each was grown on a different bootstrap sample "
                    "of the training patients and could only look at a random subset of factors at each question. The final "
                    "answer is the average vote of all the trees.")
        st.caption("Tip: train a single Decision Tree to see what one member of the forest looks like.")
    elif key == "knn":
        st.markdown(f"The model simply stores the {len(sp['y_train'])} training patients. For each new patient it looks up the "
                    f"**{model.n_neighbors} most similar** ones and takes their {'vote' if task == 'classification' else 'average'}.")
    elif key == "nb":
        st.markdown("The model learned, for each outcome group, the average and spread of every factor, plus how common each "
                    "group is (the pre-test probability).")
        if hasattr(model, "theta_"):
            names = list(r["pipe"].named_steps["prep"].get_feature_names_out())
            st.dataframe(pd.DataFrame(model.theta_.T, index=names, columns=[str(c) for c in (labels if binary else model.classes_)]).round(3),
                         width="stretch")
            st.caption("Average of each (scaled / 0-1) factor in each outcome group.")
    elif key in ("logreg", "linreg", "ridge", "lasso"):
        st.markdown("The whole model is the list of weights shown in **Which factors matter** - a risk score you could "
                    "compute with a calculator.")
        b0 = np.ravel(model.intercept_)
        st.caption(f"Intercept (starting value of the score): {', '.join(f'{x:.3f}' for x in b0)}")
    elif key == "svm":
        st.markdown("The SVM is defined by the 'support vectors' - the borderline training patients that fix the position "
                    "of the boundary. Its probabilities come from an additional calibration step (Platt scaling).")
    st.markdown("**The algorithm in brief:** " + MODELS[key]["one_line"])

ui.next_step("app_pages/predict.py", "Next: 7. Predict a new patient")
ui.footer()
