import io

import numpy as np
import pandas as pd
import streamlit as st

from mlstudio import plots, ui
from mlstudio.data_check import is_num, read_table
from mlstudio.trainer import clean_X, what_if_contributions

ui.page_head("7", "Predict a new patient", "Enter a patient's values and see the model's prediction - and the reasons for it.")
ui.need_results()
ss = st.session_state
res, sp, task = ss.results, ss.split, ss.task
labels = sp["labels"]
binary = task == "classification" and labels and len(labels) == 2
Xtr = sp["X_train"]

ui.demo_banner("Change the fasting glucose or BMI of the patient below and press **Predict**. The chart underneath shows "
               "which of this patient's values pushed the risk up (red) or down (blue).", "app_pages/report.py",
               "Demo: continue to 8. Report & Python code")
ui.box("warn", "This is a teaching and research tool. Do not use its predictions to make decisions about real patients "
               "unless the model has been properly validated (ideally on patients from another time or hospital).",
       "For learning only")

main = "AUC" if task == "classification" else "R2"
order = sorted(res, key=lambda k: -np.nan_to_num(res[k]["test"].get(main, -1), nan=-1))
key = st.selectbox("Model", order, format_func=lambda k: res[k]["name"] + f"  (test {main} {res[k]['test'].get(main, np.nan):.3f})")
r = res[key]

tab1, tab2 = st.tabs([":material/person: One patient", ":material/table_view: Many patients (from a file)"])
with tab1:
    Xte = sp["X_test"]
    starts = ["typical"] + list(range(len(Xte)))
    start = st.selectbox("Start from", starts, index=1 if len(Xte) else 0,
                         format_func=lambda s: "A typical patient (median / most common values)" if s == "typical"
                         else f"Test patient no. {s + 1} (truly: {labels[int(sp['y_test'][s])] if binary else sp['y_test'][s]})",
                         help="Pick one of the test patients the model has never seen, or a 'typical' patient, then change the values.")
    base_row = None if start == "typical" else Xte.iloc[start]
    with st.form("patient"):
        st.markdown("**Patient's values** - change any of them and press Predict")
        vals = {}
        cols = st.columns(3)
        for i, c in enumerate(Xtr.columns):
            with cols[i % 3]:
                s = Xtr[c]
                if is_num(s):
                    lo, hi, med = float(np.nanmin(s)), float(np.nanmax(s)), float(np.nanmedian(s))
                    v0 = med if base_row is None or pd.isna(base_row[c]) else float(base_row[c])
                    integer = np.allclose(s.dropna(), s.dropna().round())
                    vals[c] = st.number_input(c, value=int(round(v0)) if integer else round(v0, 2),
                                              step=1 if integer else 0.1, help=f"Training patients ranged from {lo:g} to {hi:g}.",
                                              key=f"in_{c}_{start}")
                else:
                    opts = s.dropna().value_counts().index.tolist()
                    v0 = None if base_row is None or pd.isna(base_row[c]) else base_row[c]
                    vals[c] = st.selectbox(c, opts, index=opts.index(v0) if v0 in opts else 0, key=f"in_{c}_{start}")
        st.form_submit_button("Predict", type="primary", icon=":material/psychology:")
    patient = clean_X(pd.DataFrame([vals]))
    # warn about values outside the training range
    outside = [c for c in Xtr.columns if is_num(Xtr[c]) and (patient[c].iloc[0] < np.nanmin(Xtr[c]) or patient[c].iloc[0] > np.nanmax(Xtr[c]))]
    if outside:
        ui.box("warn", f"{', '.join(outside)} {'is' if len(outside) == 1 else 'are'} outside the range seen in the training "
                       "patients. Models are unreliable when they extrapolate.")
    pipe = r["pipe"]
    if task == "regression":
        pred = float(pipe.predict(patient)[0])
        c1, c2 = st.columns([1, 2])
        with c1:
            ui.tile(f"Predicted {ss.target}", f"{pred:.2f}", f"Typical error (MAE) +/- {r['test']['MAE']:.2f}")
        base, wi = what_if_contributions(pipe, Xtr, patient, task, labels)
        with c2:
            ui.show(plots.whatif(wi, "outcome units"), key="wi_reg")
    else:
        proba = pipe.predict_proba(patient)[0]
        classes = labels if binary else list(pipe.classes_)
        probs = {str(c): float(p) for c, p in zip(classes, proba)}
        if binary:
            thr = ss.get("thresholds", {}).get(key, 0.5)
            p1 = probs[labels[1]]
            call = labels[1] if p1 >= thr else labels[0]
            c1, c2 = st.columns([1, 2])
            with c1:
                ui.tile(f"Predicted probability of '{labels[1]}'", f"{p1 * 100:.1f}%",
                        f"cut-off {thr:.2f} -> classified as '{call}'")
            with c2:
                ui.show(plots.probability_bar(probs), key="pb")
            idx = 1
        else:
            best = max(probs, key=probs.get)
            c1, c2 = st.columns([1, 2])
            with c1:
                ui.tile("Most likely group", best, f"probability {probs[best] * 100:.1f}%")
            with c2:
                ui.show(plots.probability_bar(probs), key="pbm")
            idx = list(pipe.classes_).index(best)
        base, wi = what_if_contributions(pipe, Xtr, patient, task, labels, class_index=idx)
        target_name = labels[1] if binary else best
        wi_show = wi.copy()
        wi_show["Effect"] = wi_show["Effect"] * 100
        ui.show(plots.whatif(wi_show, f"percentage points of P({target_name})"), key="wi_cls")
    if wi["Effect"].abs().max() < 1e-9:
        ui.box("note", "This patient has the typical value for every factor, so nothing pushes the prediction away from the "
                       "average. Change some values (or start from a test patient) to see their effect.")
    ui.simple("For each factor, the software replaces this patient's value with a typical value (median or most common) "
              "and checks how much the prediction changes. Long red bars = this finding raises the prediction; blue = lowers "
              "it. This is a simple 'what-if' explanation, not proof of cause.", "How the reasons are worked out")

with tab2:
    st.markdown("Upload an Excel/CSV file with the **same column names** as the predictors: "
                + ", ".join(f"`{c}`" for c in Xtr.columns))
    tmpl = io.BytesIO()
    pd.DataFrame(columns=Xtr.columns).to_excel(tmpl, index=False)
    st.download_button("Download an empty template", tmpl.getvalue(), "prediction_template.xlsx", icon=":material/download:")
    up = st.file_uploader("File with new patients", type=["xlsx", "xls", "csv"], key="batch")
    if up is not None:
        try:
            new = read_table(up.getvalue(), up.name)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the file: {e}")
            st.stop()
        missing = [c for c in Xtr.columns if c not in new.columns]
        if missing:
            ui.box("bad", f"These predictor columns are missing from the file: {', '.join(missing)}")
        else:
            Xn = clean_X(new[Xtr.columns])
            out = new.copy()
            if task == "regression":
                out[f"Predicted {ss.target}"] = r["pipe"].predict(Xn)
            else:
                pr = r["pipe"].predict_proba(Xn)
                classes = labels if binary else list(r["pipe"].classes_)
                for j, c in enumerate(classes):
                    out[f"P({c})"] = pr[:, j].round(4)
                if binary:
                    thr = ss.get("thresholds", {}).get(key, 0.5)
                    out["Predicted"] = np.where(pr[:, 1] >= thr, labels[1], labels[0])
                else:
                    out["Predicted"] = np.asarray(classes, dtype=object)[pr.argmax(axis=1)]
            st.dataframe(out, width="stretch", hide_index=True)
            buf = io.BytesIO()
            out.to_excel(buf, index=False)
            st.download_button("Download predictions (Excel)", buf.getvalue(), "predictions.xlsx", icon=":material/download:")
ui.next_step("app_pages/report.py", "Next: 8. Report & Python code")
ui.footer()
