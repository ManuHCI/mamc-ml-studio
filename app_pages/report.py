import io
import zipfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from mlstudio import APP_NAME, APP_VERSION, ui
from mlstudio.codegen import build_script
from mlstudio.report import build_docx, methods_text, results_text

ui.page_head("8", "Report & Python code", "Download a Word report, the matching Python code, and your trained models.")
ui.need_results()
ss = st.session_state
sp, res, task = ss.split, ss.results, ss.task
labels = sp["labels"]

ui.demo_banner("Download the Word report - it contains the cleaning log, a methods paragraph you can adapt for a thesis, "
               "the results with 95% confidence intervals and the charts. The Python script reproduces everything you did "
               "with point-and-click, so you can see the code behind each step.", None)

st.subheader("Word report")
c1, c2 = st.columns([1, 2])
if c1.button("Prepare the Word report", type="primary", icon=":material/description:"):
    with st.spinner("Writing the report and drawing the charts..."):
        ss.report_docx = build_docx(ss)
        ss.report_sig = ss.split_sig
if ss.get("report_docx") and ss.get("report_sig") == ss.split_sig:
    c2.download_button("Download report (.docx)", ss.report_docx, file_name="ML_analysis_report.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       icon=":material/download:")
st.caption("Contents: data and cleaning log - methods - model settings - results table with 95% CIs - ROC / calibration / "
           "confusion matrix - important predictors - fit verdicts - limitations - references.")

st.subheader("Methods and results text")
st.text_area("Methods (copy and adapt)", methods_text(ss), height=230)
st.text_area("Main result (copy and adapt)", results_text(ss), height=90)
ui.box("note", "Report prediction-model studies following **TRIPOD+AI** (Collins et al., BMJ 2024). Key items: source of data, "
               "participants, outcome definition, predictors, sample size, missing data, model building and tuning, "
               "performance measures with uncertainty (discrimination and calibration), and validation. This app performs "
               "**internal** validation (held-out test set); external validation on new patients is the next step before "
               "any clinical use.", "Reporting guideline")

st.subheader("Python code")
ui.simple("Everything you did by clicking can be written as a short Python program. The script below loads your data, "
          "repeats the same split, pipeline, settings / tuning and evaluation, and saves the charts. Open it in any editor "
          "(IDLE, VS Code, Jupyter) to see how each step looks in code - a good way to start learning Python for ML.",
          "Why export code?")
script = build_script(ss)
st.code(script, language="python", line_numbers=True, height=420)

zbuf = io.BytesIO()
with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("ml_analysis.py", script)
    z.writestr("original_data.csv", ss.df_raw.to_csv(index=False))
    z.writestr("cleaned_data.csv", ss.df.to_csv(index=False))
    z.writestr("README.txt", f"Exported from {APP_NAME} v{APP_VERSION}.\n\n1. pip install pandas numpy scikit-learn scipy "
                             "matplotlib openpyxl\n2. python ml_analysis.py\n\nThe script reads cleaned_data.csv. Set "
                             "START_FROM_ORIGINAL = True inside it to start from original_data.csv and repeat the cleaning.\n")
c1, c2 = st.columns(2)
c1.download_button("Download the analysis package (.zip: script + data)", zbuf.getvalue(), file_name="ml_analysis_package.zip",
                   icon=":material/folder_zip:", type="primary")
c2.download_button("Download only the script (.py)", script.encode("utf-8"), file_name="ml_analysis.py", icon=":material/code:")

st.subheader("Other downloads")
c1, c2 = st.columns(2)
rows = pd.DataFrame(index=sp["X_test"].index)
yte = sp["y_test"]
rows["Actual"] = [labels[int(v)] for v in yte] if (task == "classification" and len(labels) == 2) else yte
for k, r in res.items():
    if task == "regression":
        rows[f"{r['name']}: predicted"] = r["pred_test"]
    elif len(labels) == 2:
        rows[f"{r['name']}: P({labels[1]})"] = np.round(r["proba_test"], 4)
    else:
        rows[f"{r['name']}: predicted"] = np.asarray(r["classes"], dtype=object)[np.argmax(r["proba_test"], axis=1)]
out = pd.concat([sp["X_test"], rows], axis=1)
buf = io.BytesIO()
out.to_excel(buf, index=False)
c1.download_button("Test-set predictions (Excel)", buf.getvalue(), file_name="test_predictions.xlsx", icon=":material/table_view:")
mbuf = io.BytesIO()
joblib.dump({"models": {r["name"]: r["pipe"] for r in res.values()}, "features": ss.features, "target": ss.target,
             "task": task, "labels": labels, "app": f"{APP_NAME} {APP_VERSION}"}, mbuf)
c2.download_button("Trained models (.joblib)", mbuf.getvalue(), file_name="trained_models.joblib", icon=":material/save:",
                   help="Load in Python with joblib.load(). Needs scikit-learn and the 'mlstudio' folder of this app on the path.")
ui.footer()
