import io

import pandas as pd
import streamlit as st

from mlstudio import ui
from mlstudio.advisor import detect_task
from mlstudio.data_check import SEV_LABEL, apply_fix, auto_clean, health_check

ui.page_head("2", "Check & clean data", "The Data Health Check looks for the mistakes that spoil machine-learning models.")
ui.need_data()
ss = st.session_state
df = ss.df

if ss.demo and ss.clean_log:
    ui.demo_banner("Well done - the sheet is clean. Look at the **cleaning log** at the bottom: every change is recorded "
                   "(and shown as Python code), so you can report exactly what was done.", "app_pages/explore.py",
                   "Demo: continue to 3. Explore data")
else:
    ui.demo_banner("Below are the problems found in the demo sheet. Read a few of them - each one says **what** was found, "
                   "**why it matters** and offers a fix. Notice the column **On_metformin**: it is only known *after* "
                   "diabetes is diagnosed, so it would let the model 'cheat' (data leakage). Then click **Apply all "
                   "recommended fixes**.", None)

# optional outcome -> extra checks
cols = ["(not chosen yet)"] + list(df.columns)
idx = cols.index(ss.target) if ss.target in cols else 0
c1, c2 = st.columns([1.3, 1])
choice = c1.selectbox("Which column is the outcome you want to predict? (optional here - enables extra checks)", cols,
                      index=idx, help="For example Diabetes_5yr. Leave it if you only want to find patient groups.")
if choice != "(not chosen yet)" and choice != ss.target:
    ss.target = choice
    ss.task = detect_task(df[choice])[0]
    ss.features = []
    st.rerun()
if choice == "(not chosen yet)" and ss.target:
    ss.target = None
    ss.task = None
    st.rerun()
if ss.target:
    c2.markdown(f"<br><span class='small'>Outcome: <b>{ui.esc(ss.target)}</b> &rarr; {ui.esc(ss.task)}</span>",
                unsafe_allow_html=True)

with st.spinner("Checking every column..."):
    issues = health_check(df, ss.target, ss.task, ss.dismissed)

counts = {s: sum(1 for i in issues if i.severity == s) for s in ("high", "medium", "low", "info")}
t = st.columns(4)
for col, (s, lab) in zip(t, [("high", "Must fix"), ("medium", "Should fix"), ("low", "Worth checking"), ("info", "For information")]):
    with col:
        ui.tile(lab, str(counts[s]))

actionable = [i for i in issues if i.recommended != "keep"]
st.write("")
b1, b2, b3 = st.columns([1.3, 1, 1])
if b1.button(f"Apply all recommended fixes ({len(actionable)})", type="primary", disabled=not actionable,
             icon=":material/auto_fix_high:", width="stretch"):
    with st.spinner("Cleaning..."):
        new, log = auto_clean(df, ss.target, ss.task)
    ui.update_data(new, log)
    st.toast(f"{len(log)} fix(es) applied")
    st.rerun()
if b2.button("Undo everything (back to the original file)", disabled=not ss.clean_log, icon=":material/undo:", width="stretch"):
    ss.df = ss.df_raw.copy()
    ss.clean_log = []
    ss.dismissed = set()
    ss.data_version += 1
    ss.results = {}
    ss.split = None
    st.rerun()

if not issues:
    ui.box("good", "No problems found. Your data look ready for machine learning.", "All clear")


def render_issue(issue):
    with st.container(border=True):
        st.markdown(f"<span class='sev {issue.severity}'>{SEV_LABEL[issue.severity]}</span><b>{ui.esc(issue.title)}</b>",
                    unsafe_allow_html=True)
        st.markdown(ui.esc(issue.detail))
        st.markdown(f"<div class='small'><b>Why it matters:</b> {ui.esc(issue.why)}</div>", unsafe_allow_html=True)
        labels = {a: (lab + ("  (recommended)" if a == issue.recommended else "")) for a, lab in issue.options}
        if len(issue.options) == 1:
            return
        k = f"{issue.key}|{ss.data_version}"
        cc = st.columns([3, 1.4, 1])
        act = cc[0].radio("What should be done?", [a for a, _ in issue.options], format_func=labels.get,
                          index=[a for a, _ in issue.options].index(issue.recommended), key="r" + k, horizontal=False,
                          label_visibility="collapsed")
        extra = None
        if act == "rename":
            extra = cc[1].text_input("New name", issue.payload.get("suggest", ""), key="t" + k)
        if cc[2].button("Apply", key="b" + k, icon=":material/check:", width="stretch"):
            if act == "keep":
                ss.dismissed = set(ss.dismissed) | {issue.key}
                ss.clean_log = ss.clean_log + [{"text": f"Checked and kept: {issue.title}", "code": None, "key": issue.key}]
                st.rerun()
            new, entry = apply_fix(df, issue, act, extra)
            ui.update_data(new, [entry])
            st.rerun()


groups = [("high", True), ("medium", True), ("low", False), ("info", False)]
for sev, expanded in groups:
    items = [i for i in issues if i.severity == sev]
    if not items:
        continue
    if sev == "info":
        miss = [i for i in items if i.kind == "missing"]
        other = [i for i in items if i.kind != "missing"]
        with st.expander(f"For information ({len(items)})", expanded=False):
            if miss:
                st.markdown("**Blank cells that the model will fill in automatically**")
                st.dataframe(pd.DataFrame([{"Column": i.columns[0], "Blank cells": i.detail.split(" of ")[0]} for i in miss]),
                             hide_index=True, width="content")
                ui.simple("Blank cells are filled in **inside** the model: numbers get the median and categories the most "
                          "common value - both learned from the training patients only, so no information leaks from the "
                          "test patients. This is called **imputation**.", "What happens to blank cells?")
                for i in miss:
                    if i.recommended != "keep":
                        render_issue(i)
            for i in other:
                render_issue(i)
        continue
    with st.expander(f"{SEV_LABEL[sev]} ({len(items)})", expanded=expanded):
        for i in items:
            render_issue(i)

# ----------------------------------------------------------------- log
st.divider()
st.subheader("Cleaning log")
if not ss.clean_log:
    st.caption("No changes yet.")
else:
    st.dataframe(pd.DataFrame({"Step": range(1, len(ss.clean_log) + 1), "What was done": [e["text"] for e in ss.clean_log]}),
                 hide_index=True, width="stretch")
    with st.expander("Show the same cleaning as Python (pandas) code"):
        code = "\n".join(e["code"] for e in ss.clean_log if e.get("code"))
        st.code("import numpy as np\nimport pandas as pd\n# df = pd.read_excel('your_file.xlsx')\n# to_number(): see the full "
                "script on the Report page\n" + code, language="python")
    st.caption(f"Now: {df.shape[0]} rows x {df.shape[1]} columns (original: {ss.df_raw.shape[0]} x {ss.df_raw.shape[1]}).")
    c1, c2 = st.columns(2)
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    c1.download_button("Download cleaned data (Excel)", buf.getvalue(), file_name="cleaned_data.xlsx", icon=":material/download:")
    c2.download_button("Download cleaned data (CSV)", df.to_csv(index=False).encode("utf-8"), file_name="cleaned_data.csv",
                       icon=":material/download:")

ui.next_step("app_pages/explore.py", "Next: 3. Explore data")
ui.footer()
