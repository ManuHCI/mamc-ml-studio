import pandas as pd
import streamlit as st

from mlstudio import ui
from mlstudio.advisor import advice_summary, data_profile, detect_task, recommend, usable_features
from mlstudio.data_check import column_kind
from mlstudio.models import TASK_MODELS, model_name
from mlstudio.trainer import class_labels, guess_positive
from mlstudio.widgets import algorithm_card, playground

ui.page_head("4", "Choose outcome & model", "Tell the software what to predict; it recommends models and explains why.")
ui.need_data()
ss = st.session_state
df = ss.df

ui.demo_banner("The outcome **Diabetes_5yr** is already chosen and 'Yes' is the positive result. Look at the advisor's "
               "reasons below, and try the picture at the bottom: choose **Fasting_glucose** and **BMI**, then compare "
               "Logistic Regression (a straight line) with a Decision Tree (boxes) and K-Nearest Neighbours (wiggly).",
               "app_pages/train.py", "Demo: continue to 5. Train & tune")

# ---------------------------------------------------------------- A. outcome
st.subheader("A. What do you want to predict?")
cols = list(df.columns)
idx = cols.index(ss.target) if ss.target in cols else len(cols) - 1
c1, c2 = st.columns([1, 1.3])
target = c1.selectbox("Outcome (target) column", cols, index=idx,
                      help="The column with the answer you want the model to predict for future patients.")
auto_task, why = detect_task(df[target])
if target != ss.target:
    ss.target = target
    ss.task = auto_task if auto_task != "invalid" else None
    ss.features = []
    ss.positive = None
    ss.results = {}
    ss.split = None
if auto_task == "invalid":
    ui.box("bad", why, "Cannot use this column")
    st.stop()
with c2:
    st.write("")
    ui.box("note", why, "What kind of problem is this?")
opts = ["classification", "regression"]
task = st.radio("Problem type", opts, index=opts.index(ss.task or auto_task), horizontal=True,
                format_func=lambda t: {"classification": "Predict a group / category (classification)",
                                       "regression": "Predict a number (regression)"}[t])
if task == "regression" and not pd.api.types.is_numeric_dtype(df[target]):
    ui.box("bad", "A text column cannot be predicted as a number. Choose classification.")
    st.stop()
if task != ss.task:
    ss.task = task
    ss.results = {}
    ss.split = None

if task == "classification":
    labs = class_labels(df[target])
    if len(labs) == 2:
        default = ss.positive if ss.positive in labs else guess_positive(labs)
        pos = st.radio("Which value is the 'positive' result (the disease / event you want to catch)?", labs,
                       index=labs.index(default), horizontal=True,
                       help="Sensitivity, PPV etc. are calculated for this group.")
        if pos != ss.positive:
            ss.positive = pos
            ss.results = {}
            ss.split = None
    elif len(labs) > 15:
        ui.box("warn", f"The outcome has {len(labs)} different values. Classification with so many groups needs a lot of data. "
                       "Is this really a category, or should it be a number (regression)?")

ui.simple("**Classification** = the answer is a group (diabetes yes/no, type of anaemia). **Regression** = the answer is a "
          "number (length of stay, HbA1c). If your data has **no outcome** and you want to discover natural groups of "
          "patients, use **Find patient groups** (clustering) instead.", "Two kinds of supervised learning")

# ---------------------------------------------------------------- B. predictors
st.subheader("B. Which predictors (input factors) should the model use?")
usable = usable_features(df, target)
excluded = [(c, column_kind(df[c])) for c in df.columns if c != target and c not in usable]
default_feats = [f for f in ss.features if f in usable] or usable
feats = st.multiselect("Predictors", usable, default=default_feats,
                       help="Untick anything measured AFTER the outcome, or anything you would not have when using the model.")
if set(feats) != set(ss.features):
    ss.features = feats
    ss.results = {}
    ss.split = None
if excluded:
    st.caption("Not offered (cannot be used directly): " + ", ".join(f"{c} ({k})" for c, k in excluded))
text_nums = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c]) and column_kind(df[c]) == "numeric"]
if text_nums:
    ui.box("warn", f"{', '.join(text_nums)} contain numbers stored as text. They will be converted automatically, but it "
                   "is better to fix them on page **2. Check & clean data**, where you can see what changes.", "Check the data first")
if not feats:
    ui.box("warn", "Choose at least one predictor.")
    st.stop()
ui.box("warn", "Use only information that is available **at the moment you want to make the prediction**. A predictor "
               "recorded after the outcome (treatment given, discharge diagnosis, follow-up results) makes the model look "
               "brilliant in your data but useless in real life. This is called **data leakage**.", "Avoid data leakage")

# ---------------------------------------------------------------- C. advisor
st.subheader("C. Model advisor - which model, and why?")
prof = data_profile(df, feats, target, task)
for line in advice_summary(prof, task):
    st.markdown(f"- {line}")
pr_opts = ["balanced", "explain", "accuracy"]
priority = st.radio("What matters most to you?", pr_opts, index=pr_opts.index(ss.priority), horizontal=True,
                    format_func=lambda p: {"balanced": "A good balance", "explain": "I must explain the model (thesis / clinic)",
                                           "accuracy": "Best possible accuracy"}[p])
ss.priority = priority
rec = recommend(prof, task, priority)
ss.advice = rec

chosen = []
for r in rec:
    stars = "★" * r["stars"] + "☆" * (5 - r["stars"])
    with st.container(border=True):
        a, b = st.columns([0.9, 3.2])
        with a:
            tick = st.checkbox("Train this", value=(r["key"] in ss.chosen_models) if ss.chosen_models else r["recommended"],
                               key=f"pick_{r['key']}_{task}")
            st.markdown(f"<span style='color:#c98500;font-size:1.15rem'>{stars}</span>", unsafe_allow_html=True)
            if r["recommended"]:
                st.markdown("<span class='pill'>Recommended</span>", unsafe_allow_html=True)
        with b:
            st.markdown(f"**{r['name']}**")
            st.markdown("\n".join(f"- {x}" for x in r["reasons"]))
            if r["cautions"]:
                st.markdown("\n".join(f"- :orange[Caution:] {x}" for x in r["cautions"]))
        if tick:
            chosen.append(r["key"])
ss.chosen_models = chosen
st.caption(f"{len(chosen)} model(s) selected for training. Comparing a simple model with one or two flexible ones is "
           "good practice: a complex model is only worth it if it clearly beats the simple one on new patients.")

# ---------------------------------------------------------------- D. explain
st.subheader("D. How does each algorithm work?")
keys = TASK_MODELS[task]
k = st.selectbox("Explain this algorithm", keys, index=keys.index(chosen[0]) if chosen else 0,
                 format_func=lambda x: model_name(x, task))
with st.container(border=True):
    algorithm_card(k, task)

st.subheader("E. See what the algorithm does to YOUR data")
ui.simple("Pick two predictors. The algorithm is trained on just these two, and the picture shows how it divides your "
          "patients. Every algorithm 'draws' a different kind of boundary - this is the easiest way to understand them.")
playground(df[df[target].notna()], target, task, keys, "choose_pg", ss.positive)

ui.next_step("app_pages/train.py", "Next: 5. Train & tune")
ui.footer()
