import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mlstudio import plots, ui
from mlstudio.demo_data import DEMO_DATASETS, load_demo
from mlstudio.models import MODELS, TASK_MODELS, model_name
from mlstudio.widgets import algorithm_card, playground

@st.cache_data(show_spinner=False)
def demo(key):
    return load_demo(key)


ui.page_head(None, "Learn machine learning", "A short, visual textbook for medicos - no mathematics needed.")

chapters = ["1. What is ML?", "2. Kinds of ML", "3. Data for ML", "4. Training & testing", "5. Overfitting",
            "6. Settings & tuning", "7. Judging a model", "8. The algorithms", "9. Pitfalls & ethics", "Glossary"]
ch = st.radio("Chapter", chapters, horizontal=True, label_visibility="collapsed")
st.divider()

if ch == chapters[0]:
    st.subheader("What is machine learning?")
    st.markdown("""
**Traditional programming:** a person writes the rules. *"If fasting glucose >= 126 mg/dL on two occasions, diagnose diabetes."*

**Machine learning:** we give the computer many **examples with answers**, and it works out the rules itself. *"Here are
500 patients with their age, BMI, glucose ... and whether they developed diabetes. Find the pattern."*
""")
    ui.simple("Think of a first-year resident. Nobody gives them a formula for 'sick or not sick'. They see hundreds of "
              "patients, each with a final diagnosis, and gradually learn which combinations of findings matter. A machine-"
              "learning model does the same - but with numbers, very fast, and only with the information you give it.",
              "An analogy")
    st.markdown("""
**The vocabulary you need**

| Machine-learning word | Medical equivalent |
|---|---|
| **Features / predictors / inputs** | History, examination and lab findings (Age, BMI, Hb ...) |
| **Target / outcome / label** | The diagnosis or event to be predicted (Diabetes yes/no) |
| **Training** | Studying solved cases |
| **Model** | The learned 'clinical rule' (a formula, a flowchart, a set of trees ...) |
| **Prediction** | The model's answer for a new patient |
| **Test set** | The final exam with unseen cases |
""")
    ui.flow([("Data", "patients x variables"), ("Clean", "fix errors"), ("Split", "train / test"), ("Train", "learn pattern"),
             ("Tune", "best settings"), ("Test", "unseen patients"), ("Explain", "which factors")])
    st.caption("This is exactly the path that pages 1-8 of this software walk you through.")

elif ch == chapters[1]:
    st.subheader("Kinds of machine learning")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**Supervised - classification**")
            st.write("Predict a **group**. Answers are known for the training patients.")
            st.caption("Diabetes yes/no; malignant vs benign; type of anaemia; mortality.")
    with c2:
        with st.container(border=True):
            st.markdown("**Supervised - regression**")
            st.write("Predict a **number**. Answers are known for the training patients.")
            st.caption("Length of stay; HbA1c after 6 months; drug dose; eGFR.")
    with c3:
        with st.container(border=True):
            st.markdown("**Unsupervised - clustering**")
            st.write("No answers. Find **natural groups** of similar patients.")
            st.caption("Phenotypes of asthma, sepsis or diabetes; patient segments.")
    ui.simple("Other kinds you will hear about: **deep learning** (neural networks with many layers - images, ECG, text), "
              "**reinforcement learning** (learning by trial and reward), and **large language models** such as chatbots. "
              "For tables of patient data - the usual thesis dataset - the methods in this software are the right tools.",
              "And the rest?")

elif ch == chapters[2]:
    st.subheader("Data for machine learning")
    st.markdown("""
- **Tidy table:** one row per patient, one column per variable, the outcome in its own column.
- **Quality beats quantity beats algorithm.** A clean dataset with a simple model usually beats a messy dataset with a fancy model.
- **Enough patients:** what matters is the number of patients in the *smaller* outcome group (events) compared with the
  number of predictors. Flexible models need many more patients than logistic regression.
- **Representative:** the patients should look like the people the model will be used on (age, severity, setting).
- **Predictors available at prediction time:** never use information that becomes available after the outcome.
""")
    ui.box("bad", "Imagine predicting diabetes and including 'On_metformin'. The model becomes almost perfect - because "
                  "metformin is given *after* diabetes is diagnosed. In real life, at the time of prediction, nobody is on "
                  "metformin yet, and the model collapses. Other examples: 'ICU admission' to predict severity, 'discharge "
                  "diagnosis', 'number of days of antibiotics'.", "Data leakage - the most common serious mistake")
    st.markdown("**Missing values:** the software fills them in (imputation) with the median or most common value learned "
                "from the training patients. If many values are missing, or they are missing for a reason (e.g. the test "
                "was done only in sick patients), think about it before modelling.")
    st.markdown("**Categories** (e.g. Sex, Smoking) are turned into 0/1 columns (one-hot encoding): Smoking = Current -> "
                "Smoking_Current = 1, Smoking_Former = 0, Smoking_Never = 0.")
    st.markdown("**Scaling:** BMI (20-40) and triglycerides (50-500) live on different scales. Distance-based and linear "
                "models need them on the same scale, so each number column is converted to 'how many SDs from the mean'.")
    st.page_link("app_pages/samplesize.py", label="Sample size calculator", icon=":material/calculate:")

elif ch == chapters[3]:
    st.subheader("Training and testing")
    n = st.slider("Patients in the dataset", 50, 1000, 300, 50)
    t = st.slider("Percentage kept for the test", 10, 40, 20, 5)
    nt = int(np.ceil(n * t / 100))
    ui.show(plots.split_bar(n - nt, nt), key="lsplit")
    ui.simple("A model is judged on patients it has **never seen** - like an exam with new questions. The test patients are "
              "locked away and used **once**. If you keep peeking at the test set while changing the model, the test set "
              "quietly becomes part of training and the final score is too optimistic.", "The exam analogy")
    k = st.slider("Cross-validation rounds (k)", 3, 10, 5)
    ui.show(plots.kfold_diagram(k), key="lkfold")
    st.markdown(f"""
**Cross-validation** is the set of practice tests *inside* the training set. With k = {k}: the training patients are cut
into {k} parts; the model learns on {k - 1} parts and is checked on the remaining one, {k} times, so every training patient
is used once for checking. We average the {k} scores.

Why bother? (1) It gives a more stable estimate than a single split, and (2) it lets us **compare settings** (tuning)
without touching the test set.

**Stratified** splitting keeps the same proportion of each outcome group in every part - important when one group is small.
""")

elif ch == chapters[4]:
    st.subheader("Overfitting and underfitting - the most important idea in ML")
    deg = st.slider("Model complexity (flexibility of the curve)", 1, 10, 2,
                    help="1 = straight line. Higher = the curve can bend more.")
    fig, tr, te = plots.overfit_demo(deg)
    ui.show(fig, key="ovf")
    if deg <= 1:
        ui.box("warn", "A straight line is too simple to follow the curved dose-response: errors are high on both the "
                       "training and the new patients. This is **underfitting**.", "Underfitting")
    elif deg <= 3:
        ui.box("good", "The curve follows the general pattern. The error on new patients is close to the training error: "
                       "the model **generalises**.", "About right")
    else:
        ui.box("bad", "The curve now wiggles through almost every training patient (low training error) but misses the new "
                      "patients (high test error). It has memorised the noise. This is **overfitting**.", "Overfitting")
    errs = []
    for d in range(1, 11):
        _, a, b = plots.overfit_demo(d)
        errs.append((d, a, b))
    e = pd.DataFrame(errs, columns=["Complexity", "Training error", "Error on new patients"])
    f2 = go.Figure()
    f2.add_scatter(x=e["Complexity"], y=e["Training error"], mode="lines+markers", name="Training patients",
                   line=dict(color=plots.SERIES[1], dash="dash", width=2))
    f2.add_scatter(x=e["Complexity"], y=e["Error on new patients"].clip(upper=60), mode="lines+markers", name="New patients",
                   line=dict(color=plots.SERIES[0], width=2.5))
    f2.add_vline(x=deg, line=dict(color=plots.INK, dash="dot"))
    ui.show(plots._base(f2, "The U-shape: error on new patients is lowest at medium complexity", height=360,
                        xt="Model complexity", yt="Error"), key="ushape")
    ui.simple("A student who memorises last year's answers scores 100% on last year's paper and fails the new exam "
              "(overfitting). A student who reads only the chapter headings fails both (underfitting). We want the student "
              "who understands the subject. Every setting you tune, every predictor you add, moves the model along this "
              "U-shaped curve.", "The exam analogy again")

elif ch == chapters[5]:
    st.subheader("Settings (hyperparameters) and tuning")
    st.markdown("""
- **Parameters** are learned *from the data* during training (e.g. the weight of BMI in logistic regression, the questions
  in a decision tree).
- **Hyperparameters (settings)** are chosen *by you before training* (e.g. how deep the tree may grow, how many trees in a
  forest, how many neighbours KNN uses). They control how flexible the model is.
""")
    ui.simple("Choosing a hyperparameter is like finding the dose of a drug. Too little (too simple) - no effect "
              "(underfitting). Too much (too flexible) - toxicity (overfitting). **Tuning** is a dose-finding study: try "
              "several doses, measure the response by cross-validation, keep the best.", "The dose analogy")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**Grid search**")
            st.write("Tries every combination from a list (e.g. depth 2, 3, 4, 6, 8 x leaf size 1, 5, 10, 20 = 20 combinations). "
                     "Thorough but slow when many settings are tuned.")
    with c2:
        with st.container(border=True):
            st.markdown("**Random search**")
            st.write("Tries a fixed number of random combinations from wide ranges. Often finds a good setting faster when "
                     "several settings are tuned.")
    ui.box("note", "Every combination is judged by **cross-validation on the training set**. The test set is used only once, "
                   "after the best setting has been chosen.")

elif ch == chapters[6]:
    st.subheader("Judging a model - you already know most of this")
    ui.simple("A classification model is a **diagnostic test**. Everything you learned about sensitivity, specificity, PPV, "
              "NPV and ROC curves applies directly.", "Good news")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        sep = st.slider("How well does the model separate the groups?", 0.0, 3.0, 1.5, 0.1,
                        help="0 = not at all (coin toss); 3 = almost perfectly")
        thr = st.slider("Cut-off (probability above which we say 'disease')", 0.05, 0.95, 0.5, 0.01)
        prev = st.slider("Prevalence of disease", 0.05, 0.6, 0.3, 0.05)
    rng = np.random.default_rng(0)
    nd = int(1000 * prev)
    s_d = 1 / (1 + np.exp(-(rng.normal(sep / 2, 1, nd))))
    s_h = 1 / (1 + np.exp(-(rng.normal(-sep / 2, 1, 1000 - nd))))
    tp = int((s_d >= thr).sum())
    fn = nd - tp
    fp = int((s_h >= thr).sum())
    tn = (1000 - nd) - fp
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(np.r_[np.ones(nd), np.zeros(1000 - nd)], np.r_[s_d, s_h])
    with c2:
        f = go.Figure()
        f.add_histogram(x=s_h, name="Truly healthy", marker_color=plots.SERIES[0], opacity=0.6, xbins=dict(start=0, end=1, size=0.025))
        f.add_histogram(x=s_d, name="Truly diseased", marker_color=plots.SERIES[1], opacity=0.6, xbins=dict(start=0, end=1, size=0.025))
        f.update_layout(barmode="overlay")
        f.add_vline(x=thr, line=dict(color=plots.INK, width=2, dash="dash"))
        ui.show(plots._base(f, f"Predicted probabilities of 1000 patients (AUC {auc:.2f})", height=330,
                            xt="Predicted probability of disease", yt="Patients"), key="lthr")
    c = st.columns(4)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    for col, (lab, v, say) in zip(c, [("Sensitivity", sens, f"{tp} of {tp + fn} diseased caught"),
                                      ("Specificity", spec, f"{tn} of {tn + fp} healthy cleared"),
                                      ("PPV", ppv, f"{tp} of {tp + fp} positive calls correct"),
                                      ("NPV", npv, f"{tn} of {tn + fn} negative calls correct")]):
        with col:
            ui.tile(lab, ui.pct(v, 0), "", say)
    st.markdown(f"""
|  | Truly diseased | Truly healthy |
|---|---|---|
| **Model says disease** | TP = {tp} | FP = {fp} |
| **Model says healthy** | FN = {fn} | TN = {tn} |
""")
    ui.simple("Move the **cut-off**: sensitivity and specificity trade against each other - just like choosing a cut-off "
              "for troponin or PSA. Move the **separation**: the overlap shrinks and the AUC rises; AUC does not depend on "
              "the cut-off. Move the **prevalence**: PPV and NPV change although the model is the same.", "Try this")
    st.markdown("""
**Other measures**
- **AUC (area under the ROC curve):** chance that a random diseased patient gets a higher score than a random healthy one.
  0.5 = useless, 0.7-0.8 acceptable, 0.8-0.9 good, > 0.9 excellent.
- **Calibration:** are the predicted risks right? If the model says 20%, about 20 in 100 such patients should have the outcome.
- **Accuracy** can mislead: with 5% prevalence, saying 'healthy' to everyone is 95% accurate and 0% useful.
- **For numbers (regression):** MAE = average size of the error; RMSE = like MAE but punishes big errors more;
  R² = share of the variation explained (0 to 1).
- **95% confidence intervals** show how much a result could change with a different sample of test patients.
""")

elif ch == chapters[7]:
    st.subheader("The algorithms, one by one")
    task = st.radio("Problem type", ["classification", "regression"], horizontal=True,
                    format_func=lambda t: {"classification": "Classification", "regression": "Regression"}[t])
    key = st.selectbox("Algorithm", TASK_MODELS[task], format_func=lambda k: model_name(k, task))
    with st.container(border=True):
        algorithm_card(key, task)
    st.markdown("##### Try it on data")
    src_opts = ["diabetes", "anaemia", "breast"] if task == "classification" else ["stay"]
    if st.session_state.get("df") is not None and st.session_state.get("target") and st.session_state.get("task") == task:
        src_opts = ["mine"] + src_opts
    src = st.selectbox("Dataset", src_opts, format_func=lambda s: "My data" if s == "mine" else DEMO_DATASETS[s]["title"])
    if src == "mine":
        d, tgt, pos = st.session_state.df, st.session_state.target, st.session_state.positive
    else:
        d, tgt, pos = demo(src), DEMO_DATASETS[src]["target"], DEMO_DATASETS[src]["positive"]
    playground(d[d[tgt].notna()], tgt, task, [key] + [k for k in TASK_MODELS[task] if k != key], "learn_pg", pos)
    if key == "logreg":
        w = st.slider("Weight of the risk factor", 0.2, 3.0, 1.0, 0.1)
        ui.show(plots.sigmoid_demo(w), key="sig")

elif ch == chapters[8]:
    st.subheader("Pitfalls, bias and ethics")
    items = [
        ("Data leakage", "Predictors that are only known after the outcome, duplicates split across train and test, or "
                         "cleaning/scaling done on the whole dataset before splitting. The model looks great and fails in practice."),
        ("Overfitting to small data", "Hundreds of predictors and a few dozen events. Use simple models, few predictors, and "
                                      "check the sample size."),
        ("Accuracy with unbalanced outcomes", "Report sensitivity, specificity, PPV, NPV, AUC and calibration - not just accuracy."),
        ("Only internal validation", "A held-out test set from the same hospital is a start. Before clinical use, test the "
                                     "model on patients from another time period or another hospital (external validation)."),
        ("Bias and fairness", "If some groups (women, rural patients, a particular age group) are under-represented, the model "
                              "may work worse for them. Check performance in subgroups."),
        ("Correlation is not causation", "An important predictor is not necessarily a cause. Do not change treatment because "
                                         "of a feature-importance chart."),
        ("Dataset shift", "Practice changes (new assays, new guidelines, new populations) make models age. Re-check them over time."),
        ("Ethics & privacy", "Get ethics approval, remove identifiers, keep data secure. This software never sends data "
                             "anywhere - everything runs on your computer."),
        ("Reporting", "Follow TRIPOD+AI (BMJ 2024) when you write up a prediction-model study."),
    ]
    for t, d in items:
        st.markdown(f"**{t}.** {d}")

else:
    st.subheader("Glossary")
    g = {
        "Algorithm": "The learning method (e.g. random forest).",
        "AUC": "Area under the ROC curve; how well the model ranks patients with vs without the outcome.",
        "Balanced accuracy": "Average of sensitivity and specificity.",
        "Bootstrap": "Re-sampling the data with replacement many times to estimate uncertainty (confidence intervals).",
        "Calibration": "Agreement between predicted risks and observed frequencies.",
        "Class imbalance": "One outcome group is much smaller than the other.",
        "Classification": "Predicting a category.",
        "Clustering": "Finding groups of similar patients without an outcome.",
        "Cross-validation": "Repeated train/check splits inside the training data.",
        "Data leakage": "Information from the outcome or the test set sneaking into training.",
        "Feature / predictor": "An input variable.",
        "Hyperparameter": "A setting chosen before training that controls model flexibility.",
        "Imputation": "Filling in missing values.",
        "One-hot encoding": "Turning a category into 0/1 columns.",
        "Overfitting": "Memorising the training data; poor on new patients.",
        "Permutation importance": "Drop in performance when a predictor is scrambled.",
        "PCA": "Principal component analysis; squeezes many variables into a few for plotting.",
        "Regression": "Predicting a number.",
        "Regularisation": "A penalty that keeps a model simple to reduce overfitting.",
        "Scaling / standardisation": "Converting values to SD units from the mean.",
        "Test set": "Patients kept aside for the final, one-time evaluation.",
        "Training set": "Patients the model learns from.",
        "Underfitting": "Model too simple to capture the pattern.",
    }
    st.dataframe(pd.DataFrame(list(g.items()), columns=["Term", "Meaning"]), hide_index=True, width="stretch", height=600)
ui.footer()
