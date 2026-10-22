import pandas as pd
import streamlit as st

from mlstudio import sample_size as ssz
from mlstudio import ui

ui.page_head(None, "Sample size calculator", "How many patients do I need to develop a prediction model?")
ss = st.session_state

ui.simple("A model built on too few patients **memorises** them (overfits) and fails on new patients. The old rule of "
          "'10 events per variable' is a rough guide only. Riley and colleagues (BMJ 2020) give a better method that "
          "checks three things: (i) little overfitting, (ii) little optimism in the model fit and (iii) a precise estimate "
          "of the overall risk. This calculator applies their method (the same as the 'pmsampsize' package).",
          "Why sample size matters")

kind = st.radio("Type of outcome", ["binary", "continuous"], horizontal=True,
                format_func=lambda k: {"binary": "Two groups (e.g. disease yes/no)", "continuous": "A number (e.g. length of stay)"}[k])

if kind == "binary":
    c1, c2 = st.columns(2)
    prev = c1.number_input("Expected proportion with the outcome (prevalence)", 0.01, 0.99, 0.20, 0.01,
                           help="From previous studies or your pilot data, e.g. 0.20 = 20%.")
    p = c2.number_input("Number of candidate predictor parameters", 1, 200, 10,
                        help="Count every number predictor as 1, and a category with k levels as k-1. Include all you "
                             "might test, not just the final ones.")
    how = st.radio("How good do you expect the model to be?", ["auc", "nagel", "cs"], horizontal=True,
                   format_func=lambda h: {"auc": "Expected AUC (C-statistic)", "nagel": "Expected Nagelkerke R²",
                                          "cs": "Expected Cox-Snell R²"}[h])
    mx = ssz.max_r2_cs(prev)
    if how == "auc":
        auc = st.slider("Expected AUC", 0.60, 0.95, 0.75, 0.01,
                        help="Use a value reported for similar models; be conservative.")
        with st.spinner("Converting AUC to R² ..."):
            r2 = ssz.r2cs_from_auc(auc, prev)
        st.caption(f"AUC {auc:.2f} corresponds to a Cox-Snell R² of about {r2:.3f} (Riley et al., Stat Med 2021, by simulation).")
    elif how == "nagel":
        nag = st.slider("Expected Nagelkerke R²", 0.05, 0.90, 0.15, 0.01,
                        help="0.15 is the conservative default suggested by Riley et al. when nothing is known.")
        r2 = nag * mx
        st.caption(f"Cox-Snell R² = {nag:.2f} x {mx:.3f} (maximum possible) = {r2:.3f}")
    else:
        r2 = st.number_input(f"Expected Cox-Snell R² (must be below {mx:.3f})", 0.001, float(mx) - 0.001,
                             min(0.10, float(mx) / 2), 0.005, format="%.3f")
    try:
        out = ssz.riley_binary(prev, int(p), float(r2))
    except ValueError as e:
        ui.box("bad", str(e))
        st.stop()
    st.dataframe(pd.DataFrame(out["criteria"], columns=["Criterion", "Patients needed"]), hide_index=True, width="stretch")
    c = st.columns(3)
    with c[0]:
        ui.tile("Minimum sample size", f"{out['n']}", "the largest of the three criteria")
    with c[1]:
        ui.tile("Patients with the outcome", f"{out['events']}", f"prevalence {prev:.0%}")
    with c[2]:
        ui.tile("Events per parameter", f"{out['epp']:.1f}")
    st.markdown(f"For comparison, the simple rules give **{ssz.epv_rule(prev, int(p), 10)}** (10 events per variable) "
                f"and **{ssz.epv_rule(prev, int(p), 20)}** (20 events per variable) patients.")
    ui.box("warn", "These numbers are for **regression-based** models (e.g. logistic regression). Flexible machine-learning "
                   "methods such as random forests, boosting and neural networks usually need **several times more** "
                   "patients to reach stable performance (van der Ploeg et al., BMC Med Res Methodol 2014).",
           "Machine-learning models are data-hungry")
else:
    c1, c2 = st.columns(2)
    p = c1.number_input("Number of candidate predictor parameters", 1, 200, 10)
    r2 = c2.slider("Expected R² of the model", 0.05, 0.90, 0.30, 0.01)
    out = ssz.continuous_rule(int(p), float(r2))
    st.dataframe(pd.DataFrame(out["criteria"], columns=["Criterion", "Patients needed"]), hide_index=True, width="stretch")
    ui.tile("Minimum sample size (approximate)", f"{out['n']}")
    st.caption("Riley et al. (Stat Med 2019) also require a precise estimate of the mean outcome; with typical outcomes this "
               "is usually satisfied by the numbers above. Use the 'pmsampsize' package for the full calculation.")

st.divider()
st.markdown("**References**  \n"
            "Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required for developing a clinical "
            "prediction model. BMJ. 2020;368:m441.  \n"
            "Riley RD, Snell KIE, Ensor J, et al. Minimum sample size for developing a multivariable prediction model: "
            "Part I - continuous outcomes. Stat Med. 2019;38:1262-75.  \n"
            "Riley RD, Van Calster B, Collins GS. A note on estimating the Cox-Snell R2 from a reported C statistic (AUROC) "
            "to inform sample size calculations for developing a prediction model with a binary outcome. Stat Med. 2021;40:859-64.  \n"
            "van der Ploeg T, Austin PC, Steyerberg EW. Modern modelling techniques are data hungry: a simulation study for "
            "predicting dichotomous endpoints. BMC Med Res Methodol. 2014;14:137.")
ui.footer()
