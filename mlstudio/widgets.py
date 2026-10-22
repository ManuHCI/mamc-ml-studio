"""Re-usable interactive pieces: setting (hyperparameter) widgets, algorithm explainer cards and the
'what does this algorithm do to my data?' playground."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import plots, ui
from .data_check import column_kind, is_num
from .models import MODELS, fmt_value, model_name
from .trainer import _label, decision_surface

LOG_STEPS = [1e-12, 1e-9, 1e-6, 1e-5, 1e-4, 0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0,
             3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0]


def param_widget(p: dict, key: str, value=None):
    """Show one hyperparameter control with its plain-language explanation. Returns the chosen value."""
    v = p["default"] if value is None else value
    helptext = p["explain"]
    if p["kind"] == "int":
        lo, hi = p["range"]
        allow_none = p["default"] is None or (p["grid"] and None in p["grid"])
        if allow_none:
            nolimit = st.checkbox(f"{p['label']}: no limit", value=v is None, key=key + "_none", help=helptext)
            if nolimit:
                return None
            return st.slider(p["label"], int(lo), int(hi), int(v) if v is not None else int(min(max(8, lo), hi)),
                             key=key, help=helptext)
        return st.slider(p["label"], int(lo), int(hi), int(v), key=key, help=helptext)
    if p["kind"] == "float":
        lo, hi = p["range"]
        return st.slider(p["label"], float(lo), float(hi), float(v), key=key, help=helptext)
    if p["kind"] == "logfloat":
        lo, hi = p["range"]
        steps = sorted({s for s in LOG_STEPS if lo <= s <= hi} | {float(v)})
        return st.select_slider(p["label"], options=steps, value=float(v), key=key, help=helptext,
                                format_func=lambda x: fmt_value(p, x))
    opts = list(p["options"])
    idx = opts.index(v) if v in opts else 0
    return st.selectbox(p["label"], opts, index=idx, key=key, help=helptext, format_func=lambda x: fmt_value(p, x))


def param_explain_table(key: str) -> pd.DataFrame:
    rows = []
    for p in MODELS[key]["params"]:
        rows.append({"Setting": p["label"], "Starting value": fmt_value(p, p["default"]), "What it does": p["explain"],
                     "Low value": p["low"], "High value": p["high"]})
    return pd.DataFrame(rows)


def algorithm_card(key: str, task: str, expanded_how: bool = True):
    m = MODELS[key]
    st.markdown(f"#### {model_name(key, task)}")
    st.markdown(f"**In one line:** {m['one_line']}")
    ui.simple(m["analogy"], "Think of it like this")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("**How it learns, step by step**")
        st.markdown("\n".join(f"{i + 1}. {s}" for i, s in enumerate(m["how"])))
        st.markdown(f"**What it does to your data:** {m['data_effect']}")
    with c2:
        st.markdown("**Strengths**")
        st.markdown("\n".join(f"- {s}" for s in m["strengths"]))
        st.markdown("**Weaknesses**")
        st.markdown("\n".join(f"- {s}" for s in m["weaknesses"]))
    st.markdown(f"**Use it when:** {m['when']}  \n**Medical example:** {m['example']}")
    if m["params"]:
        with st.expander("Its settings (hyperparameters) explained"):
            st.dataframe(param_explain_table(key), hide_index=True, width="stretch")
    else:
        st.caption("This model has no settings to tune.")


def playground(df: pd.DataFrame, target: str, task: str, keys: list[str], kp: str = "pg", positive: str | None = None):
    """Train the chosen algorithm on just TWO numeric predictors and draw how it divides the patients."""
    nums = [c for c in df.columns if c != target and is_num(df[c]) and column_kind(df[c]) == "numeric"]
    if len(nums) < 2:
        st.info("This picture needs at least two number columns among the predictors.")
        return
    c1, c2, c3 = st.columns(3)
    key = c1.selectbox("Algorithm", keys, format_func=lambda k: model_name(k, task), key=kp + "_m")
    a = c2.selectbox("Predictor on the x-axis", nums, index=0, key=kp + "_a")
    b = c3.selectbox("Predictor on the y-axis", [c for c in nums if c != a], index=0, key=kp + "_b")
    params = {}
    main = [p for p in MODELS[key]["params"] if not p["advanced"]][:2]
    if main:
        cols = st.columns(len(main))
        for col, p in zip(cols, main):
            with col:
                params[p["name"]] = param_widget(p, f"{kp}_{key}_{p['name']}")
    d = df[[a, b, target]].dropna()
    if len(d) > 1500:
        d = d.sample(1500, random_state=0)
    X2 = d[[a, b]].astype(float)
    labels = None
    if task == "classification":
        ys = d[target].map(_label)
        cls = sorted(ys.unique())
        if len(cls) == 2:
            pos = positive if positive in cls else cls[1]
            labels = [c for c in cls if c != pos] + [pos]
            y = (ys == pos).astype(int).to_numpy()
        else:
            y = np.asarray(ys, dtype=object)
    else:
        y = d[target].astype(float).to_numpy()
    try:
        with st.spinner("Training on two predictors..."):
            ga, gb, Z, classes, score, _ = decision_surface(key, task, X2, y, params)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not draw this model: {e}")
        return
    title = f"{model_name(key, task)} using only {a} and {b}"
    ui.show(plots.decision_surface(ga, gb, Z, X2, y, classes, task, title, labels), key=kp + "_fig")
    what = "accuracy" if task == "classification" else "R²"
    if task == "classification":
        ui.simple(f"Each dot is a patient (colour = what really happened). The shaded background is what the model would "
                  f"predict for a patient at that spot; the black line is where it switches its answer. Accuracy on these "
                  f"same patients: {score:.2f} ({what} on the training data - so it flatters the model). Change the "
                  f"settings above and watch how the boundary becomes simpler or more wiggly.", "How to read this picture")
    else:
        ui.simple(f"The background colour is the number the model would predict for a patient at that spot; dots are real "
                  f"patients. {what} on these same patients: {score:.2f}.", "How to read this picture")
