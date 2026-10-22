"""All interactive charts (Plotly). Every chart has hover tooltips and a camera button for a high-resolution PNG."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Colour-blind-checked categorical order (fixed: colour follows the group, never its rank)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = [[0, "#f3f8fe"], [0.15, "#cde2fb"], [0.35, "#9ec5f4"], [0.55, "#5598e7"], [0.75, "#256abf"], [1, "#0d366b"]]
DIVERGING = [[0, "#1c5cab"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#f0a3a2"], [1, "#c93c3b"]]
GOOD, WARN, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BRAND = "#7b1c2e"
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 3},
               "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def color_map(labels) -> dict:
    return {lab: SERIES[i % len(SERIES)] for i, lab in enumerate(labels)}


def _base(fig: go.Figure, title: str | None = None, height: int = 420, legend: bool = True, xt=None, yt=None) -> go.Figure:
    top = 56 if title else 24
    if title and legend:
        top = 92
    elif legend:
        top = 44
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>" if title else None, font=dict(size=15, color=INK), x=0, xanchor="left",
                   yref="container", y=1, yanchor="top", pad=dict(t=12)),
        font=dict(family=FONT, size=13, color=INK2), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        height=height, margin=dict(l=60, r=24, t=top, b=56),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, color=INK, size=12)),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                                       bgcolor="rgba(0,0,0,0)", traceorder="normal"),
        bargap=0.35,
    )
    try:
        fig.update_layout(barcornerradius=4)
    except Exception:
        pass
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, title=dict(text=xt, font=dict(color=INK2)),
                     tickfont=dict(color=MUTED), automargin=True)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, title=dict(text=yt, font=dict(color=INK2)),
                     tickfont=dict(color=MUTED), automargin=True)
    return fig


# ---------------------------------------------------------------------------
# Data exploration
# ---------------------------------------------------------------------------
def hist(df, col, by=None, bins=30):
    fig = go.Figure()
    if by is None:
        fig.add_histogram(x=df[col], nbinsx=bins, marker_color=SERIES[0], name=col,
                          hovertemplate=f"{col}: %{{x}}<br>Patients: %{{y}}<extra></extra>")
        return _base(fig, f"Distribution of {col}", legend=False, xt=col, yt="Number of patients")
    groups = sorted(df[by].dropna().astype(str).unique())
    cm = color_map(groups)
    for g in groups:
        fig.add_histogram(x=df.loc[df[by].astype(str) == g, col], nbinsx=bins, name=g, marker_color=cm[g], opacity=0.65,
                          hovertemplate=f"{by} = {g}<br>{col}: %{{x}}<br>Patients: %{{y}}<extra></extra>")
    fig.update_layout(barmode="overlay")
    return _base(fig, f"{col} by {by}", xt=col, yt="Number of patients")


def box(df, num, cat):
    groups = sorted(df[cat].dropna().astype(str).unique())
    cm = color_map(groups)
    fig = go.Figure()
    for g in groups:
        fig.add_box(y=df.loc[df[cat].astype(str) == g, num], name=g, marker_color=cm[g], boxpoints="outliers",
                    line=dict(width=1.5))
    return _base(fig, f"{num} in each {cat} group", legend=False, xt=cat, yt=num)


def bar_counts(df, col, by=None):
    fig = go.Figure()
    if by is None:
        vc = df[col].astype(str).value_counts()
        fig.add_bar(x=vc.index, y=vc.values, marker_color=SERIES[0], text=vc.values, textposition="outside",
                    hovertemplate="%{x}: %{y} patients<extra></extra>")
        return _base(fig, f"Patients in each {col} group", legend=False, xt=col, yt="Number of patients")
    ct = pd.crosstab(df[col].astype(str), df[by].astype(str))
    cm = color_map(list(ct.columns))
    for g in ct.columns:
        pct = (ct[g] / ct.sum(axis=1) * 100).round(1)
        fig.add_bar(x=ct.index, y=ct[g], name=str(g), marker_color=cm[g], customdata=pct,
                    hovertemplate=f"{col}: %{{x}}<br>{by} = {g}: %{{y}} (%{{customdata}}% of row)<extra></extra>")
    fig.update_layout(barmode="group")
    return _base(fig, f"{col} by {by}", xt=col, yt="Number of patients")


def scatter(df, x, y, color=None):
    fig = go.Figure()
    if color is None:
        fig.add_scatter(x=df[x], y=df[y], mode="markers", marker=dict(color=SERIES[0], size=8, opacity=0.7,
                        line=dict(color="#ffffff", width=1)), hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<extra></extra>")
        return _base(fig, f"{y} vs {x}", legend=False, xt=x, yt=y)
    if pd.api.types.is_numeric_dtype(df[color]) and df[color].nunique() > 8:
        fig.add_scatter(x=df[x], y=df[y], mode="markers", marker=dict(color=df[color], colorscale=SEQ, size=8,
                        showscale=True, colorbar=dict(title=color), line=dict(color="#ffffff", width=1)),
                        hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<br>{color}: %{{marker.color}}<extra></extra>")
        return _base(fig, f"{y} vs {x}", legend=False, xt=x, yt=y)
    groups = sorted(df[color].dropna().astype(str).unique())
    cm = color_map(groups)
    for g in groups:
        d = df[df[color].astype(str) == g]
        fig.add_scatter(x=d[x], y=d[y], mode="markers", name=g, marker=dict(color=cm[g], size=8, opacity=0.75,
                        line=dict(color="#ffffff", width=1)),
                        hovertemplate=f"{color} = {g}<br>{x}: %{{x}}<br>{y}: %{{y}}<extra></extra>")
    return _base(fig, f"{y} vs {x}, coloured by {color}", xt=x, yt=y)


def corr_heatmap(df):
    num = df.select_dtypes("number")
    c = num.corr().round(2)
    fig = go.Figure(go.Heatmap(z=c.values, x=c.columns, y=c.index, zmin=-1, zmax=1, colorscale=DIVERGING,
                               text=c.values, texttemplate="%{text}", textfont=dict(size=11),
                               hovertemplate="%{y} & %{x}: r = %{z}<extra></extra>", colorbar=dict(title="r")))
    fig.update_yaxes(autorange="reversed")
    h = max(380, 36 * len(c) + 120)
    return _base(fig, "Correlation between number columns (Pearson r)", height=h, legend=False)


def missing_map(df):
    m = df.isna().astype(int)
    fig = go.Figure(go.Heatmap(z=m.T.values, x=np.arange(1, len(df) + 1), y=df.columns, colorscale=[[0, "#eef3fa"], [1, CRITICAL]],
                               showscale=False, hovertemplate="Row %{x}<br>%{y}: %{customdata}<extra></extra>",
                               customdata=np.where(m.T.values == 1, "blank", "filled")))
    h = max(300, 22 * df.shape[1] + 120)
    return _base(fig, "Where are the blank cells? (red = blank)", height=h, legend=False, xt="Row (patient)")


def class_balance(y: pd.Series, title="Outcome groups"):
    vc = y.astype(str).value_counts()
    cm = color_map(sorted(vc.index))
    pct = (vc / vc.sum() * 100).round(1)
    fig = go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color=[cm[i] for i in vc.index],
                           text=[f"{v} ({p}%)" for v, p in zip(vc.values, pct.values)], textposition="outside",
                           hovertemplate="%{x}: %{y} patients<extra></extra>"))
    fig.update_yaxes(range=[0, vc.max() * 1.2])
    return _base(fig, title, height=330, legend=False, yt="Patients")


# ---------------------------------------------------------------------------
# Training process
# ---------------------------------------------------------------------------
def split_bar(n_train, n_test):
    fig = go.Figure()
    fig.add_bar(y=["Patients"], x=[n_train], orientation="h", name=f"Training set: {n_train} patients (the model studies these)",
                marker_color=SERIES[0], text=[f"TRAINING  {n_train}"], textposition="inside", insidetextanchor="middle",
                textfont=dict(color="#ffffff", size=14), hovertemplate="Training: %{x} patients<extra></extra>")
    fig.add_bar(y=["Patients"], x=[n_test], orientation="h", name=f"Test set: {n_test} patients (locked away for the final exam)",
                marker_color=SERIES[1], text=[f"TEST  {n_test}"], textposition="inside", insidetextanchor="middle",
                textfont=dict(color="#ffffff", size=14), hovertemplate="Test: %{x} patients<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.2)
    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(title=None, showgrid=False)
    fig = _base(fig, None, height=170)
    fig.update_layout(legend=dict(orientation="h", y=-0.35, x=0, xanchor="left", yanchor="top", traceorder="normal"),
                      margin=dict(t=10, b=10, l=10, r=10))
    return fig


def kfold_diagram(k: int):
    z = np.zeros((k, k))
    text = [["" for _ in range(k)] for _ in range(k)]
    for i in range(k):
        z[i, i] = 1
        for j in range(k):
            text[i][j] = "check" if i == j else "learn"
    fig = go.Figure(go.Heatmap(z=z, x=[f"Part {j + 1}" for j in range(k)], y=[f"Round {i + 1}" for i in range(k)],
                               colorscale=[[0, SERIES[0]], [1, SERIES[3]]], showscale=False, xgap=4, ygap=4,
                               text=text, texttemplate="%{text}", textfont=dict(color="#ffffff", size=13),
                               hovertemplate="%{y}, %{x}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    fig = _base(fig, None, height=60 + 42 * k, legend=False)
    fig.update_layout(margin=dict(l=80, r=10, t=40, b=10))
    return fig


def cv_folds(results: dict, metric: str):
    fig = go.Figure()
    keys = list(results)
    cm = color_map([results[k]["name"] for k in keys])
    for k in keys:
        name = results[k]["name"]
        folds = results[k]["cv"][metric]["folds"]
        fig.add_scatter(x=[name] * len(folds), y=folds, mode="markers", name=name, marker=dict(size=10, color=cm[name],
                        line=dict(color="#ffffff", width=1.5)), hovertemplate=f"{name}<br>Round %{{customdata}}: %{{y:.3f}}<extra></extra>",
                        customdata=np.arange(1, len(folds) + 1), showlegend=False)
        fig.add_scatter(x=[name], y=[np.nanmean(folds)], mode="markers", marker=dict(symbol="line-ew-open", size=34,
                        color=INK, line=dict(width=2.5)), showlegend=False, hovertemplate=f"Mean: %{{y:.3f}}<extra></extra>")
    return _base(fig, f"Cross-validation: {metric} in each round (dots) and the average (bar)", height=380, legend=False, yt=metric)


def tuning_plot(tab: pd.DataFrame, param: str, metric: str, colour_by: str | None = None, label: str | None = None):
    d = tab.copy()
    d[param] = d[param].map(lambda v: "No limit" if v is None or (isinstance(v, float) and np.isnan(v)) else v)
    fig = go.Figure()
    numeric = pd.to_numeric(d[param], errors="coerce").notna().all()
    if colour_by and colour_by in d:
        d[colour_by] = d[colour_by].map(lambda v: "No limit" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        groups = list(dict.fromkeys(d[colour_by]))
        cm = color_map(groups)
        for g in groups:
            sub = d[d[colour_by] == g]
            sub = sub.groupby(param, sort=True, dropna=False)[[f"CV {metric}", f"Train {metric}"]].mean().reset_index() \
                if f"Train {metric}" in sub else sub.groupby(param)[[f"CV {metric}"]].mean().reset_index()
            fig.add_scatter(x=sub[param].astype(str) if not numeric else sub[param], y=sub[f"CV {metric}"], mode="lines+markers",
                            name=f"{colour_by} = {g}", line=dict(color=cm[g], width=2), marker=dict(size=8))
    else:
        cols = [f"CV {metric}"] + ([f"Train {metric}"] if f"Train {metric}" in d else [])
        sub = d.groupby(param, sort=True, dropna=False)[cols].mean().reset_index()
        x = sub[param] if numeric else sub[param].astype(str)
        fig.add_scatter(x=x, y=sub[f"CV {metric}"], mode="lines+markers", name="Unseen patients (cross-validation)",
                        line=dict(color=SERIES[0], width=2.5), marker=dict(size=9))
        if f"Train {metric}" in sub:
            fig.add_scatter(x=x, y=sub[f"Train {metric}"], mode="lines+markers", name="Training patients",
                            line=dict(color=SERIES[1], width=2, dash="dash"), marker=dict(size=7))
    if numeric:
        vals = pd.to_numeric(d[param])
        if vals.min() > 0 and vals.max() / vals.min() >= 100:
            fig.update_xaxes(type="log")
    else:
        fig.update_xaxes(type="category")
    return _base(fig, f"Effect of '{label or param}' on {metric}", height=400, xt=label or param, yt=metric)


def tuning_scatter(tab: pd.DataFrame, param: str, metric: str):
    d = tab.copy()
    x = d[param].map(lambda v: "No limit" if v is None else v)
    fig = go.Figure(go.Scatter(x=x, y=d[f"CV {metric}"], mode="markers", marker=dict(size=10, color=SERIES[0], opacity=0.8,
                                line=dict(color="#ffffff", width=1)),
                               hovertemplate=f"{param}: %{{x}}<br>CV {metric}: %{{y:.3f}}<extra></extra>"))
    best = d.iloc[0]
    fig.add_scatter(x=[best[param] if best[param] is not None else "No limit"], y=[best[f"CV {metric}"]], mode="markers",
                    marker=dict(size=16, symbol="star", color=SERIES[3], line=dict(color=INK, width=1)), name="Best")
    try:
        v = pd.to_numeric(x)
        if v.min() > 0 and v.max() / v.min() >= 100:
            fig.update_xaxes(type="log")
    except (ValueError, TypeError):
        pass
    return _base(fig, f"Random search: every trial for '{param}'", height=380, xt=param, yt=f"CV {metric}")


def learning_curve(sizes, tr, te, metric):
    fig = go.Figure()
    for arr, name, col, dash in [(tr, "Training patients", SERIES[1], "dash"), (te, "Unseen patients (cross-validation)", SERIES[0], None)]:
        m, s = np.nanmean(arr, axis=1), np.nanstd(arr, axis=1)
        fig.add_scatter(x=np.r_[sizes, sizes[::-1]], y=np.r_[m + s, (m - s)[::-1]], fill="toself", fillcolor=col,
                        opacity=0.15, line=dict(width=0), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=sizes, y=m, mode="lines+markers", name=name, line=dict(color=col, width=2.5, dash=dash),
                        marker=dict(size=8), hovertemplate=f"{name}<br>%{{x}} patients: %{{y:.3f}}<extra></extra>")
    return _base(fig, "Learning curve: does more data help?", height=400, xt="Number of training patients", yt=metric)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def confusion(cm, labels, binary=True):
    cm = np.asarray(cm)
    tot = cm.sum(axis=1, keepdims=True)
    pct = np.where(tot > 0, cm / np.maximum(tot, 1) * 100, 0)
    text = [[f"<b>{cm[i, j]}</b><br>{pct[i, j]:.0f}% of row" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    if binary and cm.shape == (2, 2):
        names = [["True negative (TN)", "False positive (FP)"], ["False negative (FN)", "True positive (TP)"]]
        text = [[f"<b>{cm[i, j]}</b><br>{names[i][j]}" for j in range(2)] for i in range(2)]
    z = pct
    fig = go.Figure(go.Heatmap(z=z, x=[f"Predicted: {l}" for l in labels], y=[f"Truly: {l}" for l in labels],
                               colorscale=SEQ, zmin=0, zmax=100, text=text, texttemplate="%{text}", showscale=False,
                               xgap=3, ygap=3, hovertemplate="%{y}<br>%{x}<br>%{z:.1f}% of this row<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", showgrid=False)
    n = len(labels)
    fig = _base(fig, None, height=150 + 70 * n, legend=False)
    # dark cells need white text
    fig.update_traces(textfont=dict(size=13))
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def roc(curves: list, point=None):
    """curves: list of (name, fpr, tpr, auc)."""
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=AXIS, dash="dot", width=1.5), name="Coin toss (AUC 0.5)",
                    hoverinfo="skip")
    cm = color_map([c[0] for c in curves])
    for name, fpr, tpr, auc in curves:
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.3f})", line=dict(color=cm[name], width=2.5),
                        hovertemplate=f"{name}<br>1 - specificity: %{{x:.2f}}<br>Sensitivity: %{{y:.2f}}<extra></extra>")
    if point is not None:
        fig.add_scatter(x=[point[0]], y=[point[1]], mode="markers", name="Current cut-off",
                        marker=dict(size=13, color=SERIES[3], symbol="circle", line=dict(color=INK, width=1.5)))
    fig.update_xaxes(range=[-0.01, 1.01])
    fig.update_yaxes(range=[-0.01, 1.01], scaleanchor="x", scaleratio=1)
    return _base(fig, "ROC curve", height=460, xt="1 - Specificity (false-positive rate)", yt="Sensitivity (true-positive rate)")


def pr_curve(curves: list, prevalence: float):
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[prevalence, prevalence], mode="lines", line=dict(color=AXIS, dash="dot"),
                    name=f"No skill (prevalence {prevalence:.2f})", hoverinfo="skip")
    cm = color_map([c[0] for c in curves])
    for name, rec, prec, ap in curves:
        fig.add_scatter(x=rec, y=prec, mode="lines", name=f"{name} (AP {ap:.3f})", line=dict(color=cm[name], width=2.5),
                        hovertemplate=f"{name}<br>Sensitivity: %{{x:.2f}}<br>PPV: %{{y:.2f}}<extra></extra>")
    fig.update_xaxes(range=[0, 1.01])
    fig.update_yaxes(range=[0, 1.02])
    return _base(fig, "Precision-recall curve (PPV vs sensitivity)", height=420, xt="Sensitivity (recall)", yt="PPV (precision)")


def calibration(curves: list):
    """curves: list of (name, mean_pred, frac_pos, counts)."""
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=AXIS, dash="dot"), name="Perfect calibration", hoverinfo="skip")
    cm = color_map([c[0] for c in curves])
    for name, mp, fp, cnt in curves:
        fig.add_scatter(x=mp, y=fp, mode="lines+markers", name=name, line=dict(color=cm[name], width=2),
                        marker=dict(size=9), customdata=cnt,
                        hovertemplate=f"{name}<br>Predicted risk: %{{x:.2f}}<br>Observed rate: %{{y:.2f}}<br>Patients: %{{customdata}}<extra></extra>")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return _base(fig, "Calibration: are the predicted risks believable?", height=420, xt="Predicted probability",
                 yt="Observed proportion with the outcome")


def threshold_plot(tab: pd.DataFrame, t: float):
    fig = go.Figure()
    for col, c, dash in [("Sensitivity", SERIES[0], None), ("Specificity", SERIES[1], None), ("PPV", SERIES[2], "dot"), ("NPV", SERIES[6], "dot")]:
        fig.add_scatter(x=tab["Threshold"], y=tab[col], mode="lines", name=col, line=dict(color=c, width=2.5 if not dash else 1.8, dash=dash),
                        hovertemplate=f"Cut-off %{{x:.2f}}: {col} %{{y:.2f}}<extra></extra>")
    fig.add_vline(x=t, line=dict(color=INK, width=1.5, dash="dash"))
    fig.update_yaxes(range=[0, 1.02])
    return _base(fig, "How the cut-off changes sensitivity and specificity", height=380, xt="Probability cut-off", yt="Value")


def importance(df: pd.DataFrame, value="Importance", label="Factor", err=None, title="Which factors matter most?", xt=None):
    d = df.head(20).iloc[::-1]
    fig = go.Figure(go.Bar(x=d[value], y=d[label].astype(str), orientation="h", marker_color=SERIES[0],
                           error_x=dict(type="data", array=d[err], color=MUTED, thickness=1.2) if err else None,
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    return _base(fig, title, height=max(320, 30 * len(d) + 110), legend=False, xt=xt or value)


def odds_ratio_plot(df: pd.DataFrame, positive: str):
    d = df.head(20).iloc[::-1]
    col = [SERIES[1] if v > 1 else SERIES[0] for v in d["Odds ratio"]]
    fig = go.Figure(go.Scatter(x=d["Odds ratio"], y=d["Term"], mode="markers", marker=dict(size=12, color=col,
                               line=dict(color="#ffffff", width=1.5)), customdata=d["Meaning"],
                               hovertemplate="%{y}<br>Odds ratio %{x:.2f} (%{customdata})<extra></extra>"))
    fig.add_vline(x=1, line=dict(color=INK, width=1, dash="dash"))
    fig.update_xaxes(type="log")
    return _base(fig, f"Odds ratios for '{positive}' (right of 1 = higher odds)", height=max(320, 30 * len(d) + 110),
                 legend=False, xt="Odds ratio (log scale)")


def pred_vs_actual(y, pred, name):
    lo, hi = float(min(np.min(y), np.min(pred))), float(max(np.max(y), np.max(pred)))
    fig = go.Figure()
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=AXIS, dash="dot"), name="Perfect prediction", hoverinfo="skip")
    fig.add_scatter(x=y, y=pred, mode="markers", name="Test patients", marker=dict(color=SERIES[0], size=8, opacity=0.75,
                    line=dict(color="#ffffff", width=1)), hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>")
    return _base(fig, f"{name}: predicted vs actual (test patients)", height=430, xt="Actual value", yt="Predicted value")


def residuals(y, pred):
    res = np.asarray(y) - np.asarray(pred)
    fig = go.Figure()
    fig.add_scatter(x=pred, y=res, mode="markers", marker=dict(color=SERIES[0], size=8, opacity=0.75, line=dict(color="#ffffff", width=1)),
                    hovertemplate="Predicted: %{x:.2f}<br>Error: %{y:.2f}<extra></extra>", showlegend=False)
    fig.add_hline(y=0, line=dict(color=INK, width=1, dash="dash"))
    return _base(fig, "Errors (actual - predicted): should scatter evenly around 0", height=380, legend=False,
                 xt="Predicted value", yt="Error")


def compare_bars(table: pd.DataFrame, metric: str, ci: dict | None = None):
    # plotly draws the first bar at the bottom, so sort worst -> best to put the best model on top
    d = table.sort_values(metric, ascending=metric not in ("MAE", "RMSE"))
    cm = color_map(list(table["Model"]))
    err_plus = err_minus = None
    if ci:
        err_plus = [ci.get(m, (np.nan, np.nan))[1] - v for m, v in zip(d["Model"], d[metric])]
        err_minus = [v - ci.get(m, (np.nan, np.nan))[0] for m, v in zip(d["Model"], d[metric])]
    fig = go.Figure(go.Bar(x=d[metric], y=d["Model"], orientation="h", marker_color=[cm[m] for m in d["Model"]],
                           error_x=dict(type="data", array=err_plus, arrayminus=err_minus, color=INK2, thickness=1.3) if ci else None,
                           text=[f"{v:.3f}" for v in d[metric]], textposition="inside", insidetextanchor="start",
                           textfont=dict(color="#ffffff", size=13),
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    shown = "R\u00b2" if metric == "R2" else metric
    return _base(fig, f"Test-set {shown}" + (" with 95% CI" if ci else ""), height=max(260, 46 * len(d) + 100),
                 legend=False, xt=shown)


def pdp(grid, ys, feature, ylabel, classes=None):
    fig = go.Figure()
    numeric = np.issubdtype(np.asarray(grid).dtype, np.number)
    if classes is None:
        if numeric:
            fig.add_scatter(x=grid, y=ys, mode="lines+markers", line=dict(color=SERIES[0], width=2.5), marker=dict(size=6),
                            hovertemplate=f"{feature} = %{{x:.2f}}<br>{ylabel}: %{{y:.3f}}<extra></extra>", showlegend=False)
        else:
            fig.add_bar(x=[str(g) for g in grid], y=ys, marker_color=SERIES[0], hovertemplate=f"%{{x}}: %{{y:.3f}}<extra></extra>",
                        showlegend=False)
    else:
        arr = np.vstack(ys)
        cm = color_map(classes)
        for i, c in enumerate(classes):
            if numeric:
                fig.add_scatter(x=grid, y=arr[:, i], mode="lines", name=str(c), line=dict(color=cm[c], width=2.5))
            else:
                fig.add_bar(x=[str(g) for g in grid], y=arr[:, i], name=str(c), marker_color=cm[c])
    return _base(fig, f"How the prediction changes with {feature}", height=380, legend=classes is not None, xt=feature, yt=ylabel)


def whatif(df: pd.DataFrame, unit: str):
    d = df.head(15).iloc[::-1]
    col = [SERIES[7] if v > 0 else SERIES[0] for v in d["Effect"]]
    labels = [f"{f} = {v}" for f, v in zip(d["Factor"], d["Patient value"])]
    fig = go.Figure(go.Bar(x=d["Effect"], y=labels, orientation="h", marker_color=col,
                           customdata=d["Typical value"].astype(str),
                           hovertemplate="%{y}<br>Effect: %{x:+.3f} " + unit + "<br>(compared with typical value %{customdata})<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    return _base(fig, "Why this prediction? (red pushes it up, blue pushes it down)", height=max(300, 28 * len(d) + 110),
                 legend=False, xt=f"Change in prediction ({unit})")


def probability_bar(probs: dict, highlight: str | None = None):
    labels = list(probs)
    cm = color_map(sorted(labels))
    fig = go.Figure(go.Bar(x=[probs[l] * 100 for l in labels], y=labels, orientation="h",
                           marker_color=[cm[l] for l in labels], text=[f"{probs[l] * 100:.1f}%" for l in labels],
                           textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_xaxes(range=[0, 110])
    return _base(fig, "Predicted probability", height=120 + 44 * len(labels), legend=False, xt="%")


# ---------------------------------------------------------------------------
# What does the algorithm do to my data?
# ---------------------------------------------------------------------------
def decision_surface(ga, gb, Z, X2: pd.DataFrame, y, classes, task, title, labels=None):
    a, b = X2.columns
    fig = go.Figure()
    if task == "regression":
        fig.add_contour(x=ga, y=gb, z=Z, colorscale=SEQ, opacity=0.85, contours=dict(showlines=False), ncontours=20,
                        colorbar=dict(title="Predicted"), hovertemplate=f"{a}: %{{x:.1f}}<br>{b}: %{{y:.1f}}<br>Predicted: %{{z:.2f}}<extra></extra>")
        fig.add_scatter(x=X2[a], y=X2[b], mode="markers", marker=dict(color=y, colorscale=SEQ, size=8, line=dict(color=INK, width=1)),
                        name="Patients", hovertemplate=f"{a}: %{{x}}<br>{b}: %{{y}}<br>Actual: %{{marker.color:.2f}}<extra></extra>")
        return _base(fig, title, height=480, xt=a, yt=b)
    names = labels or classes
    if len(classes) == 2:
        fig.add_contour(x=ga, y=gb, z=Z, colorscale=[[0, "#dbe8f8"], [0.5, "#f7f5f2"], [1, "#fbd9c9"]], zmin=0, zmax=1,
                        contours=dict(start=0, end=1, size=0.1, showlines=False), opacity=0.95,
                        colorbar=dict(title=f"P({names[1]})", tickformat=".0%"),
                        hovertemplate=f"{a}: %{{x:.1f}}<br>{b}: %{{y:.1f}}<br>P({names[1]}): %{{z:.0%}}<extra></extra>")
        fig.add_contour(x=ga, y=gb, z=Z, showscale=False, contours=dict(start=0.5, end=0.5, size=0.1, coloring="lines"),
                        line=dict(color=INK, width=2.5), colorscale=[[0, INK], [1, INK]], hoverinfo="skip", name="Boundary (50%)")
        cm = {0: SERIES[0], 1: SERIES[1]}
        yy = np.asarray(y)
        for v in [0, 1]:
            m = yy == v
            fig.add_scatter(x=X2[a][m], y=X2[b][m], mode="markers", name=f"Actually {names[v]}",
                            marker=dict(color=cm[v], size=8, line=dict(color="#ffffff", width=1.2)),
                            hovertemplate=f"{a}: %{{x}}<br>{b}: %{{y}}<br>Actual: {names[v]}<extra></extra>")
    else:
        k = len(classes)
        cm = color_map(classes)
        light = {c: _lighten(cm[c], 0.72) for c in classes}
        scale = []
        for i, c in enumerate(classes):
            scale += [[i / k, light[c]], [(i + 1) / k, light[c]]]
        fig.add_heatmap(x=ga, y=gb, z=Z, colorscale=scale, zmin=-0.5, zmax=k - 0.5, showscale=False,
                        customdata=np.vectorize(lambda i: str(classes[int(i)]))(Z),
                        hovertemplate=f"{a}: %{{x:.1f}}<br>{b}: %{{y:.1f}}<br>Model says: %{{customdata}}<extra></extra>")
        yy = np.asarray(y, dtype=object)
        for c in classes:
            m = yy == c
            fig.add_scatter(x=X2[a][m], y=X2[b][m], mode="markers", name=str(c),
                            marker=dict(color=cm[c], size=8, line=dict(color="#ffffff", width=1.2)),
                            hovertemplate=f"{a}: %{{x}}<br>{b}: %{{y}}<br>Actual: {c}<extra></extra>")
    fig.update_xaxes(range=[ga[0], ga[-1]], showgrid=False)
    fig.update_yaxes(range=[gb[0], gb[-1]], showgrid=False)
    return _base(fig, title, height=480, xt=a, yt=b)


def _lighten(hex_color, amount):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def elbow(ks, inertias, sils):
    fig = go.Figure()
    fig.add_scatter(x=ks, y=sils, mode="lines+markers", name="Silhouette score (higher = clearer groups)",
                    line=dict(color=SERIES[0], width=2.5), marker=dict(size=9),
                    hovertemplate="k = %{x}: silhouette %{y:.3f}<extra></extra>")
    best = ks[int(np.nanargmax(sils))]
    fig.add_vline(x=best, line=dict(color=SERIES[3], width=2, dash="dash"))
    fig.add_annotation(x=best, y=float(np.nanmax(sils)), text=f"best k = {best}", showarrow=True, arrowhead=2, ay=-30,
                       font=dict(color=INK))
    return _base(fig, "How many groups? (silhouette score for each k)", height=360, xt="Number of groups (k)", yt="Silhouette score")


def inertia_plot(ks, inertias):
    fig = go.Figure(go.Scatter(x=ks, y=inertias, mode="lines+markers", line=dict(color=SERIES[1], width=2.5), marker=dict(size=9),
                               hovertemplate="k = %{x}: within-group spread %{y:.0f}<extra></extra>"))
    return _base(fig, "Elbow plot (look for the bend)", height=360, legend=False, xt="Number of groups (k)",
                 yt="Within-group spread (inertia)")


def cluster_scatter(coords, labels, hover: pd.DataFrame | None = None, axis_titles=("Component 1", "Component 2")):
    fig = go.Figure()
    labs = sorted(pd.unique(labels), key=lambda v: (str(v) == "Noise", str(v)))
    cm = color_map([l for l in labs if str(l) != "Noise"])
    cm["Noise"] = AXIS
    for lab in labs:
        m = np.asarray(labels) == lab
        fig.add_scatter(x=coords[m, 0], y=coords[m, 1], mode="markers", name=str(lab),
                        marker=dict(color=cm[lab], size=9, line=dict(color="#ffffff", width=1.2), opacity=0.85),
                        hovertemplate=f"{lab}<extra></extra>")
    return _base(fig, "Patients drawn on a 2-D map (PCA), coloured by group", height=480, xt=axis_titles[0], yt=axis_titles[1])


def cluster_profile(z: pd.DataFrame):
    fig = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=z.index, colorscale=DIVERGING, zmid=0, zmin=-2, zmax=2,
                               text=z.round(2).values, texttemplate="%{text}", colorbar=dict(title="SD from<br>average"),
                               hovertemplate="%{y}<br>%{x}: %{z:+.2f} SD from the average patient<extra></extra>", xgap=2, ygap=2))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, "Profile of each group (red = higher than average, blue = lower)", height=140 + 50 * len(z), legend=False)


# ---------------------------------------------------------------------------
# Teaching charts
# ---------------------------------------------------------------------------
def sigmoid_demo(w=1.0, b=0.0):
    x = np.linspace(-8, 8, 200)
    p = 1 / (1 + np.exp(-(w * x + b)))
    fig = go.Figure()
    fig.add_scatter(x=x, y=p, mode="lines", line=dict(color=SERIES[0], width=3), name="Probability",
                    hovertemplate="Score %{x:.1f} -> probability %{y:.2f}<extra></extra>")
    fig.add_hline(y=0.5, line=dict(color=MUTED, dash="dot"))
    return _base(fig, "The logistic (S-shaped) curve turns a risk score into a probability", height=340, legend=False,
                 xt="Risk score (sum of weight x value)", yt="Probability")


def overfit_demo(degree: int, seed: int = 2):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, 14))

    def truth(t):
        return 30 + 11 * t - 0.95 * t ** 2
    y = truth(x) + rng.normal(0, 5, len(x))
    xt = np.sort(rng.uniform(0.3, 9.7, 30))
    yt = truth(xt) + rng.normal(0, 5, len(xt))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coef = np.polyfit(x, y, degree)
    grid = np.linspace(0, 10, 400)
    fit = np.polyval(coef, grid)
    tr_err = float(np.sqrt(np.mean((np.polyval(coef, x) - y) ** 2)))
    te_err = float(np.sqrt(np.mean((np.polyval(coef, xt) - yt) ** 2)))
    fig = go.Figure()
    fig.add_scatter(x=grid, y=truth(grid), mode="lines", name="True relationship", line=dict(color=MUTED, dash="dot", width=2))
    fig.add_scatter(x=grid, y=np.clip(fit, -20, 120), mode="lines", name=f"Model (complexity {degree})",
                    line=dict(color=SERIES[3], width=3))
    fig.add_scatter(x=x, y=y, mode="markers", name="Training patients", marker=dict(color=SERIES[0], size=10, line=dict(color="#fff", width=1)))
    fig.add_scatter(x=xt, y=yt, mode="markers", name="New (test) patients", marker=dict(color=SERIES[1], size=9, symbol="diamond",
                    line=dict(color="#fff", width=1)))
    fig.update_yaxes(range=[0, 100])
    return _base(fig, f"Training error {tr_err:.1f}  |  Error on new patients {te_err:.1f}", height=420,
                 xt="Dose (mg)", yt="Response"), tr_err, te_err


def kmeans_animation(X: np.ndarray, k: int = 3, seed: int = 1, max_steps: int = 6):
    rng = np.random.default_rng(seed)
    centres = X[rng.choice(len(X), k, replace=False)].copy()
    frames = []
    cols = SERIES[:k]
    for step in range(max_steps):
        d = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        frames.append((centres.copy(), lab.copy()))
        new = np.array([X[lab == j].mean(axis=0) if (lab == j).any() else centres[j] for j in range(k)])
        if np.allclose(new, centres):
            break
        centres = new

    def traces(c, lab):
        tr = [go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(color=[cols[i] for i in lab], size=8,
                         line=dict(color="#fff", width=1)), showlegend=False, hoverinfo="skip")]
        tr.append(go.Scatter(x=c[:, 0], y=c[:, 1], mode="markers", marker=dict(color=cols, size=22, symbol="x",
                             line=dict(color=INK, width=2)), showlegend=False, hoverinfo="skip"))
        return tr

    fig = go.Figure(data=traces(*frames[0]),
                    frames=[go.Frame(data=traces(c, l), name=str(i + 1)) for i, (c, l) in enumerate(frames)])
    fig.update_layout(updatemenus=[dict(type="buttons", x=0, y=-0.12, xanchor="left", showactive=False,
                                        buttons=[dict(label="Play", method="animate",
                                                      args=[None, dict(frame=dict(duration=1100, redraw=True), fromcurrent=True)])])],
                      sliders=[dict(active=0, x=0.12, len=0.88, y=-0.05, currentvalue=dict(prefix="Step "),
                                    steps=[dict(label=str(i + 1), method="animate",
                                                args=[[str(i + 1)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                                           for i in range(len(frames))])])
    return _base(fig, "K-means: centres (X) move until the groups stop changing", height=480, legend=False,
                 xt="BMI (scaled)", yt="Fasting glucose (scaled)")


def staged_curve(train_scores, test_scores, metric):
    n = np.arange(1, len(train_scores) + 1)
    fig = go.Figure()
    fig.add_scatter(x=n, y=train_scores, mode="lines", name="Training patients", line=dict(color=SERIES[1], width=2, dash="dash"))
    fig.add_scatter(x=n, y=test_scores, mode="lines", name="New (test) patients", line=dict(color=SERIES[0], width=2.5))
    return _base(fig, "Boosting: performance as trees are added one by one", height=380, xt="Number of trees", yt=metric)


def loss_curve(loss):
    fig = go.Figure(go.Scatter(x=np.arange(1, len(loss) + 1), y=loss, mode="lines", line=dict(color=SERIES[0], width=2.5),
                               hovertemplate="Pass %{x}: error %{y:.4f}<extra></extra>"))
    return _base(fig, "Neural network: training error falls with every pass (epoch)", height=360, legend=False,
                 xt="Pass over the training data (epoch)", yt="Training error (loss)")
