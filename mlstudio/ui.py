"""Shared look-and-feel and helpers for every page."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd
import streamlit as st

from . import APP_NAME, APP_VERSION
from .plots import PLOT_CONFIG

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

CSS = """
<style>
:root { --mamc:#7b1c2e; --mamc-dark:#4a0f1b; --mamc-light:#f6e9ec; --ink:#1f1f1d; --muted:#5d5c58; --line:#e6e4df; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
h1, h2, h3 { color: var(--ink); letter-spacing: -0.2px; }
.page-head { display:flex; align-items:flex-start; gap:14px; margin: 0 0 8px; }
.page-head .step { background: var(--mamc); color:#fff; border-radius: 10px; min-width: 44px; height: 44px; display:flex;
  align-items:center; justify-content:center; font-weight:800; font-size: 20px; }
.page-head h1 { font-size: 1.85rem; margin: 0; padding: 0; font-weight: 800; }
.page-head p { color: var(--muted); margin: 2px 0 0; font-size: 1.02rem; }
.hero-wrap { container-type: inline-size; }
.hero { background:#fff; border:1px solid var(--line); border-left: 8px solid var(--mamc); border-radius:16px;
  padding: 26px 28px; display:flex; gap:26px; align-items:center; margin-bottom: 18px; }
@container (max-width: 780px) {
  .hero { flex-wrap: wrap; justify-content: center; gap: 14px 40px; }
  .hero .hero-text { order: 3; flex-basis: 100%; }
  .hero img { width: 92px; }
}
.hero img { width: 120px; flex: 0 0 auto; }
.hero .hero-text { flex: 1 1 auto; }
.hero .kicker { color: var(--mamc); font-weight: 700; letter-spacing: 2.5px; font-size: 12.5px; text-transform: uppercase; }
.hero h1 { font-size: 2.6rem; margin: 2px 0 2px; font-weight: 800; color: var(--ink) !important; }
.hero h1 span { color: var(--mamc); }
.hero .inst { font-weight: 600; color: var(--ink); margin: 0 0 6px; }
.hero .lead { color: var(--muted); margin: 0 0 10px; font-size: 1.04rem; }
.pill { display:inline-block; background: var(--mamc-light); color: var(--mamc); border-radius: 20px; padding: 3px 11px;
  font-size: 12.5px; font-weight: 600; margin: 3px 6px 3px 0; }
.card { background:#fff; border:1px solid var(--line); border-radius: 14px; padding: 16px 18px; height: 100%; }
.card .num { color: var(--mamc); font-weight: 800; font-size: 12.5px; letter-spacing: 1.5px; }
.card h4 { margin: 2px 0 6px; font-size: 1.04rem; color: var(--ink); }
.card p { color: var(--muted); font-size: .93rem; margin: 0; }
.box { border-radius: 10px; padding: 11px 15px; margin: 8px 0 12px; font-size: .97rem; line-height: 1.5; }
.box b.t { display:block; margin-bottom: 2px; }
.box.note  { background:#eef5fb; border-left: 5px solid #2a78d6; }
.box.warn  { background:#fff6e0; border-left: 5px solid #e0a000; }
.box.good  { background:#e9f5ea; border-left: 5px solid #2e7d32; }
.box.bad   { background:#fdecea; border-left: 5px solid #b3261e; }
.box.learn { background:#fbf7f0; border-left: 5px solid var(--mamc); }
.box.learn b.t { color: var(--mamc); }
.issue { background:#fff; border:1px solid var(--line); border-radius: 12px; padding: 12px 15px; margin-bottom: 6px; }
.sev { display:inline-block; border-radius: 6px; padding: 1px 8px; font-size: 12px; font-weight: 700; margin-right: 8px; }
.sev.high { background:#fdecea; color:#b3261e; } .sev.medium { background:#fff2d6; color:#8a5a00; }
.sev.low { background:#eef5fb; color:#1c5cab; } .sev.info { background:#f1f1ef; color:#52514e; }
.tile { background:#fff; border:1px solid var(--line); border-radius: 12px; padding: 12px 14px; height: 100%; }
.tile .lab { color: var(--muted); font-size: .86rem; }
.tile .val { font-size: 1.65rem; font-weight: 750; color: var(--ink); line-height: 1.2; }
.tile .ci { color: var(--muted); font-size: .8rem; }
.tile .say { color: var(--ink); font-size: .86rem; margin-top: 4px; }
.verdict { border-radius: 12px; padding: 12px 16px; margin: 6px 0 12px; }
.verdict.good { background:#e9f5ea; border:1px solid #b9dfbb; } .verdict.mild { background:#fff6e0; border:1px solid #f1d58a; }
.verdict.overfit { background:#fff1e8; border:1px solid #f3c2a3; } .verdict.underfit { background:#fdecea; border:1px solid #f2b8b5; }
.flow { display:flex; flex-wrap: wrap; gap: 6px; align-items:center; margin: 6px 0 10px; }
.flow .st { background:#fff; border:1px solid var(--line); border-radius: 10px; padding: 7px 11px; font-size: .9rem; }
.flow .st b { color: var(--mamc); }
.flow .ar { color: var(--muted); font-weight: 700; }
.about-wrap { container-type: inline-size; }
.about-credit { background:#fff; border:1px solid var(--line); border-radius: 16px; padding: 22px 120px; text-align:center; position: relative; }
@container (max-width: 760px) {
  .about-credit { padding: 22px 20px; }
  .about-credit img { position: static !important; width: 70px !important; margin: 0 14px 10px; }
  .about-credit .names { grid-template-columns: 1fr; }
}
.about-credit img.l { position:absolute; left: 22px; top: 22px; width: 84px; }
.about-credit img.r { position:absolute; right: 22px; top: 22px; width: 84px; }
.about-credit .lbl { color: var(--muted); letter-spacing: 2px; text-transform: uppercase; font-size: 12.5px; margin-bottom: 8px; }
.about-credit .names { display:grid; grid-template-columns: 1fr 1fr; gap: 14px 28px; margin: 6px 0 10px; }
.about-credit .names div { text-align:center; } .about-credit .names b { display:block; font-size: 1.05rem; color: var(--ink); }
.about-credit .mic { color: var(--mamc); font-weight: 700; margin-top: 6px; }
.small { color: var(--muted); font-size: .86rem; }
.foot { color: var(--muted); font-size: .82rem; margin-top: 26px; border-top: 1px solid var(--line); padding-top: 8px; }
section[data-testid="stSidebar"] .stProgress > div > div > div > div { background-color: var(--mamc); }
div[data-testid="stExpander"] details summary p { font-weight: 600; }

</style>
"""


def img_b64(name: str) -> str:
    p = ASSETS / name
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


def setup_page():
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()


def init_state():
    ss = st.session_state
    defaults = dict(df=None, df_raw=None, data_name=None, data_source=None, clean_log=[], target=None, task=None,
                    positive=None, features=[], results={}, split=None, split_sig=None, data_version=0, demo=None,
                    priority="balanced", cluster=None, dismissed=set(), chosen_models=[])
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v.copy() if isinstance(v, (list, dict, set)) else v


def esc(s) -> str:
    return html.escape(str(s))


def page_head(step: str | None, title: str, subtitle: str = ""):
    stp = f'<div class="step">{esc(step)}</div>' if step else ""
    st.markdown(f'<div class="page-head">{stp}<div><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div></div>',
                unsafe_allow_html=True)


def box(kind: str, text: str, title: str | None = None):
    """kind: note | warn | good | bad | learn. Text may contain **bold** markdown-like markers."""
    t = f'<b class="t">{esc(title)}</b>' if title else ""
    st.markdown(f'<div class="box {kind}">{t}{_md(text)}</div>', unsafe_allow_html=True)


def simple(text: str, title: str = "In simple words"):
    box("learn", text, title)


def _md(text: str) -> str:
    import re
    s = esc(text)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", s)
    return s.replace("\n", "<br>")


def tile(label: str, value: str, ci: str = "", say: str = ""):
    st.markdown(f'<div class="tile"><div class="lab">{esc(label)}</div><div class="val">{esc(value)}</div>'
                f'<div class="ci">{esc(ci)}</div><div class="say">{_md(say)}</div></div>', unsafe_allow_html=True)


def flow(steps: list[tuple[str, str]]):
    parts = []
    for i, (a, b) in enumerate(steps):
        parts.append(f'<div class="st"><b>{esc(a)}</b> {esc(b)}</div>')
        if i < len(steps) - 1:
            parts.append('<div class="ar">&rarr;</div>')
    st.markdown(f'<div class="flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def show(fig, key=None):
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CONFIG, key=key)


def display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Make a frame safe to display: columns that mix numbers and text are shown as text."""
    d = df.copy()
    for c in d.columns:
        if d[c].dtype == object:
            types = {type(v) for v in d[c].dropna()}
            if len(types) > 1:
                d[c] = d[c].map(lambda v: v if pd.isna(v) else str(v)).astype(object)
    return d


def table(df: pd.DataFrame, fmt: dict | None = None, height=None, key=None):
    d = df.copy()
    if fmt:
        for c, f in fmt.items():
            if c in d:
                d[c] = d[c].map(lambda v: f.format(v) if isinstance(v, (int, float)) and pd.notna(v) else v)
    kw = dict(width="stretch", hide_index=True)
    if height:
        kw["height"] = height
    st.dataframe(d, key=key, **kw)


def pct(v, digits=1):
    try:
        return f"{100 * float(v):.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def num(v, digits=3):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "-"
    if f != f:
        return "-"
    return f"{f:.{digits}f}"


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def set_data(df: pd.DataFrame, name: str, source: str, demo: str | None = None, target: str | None = None):
    ss = st.session_state
    ss.df_raw = df.copy()
    ss.df = df.copy()
    ss.data_name = name
    ss.data_source = source
    ss.clean_log = []
    ss.target = None
    ss.task = None
    ss.positive = None
    ss.features = []
    ss.results = {}
    ss.split = None
    ss.split_sig = None
    ss.cluster = None
    ss.dismissed = set()
    ss.chosen_models = []
    ss.data_version += 1
    ss.demo = demo
    if target and target in df.columns:
        from .advisor import detect_task
        ss.target = target
        ss.task = detect_task(df[target])[0]


def update_data(df: pd.DataFrame, log_entries: list):
    ss = st.session_state
    ss.df = df
    ss.clean_log = ss.clean_log + [e for e in log_entries if e]
    ss.data_version += 1
    ss.results = {}
    ss.split = None
    ss.split_sig = None
    if ss.target and ss.target not in df.columns:
        ss.target = None
    ss.features = [f for f in ss.features if f in df.columns]


def need_data():
    if st.session_state.get("df") is None:
        box("warn", "No data loaded yet. Go to **1. Load data** and import your Excel/CSV file or pick a demo dataset.",
            "Load some data first")
        st.page_link("app_pages/load.py", label="Go to 1. Load data", icon=":material/upload_file:")
        st.stop()


def need_setup():
    need_data()
    ss = st.session_state
    if ss.target and ss.target in ss.df.columns and ss.task in ("classification", "regression") and not ss.features:
        from .advisor import usable_features
        ss.features = usable_features(ss.df, ss.target)
    if not ss.target or ss.task not in ("classification", "regression") or not ss.features:
        box("warn", "Choose the outcome you want to predict and the predictors first (step **4. Choose outcome & model**).",
            "Set up the prediction question first")
        st.page_link("app_pages/choose.py", label="Go to 4. Choose outcome & model", icon=":material/target:")
        st.stop()


def need_results():
    need_setup()
    if not st.session_state.results:
        box("warn", "No model has been trained yet. Train at least one model in step **5. Train & tune**.", "Train a model first")
        st.page_link("app_pages/train.py", label="Go to 5. Train & tune", icon=":material/model_training:")
        st.stop()


def demo_banner(text: str, next_page: str | None = None, next_label: str | None = None):
    if st.session_state.get("demo"):
        box("note", text, "Guided demo")
        if next_page:
            st.page_link(next_page, label=next_label or "Next step", icon=":material/arrow_forward:")


def next_step(page: str, label: str, icon=":material/arrow_forward:"):
    st.divider()
    st.page_link(page, label=label, icon=icon)


def footer():
    st.markdown(f'<div class="foot">{APP_NAME} v{APP_VERSION} &middot; Maulana Azad Medical College, New Delhi &middot; '
                'Free &amp; open source &middot; Your data never leave your computer &middot; For learning and research - '
                'not for clinical decisions on individual patients.</div>', unsafe_allow_html=True)


def sidebar_progress():
    ss = st.session_state
    with st.sidebar:
        st.markdown("**Your progress**")
        steps = [
            ("Data loaded", ss.df is not None, ss.data_name or ""),
            ("Data checked / cleaned", bool(ss.clean_log), f"{len(ss.clean_log)} fix(es)" if ss.clean_log else ""),
            ("Outcome chosen", bool(ss.target), f"{ss.target}" if ss.target else ""),
            ("Models trained", bool(ss.results), f"{len(ss.results)} model(s)" if ss.results else ""),
        ]
        for name, ok, extra in steps:
            icon = ":material/check_circle:" if ok else ":material/radio_button_unchecked:"
            st.markdown(f"{icon} {name}" + (f"  \n<span class='small'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{esc(extra)}</span>" if extra else ""),
                        unsafe_allow_html=True)
        st.caption(f"{APP_NAME} v{APP_VERSION}")
