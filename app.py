"""MAMC ML Studio - start here.

Run with:   streamlit run app.py
(or double-click the launcher for your computer).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from mlstudio import APP_NAME  # noqa: E402
from mlstudio import ui  # noqa: E402

st.set_page_config(page_title=APP_NAME, page_icon=str(ROOT / "assets" / "favicon.png"), layout="wide",
                   initial_sidebar_state="expanded")

P = "app_pages/"
pages = {
    "": [st.Page(P + "home.py", title="Home", icon=":material/home:", default=True)],
    "Build a prediction model": [
        st.Page(P + "load.py", title="1. Load data", icon=":material/upload_file:"),
        st.Page(P + "clean.py", title="2. Check & clean data", icon=":material/cleaning_services:"),
        st.Page(P + "explore.py", title="3. Explore data", icon=":material/query_stats:"),
        st.Page(P + "choose.py", title="4. Choose outcome & model", icon=":material/target:"),
        st.Page(P + "train.py", title="5. Train & tune", icon=":material/model_training:"),
        st.Page(P + "results.py", title="6. Results", icon=":material/insights:"),
        st.Page(P + "predict.py", title="7. Predict a new patient", icon=":material/person_search:"),
        st.Page(P + "report.py", title="8. Report & Python code", icon=":material/description:"),
    ],
    "Learn & other tools": [
        st.Page(P + "learn.py", title="Learn machine learning", icon=":material/school:"),
        st.Page(P + "clusters.py", title="Find patient groups", icon=":material/bubble_chart:"),
        st.Page(P + "samplesize.py", title="Sample size calculator", icon=":material/calculate:"),
        st.Page(P + "about.py", title="About", icon=":material/info:"),
    ],
}

try:
    st.logo(str(ROOT / "assets" / "mamc_logo.png"), size="large")
except TypeError:
    st.logo(str(ROOT / "assets" / "mamc_logo.png"))

ui.setup_page()
nav = st.navigation(pages, position="sidebar", expanded=True)
ui.sidebar_progress()
nav.run()
