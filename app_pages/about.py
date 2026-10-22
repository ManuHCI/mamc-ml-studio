import platform

import streamlit as st

from mlstudio import APP_NAME, APP_VERSION, ui

ui.page_head(None, "About", f"{APP_NAME} - version {APP_VERSION}")

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("#### What it is")
        st.write("A free, point-and-click machine-learning workbench for medical students, residents and researchers. "
                 "It teaches the full workflow - data cleaning, model choice, training, tuning, validation and "
                 "interpretation - with plain-language explanations at every step, and exports a Word report and the "
                 "equivalent Python code.")
    with st.container(border=True):
        st.markdown("#### Free & open source")
        st.write("Released under the GNU General Public License v3.0. Anyone may use, study, share and improve it. It runs "
                 "entirely on your computer - no data ever leave your laptop.")
with c2:
    with st.container(border=True):
        st.markdown("#### Built with")
        import numpy
        import pandas
        import plotly
        import sklearn
        st.write(f"Python {platform.python_version()} - scikit-learn {sklearn.__version__} - pandas {pandas.__version__} - "
                 f"NumPy {numpy.__version__} - Plotly {plotly.__version__} - Streamlit {st.__version__}")
    with st.container(border=True):
        st.markdown("#### Important")
        st.write("For education and research. Predictions must not be used for decisions about individual patients "
                 "unless the model has been properly validated and approved.")

st.write("")
st.markdown("<p style='text-align:center;font-weight:700;font-size:1.08rem;margin:10px 0 12px'>Developed by Faculties of "
            "Maulana Azad Medical College &amp; Govind Ballabh Pant Hospital, University of Delhi, New Delhi.</p>", unsafe_allow_html=True)
st.markdown(
    "<div class='about-wrap'><div class='about-credit'>"
    f"<img class='l' src='data:image/png;base64,{ui.img_b64('mamc_logo.png')}' alt='MAMC logo'>"
    f"<img class='r' src='data:image/png;base64,{ui.img_b64('gbpant_logo.png')}' alt='GB Pant Hospital logo'>"
    "<div class='lbl'>Contributed by</div>"
    "<div class='names'>"
    "<div><b>Dr. Pradeep Kr. Dabla</b>Director Professor, Department of Biochemistry,<br>GB&nbsp;Pant&nbsp;Hospital</div>"
    "<div><b>Dr. Manu Kumar Shetty</b>Professor, Department of Pharmacology</div>"
    "</div>"
    "<div class='mic'>Medical Innovation Centre, Maulana Azad Medical College, New Delhi</div>"
    "</div></div>", unsafe_allow_html=True)
ui.footer()
