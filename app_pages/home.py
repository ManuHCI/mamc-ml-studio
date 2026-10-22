import streamlit as st

from mlstudio import APP_VERSION, ui
from mlstudio.demo_data import DEMO_DATASETS, load_demo

logo = ui.img_b64("mamc_logo.png")
logo2 = ui.img_b64("gbpant_logo.png")
st.markdown(f"""
<div class="hero-wrap"><div class="hero">
  <img src="data:image/png;base64,{logo}" alt="MAMC logo">
  <div class="hero-text">
    <div class="kicker">Machine learning for medical students &amp; researchers</div>
    <h1>MAMC <span>ML Studio</span></h1>
    <div class="inst">Maulana Azad Medical College &amp; Govind Ballabh Pant Hospital, New Delhi</div>
    <p class="lead">Build, test and understand prediction models from your own data - without writing a single line of code.
    Every step is explained in simple words.</p>
    <span class="pill">Import Excel / CSV</span><span class="pill">Data Health Check</span>
    <span class="pill">Model advisor: which &amp; why</span><span class="pill">20 ML methods</span>
    <span class="pill">Automatic tuning</span><span class="pill">Doctor-friendly results</span>
    <span class="pill">Word report + Python code</span><span class="pill">Works offline</span>
  </div>
  <img class="r" src="data:image/png;base64,{logo2}" alt="GB Pant Hospital logo">
</div></div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Start: load my own data", type="primary", width="stretch", icon=":material/upload_file:"):
        st.switch_page("app_pages/load.py")
with c2:
    if st.button("Guided demo (328 patients, step by step)", width="stretch", icon=":material/play_circle:"):
        info = DEMO_DATASETS["messy"]
        ui.set_data(load_demo("messy"), info["title"], "demo:messy", demo="messy", target=info["target"])
        st.switch_page("app_pages/load.py")
with c3:
    if st.button("New to ML? Learn the basics first", width="stretch", icon=":material/school:"):
        st.switch_page("app_pages/learn.py")

st.write("")
steps = [
    ("STEP 1-2", "Load & clean", "Import Excel/CSV. The Data Health Check finds errors, explains why they matter and fixes them."),
    ("STEP 3-4", "Explore & choose", "See your data in charts. Pick the outcome; the advisor recommends models and explains why."),
    ("STEP 5", "Train & tune", "Watch the train/test split, cross-validation and automatic setting search, explained visually."),
    ("STEP 6-8", "Results & report", "Sensitivity, specificity, AUC with 95% CI, ROC, key factors, a Word report and Python code."),
]
cols = st.columns(4)
for col, (n, h, p) in zip(cols, steps):
    col.markdown(f'<div class="card"><div class="num">{n}</div><h4>{h}</h4><p>{p}</p></div>', unsafe_allow_html=True)

st.write("")
a, b = st.columns([1.15, 1])
with a:
    with st.container(border=True):
        st.markdown("#### Before you start - prepare your Excel sheet")
        st.markdown("""
- Row 1 = short column names (`Age`, `Sex`, `BMI`, `Fasting_glucose`); no merged cells or titles
- One row per patient, one column per variable; leave missing cells **blank**
- One **outcome** column - what you want to predict (e.g. `Diabetes_5yr` Yes/No, or `Length_of_stay_days`)
- Only use predictors that are known **before** the outcome (no 'discharge diagnosis' to predict the disease)
- Remove names, phone numbers and hospital IDs
""")
        st.page_link("app_pages/load.py", label="See the full data checklist and the demo datasets", icon=":material/checklist:")
with b:
    with st.container(border=True):
        st.markdown("#### Machine learning in 30 seconds")
        ui.simple("A machine-learning model is like a resident who learns from many previous case records (the **training "
                  "set**). After studying them, it is given new cases it has never seen (the **test set**) - just like an "
                  "exam - to check whether it really learned the pattern or only memorised the cases.")
        st.page_link("app_pages/learn.py", label="Open 'Learn machine learning'", icon=":material/school:")

st.markdown(f'<p class="small">Version {APP_VERSION} &middot; Powered by Python (scikit-learn, pandas, Plotly) &middot; '
            'Your data never leave your computer</p>', unsafe_allow_html=True)
