import hashlib
import io

import pandas as pd
import streamlit as st

from mlstudio import ui
from mlstudio.data_check import DATA_REQUIREMENTS, excel_sheets, overview, read_table
from mlstudio.demo_data import DEMO_DATASETS, load_demo

@st.cache_data(show_spinner=False)
def demo_xlsx(key: str) -> bytes:
    buf = io.BytesIO()
    DEMO_DATASETS[key]["func"]().to_excel(buf, index=False)
    return buf.getvalue()


ui.page_head("1", "Load data", "Import your Excel / CSV file, or practise with a demo dataset.")
ss = st.session_state

if ss.demo:
    ui.demo_banner("The demo dataset is loaded: **328 patients** from a 5-year diabetes follow-up study, typed into Excel "
                   "with the usual mistakes. Look at the overview at the bottom of this page (types, blanks), then go to "
                   "**2. Check & clean data**.", "app_pages/clean.py", "Demo: continue to 2. Check & clean data")

tab_up, tab_demo, tab_need = st.tabs([":material/upload_file: Upload your file", ":material/dataset: Demo datasets",
                                      ":material/checklist: What data do I need?"])

with tab_up:
    up = st.file_uploader("Excel (.xlsx, .xls) or CSV file", type=["xlsx", "xls", "xlsm", "csv", "txt"],
                          help="Your file is read only on this computer. Nothing is uploaded to the internet.")
    if up is not None:
        data = up.getvalue()
        sheets = excel_sheets(data, up.name)
        c1, c2 = st.columns(2)
        sheet = c1.selectbox("Sheet", sheets, index=0) if len(sheets) > 1 else (sheets[0] if sheets else None)
        header = c2.number_input("Column names are in row", 1, 50, 1,
                                 help="Usually row 1. Change it if your sheet has a title above the column names.")
        try:
            df_new = read_table(data, up.name, sheet if sheets else None, header)
        except Exception as e:  # noqa: BLE001
            ui.box("bad", f"Could not read this file: {e}", "Problem reading the file")
            st.stop()
        st.caption(f"Preview: {df_new.shape[0]} rows x {df_new.shape[1]} columns")
        st.dataframe(ui.display_df(df_new.head(8)), width="stretch")
        sig = hashlib.md5(data).hexdigest() + str(sheet) + str(header)
        if st.button("Use this data", type="primary", icon=":material/check:"):
            ui.set_data(df_new, up.name + (f" [{sheet}]" if sheet and len(sheets) > 1 else ""), "upload:" + sig)
            st.rerun()

with tab_demo:
    ui.box("note", "All demo datasets are synthetic (computer-generated patients that behave like real ones), except the "
                   "breast-cancer data, which is a real, public dataset. Sample sizes are fixed so every student gets the same results.")
    keys = list(DEMO_DATASETS)
    for i in range(0, len(keys), 2):
        cols = st.columns(2)
        for col, k in zip(cols, keys[i:i + 2]):
            info = DEMO_DATASETS[k]
            with col.container(border=True):
                st.markdown(f"**{info['title']}**")
                st.caption(f"{info['kind']} - **{info['n']} patients**")
                st.write(info["story"])
                b1, b2 = st.columns(2)
                if b1.button("Load this dataset", key=f"demo_{k}", icon=info["icon"], width="stretch"):
                    ui.set_data(load_demo(k), info["title"], f"demo:{k}", target=info["target"])
                    st.rerun()
                b2.download_button("Download as Excel", demo_xlsx(k), file_name=f"demo_{k}.xlsx", key=f"dl_{k}",
                                   icon=":material/download:", width="stretch",
                                   help="Practise the import step with a real Excel file.")

with tab_need:
    ui.simple("A model can only learn what is in your data. Garbage in, garbage out: a clean, well-planned sheet matters "
              "more than the choice of algorithm.", "Why this matters")
    for t, d in DATA_REQUIREMENTS:
        st.markdown(f"**{t}.** {d}")
    st.markdown("**A good layout looks like this:**")
    st.dataframe(pd.DataFrame({"Age": [54, 61, 47], "Sex": ["Male", "Female", "Female"], "BMI": [27.4, 31.2, None],
                               "Fasting_glucose": [104, 118, 96], "Family_history": ["Yes", "No", "No"],
                               "Diabetes_5yr": ["No", "Yes", "No"]}), hide_index=True, width="content")
    st.caption("One row per patient, units in the column names, a blank cell for the missing BMI, outcome in the last column.")
    st.page_link("app_pages/samplesize.py", label="How many patients do I need? Open the sample size calculator",
                 icon=":material/calculate:")

# ----------------------------------------------------------------- loaded data
if ss.df is not None:
    st.divider()
    df = ss.df
    st.subheader(f"Loaded: {ss.data_name}")
    c = st.columns(4)
    with c[0]:
        ui.tile("Rows (patients)", f"{df.shape[0]}")
    with c[1]:
        ui.tile("Columns (variables)", f"{df.shape[1]}")
    with c[2]:
        ui.tile("Blank cells", f"{int(df.isna().sum().sum())}", f"{100 * df.isna().mean().mean():.1f}% of all cells")
    with c[3]:
        ui.tile("Duplicate rows", f"{int(df.dropna(how='all').duplicated().sum())}")
    st.markdown("##### What the software sees in each column")
    ov = overview(df)
    st.dataframe(ov, hide_index=True, width="stretch", height=min(36 * (len(ov) + 1) + 4, 760),
                 column_config={"Missing %": st.column_config.ProgressColumn("Missing %", min_value=0, max_value=100, format="%.1f%%")})
    ui.simple("**Number** columns can be used directly. **Categories** (e.g. Male/Female) are turned into 0/1 columns "
              "automatically. **ID / name**, **free text**, **dates** and columns with the **same value everywhere** "
              "cannot help a model - the Health Check will suggest removing them.", "How to read this table")
    with st.expander("Show the data (first 50 rows)"):
        st.dataframe(ui.display_df(df.head(50)), width="stretch")
    ui.next_step("app_pages/clean.py", "Next: 2. Check & clean data")
ui.footer()
