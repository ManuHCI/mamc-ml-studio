import pandas as pd
import streamlit as st

from mlstudio import plots, ui
from mlstudio.data_check import column_kind, is_num

ui.page_head("3", "Explore data", "Look before you model: distributions, differences between groups and relationships.")
ui.need_data()
ss = st.session_state
df = ss.df

ui.demo_banner("Open the **Outcome vs predictors** tab and look at Fasting_glucose and BMI: patients who developed diabetes "
               "had higher values. A model will try to learn exactly these differences - all at once.",
               "app_pages/choose.py", "Demo: continue to 4. Choose outcome & model")

kinds = {c: column_kind(df[c]) for c in df.columns}
num_cols = [c for c in df.columns if is_num(df[c]) and kinds[c] in ("numeric", "binary")]
cat_cols = [c for c in df.columns if kinds[c] in ("categorical", "binary") and not is_num(df[c])]
usable = num_cols + cat_cols

t1, t2, t3, t4, t5 = st.tabs([":material/bar_chart: One variable", ":material/scatter_plot: Two variables",
                              ":material/target: Outcome vs predictors", ":material/grid_on: Correlations",
                              ":material/grid_view: Blank cells"])

with t1:
    if not usable:
        st.info("No usable columns.")
    else:
        col = st.selectbox("Variable", usable, key="ex_one")
        if col in num_cols:
            c1, c2 = st.columns([2.2, 1])
            with c1:
                ui.show(plots.hist(df, col))
            with c2:
                s = df[col].astype(float)
                st.dataframe(pd.DataFrame({"Statistic": ["Patients with a value", "Mean", "SD", "Median", "IQR (Q1-Q3)", "Min - Max", "Blank"],
                                           "Value": [str(int(s.notna().sum())), f"{s.mean():.2f}", f"{s.std():.2f}", f"{s.median():.2f}",
                                                     f"{s.quantile(.25):.2f} - {s.quantile(.75):.2f}", f"{s.min():g} - {s.max():g}",
                                                     str(int(s.isna().sum()))]}), hide_index=True, width="stretch")
                skew = s.skew()
                if abs(skew) > 1:
                    ui.box("note", f"The distribution is clearly skewed (skewness {skew:.1f}). Tree-based models do not mind; "
                                   "for linear models, extreme values have extra influence.")
        else:
            ui.show(plots.bar_counts(df, col))

with t2:
    if len(usable) < 2:
        st.info("Need at least two usable columns.")
    else:
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("Horizontal (x)", usable, key="ex_x")
        y = c2.selectbox("Vertical (y)", [c for c in usable if c != x], key="ex_y")
        colour = c3.selectbox("Colour by (optional)", ["(none)"] + [c for c in usable if c not in (x, y)], key="ex_c")
        colour = None if colour == "(none)" else colour
        if x in num_cols and y in num_cols:
            ui.show(plots.scatter(df, x, y, colour))
            r = df[[x, y]].astype(float).corr().iloc[0, 1]
            ui.simple(f"Correlation r = {r:.2f}. Values near +1 or -1 mean a strong straight-line relationship; near 0 means "
                      "no straight-line relationship (there could still be a curved one).")
        elif x in num_cols and y in cat_cols:
            ui.show(plots.box(df, x, y))
        elif x in cat_cols and y in num_cols:
            ui.show(plots.box(df, y, x))
        else:
            ui.show(plots.bar_counts(df, x, y))

with t3:
    if not ss.target:
        ui.box("note", "Choose the outcome column on page 2 or 4 to see how each predictor differs between outcome groups.")
    else:
        tgt = ss.target
        if ss.task == "classification":
            ui.show(plots.class_balance(df[tgt].dropna(), f"Outcome: {tgt}"))
        else:
            ui.show(plots.hist(df, tgt))
        preds = [c for c in usable if c != tgt]
        if preds:
            p = st.selectbox("Predictor", preds, key="ex_p")
            if ss.task == "classification":
                d = df[df[tgt].notna()].copy()
                d[tgt] = d[tgt].astype(str)
                if p in num_cols:
                    ui.show(plots.box(d, p, tgt))
                    ui.simple("Each box shows the middle half of patients (IQR) and the line inside is the median. The more the "
                              "boxes of the outcome groups are separated, the more useful this predictor is likely to be.")
                else:
                    ui.show(plots.bar_counts(d, p, tgt))
            else:
                if p in num_cols:
                    ui.show(plots.scatter(df, p, tgt))
                else:
                    ui.show(plots.box(df, tgt, p))

with t4:
    if len(num_cols) >= 2:
        ui.show(plots.corr_heatmap(df[num_cols]))
        ui.simple("Red = both go up together, blue = one goes up as the other goes down. Two predictors with |r| > 0.9 carry "
                  "almost the same information (e.g. weight and BMI).")
    else:
        st.info("Need at least two number columns.")

with t5:
    if df.isna().sum().sum() == 0:
        ui.box("good", "There are no blank cells.")
    else:
        ui.show(plots.missing_map(df))
        ui.simple("Look for patterns: if blanks cluster in some patients or some columns, ask why (test not done in OPD? "
                  "data-entry break?). Blanks that are not random can bias the model.")
ui.next_step("app_pages/choose.py", "Next: 4. Choose outcome & model")
ui.footer()
