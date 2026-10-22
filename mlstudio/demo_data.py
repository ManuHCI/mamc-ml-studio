"""Demo datasets for practice.

All datasets except the breast-cancer one are SYNTHETIC (computer-generated)
patients. They behave like real clinical data but belong to no real person.
Each generator uses a fixed random seed, so every student gets the same numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------
# 1. Diabetes risk (binary classification)
# ----------------------------------------------------------------------------
def diabetes_risk(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """Will an adult develop type 2 diabetes within 5 years? (Yes / No)."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(48, 12, n), 20, 85).round()
    sex = rng.choice(["Male", "Female"], n, p=[0.52, 0.48])
    bmi = np.clip(rng.normal(25.8, 4.3, n), 16, 46).round(1)
    waist = np.clip(0.95 * (bmi - 25.8) * 2.6 + np.where(sex == "Male", 90, 84) + rng.normal(0, 6, n), 60, 140).round()
    family = rng.choice(["Yes", "No"], n, p=[0.35, 0.65])
    activity = rng.choice(["Low", "Moderate", "High"], n, p=[0.45, 0.38, 0.17])
    smoking = rng.choice(["Never", "Former", "Current"], n, p=[0.62, 0.15, 0.23])
    fg = np.clip(rng.normal(98, 9, n) + 0.6 * (bmi - 25.8) + 0.12 * (age - 48), 70, 125).round()
    sbp = np.clip(rng.normal(126, 14, n) + 0.45 * (age - 48) + 0.7 * (bmi - 25.8), 90, 200).round()
    tg = np.clip(rng.lognormal(np.log(145), 0.38, n) + 3 * (bmi - 25.8), 50, 600).round()
    hdl = np.clip(rng.normal(np.where(sex == "Male", 42, 50), 9, n) - 0.4 * (bmi - 25.8), 20, 95).round()

    z = (-1.25
         + 0.035 * (age - 48)
         + 0.11 * (bmi - 25.8)
         + 0.085 * (fg - 98)
         + 0.95 * (family == "Yes")
         + 0.55 * (activity == "Low") - 0.55 * (activity == "High")
         + 0.30 * (smoking == "Current")
         + 0.0035 * (tg - 145)
         - 0.025 * (hdl - 46)
         + 0.012 * (sbp - 126)
         # a non-linear effect: risk climbs faster when BMI and glucose are both high
         + 0.012 * np.clip(bmi - 27, 0, None) * np.clip(fg - 100, 0, None))
    p = _sigmoid(z)
    y = np.where(rng.random(n) < p, "Yes", "No")

    return pd.DataFrame({
        "Patient_ID": [f"P{i:04d}" for i in range(1, n + 1)],
        "Age": age.astype(int),
        "Sex": sex,
        "BMI": bmi,
        "Waist_cm": waist.astype(int),
        "Fasting_glucose": fg.astype(int),
        "Systolic_BP": sbp.astype(int),
        "Triglycerides": tg.astype(int),
        "HDL": hdl.astype(int),
        "Family_history": family,
        "Physical_activity": activity,
        "Smoking": smoking,
        "Diabetes_5yr": y,
    })


# ----------------------------------------------------------------------------
# 2. Messy version of the diabetes data (for practising cleaning)
# ----------------------------------------------------------------------------
def messy_diabetes(n: int = 320, seed: int = 11) -> pd.DataFrame:
    """The diabetes data with the typical mistakes found in thesis Excel sheets."""
    rng = np.random.default_rng(seed)
    df = diabetes_risk(n, seed=seed).astype(object)

    # inconsistent spellings and stray spaces
    sex_variants = {"Male": ["Male", "male", "M", "MALE", " Male"], "Female": ["Female", "female", "F", "Female "]}
    df["Sex"] = [rng.choice(sex_variants[s]) for s in df["Sex"]]
    fam_variants = {"Yes": ["Yes", "yes", "Y"], "No": ["No", "no", "N", "No "]}
    df["Family_history"] = [rng.choice(fam_variants[s]) for s in df["Family_history"]]

    # numbers typed with units / decimal comma -> column becomes text
    idx = rng.choice(n, 14, replace=False)
    for i in idx[:8]:
        df.at[i, "Age"] = f"{df.at[i, 'Age']} yrs"
    for i in idx[8:]:
        df.at[i, "BMI"] = str(df.at[i, "BMI"]).replace(".", ",")

    # missing values written as codes
    idx = rng.choice(n, 22, replace=False)
    for k, i in enumerate(idx):
        df.at[i, "Fasting_glucose"] = ["NA", "-", "nil", 999, ""][k % 5]
    idx = rng.choice(n, 30, replace=False)
    for i in idx:
        df.at[i, "Triglycerides"] = np.nan
    idx = rng.choice(n, 18, replace=False)
    for i in idx:
        df.at[i, "HDL"] = np.nan

    # impossible values (typing errors)
    df.at[5, "Age"] = 250
    df.at[17, "BMI"] = 0
    df.at[44, "Systolic_BP"] = 1200

    # outcome missing for a few patients
    for i in [9, 63, 150]:
        df.at[i, "Diabetes_5yr"] = np.nan

    # a "leaky" column: only known AFTER diabetes is diagnosed
    y = df["Diabetes_5yr"].astype(str)
    df["On_metformin"] = np.where((y == "Yes") & (rng.random(n) < 0.97), "Yes",
                                  np.where(rng.random(n) < 0.01, "Yes", "No"))
    # useless columns
    df["Hospital"] = "LNH"
    df["Remarks"] = np.nan

    # duplicate rows and blank rows
    dups = df.iloc[[20, 21, 22, 23, 24]].copy()
    blanks = pd.DataFrame([[np.nan] * df.shape[1]] * 3, columns=df.columns)
    df = pd.concat([df, dups, blanks], ignore_index=True)
    df = df.rename(columns={"Systolic_BP": "Systolic BP (mmHg)"})
    return df


# ----------------------------------------------------------------------------
# 3. Type of anaemia (multi-class classification)
# ----------------------------------------------------------------------------
def anaemia_type(n: int = 400, seed: int = 21) -> pd.DataFrame:
    """Which type of anaemia does the blood picture suggest? (4 classes)."""
    rng = np.random.default_rng(seed)
    classes = np.array(["Normal", "Iron deficiency", "Thalassaemia trait", "B12/folate deficiency"])
    y = rng.choice(classes, n, p=[0.30, 0.35, 0.18, 0.17])
    sex = rng.choice(["Female", "Male"], n, p=[0.6, 0.4])
    age = np.clip(rng.normal(32, 13, n), 12, 80).round()
    spec = {  # mean, sd for: Hb, MCV, MCH, RDW, RBC, log-ferritin
        "Normal": (12.9, 1.4, 87, 6, 29.0, 2.2, 13.4, 1.0, 4.6, 0.45, np.log(70), 0.7),
        "Iron deficiency": (10.0, 1.7, 75, 7, 24.0, 2.8, 16.6, 2.2, 4.2, 0.55, np.log(14), 0.8),
        "Thalassaemia trait": (11.0, 1.2, 68, 5.5, 21.5, 2.0, 15.2, 1.6, 5.3, 0.6, np.log(55), 0.8),
        "B12/folate deficiency": (10.2, 1.8, 102, 8, 33.5, 2.8, 16.0, 2.2, 3.2, 0.6, np.log(100), 0.7),
    }
    cols = {k: np.zeros(n) for k in ["Hb", "MCV", "MCH", "RDW", "RBC", "Ferritin"]}
    for c in classes:
        m = y == c
        k = m.sum()
        s = spec[c]
        cols["Hb"][m] = rng.normal(s[0], s[1], k)
        cols["MCV"][m] = rng.normal(s[2], s[3], k)
        cols["MCH"][m] = rng.normal(s[4], s[5], k)
        cols["RDW"][m] = rng.normal(s[6], s[7], k)
        cols["RBC"][m] = rng.normal(s[8], s[9], k)
        cols["Ferritin"][m] = np.exp(rng.normal(s[10], s[11], k))
    cols["Hb"] += np.where(sex == "Male", 1.0, 0.0)
    # real life is messy: about 5% of reports are borderline / mixed pictures
    mixed = rng.random(n) < 0.05
    y = np.where(mixed, rng.choice(classes, n), y)
    return pd.DataFrame({
        "Sample_No": np.arange(1, n + 1),
        "Age": age.astype(int),
        "Sex": sex,
        "Hb_g_dL": np.clip(cols["Hb"], 4, 18).round(1),
        "MCV_fL": np.clip(cols["MCV"], 50, 130).round(1),
        "MCH_pg": np.clip(cols["MCH"], 14, 42).round(1),
        "RDW_percent": np.clip(cols["RDW"], 10.5, 26).round(1),
        "RBC_million": np.clip(cols["RBC"], 1.8, 7.0).round(2),
        "Ferritin_ng_mL": np.clip(cols["Ferritin"], 1, 600).round(1),
        "Anaemia_type": y,
    })


# ----------------------------------------------------------------------------
# 4. Length of hospital stay (regression)
# ----------------------------------------------------------------------------
def hospital_stay(n: int = 400, seed: int = 31) -> pd.DataFrame:
    """How many days will a patient stay in hospital? (a number)."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(52, 17, n), 18, 92).round()
    sex = rng.choice(["Male", "Female"], n)
    adm = rng.choice(["Emergency", "Elective"], n, p=[0.6, 0.4])
    surgery = rng.choice(["Yes", "No"], n, p=[0.45, 0.55])
    comorb = np.clip(rng.poisson(1.0 + 0.025 * (age - 52).clip(0), n), 0, 6)
    crp = np.clip(rng.lognormal(np.log(18), 0.9, n) * np.where(adm == "Emergency", 1.8, 1.0), 0.5, 300).round(1)
    alb = np.clip(rng.normal(3.8, 0.5, n) - 0.004 * crp, 1.8, 5.2).round(1)
    hb = np.clip(rng.normal(12.6, 1.8, n), 6, 17).round(1)
    diabetes = rng.choice(["Yes", "No"], n, p=[0.28, 0.72])
    log_los = (1.15 + 0.008 * (age - 52) + 0.30 * (adm == "Emergency") + 0.38 * (surgery == "Yes")
               + 0.10 * comorb + 0.30 * np.log(crp / 18) - 0.30 * (alb - 3.8) - 0.03 * (hb - 12.6)
               + 0.12 * (diabetes == "Yes") + rng.normal(0, 0.28, n))
    los = np.clip(np.exp(log_los), 1, 45).round(1)
    return pd.DataFrame({
        "Age": age.astype(int), "Sex": sex, "Admission_type": adm, "Surgery": surgery,
        "Comorbidities": comorb.astype(int), "CRP_mg_L": crp, "Albumin_g_dL": alb, "Hb_g_dL": hb,
        "Diabetes": diabetes, "Length_of_stay_days": los,
    })


# ----------------------------------------------------------------------------
# 5. Patient groups (clustering - no outcome column)
# ----------------------------------------------------------------------------
def metabolic_groups(n: int = 300, seed: int = 41) -> pd.DataFrame:
    """Metabolic profile of adults; hidden inside are 3 natural patient groups."""
    rng = np.random.default_rng(seed)
    g = rng.choice([0, 1, 2], n, p=[0.4, 0.35, 0.25])
    centre = np.array([  # Age, BMI, Waist, FG, TG, HDL, SBP
        [34, 22.5, 79, 88, 105, 55, 116],    # lean, healthy
        [45, 31.0, 102, 112, 230, 38, 132],  # obese, insulin resistant
        [64, 25.5, 90, 96, 140, 47, 158],    # older, hypertensive
    ])
    sd = np.array([7, 2.2, 7, 8, 35, 6, 10])
    X = centre[g] + rng.normal(0, 1, (n, 7)) * sd
    X = np.clip(X, [18, 15, 55, 65, 40, 20, 90], [90, 50, 150, 200, 500, 90, 220])
    df = pd.DataFrame(X.round(1), columns=["Age", "BMI", "Waist_cm", "Fasting_glucose", "Triglycerides", "HDL", "Systolic_BP"])
    for c in ["Age", "Waist_cm", "Fasting_glucose", "Triglycerides", "HDL", "Systolic_BP"]:
        df[c] = df[c].round().astype(int)
    df.insert(0, "Subject", [f"S{i:03d}" for i in range(1, n + 1)])
    return df


# ----------------------------------------------------------------------------
# 6. Real public dataset bundled with scikit-learn
# ----------------------------------------------------------------------------
def breast_cancer() -> pd.DataFrame:
    """Wisconsin Diagnostic Breast Cancer data (real, public, 569 FNA samples)."""
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    df = d.frame.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df["Diagnosis"] = np.where(df.pop("target") == 1, "Benign", "Malignant")
    # keep the 10 "mean" features so the data stays easy to read
    keep = [c for c in df.columns if c.startswith("mean_")] + ["Diagnosis"]
    return df[keep]


DEMO_DATASETS = {
    "diabetes": {
        "title": "Diabetes risk in 5 years",
        "func": diabetes_risk, "n": 500, "target": "Diabetes_5yr", "task": "classification",
        "kind": "Classification (Yes / No)", "icon": ":material/bloodtype:",
        "story": "500 adults attending a medicine OPD were followed for 5 years. Can age, BMI, fasting "
                 "glucose, lipids, family history and lifestyle predict who develops type 2 diabetes?",
        "positive": "Yes",
    },
    "messy": {
        "title": "Diabetes risk - MESSY version (practise cleaning)",
        "func": messy_diabetes, "n": 328, "target": "Diabetes_5yr", "task": "classification",
        "kind": "Classification (Yes / No) - with mistakes", "icon": ":material/cleaning_services:",
        "story": "The same study typed into Excel the way it often happens: 'male' vs 'M', '45 yrs', "
                 "999 for missing, duplicates, blank rows, impossible values and a column that leaks "
                 "the answer. Use the Data Health Check to fix it.",
        "positive": "Yes",
    },
    "anaemia": {
        "title": "Type of anaemia from the blood picture",
        "func": anaemia_type, "n": 400, "target": "Anaemia_type", "task": "classification",
        "kind": "Classification (4 groups)", "icon": ":material/water_drop:",
        "story": "400 haemograms with ferritin. Can a model tell normal, iron deficiency, "
                 "thalassaemia trait and B12/folate deficiency apart from Hb, MCV, MCH, RDW, RBC and ferritin?",
        "positive": None,
    },
    "stay": {
        "title": "Length of hospital stay",
        "func": hospital_stay, "n": 400, "target": "Length_of_stay_days", "task": "regression",
        "kind": "Regression (predict a number)", "icon": ":material/local_hospital:",
        "story": "400 admissions. Predict how many days a patient will stay from age, admission type, "
                 "surgery, comorbidities, CRP, albumin, Hb and diabetes.",
        "positive": None,
    },
    "groups": {
        "title": "Patient groups (metabolic profiles)",
        "func": metabolic_groups, "n": 300, "target": None, "task": "clustering",
        "kind": "Clustering (no outcome - find groups)", "icon": ":material/bubble_chart:",
        "story": "300 adults with age, BMI, waist, fasting glucose, triglycerides, HDL and BP - but no "
                 "diagnosis column. Can the computer discover natural groups of similar patients?",
        "positive": None,
    },
    "breast": {
        "title": "Breast cancer - REAL public data (Wisconsin)",
        "func": breast_cancer, "n": 569, "target": "Diagnosis", "task": "classification",
        "kind": "Classification (Malignant / Benign)", "icon": ":material/biotech:",
        "story": "Real measurements of cell nuclei from 569 breast-lump FNA images (University of "
                 "Wisconsin, public dataset). Predict malignant vs benign.",
        "positive": "Malignant",
    },
}


def load_demo(key: str) -> pd.DataFrame:
    from .data_check import normalize_types
    return normalize_types(DEMO_DATASETS[key]["func"]())
