# MAMC ML Studio

**Free, open-source, no-code machine learning for medical students, residents and researchers.**
Developed by Faculties of **Maulana Azad Medical College & Govind Ballabh Pant Hospital**, University of Delhi, New Delhi.
Contributed by Dr. Pradeep Kr. Dabla (Director Professor, Department of Biochemistry, GB Pant Hospital) and
Dr. Manu Kumar Shetty (Professor, Department of Pharmacology); Medical Innovation Centre, MAMC.

MAMC ML Studio is a point-and-click machine-learning workbench written in **Python** (Streamlit, scikit-learn, pandas,
Plotly). Students load an Excel sheet, let the **Data Health Check** clean it, get a **model recommendation with
reasons**, watch the **train/test split, cross-validation and hyperparameter tuning** happen visually, and read the
results in clinical language (sensitivity, specificity, PPV, NPV, AUC with 95% CI). Every step has a plain-language
explanation, and the whole analysis can be downloaded as a **Word report** and as the equivalent **Python script**.
It runs entirely on the user's computer: no data are uploaded anywhere.

Licence: **GNU GPL-3.0**.

---

## Download (nothing else to install)

**[Download the latest version](https://github.com/ManuHCI/mamc-ml-studio/releases/latest)** and pick the file for your computer
under **Assets**:

| Your computer | File |
|---|---|
| Windows 10 / 11 (64-bit) | `MAMC_ML_Studio_Setup_<version>_Windows10-11.exe` - installer, no administrator rights needed |
| Windows, cannot install (college PC) | `MAMC_ML_Studio_<version>_Windows_portable.zip` - Extract All, double-click `MAMC ML Studio.exe` |
| Mac with Apple chip (M1-M4) | `MAMC_ML_Studio_<version>_macOS-AppleSilicon.dmg` |
| Intel Mac | `MAMC_ML_Studio_<version>_macOS-Intel.dmg` |

Python and every library are **inside** the download, so nothing else has to be installed and no internet is needed.
The program opens in its own window after a few seconds; close the window to quit. Needs Windows 10/11 (64-bit) or
macOS 12 Monterey or newer.

The first time, Windows shows "Windows protected your PC" (click **More info -> Run anyway**) and macOS says the app
is from an unidentified developer (right-click the app -> **Open -> Open**; on macOS 15: System Settings -> Privacy &
Security -> **Open Anyway**). This is because the software is not signed with a paid certificate - the release page
explains each step.

**Windows 11 "Smart App Control":** on new Windows 11 laptops this feature may block the program (Event Viewer shows
Code Integrity event 3077 for a `.pyd` or `.dll` file). Open Windows Security -> App & browser control -> Smart App
Control settings -> **Off**, then start MAMC ML Studio again. Smart App Control only allows code signed with a
certificate from Microsoft's trusted-root programme, and MAMC ML Studio and its libraries are not signed yet.

The installers are built and tested automatically by GitHub Actions (`.github/workflows/build-installers.yml`,
scripts in `build_tools/`): every edition runs the full engine test, a click-through of every page and a
start-and-close test before it is published.

## Running from the source code (for developers)

| Computer | What to do |
|---|---|
| **Windows** | Double-click `START_MAMC_ML_Studio_Windows.bat` |
| **Mac** | Double-click `START_MAMC_ML_Studio_Mac.command` (first time: right-click -> Open) |
| **Linux** | `bash start_mamc_ml_studio_linux.sh` |

The **first start** creates a private Python environment in the folder `.venv` and installs the components
(internet needed, about 5-10 minutes). Later starts take a few seconds and work offline. The app opens in your web
browser at `http://localhost:8501`. Close the black window to stop it.

Needs Python 3.10 or newer (python.org or Anaconda). If Python is missing, install it from
https://www.python.org/downloads/ and tick **Add python.exe to PATH**.

Run from a terminal instead: `pip install -r requirements.txt` then `streamlit run app.py`.

---

## What it does

| Page | Features |
|---|---|
| **Home** | Overview, one-click **guided demo** (328 patients, every step explained) |
| **1. Load data** | Excel (.xlsx/.xls) or CSV, sheet and header-row selection; 6 demo datasets with fixed sample sizes; checklist of what the data should look like |
| **2. Check & clean** | **Data Health Check**: blank rows/columns, duplicates, numbers stored as text ('45 yrs', '27,5'), text missing-codes (NA, -, nil), 999-type codes, inconsistent spellings (Male / male / M), impossible physiological values, extreme outliers, ID / free-text / date columns, constant columns, rare categories, missing outcome, **data leakage** (a column that predicts the outcome almost perfectly), class imbalance, events per variable. Each problem: what was found, why it matters, recommended fix. One-click "apply all", undo, cleaning log with the pandas code, download of the cleaned data |
| **3. Explore** | Histograms with summary statistics, scatter / box / grouped bar charts, outcome vs predictors, correlation heat-map, missing-value map |
| **4. Choose outcome & model** | Automatic detection of classification vs regression, positive class, predictor selection with leakage warning, **model advisor** (ranks the algorithms for *your* data and explains why), algorithm explainer cards, and a **live picture of what each algorithm does to your data** (decision boundary on two predictors, with adjustable settings) |
| **5. Train & tune** | Train/test split (stratified, seed), the preprocessing pipeline explained, k-fold cross-validation diagram, settings per model: starting values / set by hand / **grid search** / **random search**, number of fits, class balancing, live training log, CV-fold chart, tuning curves (training vs unseen-patient score) |
| **6. Results** | Comparison table with 95% bootstrap CIs, ROC and calibration curves, best-model statement, over/under-fitting verdict, cut-off slider (Youden), sensitivity / specificity / PPV / NPV / LR+ / LR-, confusion matrix as a 2x2 table, PR curve, permutation importance, odds ratios, partial dependence, learning curve, the tree diagram and text rules, boosting / neural-network learning curves |
| **7. Predict** | Form for a new patient (or start from a test patient), probability, "why this prediction" chart, out-of-range warning, batch prediction from a file |
| **8. Report & code** | Word report (methods paragraph, tables, charts, limitations, references), copyable methods/results text, **reproducible Python script** (+ data, as a zip), test-set predictions, trained models (.joblib) |
| **Learn** | 9 illustrated chapters: what ML is, kinds of ML, data, training & testing, overfitting (interactive), hyperparameters, judging a model (interactive 2x2 / cut-off demo), all algorithms with live demos, pitfalls & ethics, glossary |
| **Find patient groups** | K-means, hierarchical and DBSCAN clustering, silhouette and elbow plots, PCA map, group profiles in words, comparison with the outcome, K-means animation |
| **Sample size** | Riley et al. (BMJ 2020) criteria for binary outcomes (from expected AUC, Nagelkerke or Cox-Snell R2), 10/20 EPV rules, approximate rule for continuous outcomes |

**Algorithms (20):** classification - logistic regression, decision tree, random forest, gradient boosting,
support vector machine, k-nearest neighbours, naive Bayes, neural network; regression - linear, ridge, lasso,
decision tree, random forest, gradient boosting, support vector regression, k-nearest neighbours, neural network;
clustering - k-means, hierarchical, DBSCAN.

---

## Demo datasets (`sample_data/`)

| File | Patients | Task |
|---|---|---|
| `demo_diabetes.xlsx` | 500 | Diabetes within 5 years (Yes/No) |
| `demo_messy.xlsx` | 328 | The same study with typical data-entry mistakes - for practising cleaning |
| `demo_anaemia.xlsx` | 400 | Type of anaemia from the blood picture (4 groups) |
| `demo_stay.xlsx` | 400 | Length of hospital stay (number) |
| `demo_groups.xlsx` | 300 | Metabolic profiles without an outcome (clustering) |
| `demo_breast.xlsx` | 569 | Wisconsin breast-cancer data (real, public) |

All except the breast-cancer data are synthetic patients generated with a fixed random seed.

---

## Project structure (all Python)

```
app.py                 start page and navigation
launcher.py            source edition: creates the private environment, installs, starts the app
desktop_launcher.py    stand-alone edition: starts the bundled Python, opens the app window, stops when it is closed
build_tools/           build_windows.py (Setup .exe + portable zip), build_mac.py (.dmg), pinned library versions
installer/             icons for the installers
app_pages/             one file per screen (home, load, clean, explore, choose, train, results, predict,
                       report, learn, clusters, samplesize, about)
mlstudio/
  data_check.py        file reading, Data Health Check and fixes
  advisor.py           problem type detection and model recommendations
  models.py            the algorithm catalogue: explanations, settings, estimators
  trainer.py           pipeline, split, cross-validation, tuning, metrics, importance, verdicts
  plots.py             all charts
  clustering.py        unsupervised learning
  sample_size.py       Riley et al. sample-size criteria
  report.py            Word report
  codegen.py           exported Python script
  widgets.py, ui.py    shared screen elements and styling
  demo_data.py         demo datasets
tests/                 run_engine_tests.py (models, report, exported code), apptest_flow.py (click-through),
                       desktop_smoke.py (start / close test of an installed edition)
```

To change the version number edit `APP_VERSION` in `mlstudio/__init__.py`.
Run the tests with `python tests/run_engine_tests.py` and `python tests/apptest_flow.py`.

---

## References used in the software

- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. BMJ. 2024;385:e078378.
- Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required for developing a clinical prediction model. BMJ. 2020;368:m441.
- Riley RD, Snell KIE, Ensor J, et al. Minimum sample size for developing a multivariable prediction model: Part I - continuous outcomes. Stat Med. 2019;38:1262-75.
- Riley RD, Van Calster B, Collins GS. A note on estimating the Cox-Snell R2 from a reported C statistic (AUROC). Stat Med. 2021;40:859-64.
- van der Ploeg T, Austin PC, Steyerberg EW. Modern modelling techniques are data hungry. BMC Med Res Methodol. 2014;14:137.
- Pedregosa F, et al. Scikit-learn: machine learning in Python. J Mach Learn Res. 2011;12:2825-30.

For learning and research. Predictions must not be used for decisions about individual patients unless the model
has been properly validated.
