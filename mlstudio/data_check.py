"""Data import, the Data Health Check and its fixes.

Every problem found becomes an `Issue` with:
  * what we found (with numbers),
  * why it matters for machine learning (plain words),
  * fix options, with one marked as recommended.
Every fix that is applied is written to a log together with the equivalent
pandas code, so the cleaning can be reproduced and reported in a thesis.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MISSING_TEXT = {"", "na", "n/a", "n.a.", "nan", "none", "nil", "null", "-", "--", "---", "?", ".",
                "not done", "nd", "missing", "not available", "#n/a", "#value!", "#div/0!"}
NUMERIC_CODES = [999, 9999, 99999, -9, -99, -999, -9999, 888, 8888]

SEV_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
SEV_LABEL = {"high": "Must fix", "medium": "Should fix", "low": "Worth checking", "info": "For your information"}

# Plausible physiological ranges, picked by column name. A range is only used
# when the column's median falls inside it (so different units are not flagged).
RANGES = [
    (r"\bage\b|^age|age_?y", [(0, 120)], "years"),
    (r"\bbmi\b", [(10, 80)], "kg/m2"),
    (r"systolic|\bsbp\b", [(50, 300)], "mmHg"),
    (r"diastolic|\bdbp\b", [(20, 200)], "mmHg"),
    (r"heart ?rate|\bpulse\b|\bhr\b", [(20, 250)], "beats/min"),
    (r"resp\w* ?rate|\brr\b", [(4, 80)], "breaths/min"),
    (r"spo2|sao2|o2 ?sat", [(40, 100)], "%"),
    (r"\btemp", [(25, 45), (77, 113)], "deg C or deg F"),
    (r"hba1c|\ba1c\b", [(3, 20)], "%"),
    (r"glucose|sugar|\bfbs\b|\bppbs\b|\brbs\b|\bfpg\b", [(20, 1000), (1, 60)], "mg/dL or mmol/L"),
    (r"\bhb\b|^hb|haemoglobin|hemoglobin|\bhgb\b", [(2, 25), (20, 250)], "g/dL or g/L"),
    (r"height|\bht\b", [(30, 250), (0.3, 2.5)], "cm or m"),
    (r"weight|\bwt\b", [(0.3, 350)], "kg"),
    (r"waist|\bhip\b", [(30, 200)], "cm"),
    (r"creatinine", [(0.05, 25), (5, 2000)], "mg/dL or umol/L"),
    (r"cholesterol|\bldl\b|\bhdl\b|\bvldl\b", [(1, 1000)], "mg/dL"),
    (r"triglycer|\btg\b", [(10, 5000)], "mg/dL"),
    (r"\bmcv\b", [(40, 150)], "fL"),
    (r"\bmch\b", [(10, 50)], "pg"),
    (r"\brdw\b", [(8, 40)], "%"),
    (r"ferritin", [(0.3, 20000)], "ng/mL"),
    (r"\bcrp\b", [(0, 600)], "mg/L"),
    (r"albumin", [(0.5, 7), (5, 70)], "g/dL or g/L"),
    (r"sodium|\bna\b", [(90, 200)], "mmol/L"),
    (r"potassium|\bk\b", [(1, 10)], "mmol/L"),
    (r"\bgcs\b", [(3, 15)], "points"),
    (r"apgar", [(0, 10)], "points"),
]

ID_NAME = re.compile(r"(^|[\s_\.])(id|uhid|mrd|mrn|crno|cr no|reg|regn|regno|registration|serial|s ?no|sno|sl ?no|"
                     r"sample ?no|sample ?id|subject|patient ?id|patient ?no|ipd|opd|name|phone|mobile|contact|address)"
                     r"([\s_\.]|$|no|num|number)", re.I)


# ---------------------------------------------------------------------------
# Reading files
# ---------------------------------------------------------------------------
def excel_sheets(data: bytes, filename: str) -> list[str]:
    if filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        try:
            return pd.ExcelFile(io.BytesIO(data)).sheet_names
        except Exception:
            return []
    return []


def read_table(data: bytes, filename: str, sheet: str | int | None = 0, header_row: int = 1) -> pd.DataFrame:
    """Read CSV / Excel bytes into a DataFrame. header_row is 1-based (Excel row number)."""
    name = filename.lower()
    hdr = max(int(header_row) - 1, 0)
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(io.BytesIO(data), sheet_name=sheet if sheet is not None else 0, header=hdr)
    else:
        text = None
        for enc in ("utf-8-sig", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        sep = ","
        first = text.splitlines()[hdr] if text and len(text.splitlines()) > hdr else ""
        if first.count(";") > first.count(",") and first.count(";") >= 1:
            sep = ";"
        elif first.count("\t") > first.count(","):
            sep = "\t"
        df = pd.read_csv(io.StringIO(text), sep=sep, header=hdr)
    df.columns = [str(c).strip() for c in df.columns]
    return normalize_types(df)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def is_dt(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def _num_regex():
    return re.compile(r"^\s*([-+]?\d+(?:[.,]\d+)?)\s*([a-zA-Z%/µ°\.]*\s*[a-zA-Z/0-9]*)\s*$")


_NUM = _num_regex()
_THOUSANDS = re.compile(r"^\s*[-+]?\d{1,3}(,\d{3})+(\.\d+)?\s*$")


def parse_number(v):
    """'45 yrs' -> 45, '27,5' -> 27.5, '1,234' -> 1234, 'NA' -> nan, 12 -> 12."""
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip()
    if s.lower() in MISSING_TEXT:
        return np.nan
    if _THOUSANDS.match(s):
        return float(s.replace(",", ""))
    m = _NUM.match(s)
    if not m:
        return np.nan
    return float(m.group(1).replace(",", "."))


def to_number(s: pd.Series) -> pd.Series:
    return pd.Series([parse_number(v) for v in s], index=s.index, dtype="float64")


def _text_values(s: pd.Series) -> pd.Series:
    """Non-missing values as python strings."""
    s = s[s.notna()]
    return s.map(lambda v: str(v))


def _is_missing_text(v) -> bool:
    return isinstance(v, str) and v.strip().lower() in MISSING_TEXT


def column_kind(s: pd.Series) -> str:
    """empty | constant | numeric | binary | categorical | text | date | id"""
    nn = s.dropna()
    if len(nn) == 0:
        return "empty"
    nun = nn.nunique()
    if nun == 1:
        return "constant"
    if is_dt(s):
        return "date"
    if is_num(s):
        if nun == 2:
            return "binary"
        if _looks_like_id(s):
            return "id"
        return "numeric"
    if nun == 2:
        return "binary"
    if _mostly_numeric_text(s):
        return "numeric"
    if _looks_like_id(s):
        return "id"
    if nun > 20 and nun / max(len(nn), 1) > 0.5:
        return "text"
    return "categorical"


def _looks_like_id(s: pd.Series) -> bool:
    nn = s.dropna()
    n = len(nn)
    if n < 10:
        return False
    all_unique = nn.nunique() == n
    name_hit = bool(ID_NAME.search(" " + str(s.name).replace("_", " ") + " "))
    if is_num(s):
        vals = nn.astype(float)
        integer = np.allclose(vals, vals.round())
        if not integer:
            return False
        sv = np.sort(vals.values)
        consecutive = all_unique and np.all(np.diff(sv) == 1)
        return bool(all_unique and (name_hit or consecutive))
    if all_unique and (name_hit or n >= 20):
        return True
    return bool(name_hit and nn.nunique() / n > 0.9)


_PURE_NUM = re.compile(r"^\s*[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\s*$")


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Silently turn columns that hold only clean numbers (stored as objects/strings) into numeric columns.
    Columns with units, decimal commas or text codes are left alone for the Health Check to explain."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        s = df[c]
        if is_num(s) or is_dt(s) or pd.api.types.is_bool_dtype(s):
            continue
        nn = s.dropna()
        if len(nn) == 0:
            continue
        ok = nn.map(lambda v: (isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool))
                    or (isinstance(v, str) and bool(_PURE_NUM.match(v))))
        if ok.all():
            df[c] = pd.to_numeric(s.map(lambda v: v.strip() if isinstance(v, str) else v), errors="coerce")
    return df


def _mostly_numeric_text(s: pd.Series) -> bool:
    txt = _text_values(s)
    txt = txt[~txt.map(lambda v: v.strip().lower() in MISSING_TEXT)]
    if len(txt) == 0:
        return False
    return txt.map(parse_number).notna().mean() >= 0.8 and txt.nunique() > 2


def overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        kind = column_kind(s)
        nn = s.dropna()
        if is_num(s) and len(nn):
            ex = f"{nn.min():g} to {nn.max():g}"
        else:
            vals = _text_values(s).value_counts().index[:4].tolist()
            ex = ", ".join(vals)
        rows.append({
            "Column": c,
            "Type (detected)": {"numeric": "Number", "binary": "Two groups", "categorical": "Categories",
                                "text": "Free text", "date": "Date", "id": "ID / name", "constant": "Same value everywhere",
                                "empty": "Empty"}[kind],
            "Missing": int(s.isna().sum()),
            "Missing %": round(100 * s.isna().mean(), 1),
            "Different values": int(nn.nunique()),
            "Range / examples": ex,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Issues
# ---------------------------------------------------------------------------
@dataclass
class Issue:
    key: str
    kind: str
    severity: str
    title: str
    columns: list
    detail: str
    why: str
    options: list            # list of (action, label)
    recommended: str
    payload: dict = field(default_factory=dict)

    @property
    def sev_label(self):
        return SEV_LABEL[self.severity]


def _encoded_parameter_count(df: pd.DataFrame, features: list) -> int:
    p = 0
    for c in features:
        s = df[c]
        if is_num(s) or column_kind(s) == "numeric":
            p += 1
        else:
            p += max(s.dropna().nunique() - 1, 1)
    return p


def health_check(df: pd.DataFrame, target: str | None = None, task: str | None = None,
                 dismissed: set | None = None) -> list[Issue]:
    dismissed = dismissed or set()
    issues: list[Issue] = []
    n = len(df)
    add = issues.append

    # 1. completely blank rows
    blank = int(df.isna().all(axis=1).sum())
    if blank:
        add(Issue("blank_rows", "blank_rows", "high", f"{blank} completely blank row(s)", [],
                  f"{blank} row(s) have no value in any column (often empty lines at the end of an Excel sheet).",
                  "Blank rows are not patients. They would be counted as patients with every value missing.",
                  [("drop", "Remove blank rows"), ("keep", "Keep")], "drop"))

    # 2. empty / constant columns
    for c in df.columns:
        s = df[c]
        if s.isna().all():
            add(Issue(f"empty_col:{c}", "drop_col", "high", f"'{c}' is completely empty", [c],
                      "Every cell in this column is blank.",
                      "An empty column carries no information and can only cause errors.",
                      [("drop", "Remove this column"), ("keep", "Keep")], "drop"))
        elif s.dropna().nunique() == 1:
            v = s.dropna().iloc[0]
            add(Issue(f"const_col:{c}", "drop_col", "medium", f"'{c}' has the same value for everybody", [c],
                      f"Every filled cell says '{v}'.",
                      "A column that never changes cannot help to tell patients apart, so the model cannot learn from it.",
                      [("drop", "Remove this column"), ("keep", "Keep")], "drop"))

    # 3. column header problems
    bad = [c for c in df.columns if re.match(r"^Unnamed:? ?\d*$", str(c)) or str(c).strip() == "" or str(c).lower() == "nan"]
    dup = [c for c in df.columns if re.search(r"\.\d+$", str(c)) and re.sub(r"\.\d+$", "", str(c)) in df.columns]
    for c in bad:
        if df[c].isna().all():
            continue  # already reported as empty
        add(Issue(f"no_header:{c}", "rename", "medium", f"A column has no heading ('{c}')", [c],
                  "This column has data but its first-row heading is blank.",
                  "Without a name you cannot tell the model (or your reader) what the column means.",
                  [("rename", "Give it a name (type below)"), ("drop", "Remove this column"), ("keep", "Keep as is")],
                  "rename", {"suggest": f"Column_{list(df.columns).index(c) + 1}"}))
    for c in dup:
        base = re.sub(r"\.\d+$", "", str(c))
        add(Issue(f"dup_header:{c}", "rename", "medium", f"Two columns are both called '{base}'", [c],
                  f"The second copy was renamed to '{c}' while reading the file.",
                  "Duplicate names cause confusion. Check if it is a repeated measurement (e.g. week 0 / week 4) and rename it.",
                  [("rename", "Rename it (type below)"), ("drop", "Remove this copy"), ("keep", "Keep as is")],
                  "rename", {"suggest": re.sub(r"\.(\d+)$", r"_\1", c)}))

    # 4. exact duplicate rows
    content = df.dropna(how="all")
    ndup = int(content.duplicated().sum())
    if ndup:
        add(Issue("dup_rows", "dup_rows", "high", f"{ndup} duplicate row(s)", [],
                  f"{ndup} row(s) are exact copies of another row (same value in every column).",
                  "The same patient counted twice gets double weight. Worse, one copy may land in the training set and "
                  "the other in the test set, so the test becomes too easy and the results look better than they are.",
                  [("drop", "Keep only the first copy"), ("keep", "Keep all (they are different patients)")], "drop"))

    for c in df.columns:
        s = df[c]
        nn = s.dropna()
        if len(nn) == 0 or nn.nunique() == 1:
            continue

        # 5. text missing-codes (NA, -, nil) inside a column
        if not is_num(s):
            txt = _text_values(s)
            codes = sorted({v for v in txt if v.strip().lower() in MISSING_TEXT})
            if codes:
                cnt = int(txt.map(lambda v: v.strip().lower() in MISSING_TEXT).sum())
                shown = ", ".join(repr(x) for x in codes)
                add(Issue(f"miss_text:{c}", "miss_text", "high", f"'{c}': missing values typed as text", [c],
                          f"{cnt} cell(s) contain {shown} instead of being left blank.",
                          "The computer reads 'NA' or '-' as a real word, not as 'missing'. A number column then turns into a text column.",
                          [("fix", "Convert these to blank (missing)"), ("keep", "Keep")], "fix", {"codes": codes}))

        # 6. numbers stored as text ('45 yrs', '27,5')
        if not is_num(s) and not is_dt(s):
            txt = _text_values(s)
            txt = txt[~txt.map(lambda v: v.strip().lower() in MISSING_TEXT)]
            if len(txt):
                parsed = txt.map(parse_number)
                ok = parsed.notna().mean()
                if ok >= 0.8 and txt.nunique() > 2:
                    odd = txt[~txt.str.strip().str.match(r"^[-+]?\d+(\.\d+)?$")]
                    ex = ", ".join(repr(x) for x in odd.unique()[:5])
                    fail = txt[parsed.isna()].unique()[:5]
                    if len(odd):
                        det = f"{int(ok * 100)}% of the cells are numbers, but some are written as text: {ex}."
                    else:
                        det = ("All filled cells are numbers, but the column is stored as text - usually because some "
                               "cells contain codes such as 'NA' or '-', or numbers were typed as text.")
                    if len(fail):
                        det += f" These cannot be read as numbers and will become blank: {', '.join(map(repr, fail))}."
                    add(Issue(f"num_text:{c}", "num_text", "high", f"'{c}' contains numbers stored as text", [c], det,
                              "A model can only do arithmetic on real numbers. Units ('yrs', 'mg/dL') or a decimal comma make "
                              "the whole column text. Put units in the column heading, not in the cells.",
                              [("fix", "Convert to numbers (remove units, fix decimal comma)"), ("keep", "Keep as text")], "fix"))

        # 7. numeric missing codes (999, -99)
        if is_num(s):
            vals = nn.astype(float)
            found = []
            for code in NUMERIC_CODES:
                k = int((vals == code).sum())
                if k:
                    rest = vals[vals != code]
                    if len(rest) and (abs(code) > 3 * max(abs(rest.max()), abs(rest.min()), 1e-9) or
                                      (code > 0 and code > rest.quantile(0.99) * 3) or (code < 0 and rest.min() >= 0)):
                        found.append((code, k))
            if found:
                desc = ", ".join(f"{code} ({k} times)" for code, k in found)
                add(Issue(f"miss_code:{c}", "miss_code", "high", f"'{c}': possible missing-value codes", [c],
                          f"The value(s) {desc} look like codes for 'not available', far away from the other values.",
                          "If 999 means 'not done', the model will believe the patient had a value of 999 - a huge error.",
                          [("fix", "Treat these as missing"), ("keep", "Keep (they are real values)")], "fix",
                          {"codes": [code for code, _ in found]}))

        # 8. inconsistent spellings
        if not is_num(s) and nn.nunique() <= 60:
            txt = _text_values(s)
            txt = txt[~txt.map(lambda v: v.strip().lower() in MISSING_TEXT)]
            mapping = _category_mapping(txt)
            if mapping:
                groups = {}
                for orig, canon in mapping.items():
                    groups.setdefault(canon, []).append(orig)
                desc = "; ".join(f"{', '.join(repr(o) for o in sorted(v))} -> '{k}'" for k, v in groups.items())
                add(Issue(f"cat_spell:{c}", "cat_spell", "high", f"'{c}': the same category is written in different ways", [c],
                          f"Will be merged as: {desc}.",
                          "To a computer 'Male', 'male' and 'M ' are three different groups. Each group then gets too few "
                          "patients and the model learns nonsense.",
                          [("fix", "Merge the spellings"), ("keep", "Keep as they are")], "fix", {"mapping": mapping}))

    # re-scan with numeric view for range / outlier checks
    for c in df.columns:
        s = df[c]
        if not is_num(s) or s.dropna().nunique() < 3:
            continue
        vals = s.dropna().astype(float)

        # 9. physiologically impossible values
        rng_hit = _range_for(c, vals)
        if rng_hit:
            lo, hi, unit = rng_hit
            bad_mask = (vals < lo) | (vals > hi)
            k = int(bad_mask.sum())
            if k:
                ex = ", ".join(f"{v:g}" for v in vals[bad_mask].unique()[:6])
                add(Issue(f"implaus:{c}", "implaus", "high", f"'{c}': impossible values", [c],
                          f"{k} value(s) outside the plausible range {lo:g}-{hi:g} {unit}: {ex}.",
                          "Typing errors (e.g. age 250, BMI 0) distort what the model learns. Check the case record; if you "
                          "cannot correct it, treat it as missing.",
                          [("fix", "Treat these values as missing"), ("keep", "Keep (I have checked them)")], "fix",
                          {"lo": lo, "hi": hi}))
                continue

        # 10. extreme outliers (IQR x 3)
        if len(vals) >= 20 and c != target:
            q1, q3 = vals.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
                m = (vals < lo) | (vals > hi)
                k = int(m.sum())
                if k:
                    ex = ", ".join(f"{v:g}" for v in vals[m].unique()[:6])
                    p1, p99 = vals.quantile([0.01, 0.99])
                    add(Issue(f"outlier:{c}", "outlier", "low", f"'{c}': {k} extreme value(s)", [c],
                              f"Values far from the rest (beyond 3 x IQR): {ex}.",
                              "Extreme values can be real (a very sick patient) or errors. Tree-based models are hardly affected; "
                              "linear models, KNN, SVM and neural networks are more sensitive. Do not delete real patients.",
                              [("keep", "Keep (real values)"), ("cap", f"Cap at 1st-99th percentile ({p1:g} to {p99:g})"),
                               ("missing", "Treat as missing")], "keep", {"p1": float(p1), "p99": float(p99), "lo": lo, "hi": hi}))

    # 11. ID columns, free text, dates
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        kind = column_kind(s)
        if kind == "id":
            add(Issue(f"id_col:{c}", "drop_col", "high", f"'{c}' looks like an ID / name column", [c],
                      "Every row has a different value - typical of patient IDs, serial numbers or names.",
                      "An ID tells nothing about the disease. Worse, the model may memorise it. Keep IDs out of the model "
                      "(and remove names / phone numbers for confidentiality).",
                      [("drop", "Remove from the analysis"), ("keep", "Keep")], "drop"))
        elif kind == "text":
            add(Issue(f"free_text:{c}", "drop_col", "medium", f"'{c}' looks like free text", [c],
                      f"{s.dropna().nunique()} different entries for {s.notna().sum()} patients (e.g. "
                      f"{', '.join(map(repr, _text_values(s).unique()[:3]))}).",
                      "Free text (remarks, diagnoses typed in words) cannot be used directly. Recode it into a few "
                      "categories in Excel, or remove it.",
                      [("drop", "Remove this column"), ("keep", "Keep")], "drop"))
        elif kind == "date" or (not is_num(s) and _looks_like_date(s)):
            add(Issue(f"date_col:{c}", "drop_col", "medium", f"'{c}' contains dates", [c],
                      "A calendar date (e.g. date of admission) is not useful to a model by itself.",
                      "Convert dates into something meaningful in Excel first - age, duration of illness, days to "
                      "surgery - or remove the column.",
                      [("drop", "Remove this column"), ("keep", "Keep")], "drop"))

    # 12. rare categories
    for c in df.columns:
        s = df[c]
        if c == target or is_num(s):
            continue
        kind = column_kind(s)
        if kind != "categorical":
            continue
        vc = _text_values(s).value_counts()
        rare = vc[vc < 5]
        if len(rare) and len(vc) > 2:
            add(Issue(f"rare_cat:{c}", "rare_cat", "low", f"'{c}': {len(rare)} rare category(ies)", [c],
                      f"Categories with fewer than 5 patients: {', '.join(f'{k} ({v})' for k, v in rare.items())}.",
                      "The model cannot learn reliably from 2-3 patients. Grouping rare categories into 'Other' often helps.",
                      [("keep", "Keep"), ("group", "Group them into 'Other'")], "keep", {"levels": rare.index.tolist()}))

    # 13. missing values (per column)
    for c in df.columns:
        s = df[c]
        miss = int(s.isna().sum())
        if not miss or s.isna().all() or miss == blank:
            continue
        pct = 100 * miss / max(n, 1)
        if c == target:
            add(Issue(f"miss_target:{c}", "miss_target", "high", f"Outcome '{c}' is missing for {miss} patient(s)", [c],
                      f"{miss} row(s) ({pct:.1f}%) have no outcome.",
                      "A patient without an outcome cannot teach the model anything (we do not know the right answer).",
                      [("drop_rows", "Remove these rows"), ("keep", "Keep")], "drop_rows"))
            continue
        if pct > 50:
            sev, rec = "medium", "drop"
        elif pct > 20:
            sev, rec = "low", "keep"
        else:
            sev, rec = "info", "keep"
        add(Issue(f"missing:{c}", "missing", sev, f"'{c}': {miss} missing value(s) ({pct:.0f}%)", [c],
                  f"{miss} of {n} cells are blank.",
                  "The software fills blanks automatically inside the model (median for numbers, most common value for "
                  "categories), learning the fill-in values from the TRAINING patients only. If more than half is "
                  "missing, the column is usually not worth keeping.",
                  [("keep", "Let the model fill them in (imputation)"), ("drop", "Remove this column"),
                   ("drop_rows", "Remove the patients with a blank here")], rec))

    # 14. target-specific checks
    if target and target in df.columns:
        issues.extend(_target_checks(df, target, task))

    # 15. highly correlated pairs
    num_cols = [c for c in df.columns if c != target and is_num(df[c]) and column_kind(df[c]) in ("numeric", "binary")]
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        pairs = []
        for i, a in enumerate(num_cols):
            for b in num_cols[i + 1:]:
                r = corr.loc[a, b]
                if pd.notna(r) and r >= 0.9:
                    pairs.append((a, b, r))
        if pairs:
            add(Issue("high_corr", "high_corr", "info", f"{len(pairs)} pair(s) of almost identical columns", [p[0] for p in pairs],
                      "; ".join(f"{a} & {b} (r = {r:.2f})" for a, b, r in pairs[:6]) + ".",
                      "Columns that measure the same thing (e.g. weight and BMI) do not hurt prediction much, but they "
                      "split the importance between them and make logistic-regression weights unstable.",
                      [("keep", "Keep both (fine for prediction)")], "keep"))

    issues = [i for i in issues if i.key not in dismissed]
    issues.sort(key=lambda i: SEV_ORDER[i.severity])
    return issues


def _target_checks(df: pd.DataFrame, target: str, task: str | None) -> list[Issue]:
    out = []
    y = df[target]
    yy = y.dropna()
    n = len(yy)
    feats = [c for c in df.columns if c != target and column_kind(df[c]) in ("numeric", "binary", "categorical")]
    if task == "classification":
        vc = _text_values(y).value_counts()
        small = vc[vc < 10]
        if len(small):
            out.append(Issue("tiny_class", "info_only", "high", "Some outcome groups are too small", [target],
                             f"Groups with fewer than 10 patients: {', '.join(f'{k} ({v})' for k, v in small.items())}.",
                             "The model cannot learn a group it has seen only a few times, and the test set may not "
                             "contain it at all. Collect more patients or merge groups.",
                             [("keep", "Understood")], "keep"))
        if len(vc) >= 2:
            share = vc.min() / vc.sum()
            if vc.min() / vc.max() < 0.25:
                out.append(Issue("imbalance", "info_only", "medium", "The outcome groups are unbalanced", [target],
                                 f"The smallest group '{vc.idxmin()}' is only {100 * share:.0f}% of patients.",
                                 "With 10% disease, a lazy model that says 'no disease' for everyone is 90% 'accurate'. "
                                 "Judge models by sensitivity, specificity, balanced accuracy and AUC - not accuracy. "
                                 "Turn on 'balance the groups' in the training step.",
                                 [("keep", "Understood")], "keep"))
        p = _encoded_parameter_count(df, feats)
        if len(vc) == 2 and p:
            events = vc.min()
            epv = events / p
            if epv < 10:
                out.append(Issue("epv", "info_only", "high" if epv < 5 else "medium", f"Few patients per predictor (EPV = {epv:.1f})", [target],
                                 f"{events} patients in the smaller outcome group and about {p} predictor terms -> {epv:.1f} events per variable.",
                                 "A common rule is at least 10-20 events per variable for regression models; flexible ML models "
                                 "need far more. With fewer, the model 'memorises' (overfits). Use fewer predictors or collect more "
                                 "data. See the sample-size calculator on the Data page.",
                                 [("keep", "Understood")], "keep"))
    if n < 50:
        out.append(Issue("small_n", "info_only", "high", f"Very small dataset ({n} patients)", [target],
                         f"Only {n} patients have an outcome.",
                         "Machine learning needs many examples. With fewer than ~100 patients, results will change a lot "
                         "with every split; treat any model as a learning exercise only.",
                         [("keep", "Understood")], "keep"))

    # leakage: one column alone predicts the outcome almost perfectly
    for c in feats:
        score = _single_feature_score(df, c, target, task)
        if score is not None and score >= 0.95:
            out.append(Issue(f"leak:{c}", "drop_col", "high", f"'{c}' predicts the outcome almost perfectly - possible leakage", [c],
                             f"This one column alone predicts '{target}' with a score of {score:.2f} (1.00 = perfect).",
                             "Real predictors are never this perfect. Usually the column is recorded AFTER the outcome is known "
                             "(e.g. 'on metformin' for diabetes, 'discharge diagnosis'), or it is the outcome in another form. "
                             "A model using it will look excellent but be useless for new patients.",
                             [("drop", "Remove this column"), ("keep", "Keep (it is truly measured before the outcome)")], "drop"))
    return out


def _single_feature_score(df, c, target, task):
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    d = df[[c, target]].dropna()
    if len(d) < 30 or d[target].nunique() < 2:
        return None
    if is_num(d[c]):
        X = d[[c]].astype(float).to_numpy()
    else:
        X = pd.get_dummies(d[c].astype(str)).to_numpy(dtype=float)
        if X.shape[1] > 30:
            return None
    try:
        if task == "regression":
            yv = d[target].astype(float).to_numpy()
            return float(np.mean(cross_val_score(DecisionTreeRegressor(max_depth=4, random_state=0), X, yv, cv=3, scoring="r2")))
        yv = np.asarray(d[target].astype(str), dtype=object)
        if pd.Series(yv).value_counts().min() < 3:
            return None
        return float(np.mean(cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=0), X, yv, cv=3,
                                             scoring="balanced_accuracy")))
    except Exception:
        return None


def _range_for(col: str, vals: pd.Series):
    name = " " + re.sub(r"[_\.\(\)\-/]+", " ", str(col).lower()) + " "
    med = float(vals.median())
    for pat, ranges, unit in RANGES:
        if re.search(pat, name):
            for lo, hi in ranges:
                if lo <= med <= hi:
                    return lo, hi, unit
            return None
    return None


def _looks_like_date(s: pd.Series) -> bool:
    txt = _text_values(s)
    if len(txt) < 5:
        return False
    sample = txt.iloc[:200]
    if not sample.str.contains(r"\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}", regex=True).mean() > 0.8:
        return False
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
    return parsed.notna().mean() > 0.8


def _category_mapping(txt: pd.Series) -> dict:
    """Map each spelling variant to one canonical label. Returns only values that change."""
    counts = txt.value_counts()
    groups: dict[str, list] = {}
    for v in counts.index:
        k = re.sub(r"\s+", " ", v.strip().lower())
        groups.setdefault(k, []).append(v)
    # one-letter abbreviations (M -> male, Y -> yes) when exactly one full word matches
    keys = list(groups)
    for k in keys:
        if len(k) == 1:
            longer = [o for o in keys if len(o) > 1 and o.startswith(k)]
            if len(longer) == 1:
                groups[longer[0]].extend(groups.pop(k))
    mapping = {}
    for k, variants in groups.items():
        stripped = [v.strip() for v in variants]
        # canonical = most frequent stripped variant that is not a single letter, prefer Title case
        best = max(variants, key=lambda v: (len(v.strip()) > 1, counts.get(v, 0), v.strip()[:1].isupper()))
        canon = re.sub(r"\s+", " ", best.strip())
        if canon.islower() or canon.isupper() and len(canon) > 3:
            canon = canon[:1].upper() + canon[1:].lower()
        for v in variants:
            if v != canon:
                mapping[v] = canon
        del stripped
    return mapping


# ---------------------------------------------------------------------------
# Applying fixes - each returns (new_df, log_entry)
# ---------------------------------------------------------------------------
def apply_fix(df: pd.DataFrame, issue: Issue, action: str, extra: str | None = None):
    df = df.copy()
    c = issue.columns[0] if issue.columns else None
    cq = repr(c)
    if action == "keep":
        return df, {"text": f"Kept as is: {issue.title}", "code": None, "key": issue.key}

    if issue.kind == "blank_rows":
        before = len(df)
        df = df.dropna(how="all").reset_index(drop=True)
        return df, _log(issue, f"Removed {before - len(df)} blank rows", "df = df.dropna(how='all').reset_index(drop=True)")
    if issue.kind == "dup_rows":
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        return df, _log(issue, f"Removed {before - len(df)} duplicate rows", "df = df.drop_duplicates().reset_index(drop=True)")
    if action == "drop" and c is not None:
        df = df.drop(columns=issue.columns[:1])
        return df, _log(issue, f"Removed column '{c}' ({issue.title})", f"df = df.drop(columns=[{cq}])")
    if action == "drop_rows":
        before = len(df)
        df = df[df[c].notna()].reset_index(drop=True)
        return df, _log(issue, f"Removed {before - len(df)} rows with blank '{c}'",
                        f"df = df[df[{cq}].notna()].reset_index(drop=True)")
    if action == "rename":
        new = (extra or issue.payload.get("suggest") or f"{c}_renamed").strip()
        df = df.rename(columns={c: new})
        return df, _log(issue, f"Renamed column '{c}' to '{new}'", f"df = df.rename(columns={{{cq}: {new!r}}})")
    if issue.kind == "miss_text":
        codes = issue.payload["codes"]
        mask = df[c].map(_is_missing_text)
        df[c] = df[c].astype(object).where(~mask, np.nan)
        return df, _log(issue, f"'{c}': converted {int(mask.sum())} text codes {codes} to missing",
                        f"df[{cq}] = df[{cq}].replace({codes!r}, np.nan)")
    if issue.kind == "num_text":
        before_missing = int(df[c].isna().sum())
        df[c] = to_number(df[c])
        lost = int(df[c].isna().sum()) - before_missing
        msg = f"'{c}': converted text to numbers (removed units / fixed decimal commas)"
        if lost > 0:
            msg += f"; {lost} unreadable cell(s) became missing"
        return df, _log(issue, msg, f"df[{cq}] = to_number(df[{cq}])")
    if issue.kind == "miss_code":
        codes = issue.payload["codes"]
        k = int(df[c].isin(codes).sum())
        df[c] = df[c].where(~df[c].isin(codes), np.nan).astype(float)
        return df, _log(issue, f"'{c}': treated {k} code value(s) {codes} as missing",
                        f"df[{cq}] = df[{cq}].replace({codes!r}, np.nan)")
    if issue.kind == "cat_spell":
        mp = issue.payload["mapping"]
        s = df[c]
        df[c] = s.map(lambda v: mp.get(str(v), v) if pd.notna(v) else v)
        df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
        return df, _log(issue, f"'{c}': merged spellings {mp}",
                        f"df[{cq}] = df[{cq}].replace({mp!r}).str.strip()")
    if issue.kind == "implaus":
        lo, hi = issue.payload["lo"], issue.payload["hi"]
        m = (df[c] < lo) | (df[c] > hi)
        k = int(m.sum())
        df.loc[m, c] = np.nan
        return df, _log(issue, f"'{c}': {k} impossible value(s) outside {lo:g}-{hi:g} set to missing",
                        f"df.loc[(df[{cq}] < {lo!r}) | (df[{cq}] > {hi!r}), {cq}] = np.nan")
    if issue.kind == "outlier":
        if action == "cap":
            p1, p99 = issue.payload["p1"], issue.payload["p99"]
            df[c] = df[c].clip(p1, p99)
            return df, _log(issue, f"'{c}': capped at {p1:g} - {p99:g} (1st-99th percentile)",
                            f"df[{cq}] = df[{cq}].clip({p1!r}, {p99!r})")
        if action == "missing":
            lo, hi = issue.payload["lo"], issue.payload["hi"]
            m = (df[c] < lo) | (df[c] > hi)
            df.loc[m, c] = np.nan
            return df, _log(issue, f"'{c}': {int(m.sum())} extreme value(s) set to missing",
                            f"df.loc[(df[{cq}] < {lo!r}) | (df[{cq}] > {hi!r}), {cq}] = np.nan")
    if issue.kind == "rare_cat" and action == "group":
        lv = issue.payload["levels"]
        df[c] = df[c].map(lambda v: "Other" if pd.notna(v) and str(v) in lv else v)
        return df, _log(issue, f"'{c}': grouped rare categories {lv} into 'Other'",
                        f"df[{cq}] = df[{cq}].replace({lv!r}, 'Other')")
    return df, {"text": f"No change: {issue.title}", "code": None, "key": issue.key}


def _log(issue, text, code):
    return {"text": text, "code": code, "key": issue.key}


AUTO_PRIORITY = ["blank_rows", "drop_col", "rename", "dup_rows", "miss_text", "miss_code", "num_text",
                 "cat_spell", "implaus", "miss_target"]


def auto_clean(df: pd.DataFrame, target: str | None = None, task: str | None = None, max_steps: int = 80):
    """Apply every *recommended* fix (except 'keep'), re-checking after each one."""
    log = []
    done = set()
    for _ in range(max_steps):
        issues = health_check(df, target, task)
        todo = [i for i in issues if i.recommended not in ("keep",) and i.key not in done and i.kind != "rename"]
        if not todo:
            break
        todo.sort(key=lambda i: AUTO_PRIORITY.index(i.kind) if i.kind in AUTO_PRIORITY else 99)
        issue = todo[0]
        done.add(issue.key)
        df, entry = apply_fix(df, issue, issue.recommended)
        log.append(entry)
    return df, log


DATA_REQUIREMENTS = [
    ("One row = one patient, one column = one variable",
     "Put each patient on a single row. Do not merge cells, do not add titles above the headings, and do not put "
     "totals at the bottom."),
    ("Short, clear column names in row 1",
     "e.g. Age, Sex, BMI, Fasting_glucose. Put units in the name (Hb_g_dL), never inside the cells."),
    ("An outcome (target) column",
     "What you want to predict: a category (Diabetes Yes/No, type of anaemia) or a number (length of stay). "
     "For finding patient groups (clustering) no outcome is needed."),
    ("Predictors measured BEFORE the outcome",
     "Use only information you would have at the moment you want to predict. 'Discharge diagnosis' or 'treatment "
     "given' cannot predict the disease - that is data leakage."),
    ("Consistent categories",
     "Write each category the same way (Male / Female, not M, male, MALE). Use the same reference for all patients."),
    ("Leave missing cells blank",
     "Do not write NA, -, nil, 0 or 999 for 'not done'. Blank is understood by every software."),
    ("Enough patients",
     "Rough guide: at least 10-20 patients in the SMALLER outcome group for every predictor for regression models; "
     "flexible ML models (random forest, boosting, neural networks) need many more. Use the sample-size calculator."),
    ("No personal identifiers",
     "Remove names, phone numbers and hospital numbers before analysis (ethics & confidentiality). Use a study code."),
]
