"""Headless click-through of the whole app with Streamlit's AppTest."""
import sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest

def check(at, where):
    errs = [e.value for e in at.exception]
    if errs:
        print(f"!! EXCEPTION on {where}:\n", "\n".join(str(e)[:3000] for e in errs))
        raise SystemExit(1)
    print(f"ok  {where}  (markdown blocks: {len(at.markdown)}, buttons: {len(at.button)})")

def btn(at, label_start):
    for b in at.button:
        if b.label.startswith(label_start):
            return b
    raise KeyError(f"button '{label_start}' not found: {[b.label for b in at.button]}")

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
at.run(); check(at, "home")
btn(at, "Guided demo").click().run(); check(at, "after demo click")
print("data:", at.session_state["data_name"], at.session_state["df"].shape, "target:", at.session_state["target"])
at.switch_page("app_pages/load.py").run(); check(at, "load")
at.switch_page("app_pages/clean.py").run(); check(at, "clean")
btn(at, "Apply all recommended fixes").click().run(); check(at, "clean after auto-fix")
print("clean log:", len(at.session_state["clean_log"]), "shape", at.session_state["df"].shape)
at.switch_page("app_pages/explore.py").run(); check(at, "explore")
at.switch_page("app_pages/choose.py").run(); check(at, "choose")
print("features:", at.session_state["features"], "chosen:", at.session_state["chosen_models"])
at.switch_page("app_pages/train.py").run(); check(at, "train page")
# set RF to grid search if present
for r in at.radio:
    if r.key == "mode_rf":
        r.set_value("grid").run(); check(at, "rf grid mode")
t0 = time.time()
btn(at, "Train").click().run(); check(at, f"train click ({time.time()-t0:.1f}s)")
print("results:", list(at.session_state["results"].keys()))
at.switch_page("app_pages/results.py").run(); check(at, "results")
at.switch_page("app_pages/predict.py").run(); check(at, "predict")
at.switch_page("app_pages/report.py").run(); check(at, "report")
btn(at, "Prepare the Word report").click().run(); check(at, "report docx")
print("docx bytes:", len(at.session_state["report_docx"]))
for p in ["learn", "clusters", "samplesize", "about"]:
    at.switch_page(f"app_pages/{p}.py").run(); check(at, p)
print("ALL OK")
