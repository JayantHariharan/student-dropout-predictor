"""Headless smoke test for app.py using streamlit.testing.v1.AppTest.
Run: python tests/test_app_smoke.py (from project root, with .venv active)"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=60)
at.run()
assert not at.exception, f"App raised on cold start: {at.exception}"
print("PASS cold start (no models uploaded yet) - no exception")

# tick the "use bundled test_data.csv" checkbox
at.sidebar.checkbox[0].set_value(True).run()
assert not at.exception, f"App raised with sample data ticked: {at.exception}"
assert any("Loaded" in str(el.value) for el in at.success), "Expected a success message after loading sample data"
print("PASS sample data loads, no exception, success message shown")

# switch model dropdown through all 5 models
select = at.sidebar.selectbox[0]
for opt in select.options:
    select.set_value(opt).run()
    assert not at.exception, f"App raised selecting model {opt}: {at.exception}"
print(f"PASS cycled through all models: {select.options}")

print("ALL APP SMOKE TESTS PASSED")
