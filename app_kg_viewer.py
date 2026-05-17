from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


HTML_PATH = Path("outputs/agent/kg_viewer.html")


st.set_page_config(page_title="Aquaculture KG Viewer", layout="wide")
st.title("Aquaculture Literature Knowledge Graph")

st.markdown(
    """
Run these commands first:

```powershell
python scripts/agent/run_agent_pipeline.py "How does temperature affect sex determination in fish?" --skip-vector
python scripts/agent/07_export_kg_viewer_data.py --bundle data/agent/evidence_bundle.json
python scripts/agent/08_make_kg_viewer.py
streamlit run app_kg_viewer.py
```
"""
)

if not HTML_PATH.exists():
    st.error(f"Viewer HTML not found: {HTML_PATH}")
    st.stop()

components.html(HTML_PATH.read_text(encoding="utf-8"), height=850, scrolling=True)
