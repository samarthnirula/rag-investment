#!/usr/bin/env bash
set -euo pipefail

source venv/bin/activate
streamlit run src/insightlens/ui/streamlit_app.py
