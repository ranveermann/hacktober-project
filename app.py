"""GitTrace Streamlit entry point.

    cd /path/to/your/cloned/repo
    streamlit run /path/to/gittrace/app.py

The app analyses the repository in the current working directory (override with
`streamlit run app.py -- --repo /path/to/repo` or the sidebar).
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))  # works even without `pip install -e .`

from gittrace.webapp import main  # noqa: E402

main()
