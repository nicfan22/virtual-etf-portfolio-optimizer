"""Generate the GitHub Pages static dashboard.

Run from the repository root:
    python scripts/generate_static_report.py

The output is written to docs/index.html. Push the repository to GitHub and
publish the docs folder with GitHub Pages to get a public URL.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.report_builder import generate_demo_report  # noqa: E402


if __name__ == "__main__":
    output = ROOT / "docs" / "index.html"
    path = generate_demo_report(output)
    print(f"Static LinkedIn-ready report written to: {path}")
