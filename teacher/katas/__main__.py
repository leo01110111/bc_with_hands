"""Entry point: `python -m katas`.

Deliberately thin -- running runner.py as __main__ would import katas.runner a
second time under a different module identity, and the KataFail raised by the
levels would no longer match the KataFail caught by the runner.
"""

import argparse
import sys

from katas.runner import run, show

ap = argparse.ArgumentParser(prog="katas")
ap.add_argument("--level", type=int, default=None)
ap.add_argument("--show", choices=["hint", "peek", "brief"], default=None)
ap.add_argument("--verbose", "-v", action="store_true")
ap.add_argument("--no-color", action="store_true")
a = ap.parse_args()

sys.exit(show(a.show, a.level) if a.show else run(a.level, a.verbose, a.no_color))
