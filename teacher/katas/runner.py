"""The `make check` runner.

Runs levels in order. The first failure stops the run and everything after it
shows as locked -- you always have exactly one thing to work on.
"""

import argparse
import importlib
import io
import sys
import time
import traceback
from contextlib import redirect_stdout

C_GREEN = "\033[32m"
C_RED = "\033[31m"
C_DIM = "\033[2m"
C_BOLD = "\033[1m"
C_YELLOW = "\033[33m"
C_CYAN = "\033[36m"
C_OFF = "\033[0m"


class KataFail(AssertionError):
    """Raised by a level check. `hint` is shown under the failure."""

    def __init__(self, message: str, expected=None, got=None, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.expected = expected
        self.got = got
        self.hint = hint


class NotImplementedYet(KataFail):
    def __init__(self, what: str):
        super().__init__(f"{what} is still a stub", hint="open the file and start there")


def _fmt(v) -> str:
    if v is None:
        return ""
    if hasattr(v, "shape") and hasattr(v, "dtype"):
        return f"array{tuple(v.shape)} {v.dtype}"
    return str(v)


def run(only: int | None = None, verbose: bool = False, no_color: bool = False) -> int:
    from katas.levels import LEVELS

    if no_color or not sys.stdout.isatty():
        globals().update({k: "" for k in list(globals()) if k.startswith("C_")})

    print()
    print(f"  {C_BOLD}wuji_bc kata{C_OFF}  {C_DIM}behaviour cloning for a 16-DoF LEAP hand{C_OFF}")
    print()

    passed = 0
    blocked = False
    failure = None
    t_start = time.time()

    for lvl in LEVELS:
        label = f"  {C_DIM}L{lvl.num:02d}{C_OFF}  {lvl.title:<34s}"

        if only is not None and lvl.num != only:
            if lvl.num < only:
                print(f"{label} {C_DIM}skipped{C_OFF}")
                continue
            else:
                break

        if blocked:
            print(f"{label} {C_DIM}locked{C_OFF}")
            continue

        buf = io.StringIO()
        try:
            t0 = time.time()
            with redirect_stdout(buf):
                lvl.check()
            dt = time.time() - t0
            dots = "." * max(1, 26 - len(lvl.title))
            timing = f" {C_DIM}{dt:.1f}s{C_OFF}" if dt > 1.0 else ""
            print(f"{label} {C_GREEN}PASS{C_OFF}{timing}")
            passed += 1
        except KataFail as e:
            print(f"{label} {C_RED}FAIL{C_OFF}")
            print(f"        {C_RED}{e.message}{C_OFF}")
            if e.expected is not None or e.got is not None:
                print(f"        expected  {_fmt(e.expected)}")
                print(f"        got       {_fmt(e.got)}")
            if e.hint:
                print(f"        {C_YELLOW}hint:{C_OFF} {e.hint}")
            failure = lvl
            blocked = True
        except Exception:
            print(f"{label} {C_RED}ERROR{C_OFF}")
            tb = traceback.format_exc().rstrip().splitlines()
            for line in tb[-6:]:
                print(f"        {C_DIM}{line}{C_OFF}")
            failure = lvl
            blocked = True

        if verbose and buf.getvalue():
            for line in buf.getvalue().rstrip().splitlines():
                print(f"        {C_DIM}{line}{C_OFF}")

    total = len(LEVELS)
    print()
    bar_n = 22
    filled = int(bar_n * passed / total)
    bar = "█" * filled + "░" * (bar_n - filled)
    print(f"  {C_CYAN}{bar}{C_OFF}  {passed}/{total} levels", end="")
    if failure is not None:
        print(f"  {C_DIM}·{C_OFF}  next: {C_BOLD}L{failure.num:02d} {failure.title}{C_OFF}")
        print()
        print(f"  {C_DIM}edit{C_OFF}  {failure.file}")
        print(f"  {C_DIM}help{C_OFF}  make hint LEVEL={failure.num}   "
              f"{C_DIM}·{C_OFF}   make peek LEVEL={failure.num}   "
              f"{C_DIM}·{C_OFF}   make brief LEVEL={failure.num}")
    elif passed == total:
        print()
        print()
        print(f"  {C_GREEN}{C_BOLD}all levels cleared.{C_OFF} you wrote a flow-matching BC "
              f"policy that grasps.")
    print(f"  {C_DIM}{time.time() - t_start:.1f}s total{C_OFF}")
    print()
    return 0 if failure is None else 1


def show(kind: str, num: int) -> int:
    from katas.levels import LEVELS

    lvl = next((l for l in LEVELS if l.num == num), None)
    if lvl is None:
        print(f"no level {num}")
        return 1
    print()
    print(f"  {C_BOLD}L{lvl.num:02d}  {lvl.title}{C_OFF}")
    print(f"  {C_DIM}{lvl.file}{C_OFF}")
    print()
    if kind == "brief":
        for line in lvl.brief.strip().splitlines():
            print(f"  {line}")
    elif kind == "hint":
        for i, h in enumerate(lvl.hints, 1):
            print(f"  {C_YELLOW}hint {i}/{len(lvl.hints)}{C_OFF}  {h}")
            print()
    elif kind == "peek":
        print(f"  {C_CYAN}how OGPO does it:{C_OFF}")
        for ref in lvl.ogpo_refs:
            print(f"    {ref}")
        print()
        print(f"  {C_DIM}read it, then close it and write your own.{C_OFF}")
    print()
    return 0


if __name__ == "__main__":  # pragma: no cover -- use `python -m katas`
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--show", choices=["hint", "peek", "brief"], default=None)
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args()
    if a.show:
        sys.exit(show(a.show, a.level))
    sys.exit(run(a.level, a.verbose, a.no_color))
