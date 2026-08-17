#!/usr/bin/env python3
"""
run_logging.py — the run-artifact policy for this repo.

WHY
---
A survey run used to leave ~700 MB of per-condition COCO dump JSONs behind
(`clean_jpeg.json`, `pgd_eps0.03_median.json`, ...). They are regenerable,
useless to a reader, and they made the results directories impossible to share.
`.gitignore` hides them from git but they still fill the disk and still have to
be skipped by hand when zipping a run for a teammate.

This module makes every run produce exactly THREE kinds of artifact:

    summary*.json   the numbers            — committed, a few KB
    *.md            the readable table     — committed, a few KB
    run.log         the console transcript — committed, tens of KB

plus, on disk only (gitignored, never shared):

    checkpoint_*.pkl / state_*.json   resume state
    per_image_*.json                  per-image audit trail

Everything else — above all the COCO eval dumps — is written to a temporary
directory and deleted when the run ends, unless `--keep-eval-dumps` is passed.

USAGE (both new scripts do exactly this)

    from run_logging import run_log, eval_dump_dir, artifact_report

    with run_log(os.path.join(out_dir, "run.log"), argv=sys.argv):
        ...
        with eval_dump_dir(out_dir, keep=args.keep_eval_dumps) as dump_dir:
            eval_stats = pc.evaluate_all_conditions(..., output_dir=dump_dir)
        ...
        artifact_report(out_dir)

Stdlib only — importable with no torch, so `--preflight` still works.
"""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

# Filenames matching these are the ones .gitignore keeps (see the results/
# block there: `!results/**/summary*.json`, `!results/**/run.log`, and *.md is
# never ignored). Kept in sync by test_offline_logic.py.
COMMITTED_PATTERNS = ("summary", ".md", "run.log")
# Anything bigger than this is flagged: GitHub warns at 50 MB and blocks at
# 100 MB, and a shared run should be nowhere near either.
LARGE_FILE_MB = 5.0


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        commit = out.stdout.strip() or "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5)
        return commit + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:  # noqa: BLE001
        return "unknown"


class _Tee:
    """Duplicate a text stream into a file handle."""

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, data):
        self._stream.write(data)
        self._handle.write(data)
        self._handle.flush()          # tmux-friendly: tail -f sees it live
        return len(data)

    def flush(self):
        self._stream.flush()
        self._handle.flush()

    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        # Real fd, so a subprocess (nvidia-smi) still writes to the terminal.
        return self._stream.fileno()


@contextlib.contextmanager
def run_log(path, argv=None, extra=None):
    """Tee stdout+stderr into `path` (append) with a session header.

    Append, not truncate: a tmux session that resumes a checkpointed run adds
    a new stamped block instead of destroying the earlier one.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    handle = open(path, "a", encoding="utf-8", buffering=1)
    started = time.time()
    header = [
        "=" * 78,
        f"RUN START  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  host    : {platform.node()}",
        f"  python  : {platform.python_version()}",
        f"  git     : {_git_commit()}",
        f"  cwd     : {os.getcwd()}",
        f"  command : {' '.join(argv) if argv else '(not recorded)'}",
    ]
    for k, v in (extra or {}).items():
        header.append(f"  {k:<8}: {v}")
    header.append("=" * 78)
    handle.write("\n".join(header) + "\n")

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_out, handle)
    sys.stderr = _Tee(old_err, handle)
    status = "OK"
    try:
        yield path
    except BaseException as exc:  # noqa: BLE001 — record, then re-raise
        status = f"FAILED ({type(exc).__name__}: {exc})"
        raise
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        handle.write(
            f"{'=' * 78}\nRUN END    {time.strftime('%Y-%m-%d %H:%M:%S')}  "
            f"| {(time.time() - started) / 60:.1f} min | {status}\n"
            f"{'=' * 78}\n")
        handle.close()


@contextlib.contextmanager
def eval_dump_dir(output_dir, keep=False):
    """Where pycocotools' per-condition detection dumps go.

    keep=False (default): a temp dir, deleted on exit — the dumps never touch
    the results directory, so nothing large is ever produced.
    keep=True: `<output_dir>/eval_dumps/`, for debugging a suspicious mAP.
    COCOeval needs a real file on disk either way (`coco_gt.loadRes(path)`),
    so the dumps cannot simply be skipped.
    """
    if keep:
        path = os.path.join(output_dir, "eval_dumps")
        os.makedirs(path, exist_ok=True)
        print(f"[artifacts] --keep-eval-dumps: COCO dumps kept in {path} "
              f"(large, gitignored — delete before sharing).")
        yield path
        return
    tmp = tempfile.mkdtemp(prefix="coco_eval_dump_")
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def is_committed(name) -> bool:
    """True if .gitignore's results/ rules keep this filename."""
    return any(p in name for p in COMMITTED_PATTERNS)


def artifact_report(output_dir) -> dict:
    """Print what the run left behind, split into committed vs local-only."""
    committed, local, total = [], [], 0
    for root, _, files in os.walk(output_dir):
        for f in sorted(files):
            full = os.path.join(root, f)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            total += size
            rel = os.path.relpath(full, output_dir)
            (committed if is_committed(f) else local).append((rel, size))

    def _fmt(n):
        return f"{n / 1024:.0f} KB" if n < 1024 ** 2 else f"{n / 1024 ** 2:.1f} MB"

    print("\n" + "=" * 70)
    print("RUN ARTIFACTS".center(70))
    print("=" * 70)
    print(f"  {output_dir}")
    print("\n  committed to git (share these):")
    for rel, size in sorted(committed):
        print(f"    {_fmt(size):>9}  {rel}")
    if not committed:
        print("    (none)")
    print("\n  local only (gitignored, do not share):")
    for rel, size in sorted(local):
        print(f"    {_fmt(size):>9}  {rel}")
    if not local:
        print("    (none)")

    committed_bytes = sum(s for _, s in committed)
    print(f"\n  committed total: {_fmt(committed_bytes)} | "
          f"directory total: {_fmt(total)}")
    big = [(r, s) for r, s in committed if s > LARGE_FILE_MB * 1024 ** 2]
    if big:
        print(f"  ⚠ {len(big)} committed file(s) exceed {LARGE_FILE_MB} MB — "
              f"trim before pushing:")
        for rel, size in big:
            print(f"      {_fmt(size):>9}  {rel}")
    else:
        print("  ✓ every committed artifact is small enough to push and share.")
    print("=" * 70)
    return {"committed_bytes": committed_bytes, "total_bytes": total,
            "committed_files": [r for r, _ in committed],
            "local_files": [r for r, _ in local]}
