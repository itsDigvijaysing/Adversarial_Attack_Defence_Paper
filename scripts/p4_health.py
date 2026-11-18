#!/usr/bin/env python3
"""Read-only health check of the Phase-4 queues started by p4_launch.sh.

    python3 scripts/p4_health.py full          # or: test
    python3 scripts/p4_health.py full --log    # also append the report to
                                               # logs/phase4_queue/<mode>/health.log

For every queue window: the job it is on, its progress and speed from the
last heartbeat, an ETA for that job, how many jobs are left, and a verdict:
  OK        progressing
  WAITING   between jobs / loading a model (no heartbeat yet)
  STALLED?  no console output for much longer than its heartbeat interval
  DEAD?     the queue says a job is running but no such process exists
  FINISHED  queue done
Plus failed attempts, GPU memory/utilisation, disk space and whether the
tmux session is alive. It never starts, stops or changes anything; the exit
code is 0 when all is well and 1 when something needs a look. Stdlib only.
"""
import glob
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEAT = re.compile(r"\] (\d+)/(\d+) \| ([\d.]+)m \| ([\d.]+)s/img")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def fmt_h(seconds):
    if seconds is None:
        return "?"
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}h{m:02d}m"


def queue_state(name, logs, now, procs):
    con = os.path.join(logs, f"{name}.console.log")
    qfile = os.path.join(logs, "queues", f"{name}.txt")
    total = sum(1 for l in open(qfile) if l.strip() and not l.lstrip().startswith("#"))
    if not os.path.exists(con):
        return "WAITING", f"not started ({total} jobs queued)", None
    text = open(con, errors="replace").read()
    events = re.findall(r"^\[queue \S+\] (\S+ \S+) (START|END|FAIL|SKIP|QUEUE FINISHED)"
                        r"(?: \[(\d+)/(\d+)\])?(.*)$", text, re.M)
    finished = sum(1 for e in events if e[1] in ("END", "SKIP"))
    if "QUEUE FINISHED" in text:
        return "FINISHED", f"{finished}/{total} jobs done", None
    starts = [e for e in events if e[1] == "START"]
    if not starts:
        return "WAITING", "starting", None
    cur = starts[-1]
    cmd = cur[4].split(") ", 1)[-1].strip()
    short = re.sub(r"--(deterministic|gpu \d+|attack-space native|pgd-select best|"
                   r"results-root \S+|log-name \S+|experiment )", "", cmd)
    short = re.sub(r"python phase4/|\s+", " ", short).strip()[:95]
    tail = text[text.rfind(cur[0]):]
    beats = BEAT.findall(tail)
    idle = now - os.path.getmtime(con)
    alive = any(cmd.split()[1] in p and all(tok in p for tok in cmd.split()[2:8])
                for p in procs) if cmd.startswith("python") else True
    if beats:
        done, n, mins, spi = int(beats[-1][0]), int(beats[-1][1]), float(beats[-1][2]), float(beats[-1][3])
        step = (int(beats[-1][0]) - int(beats[-2][0])) if len(beats) > 1 else done
        # The heartbeat's s/img is the average since the job started; when
        # the GPU load changes, the rate over the last interval is the
        # better predictor of the time left.
        recent = ((mins - float(beats[-2][2])) * 60 / step
                  if len(beats) > 1 and step > 0 else spi)
        gap = recent * max(step, 1)
        eta = (n - done) * recent
        prog = (f"{done}/{n}, {spi:.1f}s/img overall, {recent:.1f}s/img lately, "
                f"job ETA {fmt_h(eta)}")
        limit = 2.5 * gap + 600
    else:
        prog, limit = "no heartbeat yet (loading or first batch)", 5400
    verdict = "OK" if beats else "WAITING"
    if not alive:
        verdict = "DEAD?"
    elif idle > limit:
        verdict = "STALLED?"
    detail = (f"job {cur[2]}/{total}: {short}\n      {prog}; last output "
              f"{fmt_h(idle)} ago; {finished} done")
    return verdict, detail, None


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "full"
    logs = os.path.join(REPO, "logs", "phase4_queue", mode)
    now = time.time()
    out = []
    session = f"p4_{mode}"
    alive = sh(f"tmux has-session -t {session} 2>/dev/null && echo yes") == "yes"
    out.append(f"=== Phase-4 {mode} health | {time.strftime('%Y-%m-%d %H:%M:%S')} | "
               f"tmux {session}: {'alive' if alive else 'NOT RUNNING'}")
    out.append("GPU: " + sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
                            "--format=csv,noheader"))
    out.append("disk free: " + sh(f"df -h {REPO} | tail -1 | awk '{{print $4}}'"))
    procs = sh("ps -eo args | grep 'phase4/p4_' | grep -v grep").splitlines()
    out.append(f"python jobs running: {len(procs)}")
    bad = not alive
    for qf in sorted(glob.glob(os.path.join(logs, "queues", "*.txt"))):
        name = os.path.basename(qf)[:-4]
        verdict, detail, _ = queue_state(name, logs, now, procs)
        bad |= verdict in ("STALLED?", "DEAD?")
        out.append(f"[{verdict:8}] {name}: {detail}")
    status = os.path.join(logs, "status.tsv")
    fails = []
    if os.path.exists(status):
        for line in open(status):
            f = line.rstrip("\n").split("\t")
            if len(f) > 5 and f[2] == "END" and f[3] not in ("", "0"):
                fails.append(f"  FAIL {f[0]} {f[1]} exit {f[3]}: {f[5][:110]}")
    out.append(f"failed attempts: {len(fails)}")
    out += fails
    bad |= bool(fails)
    out.append("verdict: " + ("NEEDS A LOOK" if bad else "all good"))
    report = "\n".join(out)
    print(report)
    if "--log" in sys.argv:
        with open(os.path.join(logs, "health.log"), "a") as f:
            f.write(report + "\n\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
