"""Run each experiment in its OWN process so a hard CUDA/driver crash cannot take the
rest of the plan down, with one automatic retry per experiment.

    python -m dermai_exp.drive_plan --plan results/experiments/plan_c.json --only c02_pw4
"""

import argparse
import json
import os
import subprocess
import sys
import time

from .common import EXP_ROOT, PROJECT, ensure_dirs, get_logger

PY = sys.executable


def run(name, plan_path, retries=1):
    cmd = [PY, "-u", "-m", "dermai_exp.sweep", "--plan", plan_path, "--only", name]
    for attempt in range(retries + 1):
        t0 = time.time()
        p = subprocess.run(cmd, cwd=PROJECT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        tail = (p.stdout or "").strip().splitlines()[-6:]
        if p.returncode == 0 and any("-> selection" in l for l in (p.stdout or "").splitlines()):
            return True, (time.time() - t0) / 60, "\n".join(tail)
        sys.stdout.write("\n".join(tail) + "\n")
        if p.returncode != 0:
            sys.stdout.write((p.stderr or "").strip().splitlines()[-8:] and
                             "\n".join((p.stderr or "").strip().splitlines()[-8:]) + "\n")
        if attempt < retries:
            time.sleep(20)
    return False, (time.time() - t0) / 60, "see results/experiments/sweep_log.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--retries", type=int, default=1)
    args = ap.parse_args()
    ensure_dirs()
    plan_path = os.path.abspath(args.plan)
    log = get_logger("dermai.drive",
                     logfile=os.path.join(EXP_ROOT, "drive_log.txt"))
    plan = json.load(open(plan_path))
    todo = [s["name"] for s in plan if not args.only or s["name"] in args.only]
    log("")
    log("#" * 78)
    log("DRIVE %s -> %s" % (os.path.basename(plan_path), ", ".join(todo)))
    log("#" * 78)
    results = []
    for name in todo:
        log(">>> %s" % name)
        ok, mins, tail = run(name, plan_path, args.retries)
        log("    %s %s (%.1f min)" % ("OK  " if ok else "FAIL", name, mins))
        for line in tail.splitlines():
            log("      | %s" % line)
        results.append((name, ok, mins))
    log("")
    log("PLAN RESULT")
    for name, ok, mins in results:
        log("  %-24s %-5s %6.1f min" % (name, "ok" if ok else "FAIL", mins))
    with open(os.path.join(EXP_ROOT, "drive_%s.json"
                           % os.path.splitext(os.path.basename(plan_path))[0]), "w") as fh:
        json.dump(results, fh, indent=2, default=float)


if __name__ == "__main__":
    main()