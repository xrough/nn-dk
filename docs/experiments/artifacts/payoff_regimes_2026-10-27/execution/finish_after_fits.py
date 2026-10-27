"""One-shot continuation of this fixed study; never selects or refits models."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO/".venv/bin/python"
ENVIRONMENT = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "MPLCONFIGDIR": "/private/tmp/nn-dk-mpl-cache"}
PACKET = {"kind": "one-shot-frozen-regime-continuation-v1", "pid": os.getpid(),
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "started_unix": time.time(), "stage": "waiting-for-all-fit-workers",
          "calls": [], "complete": False, "no_fitting_or_selection": True,
          "final_timing_launched": False}


def save():
    path = ROOT/"execution/continuation.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(PACKET, indent=2)+"\n")
    temporary.replace(path)


def run(label, args):
    started = time.time()
    with (ROOT/"execution"/(label+".log")).open("a") as stream:
        process = subprocess.run([str(PYTHON), *args], cwd=REPO, env=ENVIRONMENT,
                                 stdout=stream, stderr=subprocess.STDOUT)
    return {"label": label, "arguments": args, "returncode": process.returncode,
            "started_unix": started, "finished_unix": time.time()}


def checked(label, args):
    result = run(label, args)
    PACKET["calls"].append(result)
    save()
    if result["returncode"]:
        raise RuntimeError(f"{label} failed; inspect its complete retained execution log")


save()
try:
    while True:
        workers = [ROOT/f"payoffs/execution/worker_seed{s}.json" for s in (0, 1, 2)]
        if all(p.exists() and json.loads(p.read_text()).get("complete") for p in workers):
            break
        time.sleep(10)
    # Worker ownership locks are released immediately after these final packets.
    time.sleep(2)
    PACKET["stage"] = "strict-fit-lock-and-pre-reference-validation"
    save()
    checked("fit_lock", ["analysis/run_payoff_regimes.py", "lock", "--root", str(ROOT)])
    checked("postfit_validation", ["analysis/audit_payoff_regimes_validation.py", "--root", str(ROOT)])
    PACKET["stage"] = "independent-native-reference"
    save()
    cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    cases = list(json.loads((ROOT/"payoffs/preregistration.json").read_text())["configs"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run, "reference_"+case, ["analysis/run_payoff_regimes.py", "reference",
                    "--root", str(ROOT), "--case", case]) for case in cases]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            PACKET["calls"].append(result)
            save()
            if result["returncode"]:
                raise RuntimeError(f"Reference {result['label']} failed; no survivor-conditioned estimate permitted")
    cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    PACKET["native_reference_children_CPU_seconds"] = ((cpu_after.ru_utime+cpu_after.ru_stime)
        -(cpu_before.ru_utime+cpu_before.ru_stime))
    PACKET["stage"] = "independent-accuracy"
    save()
    checked("accuracy", ["analysis/evaluate_payoff_regimes.py", "evaluate", "--root", str(ROOT)])
    PACKET.update(stage="accuracy-complete-ready-for-uncontended-timing", complete=True,
                  finished_unix=time.time())
    save()
except Exception as error:
    PACKET.update(stage="failed", failure={"type": type(error).__name__, "message": str(error)},
                  finished_unix=time.time())
    save()
    raise
