"""One-shot independent initial-law evaluation after the locked main accuracy run."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.parent
packet = {"kind": "one-shot-independent-deployment-continuation-v1", "complete": False,
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "pid": os.getpid(), "started_unix": time.time(), "stage": "waiting-main-accuracy",
          "no_fitting_or_selection": True, "final_timing_launched": False}


def save():
    p = ROOT/"execution/independent_continuation.json"
    q = p.with_suffix(".tmp")
    q.write_text(json.dumps(packet, indent=2)+"\n")
    q.replace(p)


save()
while True:
    p = json.loads((ROOT/"execution/continuation.json").read_text())
    if p["stage"] == "failed":
        raise RuntimeError("Main accuracy failed; no independent continuation is allowed")
    if p.get("complete"):
        break
    time.sleep(10)
time.sleep(2)
packet.update(stage="independent-initial-law-accuracy", evaluation_started_unix=time.time())
save()
before = resource.getrusage(resource.RUSAGE_CHILDREN)
start = time.perf_counter()
arguments = [str(REPO/".venv/bin/python"), "analysis/evaluate_payoff_regimes_deployment.py",
             "evaluate", "--root", str(ROOT)]
with (ROOT/"execution/independent_deployment.log").open("w") as stream:
    result = subprocess.run(arguments, cwd=REPO, env={**os.environ,
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "MPLCONFIGDIR": "/private/tmp/nn-dk-mpl-cache"}, stdout=stream, stderr=subprocess.STDOUT)
after = resource.getrusage(resource.RUSAGE_CHILDREN)
packet.update(arguments=arguments, returncode=result.returncode,
              evaluation_wall_seconds=time.perf_counter()-start,
              evaluation_CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
              finished_unix=time.time(), complete=result.returncode == 0,
              stage="complete" if result.returncode == 0 else "failed")
save()
if result.returncode:
    raise RuntimeError("Independent initial-law evaluation failed; inspect its retained log")
