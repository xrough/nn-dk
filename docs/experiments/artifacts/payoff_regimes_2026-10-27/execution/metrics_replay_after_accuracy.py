"""Single read-only replay after both immutable accuracy inventories complete."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

repository = Path(__file__).resolve().parents[3]
root = Path(__file__).resolve().parents[1]
script = repository/"analysis/audit_payoff_regimes_metrics.py"
expected_sha = "0c30d30fc32c17d3a6b0d15813becf21615a346653b7418bcb00eb43406d3d66"
indices = (root/"comparison/accuracy_index.json", root/"deployment/accuracy_index.json")
while not all(path.exists() and json.loads(path.read_text()).get("complete") for path in indices):
    time.sleep(30)
assert hashlib.sha256(script.read_bytes()).hexdigest() == expected_sha
assert not (root/"audit/metrics_replay.json").exists()
environment = os.environ.copy()
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    environment[name] = "1"
environment["MPLCONFIGDIR"] = "/private/tmp/nn-dk-mpl-cache"
with (root/"execution/metrics_replay.log").open("w") as stream:
    stream.write("Started read-only replay after both complete accuracy indices.\n")
    stream.flush()
    result = subprocess.run([str(repository/".venv/bin/python"), str(script), "--root", str(root)],
                            cwd=repository, env=environment, stdout=stream, stderr=subprocess.STDOUT)
raise SystemExit(result.returncode)
