"""One-shot read-only fit review, gated on the complete pre-reference audit."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
HELPER = REPO/"analysis/review_payoff_boundary_fit.py"
EXPECTED_HELPER_SHA256 = "63b017824281825c208e6b753cee2c4e947cf8e783201eb6ede7ddc6b41554f5"
EXPECTED_SOURCE = "0d9bda04e5da1cc00cc3ec5917d23a4352e6ceb4585dd1a066779b364cf79576"
PACKET = ROOT/"execution/fit_review_continuation.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(packet):
    temporary = PACKET.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(packet, indent=2, sort_keys=True)+"\n")
    temporary.replace(PACKET)


def read_if_present(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    if PACKET.exists():
        raise FileExistsError("Do not duplicate the one-shot review continuation")
    started = time.monotonic()
    command = [sys.executable, str(HELPER), "--root", str(ROOT)]
    packet = {"status": "waiting-for-complete-postfit-validation", "pid": os.getpid(),
              "script_sha256": sha(__file__), "helper_sha256": EXPECTED_HELPER_SHA256,
              "source_hash": EXPECTED_SOURCE, "started_utc": utc(), "command": command,
              "environment": {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
              "fresh_outcome_access": False, "no_fitting_or_selection": True}
    write(packet)
    print(json.dumps({"pid": os.getpid(), "status": packet["status"]}), flush=True)
    try:
        while True:
            if sha(HELPER) != EXPECTED_HELPER_SHA256:
                raise ValueError("Report-only review helper changed after its wait was launched")
            audit = read_if_present(ROOT/"audit/postfit_validation_diagnostic.json")
            lock = read_if_present(ROOT/"payoffs/fit_lock.json")
            if audit and audit.get("complete"):
                if (not lock or not lock.get("complete") or lock.get("source_hash") != EXPECTED_SOURCE
                        or audit["identity"]["source_hash"] != EXPECTED_SOURCE
                        or audit["identity"]["fit_lock_sha256"] != sha(ROOT/"payoffs/fit_lock.json")
                        or len(audit["records"]) != len(lock["models"])):
                    raise ValueError("Complete validation audit and strict all-model fit lock disagree")
                break
            if time.monotonic()-started > 8*60*60:
                raise TimeoutError("No complete validation audit within the local eight-hour wait")
            time.sleep(5)
        packet.update(status="review-running", audit_sha256=sha(ROOT/"audit/postfit_validation_diagnostic.json"),
                      fit_lock_sha256=sha(ROOT/"payoffs/fit_lock.json"), review_started_utc=utc(),
                      recorded_wait_seconds=time.monotonic()-started)
        write(packet)
        environment = os.environ.copy()
        environment.update(packet["environment"])
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        tick = time.monotonic()
        with (ROOT/"execution/fit_review.log").open("w") as log:
            result = subprocess.run(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        packet.update(return_code=result.returncode, command_wall_seconds=time.monotonic()-tick,
                      command_cpu_user_seconds=after.ru_utime-before.ru_utime,
                      command_cpu_system_seconds=after.ru_stime-before.ru_stime,
                      finished_utc=utc(), elapsed_with_wait_seconds=time.monotonic()-started,
                      log_sha256=sha(ROOT/"execution/fit_review.log"))
        if result.returncode:
            raise RuntimeError(f"Read-only fit review exited {result.returncode}; see fit_review.log")
        review = json.loads((ROOT/"audit/fit_interpretation_review.json").read_text())
        if not review.get("complete") or len(review["records"]) != len(lock["models"]):
            raise ValueError("Read-only fit review omitted a locked model")
        packet.update(status="complete", review_sha256=sha(ROOT/"audit/fit_interpretation_review.json"),
                      review_models=len(review["records"]), review_source_hash=review["identity"]["source_hash"])
        write(packet)
        print(json.dumps({"status": "complete", "models": len(review["records"])}), flush=True)
    except BaseException as error:
        packet.update(status="failed", error=repr(error), finished_utc=utc(),
                      elapsed_with_wait_seconds=time.monotonic()-started)
        write(packet)
        raise


if __name__ == "__main__":
    main()
