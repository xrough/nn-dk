"""Existing validation-only loss decomposition; no final outcome input."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root/"audit/postfit_validation_diagnostic.json"
payload = json.loads(source.read_text())
records = []
for item in payload["records"]:
    strata = item["full_selected_stratified_residuals"]
    all_rows = strata["all"]["mean"]["rows"]
    partitions = {}
    for prefix in ("minimum_mass/", "remaining_horizon_quartile/", "snapshot_fraction/", "profile/"):
        partition = []
        for name, value in strata.items():
            if not name.startswith(prefix):
                continue
            row = {"stratum": name, "rows": value["mean"]["rows"]}
            for stage in ("mean", "covariance"):
                summary = value[stage]
                rms = summary.get("normalized_rms", 0.)
                row[stage+"_normalized_rms"] = rms
                row[stage+"_squared_loss_fraction"] = summary["rows"]*rms*rms/(
                    all_rows*strata["all"][stage]["normalized_rms"]**2)
            partition.append(row)
        assert sum(row["rows"] for row in partition) == all_rows
        for stage in ("mean", "covariance"):
            assert abs(sum(row[stage+"_squared_loss_fraction"] for row in partition)-1) < 1e-12
        partitions[prefix] = partition
    records.append({"case": item["case"], "seed": item["seed"], "rows": all_rows,
                    "full_active_counts": item["full_selected_active_counts"],
                    "full_mean_normalized_rms": strata["all"]["mean"]["normalized_rms"],
                    "full_covariance_normalized_rms": strata["all"]["covariance"]["normalized_rms"],
                    "partitions": partitions})
result = {"complete": True, "kind": "independent-validation-strata-squared-loss-decomposition-v1",
          "source_hash": payload["identity"]["source_hash"],
          "postfit_audit_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "replay_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "records": records, "no_final_test_outcomes_read": True,
          "all_fixed_seeds_retained": True,
          "interpretation": "Loss concentration and interior residuals diagnose this fit; no unique error-cause or independent-accuracy claim."}
target = root/"audit/postfit_strata_review.json"
assert not target.exists()
target.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
for row in records:
    if row["case"] == "T1_k6":
        print(row["seed"], [(v["stratum"], round(v["mean_squared_loss_fraction"], 6),
                             round(v["covariance_squared_loss_fraction"], 6))
                            for v in row["partitions"]["minimum_mass/"]])
print("sha256", hashlib.sha256(target.read_bytes()).hexdigest())
