"""Compact diagnostic archives retain source and reject incomplete evidence."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from nn_dk.data import source_hash
from nn_dk.types import stable_hash

_spec = importlib.util.spec_from_file_location(
    "archive_payoff_diagnostics",Path(__file__).parents[1]/"analysis/archive_payoff_diagnostics.py")
archive = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = archive
_spec.loader.exec_module(archive)


def _json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")


def _fake_completed(tmp_path,monkeypatch):
    monkeypatch.setattr(archive,"COMPRESS_THRESHOLD",64)
    historical = tmp_path/"historical"
    fit_lock = historical/"payoffs/fit_lock.json"
    _json(fit_lock,{"complete":True})
    validation = historical/"validation.h5"
    validation.write_bytes(b"full validation states remain local")
    models = []
    for seed in (0,1,2):
        weights = historical/f"weights{seed}.pt"
        weights.write_bytes(f"full original weights {seed}".encode())
        sidecar = weights.with_suffix(".json")
        _json(sidecar,{"complete":True,"seed":seed})
        models.append({"case":"tiny","seed":seed,"checkpoint_path":str(weights),
                       "checkpoint_sha256":archive.sha(weights),"metadata_path":str(sidecar),
                       "metadata_sha256":archive.sha(sidecar)})
    roots = {}
    for label,script in (("mean_gradient","audit_payoff_mean_gradients.py"),
                         ("residual_transport","audit_payoff_residual_transport.py")):
        root = tmp_path/label
        origins = root/"anchors/tiny.npz"
        origins.parent.mkdir(parents=True)
        np.savez(origins,states=np.full((1,8),.125),profiles=np.array(["uniform"]))
        identity = {"kind":label,"source_hash":source_hash(),"study_root":str(historical),
            "models":models,"cases":["tiny"],"script_sha256":archive.sha(archive.REPO/"analysis"/script),
            "helpers":{"compiled_native.py":archive.sha(archive.REPO/"analysis/compiled_native.py")},
            "fit_lock_sha256":archive.sha(fit_lock),
            "plan":{"paths":4,"chunk_paths":2,"query_fractions":[1.]}}
        identity_hash = stable_hash(identity)
        prepared = {"identity":identity,"identity_hash":identity_hash,"anchors":{
            "tiny":{"path":str(origins),"sha256":archive.sha(origins),"rows":1,"queries":[.1],
                    "state_bank_path":str(validation),"state_bank_sha256":archive.sha(validation)}}}
        identity_hash = archive._plan_digest(prepared,label)
        prepared["identity_hash"] = identity_hash
        _json(root/"plan.json",prepared)
        chunks = []
        for start in (0,2):
            path = root/f"chunks/tiny/anchor0_{start}.npz"
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez(path,family_ids=np.array([f"{label}_{start}",f"{label}_{start+1}"]),
                     full_native_polygon=np.full((2,5,8),.125))
            row = {"case":"tiny","anchor":0,"start":start,"families":2,"path":str(path),
                   "identity_hash":identity_hash,"sha256":archive.sha(path)}
            _json(path.with_suffix(".json"),row)
            chunks.append(row)
        summary = {"complete":True,"identity_hash":identity_hash,"records":[
            {"case":"tiny","anchor":0,"seed":seed,"families":4,"queries":[.1]}
            for seed in (0,1,2)]}
        if label=="residual_transport":
            shapes = {"mean_discrepancy":(1,4),"raw_second_discrepancy":(1,4,4),
                "sample_covariance_discrepancy_unbiased":(1,4,4),"mean_residual_integral":(1,4),
                "raw_second_residual_integral":(1,4,4),"covariance_residual_integral_alone":(1,4,4),
                "mean_transport_defect":(1,4),"raw_second_transport_defect":(1,4,4)}
            for row in summary["records"]:
                row["levels"]={level:{stride:{"families":4,
                    "point":{k:np.zeros(s).tolist() for k,s in shapes.items()},
                    "pointwise_percentile95":{k:np.zeros((2,*s)).tolist() for k,s in shapes.items()}}
                    for stride in ("declared","half_")} for level in ("coarse","fine")}
        if label=="mean_gradient":
            evidence = root/"gradient_evidence.npz"
            np.savez(evidence,gradient=np.ones((1,4,8)))
            summary["numeric_evidence"]={"path":str(evidence),"sha256":archive.sha(evidence)}
        _json(root/"summary.json",summary)
        _json(root/"index.json",{"complete":True,"expected_chunks":2,"identity_hash":identity_hash,
            "summary_sha256":archive.sha(root/"summary.json"),"chunks":chunks})
        roots[label]=root

    def load_complete(root):
        root = Path(root)
        index = json.loads((root/"index.json").read_text())
        if not index["complete"]:
            raise ValueError("Incomplete diagnostic")
        if archive.sha(root/"summary.json")!=index["summary_sha256"]:
            raise ValueError("Diagnostic summary changed")
        for row in index["chunks"]:
            if archive.sha(row["path"])!=row["sha256"]:
                raise ValueError("Raw chunk changed")
        return json.loads((root/"summary.json").read_text())

    monkeypatch.setattr(archive.gradient,"load_complete",load_complete)
    monkeypatch.setattr(archive.transport,"load_complete",load_complete)
    report = roots["mean_gradient"]/"report"
    identity = {"audit_root":str(roots["mean_gradient"]),
        "audit_script_sha256":archive.sha(archive.REPO/"analysis/audit_payoff_mean_gradients.py"),
        "report_script_sha256":archive.sha(archive.REPO/"analysis/report_payoff_mean_gradients.py"),
        "inputs":{str(roots["mean_gradient"]/name):archive.sha(roots["mean_gradient"]/name)
                  for name in ("plan.json","index.json","summary.json")}}
    _json(report/"report.json",{"complete":True,"identity":identity})
    (report/"report.md").write_text("# Synthetic software fixture; no scientific result\n")
    (report/"contrast_table.csv").write_text("case,seed\ntiny,0\n")
    np.savez(report/"contrasts.npz",signed=np.ones(2))
    _json(report/"output_manifest.json",{"complete":True,"identity":identity,"outputs":{
        name:archive.sha(report/name) for name in ("report.json","report.md","contrast_table.csv","contrasts.npz")}})
    monkeypatch.setattr(archive.signed,"load_report",lambda p:json.loads((Path(p)/"report.json").read_text()))
    return roots,historical


def _make_archive(tmp_path,monkeypatch):
    roots,historical = _fake_completed(tmp_path,monkeypatch)
    destination = tmp_path/"compact"
    result = archive.archive(roots["mean_gradient"],roots["residual_transport"],destination)
    return roots,historical,destination,result


def test_lossless_compact_archive_source_snapshots_and_offline_verification(tmp_path,monkeypatch):
    roots,historical,destination,result = _make_archive(tmp_path,monkeypatch)
    manifest = json.loads((destination/"archive_manifest.json").read_text())
    assert result["complete"] and result["diagnostics_only"]
    assert not manifest["active_readme_goal_complete"]
    assert manifest["source_snapshot_may_include_uncommitted_or_untracked_files"]
    assert manifest["diagnostics"]["mean_gradient"]["allocation"]["records"]==3
    copied=manifest["copied_artifacts"]
    assert any(name.endswith("summary.json.gz") for name in copied)
    assert any(name.endswith("gradient_evidence.npz") for name in copied)
    assert any(name.endswith("contrasts.npz") for name in copied)
    assert not any("/chunks/" in name and name.endswith(".npz") for name in copied)
    assert not any(name.endswith((".pt",".h5")) for name in copied)
    assert any(row["roles"]==["binary-checkpoint-local-only"] for row in manifest["external_inputs"].values())
    assert "src/nn_dk/payoff_boundary.py" in manifest["source_snapshot"]
    assert "analysis/audit_payoff_residual_transport.py" in manifest["source_snapshot"]
    assert "R4" not in (destination/"README.md").read_text()  # Explain scope in words, not unexplained IDs.
    for root in roots.values():
        shutil.rmtree(root)
    shutil.rmtree(historical)
    assert archive.verify_archive(destination)==result


@pytest.mark.parametrize("mutation",["missing","compressed","extra","restored"])
def test_offline_verifier_rejects_missing_changed_extra_or_bad_restored_payload(tmp_path,monkeypatch,mutation):
    _,_,destination,_ = _make_archive(tmp_path,monkeypatch)
    manifest_path = destination/"archive_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    name = next(n for n,v in manifest["copied_artifacts"].items() if v["compressed"])
    path = destination/name
    if mutation=="missing":
        path.unlink()
    elif mutation=="compressed":
        path.write_bytes(path.read_bytes()+b"tampered")
    elif mutation=="extra":
        (destination/"unexpected.txt").write_text("not declared")
    else:
        path.write_bytes(b"not a gzip stream")
        manifest["copied_artifacts"][name]["sha256"]=archive.sha(path)
        _json(manifest_path,manifest)
    with pytest.raises(ValueError):
        archive.verify_archive(destination)


@pytest.mark.parametrize("mutation",["partial","seed","component","chunk","extra_chunk","report","family_reuse","source"])
def test_readiness_rejects_partial_missing_corrupt_or_incompatible_allocation(tmp_path,monkeypatch,mutation):
    roots,_ = _fake_completed(tmp_path,monkeypatch)
    root=roots["residual_transport"]
    index=json.loads((root/"index.json").read_text())
    if mutation=="partial":
        index["complete"]=False
        _json(root/"index.json",index)
    elif mutation in ("seed","component"):
        summary=json.loads((root/"summary.json").read_text())
        if mutation=="seed":
            summary["records"].pop()
        else:
            summary["records"][0]["levels"]["fine"]["half_"]["point"]["raw_second_transport_defect"]=[]
        _json(root/"summary.json",summary)
        index["summary_sha256"]=archive.sha(root/"summary.json")
        _json(root/"index.json",index)
    elif mutation=="chunk":
        Path(index["chunks"][0]["path"]).write_bytes(b"corrupted raw families")
    elif mutation=="extra_chunk":
        np.savez(root/"chunks/tiny/uncommitted.npz",family_ids=np.array(["unknown"]))
    elif mutation=="report":
        (roots["mean_gradient"]/"report/report.md").write_text("changed interpretation")
    elif mutation=="family_reuse":
        path=Path(index["chunks"][0]["path"])
        np.savez(path,family_ids=np.array(["mean_gradient_0","mean_gradient_1"]))
        index["chunks"][0]["sha256"]=archive.sha(path)
        _json(path.with_suffix(".json"),index["chunks"][0])
        _json(root/"index.json",index)
    else:
        plan=json.loads((root/"plan.json").read_text())
        plan["identity"]["helpers"]["compiled_native.py"]="0"*64
        plan["identity_hash"]=stable_hash(plan["identity"])
        # Keep the outer records coherent: the source bytes are the defect.
        _json(root/"plan.json",plan)
        summary=json.loads((root/"summary.json").read_text())
        summary["identity_hash"]=plan["identity_hash"]
        _json(root/"summary.json",summary)
        index["identity_hash"]=plan["identity_hash"]
        index["summary_sha256"]=archive.sha(root/"summary.json")
        for row in index["chunks"]:
            row["identity_hash"]=plan["identity_hash"]
            _json(Path(row["path"]).with_suffix(".json"),row)
        _json(root/"index.json",index)
    with pytest.raises((ValueError,AssertionError)):
        archive.verify_ready(roots["mean_gradient"],root)


def test_existing_or_overlapping_archive_destination_is_rejected(tmp_path,monkeypatch):
    roots,_ = _fake_completed(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match="new archive"):
        archive.archive(roots["mean_gradient"],roots["residual_transport"],roots["mean_gradient"]/"archive")
    with pytest.raises(ValueError,match="new archive"):
        archive.archive(roots["mean_gradient"],roots["residual_transport"],tmp_path)


def test_exact_snapshot_guard_rejects_manifest_rehashed_source_changes(tmp_path,monkeypatch):
    _,_,destination,_ = _make_archive(tmp_path,monkeypatch)
    path=destination/"archive_manifest.json"
    manifest=json.loads(path.read_text())
    name=manifest["source_snapshot"]["src/nn_dk/native.py"]["archived_path"]
    payload=destination/name
    payload.write_bytes(payload.read_bytes()+b"\n# changed source\n")
    row=manifest["copied_artifacts"][name]
    row["sha256"]=row["original_sha256"]=archive.sha(payload)
    row["original_bytes"]=payload.stat().st_size
    manifest["source_snapshot"]["src/nn_dk/native.py"]["sha256"]=hashlib.sha256(payload.read_bytes()).hexdigest()
    _json(path,manifest)
    with pytest.raises(ValueError,match="frozen source hash"):
        archive.verify_archive(destination)
