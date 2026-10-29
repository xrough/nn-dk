"""Lossless compact evidence for the two completed frozen-v3 diagnostics.

Native polygons, Brownian arrays, raw family JVPs and binary checkpoints remain
local with complete SHA-256 inventories. The archive snapshots actual source
bytes; its informational Git HEAD is not a claim that untracked code is in Git.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

import numpy as np  # noqa: E402

from analysis import audit_payoff_mean_gradients as gradient  # noqa: E402
from analysis import audit_payoff_residual_transport as transport  # noqa: E402
from analysis import report_payoff_mean_gradients as signed  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.types import stable_hash  # noqa: E402

KIND = "frozen-v3-mean-gradient-and-residual-transport-compact-evidence-v1"
COMPRESS_THRESHOLD = 100_000
SUMMARY_SCOPE = "Diagnostic-only intermediate evidence; no new fitted formulation or final independent accuracy/cost study."


def read(path):
    return json.loads(Path(path).read_text())


def sha(path, *, compressed=False):
    digest = hashlib.sha256()
    with (gzip.open(path,"rb") if compressed else Path(path).open("rb")) as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")


def _safe(base,name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not (base/path).resolve().is_relative_to(base.resolve()):
        raise ValueError("Archive path escapes its declared directory")
    return base/path


def _plan_digest(prepared,label):
    # The frozen gradient preparation binds its outer anchors; the frozen
    # transport preparation hashes its inner identity. Preserve both formats.
    return stable_hash({"identity":prepared["identity"],"anchors":prepared["anchors"]}
                       if label=="mean_gradient" else prepared["identity"])


def _allocation(root,prepared,index,summary,label):
    identity,plan = prepared["identity"],prepared["identity"]["plan"]
    if (not index.get("complete") or not summary.get("complete")
            or index.get("identity_hash")!=prepared["identity_hash"]
            or summary.get("identity_hash")!=prepared["identity_hash"]
            or _plan_digest(prepared,label)!=prepared["identity_hash"]
            or identity["source_hash"]!=source_hash()):
        raise ValueError("A complete matching diagnostic identity is required")
    cases = identity["cases"]
    if not cases or len(set(cases))!=len(cases) or set(cases)!=set(prepared["anchors"]):
        raise ValueError("Incomplete diagnostic case allocation")
    expected_records,expected_chunks,all_ids = set(),set(),set()
    if set(v["case"] for v in identity["models"])!=set(cases):
        raise ValueError("Unexpected or missing model cases")
    for case in cases:
        models = [v for v in identity["models"] if v["case"]==case]
        if len(models)!=3 or {v["seed"] for v in models}!={0,1,2}:
            raise ValueError("All three optimizer seeds must be retained")
        origin = prepared["anchors"][case]
        queries = np.asarray(origin["queries"],dtype=float)
        if (queries.ndim!=1 or len(queries)!=len(plan["query_fractions"])
                or not np.isfinite(queries).all() or (queries<=0).any() or (np.diff(queries)<=0).any()):
            raise ValueError("Incomplete or invalid physical query allocation")
        if sha(origin["path"])!=origin["sha256"]:
            raise ValueError("Anchor evidence changed")
        with np.load(origin["path"]) as store:
            states = store["states"]
        if states.ndim!=2 or len(states)!=origin["rows"] or not np.isfinite(states).all():
            raise ValueError("Incomplete validation anchor allocation")
        for anchor in range(origin["rows"]):
            expected_records.update((case,anchor,seed) for seed in (0,1,2))
            expected_chunks.update((case,anchor,start) for start in range(0,plan["paths"],plan["chunk_paths"]))
    actual = [(v["case"],v["anchor"],v["seed"]) for v in summary["records"]]
    chunks = [(v["case"],v["anchor"],v["start"]) for v in index["chunks"]]
    if len(actual)!=len(expected_records) or set(actual)!=expected_records:
        raise ValueError("Incomplete or duplicated model/anchor/seed summary allocation")
    if (len(chunks)!=len(expected_chunks) or set(chunks)!=expected_chunks
            or index["expected_chunks"]!=len(expected_chunks)):
        raise ValueError("Incomplete or duplicated continuation chunk allocation")
    expected_payloads = {Path(v["path"]).resolve() for v in index["chunks"]}
    if ({p.resolve() for p in (root/"chunks").rglob("*.npz")}!=expected_payloads
            or {p.resolve() for p in (root/"chunks").rglob("*.json")}
            !={p.with_suffix(".json") for p in expected_payloads}):
        raise ValueError("Raw chunk directory differs from the declared complete allocation")
    for row in summary["records"]:
        if row["families"]!=plan["paths"] or row["queries"]!=prepared["anchors"][row["case"]]["queries"]:
            raise ValueError("Summary query or family allocation differs")
        if label=="residual_transport":
            q = len(row["queries"])
            shapes = {"mean_discrepancy":(q,4),"raw_second_discrepancy":(q,4,4),
                "sample_covariance_discrepancy_unbiased":(q,4,4),"mean_residual_integral":(q,4),
                "raw_second_residual_integral":(q,4,4),"covariance_residual_integral_alone":(q,4,4),
                "mean_transport_defect":(q,4),"raw_second_transport_defect":(q,4,4)}
            if set(row.get("levels",{}))!={"coarse","fine"}:
                raise ValueError("Both physical Euler levels must remain in every model row")
            for level in row["levels"].values():
                if set(level)!={"declared","half_"}:
                    raise ValueError("Both residual integration strides must remain")
                for value in level.values():
                    if value.get("families")!=plan["paths"] or set(value.get("point",{}))!=set(shapes):
                        raise ValueError("Incomplete transported-moment allocation")
                    for key,shape in shapes.items():
                        point = np.asarray(value["point"][key])
                        interval = np.asarray(value["pointwise_percentile95"][key])
                        if (point.shape!=shape or interval.shape!=(2,*shape)
                                or not np.isfinite(point).all() or not np.isfinite(interval).all()
                                or (interval[0]>interval[1]).any()):
                            raise ValueError("Incomplete query/component/bootstrap transport allocation")
    for row in index["chunks"]:
        path = Path(row["path"])
        if (not path.resolve().is_relative_to(root) or not path.exists()
                or sha(path)!=row["sha256"] or row["families"]!=plan["chunk_paths"]):
            raise ValueError("A committed raw diagnostic chunk changed")
        if read(path.with_suffix(".json"))!=row:
            raise ValueError("Committed raw chunk sidecar and index differ")
        with np.load(path) as store:
            ids = store["family_ids"].astype(str)
        if len(ids)!=plan["chunk_paths"] or len(set(ids))!=len(ids) or all_ids.intersection(ids):
            raise ValueError("Continuation families are missing, duplicated or reused")
        all_ids.update(ids.tolist())
    return {"cases":cases,"anchors":sum(v["rows"] for v in prepared["anchors"].values()),
            "models":len(identity["models"]),"records":len(actual),"chunks":len(chunks),
            "families":len(all_ids),"family_ids_sha256":stable_hash(sorted(all_ids))},all_ids


def _report(directory,audit_root):
    """Validate the shared human-report manifest schema without rewriting it."""
    directory = Path(directory).resolve()
    manifest = read(directory/"output_manifest.json")
    if not manifest.get("complete") or not manifest.get("outputs"):
        raise ValueError("A complete human diagnostic report is required")
    identity = manifest["identity"]
    if Path(identity["audit_root"]).resolve()!=audit_root:
        raise ValueError("Human report belongs to a different diagnostic")
    for path,digest in identity["inputs"].items():
        if sha(path)!=digest:
            raise ValueError("A human report input changed")
    actual = {v.name for v in directory.iterdir() if v.is_file()}
    if any(v.is_symlink() for v in directory.iterdir()):
        raise ValueError("Human report symlinks are not valid evidence")
    if actual!=set(manifest["outputs"])|{"output_manifest.json"}:
        raise ValueError("Human report output inventory changed")
    for name,digest in manifest["outputs"].items():
        if Path(name).name!=name or sha(_safe(directory,name))!=digest:
            raise ValueError("A human report output changed")
    return manifest


def verify_ready(gradient_root,transport_root,gradient_report=None,transport_report=None):
    roots = {"mean_gradient":Path(gradient_root).resolve(),"residual_transport":Path(transport_root).resolve()}
    if roots["mean_gradient"]==roots["residual_transport"]:
        raise ValueError("Two distinct diagnostic roots are required")
    summaries = {"mean_gradient":gradient.load_complete(roots["mean_gradient"]),
                 "residual_transport":transport.load_complete(roots["residual_transport"])}
    report_dir = Path(gradient_report).resolve() if gradient_report else roots["mean_gradient"]/"report"
    result = signed.load_report(report_dir)
    if not result.get("complete") or result.get("identity",{}).get("audit_root")!=str(roots["mean_gradient"]):
        raise ValueError("A matching signed-component gradient report is required")
    prepared,indexes,allocation,ids = {},{},{},{}
    for label,root in roots.items():
        prepared[label],indexes[label] = read(root/"plan.json"),read(root/"index.json")
        allocation[label],ids[label] = _allocation(root,prepared[label],indexes[label],summaries[label],label)
    if ids["mean_gradient"].intersection(ids["residual_transport"]):
        raise ValueError("The two diagnostics reused continuation families")
    reports = {"mean_gradient":report_dir}
    report_manifests = {"mean_gradient":_report(report_dir,roots["mean_gradient"])}
    if transport_report is not None:
        reports["residual_transport"] = Path(transport_report).resolve()
    elif (roots["residual_transport"]/"report/output_manifest.json").exists():
        reports["residual_transport"] = roots["residual_transport"]/"report"
    if "residual_transport" in reports:
        report_manifests["residual_transport"] = _report(reports["residual_transport"],roots["residual_transport"])
    sources = {"analysis/archive_payoff_diagnostics.py":sha(__file__)}
    external = {}

    def source(name,digest):
        if name in sources and sources[name]!=digest:
            raise ValueError("Frozen diagnostic source snapshots disagree")
        path = _safe(REPO,name)
        if sha(path)!=digest:
            raise ValueError(f"Frozen source differs from requested snapshot: {name}")
        sources[name] = digest

    def outside(path,digest,role,copy=False):
        path = str(Path(path).resolve())
        if sha(path)!=digest:
            raise ValueError("A frozen external input changed")
        if path in external and external[path]["sha256"]!=digest:
            raise ValueError("External input identities disagree")
        external.setdefault(path,{"sha256":digest,"bytes":Path(path).stat().st_size,"roles":[],"copy":copy})
        external[path]["copy"] |= copy
        if role not in external[path]["roles"]:
            external[path]["roles"].append(role)

    for label,packet in prepared.items():
        identity = packet["identity"]
        script = "audit_payoff_mean_gradients.py" if label=="mean_gradient" else "audit_payoff_residual_transport.py"
        source("analysis/"+script,identity["script_sha256"])
        for name,digest in identity["helpers"].items():
            source("analysis/"+name,digest)
        study = Path(identity["study_root"])
        outside(study/"payoffs/fit_lock.json",identity["fit_lock_sha256"],"frozen-fit-lock",True)
        for name in ("plan.json","payoffs/preregistration.json","payoffs/data_lock.json"):
            p = study/name
            if p.exists():
                outside(p,sha(p),"historical-study-metadata",True)
        preregistration = study/"payoffs/preregistration.json"
        if preregistration.exists():
            frozen = read(preregistration)
            if frozen["source_hash"]!=identity["source_hash"]:
                raise ValueError("Historical model package source differs")
            for name,digest in frozen["helpers"].items():
                source("analysis/"+name,digest)
            source("analysis/run_payoff_boundary_regimes.py",frozen["runner_sha256"])
        for model in identity["models"]:
            outside(model["checkpoint_path"],model["checkpoint_sha256"],"binary-checkpoint-local-only")
            outside(model["metadata_path"],model["metadata_sha256"],"checkpoint-sidecar",True)
        for anchor in packet["anchors"].values():
            outside(anchor["state_bank_path"],anchor["state_bank_sha256"],"validation-bank-local-only")
    for label,manifest in report_manifests.items():
        identity = manifest["identity"]
        audit_name = "audit_payoff_mean_gradients.py" if label=="mean_gradient" else "audit_payoff_residual_transport.py"
        report_name = "report_payoff_mean_gradients.py" if label=="mean_gradient" else "report_payoff_residual_transport.py"
        source("analysis/"+audit_name,identity["audit_script_sha256"])
        source("analysis/"+report_name,identity["report_script_sha256"])
    for path in sorted((REPO/"src/nn_dk").rglob("*.py")):
        source(str(path.relative_to(REPO)),sha(path))
    for name in ("pyproject.toml","uv.lock","LICENSE","LICENSE-DOCS","LICENSING.md"):
        source(name,sha(REPO/name))
    return {"roots":roots,"summaries":summaries,"prepared":prepared,"indexes":indexes,
            "reports":reports,"report_manifests":report_manifests,"allocation":allocation,
            "sources":sources,"external":external,"source_hash":source_hash()}


def _copy(source,target):
    original = sha(source)
    compressed = source.suffix==".json" and source.stat().st_size>COMPRESS_THRESHOLD
    if compressed:
        target = target.with_name(target.name+".gz")
    target.parent.mkdir(parents=True,exist_ok=True)
    if compressed:
        with source.open("rb") as stream,target.open("wb") as sink:
            with gzip.GzipFile(fileobj=sink,mode="wb",filename="",mtime=0) as zipper:
                shutil.copyfileobj(stream,zipper)
    else:
        shutil.copyfile(source,target)
    if sha(target,compressed=compressed)!=original or sha(source)!=original:
        raise ValueError("Evidence changed while copying")
    return target,{"sha256":sha(target),"original_sha256":original,"compressed":compressed,
                   "original_bytes":source.stat().st_size,"archive_bytes":target.stat().st_size,
                   "original_path":str(source.resolve())}


def archive(gradient_root,transport_root,destination,gradient_report=None,transport_report=None):
    """Archive only completed diagnostics; never modify their plans or inputs."""
    ready = verify_ready(gradient_root,transport_root,gradient_report,transport_report)
    destination = Path(destination).resolve()
    if destination.exists() or any(destination.is_relative_to(root) or root.is_relative_to(destination)
                                   for root in ready["roots"].values()):
        raise ValueError("Use a new archive outside both authoritative diagnostic roots")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".payoff-diagnostic-archive-",dir=destination.parent) as temporary:
        staging = Path(temporary)/"archive"
        staging.mkdir()
        copied,inventory,source_files = {},{},{}

        def preserve(path,relative,role):
            target,packet = _copy(path,staging/relative)
            name = str(target.relative_to(staging))
            copied[name] = {**packet,"role":role}
            return name

        for label,root in ready["roots"].items():
            local = {}
            anchor_paths = {Path(v["path"]).resolve() for v in ready["prepared"][label]["anchors"].values()}
            numeric = (Path(ready["summaries"][label]["numeric_evidence"]["path"]).resolve()
                       if "numeric_evidence" in ready["summaries"][label] else None)
            report_dir = ready["reports"].get(label)
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise ValueError("Diagnostic evidence must not use symlinks")
                if not path.is_file() or path.name==".run.lock" or "__pycache__" in path.parts:
                    continue
                if path.name.endswith(".tmp"):
                    raise ValueError("Incomplete temporary diagnostic evidence remains")
                relative = str(path.relative_to(root))
                local[relative] = {"sha256":sha(path),"bytes":path.stat().st_size,"copied":False}
                selected = (path.suffix in (".json",".md",".csv") or path.resolve() in anchor_paths
                            or path.resolve()==numeric or (report_dir is not None and path.parent==report_dir))
                if selected:
                    stored = preserve(path,Path(label)/relative,"diagnostic-metadata-or-compact-evidence")
                    local[relative].update(copied=True,archived_path=stored)
            # Custom signed/report destinations can be outside their audit root.
            if report_dir is not None and not report_dir.is_relative_to(root):
                for path in sorted(report_dir.iterdir()):
                    if path.is_file():
                        preserve(path,Path(label)/"report"/path.name,"human-report")
            inventory[label] = {"root":str(root),"files":local,"bytes":sum(v["bytes"] for v in local.values())}
        external = {}
        for path,record in ready["external"].items():
            value = {k:v for k,v in record.items() if k!="copy"}
            if record["copy"]:
                key = hashlib.sha256(path.encode()).hexdigest()[:16]
                value["archived_path"] = preserve(Path(path),Path("historical_metadata")/key/Path(path).name,"historical-metadata")
            external[path] = value
        for name,digest in ready["sources"].items():
            stored = preserve(REPO/name,Path("code")/name,"exact-source-snapshot")
            if copied[stored]["original_sha256"]!=digest:
                raise ValueError("Frozen source changed while archiving")
            source_files[name] = {"archived_path":stored,"sha256":digest}
        readme = ("# Frozen-v3 diagnostic evidence\n\n"+SUMMARY_SCOPE+"\n\n"
            "The mean-gradient signed-component report is under `mean_gradient/report/`. "
            "Residual transport plans and complete summaries are under `residual_transport/`; "
            "a human report is included when supplied. Large JSON is losslessly gzipped. "
            "No covariance PDE residual alone is a prediction-error certificate.\n\n"
            "All declared optimizer seeds, validation-only anchors, fresh continuation families, "
            "paired Euler/derivative refinement, stopping and quadrature limitations remain recorded. "
            "Intervals are pointwise diagnostics, not simultaneous or continuum guarantees.\n\n"
            "Exact frozen source bytes and packaging metadata are under `code/`. The informational "
            "Git HEAD does not assert that uncommitted or untracked scripts existed in that commit. "
            "Full native polygons, Brownian arrays, raw family JVPs, compiled libraries, validation "
            "banks and binary model weights remain local with checksummed inventories. Original "
            "absolute metadata paths and original bytes have not been rewritten.\n\n"
            "This archive does not complete controlled boundary-layer sampling, learned-correction "
            "candidate testing, or a new final independent accuracy/cost study. Code is MIT; "
            "authored prose and generated diagnostic evidence are CC BY 4.0.\n")
        (staging/"README.md").write_text(readme)
        try:
            head = subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        except (OSError,subprocess.CalledProcessError):
            head = None
        manifest = {"kind":KIND,"complete":True,"diagnostics_only":True,"active_readme_goal_complete":False,
            "scope":SUMMARY_SCOPE,"created_utc":datetime.now(timezone.utc).isoformat(),
            "source_hash":ready["source_hash"],"source_snapshot":source_files,"informational_git_head":head,
            "source_snapshot_may_include_uncommitted_or_untracked_files":True,
            "diagnostics":{label:{"identity_hash":packet["identity_hash"],
                                  "whole_plan_sha256":sha(ready["roots"][label]/"plan.json"),
                                  "allocation":ready["allocation"][label]}
                           for label,packet in ready["prepared"].items()},
            "copied_artifacts":copied,"raw_inventory":inventory,"external_inputs":external,
            "generated_files":{"README.md":sha(staging/"README.md")},
            "raw_artifacts_retained_locally":True,"immutable_original_metadata_rewritten":False}
        _write(staging/"archive_manifest.json",manifest)
        verify_archive(staging)
        for label,root in ready["roots"].items():
            for name,row in inventory[label]["files"].items():
                if sha(root/name)!=row["sha256"]:
                    raise ValueError("Diagnostic changed while archiving")
        if source_hash()!=ready["source_hash"]:
            raise ValueError("Numerical package changed while archiving")
        staging.rename(destination)
    return verify_archive(destination)


def _original(path,packet):
    try:
        return gzip.decompress(path.read_bytes()) if packet["compressed"] else path.read_bytes()
    except (OSError,EOFError,zlib.error) as error:
        raise ValueError("Invalid losslessly compressed archive payload") from error


def verify_archive(destination):
    """Verify a compact snapshot offline; original local raw files are unneeded."""
    destination = Path(destination).resolve()
    manifest = read(destination/"archive_manifest.json")
    if (manifest.get("kind")!=KIND or not manifest.get("complete") or manifest.get("diagnostics_only") is not True
            or manifest.get("active_readme_goal_complete") is not False
            or set(manifest.get("diagnostics",{}))!={"mean_gradient","residual_transport"}):
        raise ValueError("A complete diagnostic-only archive is required")
    expected = set(manifest["copied_artifacts"])|set(manifest["generated_files"])|{"archive_manifest.json"}
    paths = list(destination.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise ValueError("Archive symlinks are not valid evidence")
    actual = {str(p.relative_to(destination)) for p in paths if p.is_file()}
    if actual!=expected:
        raise ValueError("Archive exact file inventory changed")
    originals = {}
    for name,packet in manifest["copied_artifacts"].items():
        path = _safe(destination,name)
        if sha(path)!=packet["sha256"]:
            raise ValueError("Archived evidence checksum changed")
        original = _original(path,packet)
        if hashlib.sha256(original).hexdigest()!=packet["original_sha256"] or len(original)!=packet["original_bytes"]:
            raise ValueError("Restored original evidence checksum changed")
        originals[name] = original
    for name,digest in manifest["generated_files"].items():
        if sha(_safe(destination,name))!=digest:
            raise ValueError("Archive documentation changed")
    original_paths = {}
    for name,packet in manifest["copied_artifacts"].items():
        key = str(Path(packet["original_path"]))
        if key in original_paths and original_paths[key]["original_sha256"]!=packet["original_sha256"]:
            raise ValueError("Copied original identities disagree")
        original_paths[key] = packet
        if Path(key).name=="output_manifest.json":
            report = json.loads(originals[name])
            if not report.get("complete"):
                raise ValueError("Archived human report is incomplete")
            for output,digest in report["outputs"].items():
                path = str(Path(key).parent/output)
                matching = [v for v in manifest["copied_artifacts"].values() if v["original_path"]==path]
                if len(matching)!=1 or matching[0]["original_sha256"]!=digest:
                    raise ValueError("Archived human-report outputs differ from its manifest")
    for label,inventory in manifest["raw_inventory"].items():
        for name,row in inventory["files"].items():
            if row["copied"]:
                packet = manifest["copied_artifacts"].get(row["archived_path"])
                if (packet is None or packet["original_sha256"]!=row["sha256"]
                        or packet["original_path"]!=str(Path(inventory["root"])/name)):
                    raise ValueError("Compact evidence and full raw inventory disagree")
    source = manifest["source_snapshot"]
    package = {}
    for name,record in source.items():
        raw = originals[record["archived_path"]]
        if hashlib.sha256(raw).hexdigest()!=record["sha256"]:
            raise ValueError("Exact source snapshot changed")
        if name.startswith("src/nn_dk/") and name.endswith(".py"):
            package[name.removeprefix("src/nn_dk/")] = raw.decode()
    if not package or stable_hash(package)!=manifest["source_hash"]:
        raise ValueError("Archived numerical package does not match frozen source hash")
    for label,record in manifest["diagnostics"].items():
        plan_name = next((name for name,packet in manifest["copied_artifacts"].items()
                          if name.startswith(label+"/") and Path(packet["original_path"]).name=="plan.json"
                          and Path(packet["original_path"]).parent==Path(manifest["raw_inventory"][label]["root"])),None)
        if plan_name is None:
            raise ValueError("Archived diagnostic preparation is missing")
        prepared = json.loads(originals[plan_name])
        identity = prepared["identity"]
        if (hashlib.sha256(originals[plan_name]).hexdigest()!=record["whole_plan_sha256"]
                or _plan_digest(prepared,label)!=record["identity_hash"]
                or prepared["identity_hash"]!=record["identity_hash"]
                or identity["source_hash"]!=manifest["source_hash"]):
            raise ValueError("Archived diagnostic identity changed")
        script = "audit_payoff_mean_gradients.py" if label=="mean_gradient" else "audit_payoff_residual_transport.py"
        required = {"analysis/"+script:identity["script_sha256"],**{"analysis/"+n:h for n,h in identity["helpers"].items()}}
        for name,digest in required.items():
            if name not in source or source[name]["sha256"]!=digest:
                raise ValueError("Frozen script/helper source snapshot is missing or differs")
        for item in identity["models"]:
            for key in ("checkpoint","metadata"):
                external = manifest["external_inputs"].get(str(Path(item[key+"_path"]).resolve()))
                if external is None or external["sha256"]!=item[key+"_sha256"]:
                    raise ValueError("Frozen model is missing from the external raw inventory")
        for item in prepared["anchors"].values():
            anchor = original_paths.get(str(Path(item["path"])))
            if anchor is None or anchor["original_sha256"]!=item["sha256"]:
                raise ValueError("Frozen validation anchor snapshot is missing or differs")
    return {"complete":True,"diagnostics_only":True,"archive":str(destination),
            "copied_artifacts_verified":len(manifest["copied_artifacts"]),
            "indexed_raw_files":sum(len(v["files"]) for v in manifest["raw_inventory"].values()),
            "external_inputs_indexed":len(manifest["external_inputs"]),
            "manifest_sha256":sha(destination/"archive_manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradient-root",type=Path)
    parser.add_argument("--transport-root",type=Path)
    parser.add_argument("--gradient-report",type=Path)
    parser.add_argument("--transport-report",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--verify-archive",type=Path)
    args = parser.parse_args()
    if args.verify_archive:
        result = verify_archive(args.verify_archive)
    else:
        if args.gradient_root is None or args.transport_root is None or args.output is None:
            parser.error("Both completed diagnostic roots and a new output directory are required")
        result = archive(args.gradient_root,args.transport_root,args.output,args.gradient_report,args.transport_report)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
