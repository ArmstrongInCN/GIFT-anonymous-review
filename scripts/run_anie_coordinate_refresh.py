"""Retrain only the coordinate-corrected ANIE, then refresh the fixed W1 record.

The NS networks, initialization, data, loss and optimizer recipe are unchanged.
Six other W1 measurements and all W2 measurements remain immutable. Scientific
costs close before independent CPU auditing and document engineering begins.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_w1_single import (
    ROOT, atomic_json, call_module, execute_job, file_hash, fixed_json,
    now, queue_lease, read_json, status, wait_surviving_child,
)

from experiments.provenance import external_source
from scripts.queue_runtime import child_environment, configured_source

BASE = external_source('prediction_native')
PREVIOUS = external_source('prediction_extension')
SOURCE_PROFILE = "aligned_ns_single_v1"


def prepare(root, *, anie_source=None):
    """Freeze a mechanical copy of the previous job before any initialization."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    existing = read_json(root / "RUN_PLAN.json")
    if existing is not None:
        if (existing.get("schema") != "gift.anie-coordinate-refresh.plan.v1"
                or existing.get("source_profile") != SOURCE_PROFILE):
            raise ValueError("existing queue's scientific plan differs")
        if anie_source is not None:
            os.environ["GIFT_ANIE_SOURCE_ROOT"] = configured_source(anie_source, "GIFT_ANIE_SOURCE_ROOT")
        return existing
    old = read_json(BASE / "PROTOCOL_LOCK.json")
    job = copy.deepcopy(old["training_configurations"]["anie_ns"])
    job["output"] = str(root / "training/anie_ns_seed7")
    job["model_spec"]["source_profile"] = SOURCE_PROFILE
    # Relocate only this new job; the historical source lock stays read-only.
    selected_source = configured_source(anie_source, "GIFT_ANIE_SOURCE_ROOT")
    if selected_source is not None:
        job["model_spec"]["source_root"] = selected_source
    plan = dict(schema="gift.anie-coordinate-refresh.plan.v1", profile=old["profile"],
                source_profile=SOURCE_PROFILE, root=str(root), base_root=str(BASE),
                old_root=str(PREVIOUS), project_root=str(ROOT),
                old_native_job=old["training_configurations"]["anie_ns"], training_job=job,
                methods_all=["GIFT", "fno2d", "fno3d", "anie_ns", "uno", "unet", "GIFT-N50"],
                newly_trained_methods=["anie_ns"], training_runs_per_method=1,
                developer_runs=0, parameter_search=False, training_seed=7,
                evaluation_batch=4, device="cuda", job_events_file=str(root / "JOB_EVENTS.jsonl"),
                publication_package="W1_flow_prediction_20261007",
                cohorts=["confirmation", "old_test"], confirmation_is_new_blind_test=False,
                W2_ANIE_included=False, scientific_failures_preserved=True,
                final_visual_review_required=True)
    fixed_json(root / "RUN_PLAN.json", plan)
    lock = copy.deepcopy(old)
    lock["training_configurations"]["anie_ns"] = job
    lock["protocol_file"] = str(root / "PROTOCOL_LOCKED.md")
    lock["job_events_file"] = str(root / "JOB_EVENTS.jsonl")
    lock["historical_events_file"] = str(BASE / "JOB_EVENTS.jsonl")
    fixed_json(root / "PROTOCOL_LOCK.json", lock)
    fixed_json(root / "DATA_RECEIPT.json", dict(window=job["window"],
               data_binding=old["data_binding"], read_only_reuse=True, weights_reused_for_ANIE=False))
    (root / "configs").mkdir(exist_ok=True)
    fixed_json(root / "configs/anie_ns_seed7.json", job)
    protocol = root / "PROTOCOL_LOCKED.md"
    if not protocol.exists():
        with protocol.open("x", encoding="utf-8") as f:
            f.write("""# ANIE training protocol for aligned spatiotemporal coordinates

ANIE uses the original NS network at fixed official commit 2071ad709b260a6c305450affcf7168fd825f692.
States, spatial coordinates and temporal coordinates share space-major/time-minor order. The declared
evaluation-dropout correction is retained; the original training functional-dropout probability remains 0.5.

The training pool contains 300 fixed trajectories; validation uses 20 trajectories, IDs1000–1019. There are 76 points over t5–8 at 0.04-second intervals,
with only the true t5 field as input. The original encoder, 8192→64→256→4096 global decoder, integral iterations and conditional branch are retained.
Reinitialize with the same seed7, without existing model weights. Use original-unit MSE over all 76 points, Adam lr1e-4,
coupled weight decay1e-4, effective batch32/micro6, retaining the final 12 samples. The limit is 5000 epochs,
with the original validation MSE-best selection and patience1000 early stopping; CosineT101 with eta_min1e-9 advances starting after epoch17.
No parameter selection, multiple seeds or test-based tuning is used; no representation pretraining, local decoder, normalization or initial-state anchoring is added.

The other six W1 methods reuse their completed original weights, predictions and metrics. Each evaluation cohort contains 180 trajectories,
with confirmation IDs100000–100179 and exploratory old-test IDs1040–1219. Existing results for both cohorts were exposed, so this is not a new blind test.
The primary metric is future75 mean framewise relative L2; all failures are retained, with t8/MSE/costs reported separately.
W2 excludes ANIE under the existing protocol, retaining its data, results and figures. New results use the current W1 plotting structure and methods.
""")
    forecast = root / "COST_FORECAST.json"
    if not forecast.exists():
        history = read_json(BASE / "training/anie_ns_seed7/HISTORY.json")
        seconds = [row["epoch_total_seconds"] for row in history["training"][-60:]]
        fixed_json(forecast, dict(schema="gift.anie-coordinate-refresh.cost.v1",
                   source=str(BASE / "training/anie_ns_seed7/HISTORY.json"),
                   historical_seconds_per_epoch=sum(seconds)/len(seconds),
                   same_shape_different_coordinate_profile=True, max_epochs=5000,
                   postprocessing_reserve_hours=[.5, 2.],
                   final_low_frequency_review_reserve_hours=[0., 4.],
                   early_stopping_is_not_promised=True))
    return plan


def cpu_step(root, command, label):
    """Time CPU engineering outside the already closed scientific ledger."""
    environment = child_environment()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    started = time.perf_counter()
    started_utc = now()
    status(root, "cpu_postprocessing", active_job=label, command=command, child_pid=None)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / (label + ".log")).open("a", encoding="utf-8") as log:
        code = subprocess.call(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    event = dict(label=label, command=command, exit_code=code, started_utc=started_utc,
                 finished_utc=now(), process_seconds=time.perf_counter()-started)
    with (root / "CPU_EXECUTION_EVENTS.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, allow_nan=False) + "\n")
        f.flush()
    if code:
        raise RuntimeError(f"CPU step {label} exited {code}; see its preserved log")


def run(root, plan):
    """Resume only this run's complete state; never restart a completed old task."""
    with queue_lease(root):
        wait_surviving_child(root)
        if (root / "COMPLETION_RECEIPT.json").exists():
            return
        status(root, "running", active_job="prepare_training", child_pid=None)
        slot = execute_job(root, plan["training_job"])
        fixed_json(root / "FORMAL.json", slot)
        from experiments.additional.w1_anie_coordinate_refresh import complete_evaluation
        for cohort in plan["cohorts"]:
            if complete_evaluation(root / "w1_evaluation" / cohort):
                continue
            args = ["--root", root, "--cohort", cohort, "--resume"]
            if call_module(root, "experiments.additional.w1_anie_coordinate_refresh", args, "evaluation_" + cohort):
                raise RuntimeError("W1 coordinate-refresh evaluation failed: " + cohort)
        ledger = root / "SCIENTIFIC_LEDGER_CLOSED.json"
        if ledger.exists():
            if read_json(ledger)["sha256"] != file_hash(root / "JOB_EVENTS.jsonl"):
                raise ValueError("closed scientific cost ledger changed")
        else:
            atomic_json(ledger, dict(status="complete", file="JOB_EVENTS.jsonl",
                        sha256=file_hash(root / "JOB_EVENTS.jsonl"), closed_utc=now(),
                        scope="ANIE retraining and two new W1 inference/evaluation attempts"))
        audit = root / "audit/audit_coordinate_refresh.py"
        if not all((root / name).is_file() for name in ("FINAL_TRAINING_AUDIT.json", "FINAL_RESULT_AUDIT.json")):
            cpu_step(root, [sys.executable, "-u", str(audit), "--root", str(root), "--mode", "final"], "final_scientific_audit")
        if not (root / "FINAL_DOCUMENT_AUDIT.json").is_file():
            cpu_step(root, [sys.executable, "-u", "-m", "scripts.publish_anie_coordinate_refresh",
                           "--root", str(root)], "publish_and_update_documents")
        status(root, "published_visual_qa_pending", active_job=None, child_pid=None,
               reason="Results and documents published; actual image/reference review remains")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--anie-source", type=Path, help="ANIE clone for the new job; overrides GIFT_ANIE_SOURCE_ROOT")
    args = parser.parse_args()
    if args.anie_source is not None:
        os.environ["GIFT_ANIE_SOURCE_ROOT"] = configured_source(args.anie_source, "GIFT_ANIE_SOURCE_ROOT")
    root = args.root.resolve()
    plan = prepare(root, anie_source=args.anie_source)
    if not args.run:
        print(json.dumps(dict(status="prepared", root=str(root), training_job=plan["training_job"]), ensure_ascii=False))
        return
    try:
        run(root, plan)
    except Exception as error:
        prior = read_json(root / "ORCHESTRATOR.json", {})
        status(root, "needs_attention", reason=repr(error), active_job=prior.get("active_job"),
               child_pid=prior.get("child_pid"), command=prior.get("command"))
        raise


if __name__ == "__main__":
    main()
