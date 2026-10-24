"""Publish the single ANIE coordinate-aligned W1 model and activate its records.

Only completed, independently audited measurements may be published. The other
six W1 methods remain byte-identical source measurements; W2 is not changed.
Figures call the existing manuscript renderer, including its fixed trajectory,
PCHIP guide, field plate, signed residuals, and shared colour limits.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import numpy as np

from experiments.formal._shared.figure_evidence import file_record, sha256, verify_svg
from scripts import plot_w1_original_style as plot
from scripts.publish_w1_w2_single import csv_text, portable, summary_rows
from scripts.publish_original_style_figures import copy_once, numeric_source, write_once
from scripts.verify_published_results import verify_package
from scripts.run_w1_single import atomic_json

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT.parent / "runs/w1w2_native_single76_n300_v6_20261005"
PACKAGE_NAME = "W1_flow_prediction_20261007"
OLD_PACKAGE_NAME = "W1_native_single76_n300_original_style_20261005"
COHORTS = ("confirmation", "old_test")
METHODS = {"GIFT", "GIFT-N50", "fno2d", "fno3d", "uno", "unet", "anie_ns"}
DOCS = (
    "README.md", "EXPERIMENTS.md", "docs/RESULTS.md", "docs/FIGURES.md",
    "docs/CHECKPOINTS.md", "docs/TRAINING_PROTOCOL.md", "docs/PERFORMANCE.md",
    "docs/SETUP.md", "docs/VALIDATION.md", "docs/EXPERIMENT_EXECUTION.md",
    "docs/EXTERNAL_ADAPTATIONS.md",
    "docs/W1_W2_PROTOCOL.md",
)
from experiments.provenance import external_source

OLD_CHECKPOINT = str(external_source('reference_anie_checkpoint'))
OLD_CHECKPOINT_SHA = "7fdd7e521db2f5a70139ea550b46038d322b672319c17aebb920d1b95c0191ae"
PENDING_VISUAL_STATEMENT = "Figure appearance requires separate visual inspection recorded in VISUAL_QA.json."
COMPLETED_VISUAL_STATEMENT = "Four PNGs were visually inspected: two curves with seven methods and seven measured nodes, and two 8-row, 74-axis field plates. Fixed IDs100005/1045, t5/6/7/8 and shared color scales are retained; labels are clear and finite values are not clipped. VISUAL_QA.json separately binds the appearance-inspection record."


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def text_once(path, content):
    """Create a scientific/document artifact, accepting only an identical retry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"existing artifact differs: {path}")
    else:
        path.write_text(content, encoding="utf-8")


def completed_inputs(root, base):
    """Bind completed training, full180 cohorts and unchanged six-method sources."""
    status_path = root / "training/anie_ns_seed7/STATUS.json"
    status = read(status_path)
    if status.get("method") != "anie_ns" or status.get("seed") != 7:
        raise ValueError("ANIE publication requires the unique seed7 training slot")
    failed = status.get("status") != "complete"
    if failed:
        failure = read(status_path.parent / "FAILURE.json")
        if (failure.get("failure_kind") not in {"numerical", "resource"}
                or failure.get("method") != "anie_ns" or failure.get("seed") != 7):
            raise ValueError("only a documented scientific failure may replace a complete training outcome")
        checkpoint, checkpoint_hash = None, None
    else:
        failure = None
        if not (0 < status["best_epoch"] <= status["epoch"] <= 5000) or status["updates"] != 10 * status["epoch"]:
            raise ValueError("ANIE fixed-budget progress is inconsistent")
        checkpoint = Path(status["selected_checkpoint"]).resolve(strict=True)
        if checkpoint.parent != status_path.parent.resolve() or checkpoint.name != "best.pt":
            raise ValueError("selected checkpoint is not this run's validation-MSE-best")
        checkpoint_hash = sha256(checkpoint)
    audits = {}
    for filename in ("FINAL_RESULT_AUDIT.json", "FINAL_TRAINING_AUDIT.json"):
        audit = read(root / filename)
        if str(audit.get("status", "")).lower() not in {"passed", "pass", "complete"}:
            raise ValueError(f"publication requires a passed audit: {filename}")
        audits[filename] = sha256(root / filename)
    reports = {}
    for cohort in COHORTS:
        path = root / "w1_evaluation" / cohort / "numeric_report.json"
        current, original = read(path), read(base / "w1_evaluation" / cohort / "numeric_report.json")
        if (current.get("status") != "numerical_complete" or current.get("population") != 180
                or current.get("cohort") != cohort or set(current["methods"]) != METHODS):
            raise ValueError("W1 publication requires exactly seven complete population slots")
        for field in ("trajectory_ids", "absolute_times", "query_times"):
            if current[field] != original[field]:
                raise ValueError("cohort or observed nodes changed")
        for key in METHODS - {"anie_ns"}:
            for field in ("status", "seed", "checkpoint_sha256", "raw_prediction_sha256", "summary"):
                if current["methods"][key][field] != original["methods"][key][field]:
                    raise ValueError(f"the reused {key} measurement changed: {field}")
        anie = current["methods"]["anie_ns"]
        if failed:
            if (anie.get("seed") != 7 or anie.get("status") != "training_failed"
                    or anie["summary"]["failed_count"] != 180 or anie["summary"].get("metrics")
                    or anie["summary"].get("primary_metric") is not None):
                raise ValueError("the scientific failure must retain the full180 slot without finite metrics")
        elif (anie["seed"] != 7 or Path(anie["checkpoint"]).resolve() != checkpoint
              or anie["checkpoint_sha256"].upper() != checkpoint_hash):
            raise ValueError("new ANIE report belongs to a different trained model")
        reports[cohort] = current
    events_path = root / "JOB_EVENTS.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [(event["label"], event["started_utc"]) for event in events]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate process attempts in cost journal")
    training_events = [event for event in events if event["label"].lower().startswith("train") and "anie" in event["label"].lower()]
    if not training_events or not all("process_seconds" in event and "exit_code" in event for event in training_events):
        raise ValueError("ANIE training process cost is not closed")
    reused_cost_path = base / "FINAL_COST_CLOSURE.json"
    reused_cost = read(reused_cost_path) if reused_cost_path.is_file() else None
    unchanged_training_seconds = (sum(value for key,value in reused_cost["reused_model_original_training_process_seconds"].items() if key != "anie_ns")
        + sum(record["process_seconds"] for key,record in reused_cost["grouped_process_wall"].items() if key.startswith("train_"))
        if reused_cost is not None else None)
    return dict(status=status, status_path=status_path, checkpoint=checkpoint,
                checkpoint_hash=checkpoint_hash, reports=reports, audits=audits,
                training_process_seconds=sum(event["process_seconds"] for event in training_events),
                events_hash=sha256(events_path), training_attempts=training_events, failed=failed, failure=failure,
                unchanged_training_seconds=unchanged_training_seconds)


def render_figures(root, base, inputs, figures):
    """Use the same renderer; a scientifically failed model gets explicit tiles."""
    if not inputs["failed"]:
        plot.main(["--input", str(root / "w1_evaluation"), "--output", str(figures)])
        return
    sources = plot.renderer_sources()
    measured = {}
    for cohort, trajectory_id in (("confirmation", 100005), ("old_test", 1045)):
        data = plot.read_cohort(base / "w1_evaluation" / cohort, cohort, trajectory_id)
        report_path = root / "w1_evaluation" / cohort / "numeric_report.json"
        old_anie_source = next(record["actual_read_path"] for record in data["raw_provenance"] if record["raw_method_key"] == "anie_ns")
        data.update(report_path=report_path, report_sha256=sha256(report_path), report=inputs["reports"][cohort])
        data["predictions"]["anie"] = np.full_like(data["truth"], np.nan)
        data["residuals"]["anie"] = np.full_like(data["truth"], np.nan)
        data["errors"]["anie"] = np.full(4, np.nan)
        data["sources"] = [record for record in data["sources"] if record["path"] != old_anie_source]
        data["sources"].append(file_record(root / "training/anie_ns_seed7/FAILURE.json", root))
        data["raw_provenance"] = [record for record in data["raw_provenance"] if record["raw_method_key"] != "anie_ns"]
        data["raw_provenance"].append(dict(label="ANIE", raw_method_key="anie_ns", panel_slug="anie",
            scientific_status="training_failed", reason=inputs["failure"]["reason"],
            initial_and_future_fields="NaN failure placeholders; no prediction was fabricated"))
        for row in data["curve_rows"]:
            if row["raw_method_key"] == "anie_ns":
                row.update(mean_relative_l2=None, complete_population=False, finite_count_min=0)
        measured[cohort] = data
    shared_y = max(1.28, 1.08*max(row["mean_relative_l2"] for data in measured.values()
                                for row in data["curve_rows"] if row["mean_relative_l2"] is not None))
    for cohort in COHORTS:
        plot.plot_cohort(figures / cohort, cohort, measured[cohort], sources, shared_y)


def publish(root, base):
    """Export a new scoped package through the existing original-style renderers."""
    inputs = completed_inputs(root, base)
    reports = inputs["reports"]
    figures = root / "original_style_figures/W1"
    if not all((figures / cohort / "figure_manifest.json").is_file() for cohort in COHORTS):
        if any((figures / cohort).exists() for cohort in COHORTS):
            raise FileExistsError("partial plot export is preserved; inspect before another export")
        render_figures(root, base, inputs, figures)
    package = PROJECT / "results/formal" / PACKAGE_NAME
    protocol = root / "PROTOCOL_LOCK.json"
    measurement = dict(schema="gift.w1w2-native.published-measurement.v1", experiment="W1",
        profile="single_native76_n300", status="numerical_complete", single_model_values=True,
        new_blind_test=False, anie_in_w2=False, cohorts=portable(reports),
        anie_coordinates="space_major_time_minor; state and (x,y,t) share each token index",
        source_numeric_report_sha256={cohort: sha256(root / "w1_evaluation" / cohort / "numeric_report.json") for cohort in COHORTS},
        protocol_lock_sha256=sha256(protocol), result_audit_sha256=inputs["audits"]["FINAL_RESULT_AUDIT.json"])
    write_once(package / "summary/summary.json", measurement)
    rows = summary_rows(reports, "W1")
    text_once(package / "summary/metrics.csv", csv_text(rows, list(rows[0])))
    report_hash = sha256(package / "summary/summary.json")
    figure_directories = []
    for cohort in COHORTS:
        source = figures / cohort
        source_manifest = read(source / "figure_manifest.json")
        report_path = root / "w1_evaluation" / cohort / "numeric_report.json"
        if (source_manifest.get("status") != "complete"
                or source_manifest["evidence"]["report_sha256"].upper() != sha256(report_path)):
            raise ValueError("figure is not bound to the current numerical report")
        for record in source_manifest["files"]:
            item = source / record["path"]
            if item.stat().st_size != record["bytes"] or sha256(item) != record["sha256"].upper():
                raise ValueError(f"source figure bytes changed: {item}")
        destination = package / f"figures/{cohort}"
        destination.mkdir(parents=True, exist_ok=True)
        previews, metadata = [], {}
        for item in sorted(source.rglob("*")):
            if not item.is_file() or item.name == "figure_manifest.json":
                continue
            if item.suffix == ".png":
                asset = PROJECT / "assets/figures" / f"w1_aligned_anie_single76_n300_{cohort}_{item.stem}_20261006.png"
                copy_once(item, asset)
                previews.append(file_record(asset, PROJECT))
            elif item.suffix == ".npz":
                metadata[item.name] = numeric_source(item, destination / item.relative_to(source))
            elif item.suffix in {".svg", ".csv", ".json"}:
                copy_once(item, destination / item.relative_to(source))
            else:
                raise ValueError(f"unexpected source plot file: {item}")
        files = sorted(path for path in destination.rglob("*") if path.is_file() and path.name != "figure_manifest.json")
        svg_checks = {path.relative_to(destination).as_posix(): verify_svg(path, editable_text="panels" not in path.parts)
                      for path in files if path.suffix == ".svg"}
        evidence = dict(source_manifest["evidence"], report_sha256=report_hash,
            source_numeric_report_sha256=sha256(report_path),
            source_figure_manifest_sha256=sha256(source / "figure_manifest.json"),
            sources=[file_record(package / "summary/metrics.csv", package)])
        payload = dict(source_manifest, evidence=evidence, source_array_metadata=metadata,
            figure_contract=dict(backend="Python matplotlib", archetype="original manuscript comparison templates",
                editable_vector_curves=True, editable_vector_fields=True, field_previews=previews,
                residual_definition="prediction minus truth", single_training_instance=True, ANIE_in_W2=False),
            files=[file_record(path, destination) for path in files], svg_checks=svg_checks)
        write_once(destination / "figure_manifest.json", payload)
        figure_directories.append(f"figures/{cohort}")
    inventory = sorted(path for path in package.rglob("*") if path.is_file() and path.name != "published.json")
    write_once(package / "published.json", dict(schema="gift.published-results.v1", experiment="W1",
        scope="summaries_and_selected_figures", full_raw_outputs_included=False, resumable_experiment=False,
        protocol_version="single_native76_n300_aligned_anie_20261006", single_model_values=True,
        measurement=dict(report_sha256=report_hash, source_measurement=measurement),
        figure_directories=figure_directories, files=[file_record(path, package) for path in inventory]))
    integrity = verify_package(package, "W1")
    receipt = dict(status="complete", package=str(package), package_manifest_sha256=sha256(package / "published.json"),
                   source_audits=inputs["audits"], reused_methods_unchanged=sorted(METHODS-{"anie_ns"}),
                   ANIE_in_W2=False, integrity=integrity)
    write_once(root / "PUBLICATION.json", receipt)
    render_result_record(root, package, inputs)
    return inputs, package


def metric(record, name):
    value = record["summary"].get("metrics", {}).get(name)
    return value["mean"] if isinstance(value, dict) else value


def render_result_record(root, package, inputs):
    """Write the complete measured W1 table with native budgets and figure links."""
    status = inputs["status"]
    selection = ("Training encountered a declared scientific failure, leaving no evaluable model. All 180 failed trajectories are retained in the original slot." if inputs["failed"] else
                 f"ANIE completed {status['epoch']} epochs and {status['updates']} updates, using the validation-MSE-best checkpoint from epoch{status['best_epoch']}.")
    lines = ["# W1: N300, 76 points, a single initial state and fixed recipes", "",
        "Measured single-model values are compared for GIFT, GIFT-N50, FNO2D, FNO3D, U-NO, U-Net and ANIE; each method has one declared seed, without cross-seed standard deviations.",
        "Shared observations cover t5–8 every 0.04 seconds for 76 points. Each trajectory receives only its own true t5 state; the primary metric is future75 mean framewise relative L2, excluding the known initial state.",
        "Both held-out cohorts have been inspected. This evaluation is a non-blind post hoc measurement, not used for parameter or model selection; confirmation and exploratory cohorts are reported separately.", "",
        "ANIE states and (x,y,t) coordinates share space-major/time-minor token order during training and evaluation. The NS architecture, c=0/y_init conditional path, physical MSE over all 76 points and original optimizer/scheduler/early stopping are retained.",
        selection, f"ANIE training process wall time: {inputs['training_process_seconds']:.3f} seconds ({inputs['training_process_seconds']/3600:.4f} hours), including all registered attempts, preparation, training, internal validation and checkpoint I/O.", "",
        "GIFT uses supervision targets of 0.08/0.48 seconds (nominal targets 0.1/0.5 seconds), internal RK4 dt=0.02 and saving every 0.04 seconds. U-NO is supervised for 20 steps through t5.80, and U-Net for 4 through t5.16. GIFT-N50 uses the first 50 trajectories in the ordered N300 list, retaining stage epochs but reducing total updates. Fixed losses, supervision coverage and computation budgets differ, so results do not represent an equal-budget architecture-only comparison.",
        "ANIE participates only in W1; the six W2 methods retain their declared weights and results.", ""]
    labels = {spec['group']: spec['label'] for spec in plot.SPECS}
    for cohort in COHORTS:
        report = inputs["reports"][cohort]
        lines += [f"## {cohort}", "", "| Method | Ntrain | seed | future75 mean relative L2 | t8 relative L2 | future MSE | Failed/180 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for spec in plot.SPECS:
            key = spec["group"]
            record = report["methods"][key]
            values = [metric(record, field) for field in ("future_mean_relative_l2", "t8_relative_l2", "future_mse")]
            formatted = ["no aggregate value" if value is None else f"{value:.9g}" for value in values]
            lines.append(f"| {labels[key]} | {50 if key == 'GIFT-N50' else 300} | {record['seed']} | {' | '.join(formatted)} | {record['summary']['failed_count']}/180 |")
        figure_manifest = read(package / "figures" / cohort / "figure_manifest.json")
        lines += ["", f"[Complete numeric report]({(root/'w1_evaluation'/cohort/'numeric_report.json').as_posix()})", ""]
        for preview in figure_manifest["figure_contract"]["field_previews"]:
            lines += [f"![{cohort} {Path(preview['path']).stem}]({(PROJECT/preview['path']).as_posix()})", ""]
    lines += ["## Evidence and figures", "", f"[Publication package]({(package/'published.json').as_posix()}). Curves and field plates retain the original manuscript plotting structure, methods, fixed IDs100005/1045 and t5/6/7/8; residuals are prediction−truth, without clipping finite values.",
        "FINAL_TRAINING_AUDIT.json and FINAL_RESULT_AUDIT.json bind actual training slots, each 180-trajectory cohort, raw predictions and recomputed norms; PUBLICATION.json verifies the publication package. Figure appearance requires separate visual inspection recorded in VISUAL_QA.json.",
        "This experiment transfers a fixed recipe from the public NS implementation to this task; it is not a complete reproduction of all data and settings in the original paper."]
    content = "\n".join(lines) + "\n"
    record_path = root / "RESULTS.md"
    if record_path.exists() and record_path.read_text(encoding="utf-8") == content.replace(PENDING_VISUAL_STATEMENT, COMPLETED_VISUAL_STATEMENT):
        return
    text_once(record_path, content)


def measured_curve_table(report):
    """Render all seven real saved report nodes without re-evaluating a model."""
    method_keys = ("GIFT", "GIFT-N50", "fno2d", "uno", "unet", "anie_ns", "fno3d")
    indices = plot.nearest_times(np.asarray(report["absolute_times"]), plot.REQUESTED_TIMES)
    lines = ["| Time | GIFT | GIFT-Lite | FNO-2D | U-NO | U-Net | ANIE | FNO-3D |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for index in indices:
        cells = []
        for key in method_keys:
            summary = report["methods"][key]["summary"]
            curve = summary.get("mean_relative_l2_by_time")
            value = curve[int(index)] if curve is not None and summary["failed_count"] == 0 else None
            cells.append("no aggregate value" if value is None else f"{value:.6f}")
        lines.append(f"| {report['absolute_times'][int(index)]:g} | {' | '.join(cells)} |")
    return "\n".join(lines)


def refresh_curve_tables(text, reports):
    """Update only the existing cohort time tables, retaining the record structure."""
    pattern = r"(?m)^\| Time \| GIFT \| GIFT-Lite \| FNO-2D \| U-NO \| U-Net \| ANIE \| FNO-3D \|\n(?:^\|[^\n]*\|\n?)+"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return text
    if len(matches) > 2:
        raise ValueError("the current W1 record has duplicate cohort time tables")
    for match, cohort in reversed(list(zip(matches, COHORTS))):
        text = text[:match.start()] + measured_curve_table(reports[cohort]) + "\n" + text[match.end():]
    return text


def reconcile_record_wording(relative, text, root=None):
    """Keep final records precise about coordinates, exposure and source scope."""
    text = text.replace("but the results of those two evaluation cohorts have been inspected", "but the results of both evaluation cohorts have been inspected")
    text = text.replace("The confirmation set is independent of training and validation, but results for four methods were inspected previously, so it is not a blind test for all methods.",
                        "Both evaluation cohorts are independent of training and validation, and both sets of results have been inspected; this evaluation is therefore a non-blind post hoc measurement.")
    text = text.replace("results for the four existing methods were known before the extension, so this is not a new blind confirmation", "results of both evaluation cohorts have been inspected, so this evaluation is a non-blind post hoc measurement")
    text = text.replace("native token coordinates and 76 query points", "spatiotemporal coordinates aligned token by token with states and 76 query points")
    if relative == "docs/RESULTS.md":
        text = text.replace("# Interpretation of current single-run fixed-recipe W1/W2 results (2026-10-05)", "# Interpretation of current single-run fixed-recipe W1/W2 results")
        text = re.sub(r"(?m)^- v6: (?=`w1_evaluation)", f"- Numerical sources for non-ANIE methods and W2 (external directory `{BASE.as_posix()}`): ", text)
        text = re.sub(r"(?m)^- v6: (?=`training)", f"- Training sources for U-NO/U-Net (external directory `{BASE.as_posix()}`): ", text)
        text = re.sub(r"(?m)^- v6: ", "- Sources in the declared run directories: ", text)
        if root is not None and "- ANIE numerical and training sources" not in text:
            marker = "- Project: "
            source = f"- ANIE numerical and training sources (external directory `{root.as_posix()}`): `w1_evaluation/confirmation/numeric_report.json`, `w1_evaluation/old_test/numeric_report.json`, `training/anie_ns_seed7/STATUS.json`, `HISTORY.json`, `best.pt` and `JOB_EVENTS.jsonl`.\n"
            text = text.replace(marker, source + marker)
        text = text.replace("This run did not select validation-best: ", "This fixed recipe selects the terminal state: ")
        text = text.replace("Previous CPU tests verified", "CPU tests verified")
        text = text.replace("This record did not rerun inference or add training to change these results.", "The verification above is limited to the sources and interfaces of these declared models.")
        text = text.replace("Retained under the current standard and changed to N300 with a single initial state", "This experiment uses this configuration with N300 and a single initial state")
    if relative == "docs/EXTERNAL_ADAPTATIONS.md":
        text = text.replace("## Current single-initial-state W1/W2 profile (2026-10-05)", "## Current single-initial-state W1/W2 profile")
        text = text.replace("[v5 implementation record]", "[Single-run fixed-recipe implementation]")
        text = re.sub(r"The three new models have completed their fixed training;[^\n]*full scope is in \[current costs\]\(PERFORMANCE\.md\)\.",
            "Training and evaluation sources for each method are bound to its result package; independent training and numerical audits are in the declared run directories. Actual computation costs and scope are in [PERFORMANCE.md](PERFORMANCE.md).", text)
    if relative == "docs/CHECKPOINTS.md" and root is not None:
        text = text.replace("Model identities, recipes, seeds and data bindings are stored in", "Model identities, recipes, seeds and data bindings for the other six methods are stored in") if "Model identities, recipes, seeds and data bindings for the other six methods" not in text else text
        marker = "The following external experiment files are not included in the Git result package."
        if "ANIE training configuration and model identity are stored in" not in text:
            text = text.replace(marker, f"ANIE training configuration and model identity are stored in `{(root/'PROTOCOL_LOCK.json').as_posix()}` and `{(root/'training/anie_ns_seed7/ATTEMPT.json').as_posix()}`. " + marker)
    if relative == "docs/TRAINING_PROTOCOL.md" and root is not None:
        text = text.replace("Configuration sources are", "Configuration sources for the other methods are") if "Configuration sources for the other methods are" not in text else text
        if "ANIE coordinate and training configuration sources are" not in text:
            marker = "## Independent N1000"
            sentence = f"ANIE coordinate and training configuration sources are `{(root/'PROTOCOL_LOCK.json').as_posix()}`; source adaptations are detailed in [ANIE implementation notes](ANIE_ADAPTATION.md).\n\n"
            if marker in text:
                text = text.replace(marker, sentence + marker)
            else:
                first_history = text.find("\n## ", text.find("\n## ")+1)
                if first_history < 0:
                    text += "\n" + sentence
                else:
                    text = text[:first_history] + "\n" + sentence + text[first_history:]
    return text


def document_contents(root, inputs):
    """Compute only current-protocol record changes from actual completed results."""
    epoch = inputs["status"].get("epoch", inputs["status"].get("completed_epochs", 0))
    best = inputs["status"].get("best_epoch")
    seconds = inputs["training_process_seconds"]
    path = inputs["checkpoint"].as_posix() if inputs["checkpoint"] is not None else "no evaluable checkpoint"
    digest = inputs["checkpoint_hash"].lower() if inputs["checkpoint_hash"] is not None else "none"
    records = {key: inputs["reports"][key]["methods"]["anie_ns"] for key in COHORTS}
    changed = {}
    for relative in DOCS:
        before = (PROJECT / relative).read_text(encoding="utf-8")
        after = before.replace(OLD_PACKAGE_NAME, PACKAGE_NAME)
        for old, new in (
            ("Results for four methods in this cohort were inspected previously", "Results of both evaluation cohorts have been inspected"),
            ("Confirmation results for four methods were inspected previously", "Results of both cohorts have been inspected"),
            ("Results for four methods were inspected previously", "Results of both evaluation cohorts have been inspected"),
            ("Results for some methods in the confirmation set were inspected previously", "Results of both evaluation cohorts have been inspected"),
            ("Results for some methods in the confirmation set were inspected previously", "Results of both evaluation cohorts have been inspected"),
            ("some method results on it had already been examined", "results on both evaluation populations had already been examined"),
        ):
            after = after.replace(old, new)
        after = re.sub(r"w1_single76_n300_(confirmation|old_test)_(\w+)_20261005\.png",
                       r"w1_aligned_anie_single76_n300_\1_\2_20261006.png", after)
        after = after.replace(OLD_CHECKPOINT, path).replace(OLD_CHECKPOINT_SHA, digest)
        for old, new in (("actual1028", f"actual{epoch}"), ("epoch1028", f"epoch{epoch}"),
                         ("through epoch1028", f"through epoch{epoch}"), ("at epoch1028", f"at epoch{epoch}"), ("training for 1028 epochs", f"training for {epoch} epochs"),
                         ("10280 updates", f"{10*epoch} updates"), ("epoch28", f"epoch{best}"),
                         ("epoch28", f"epoch{best}")):
            after = after.replace(old, new)
        if relative == "EXPERIMENTS.md":
            after = re.sub(r"^\| ANIE \| 20810\.388 \| 5\.7807 \|$",
                           f"| ANIE | {seconds:.3f} | {seconds/3600:.4f} |", after, flags=re.M)
            matches = list(re.finditer(r"^\| 64 \| ANIE \| 300 \| 7 \|.*$", after, re.M))
            if len(matches) != 2:
                raise ValueError("current experiment record must contain exactly two ANIE W1 result rows")
            for match, cohort in reversed(list(zip(matches, COHORTS))):
                record = records[cohort]
                values = [metric(record, key) for key in ("future_mean_relative_l2", "t8_relative_l2", "future_mse")]
                number = lambda value: "no aggregate value" if value is None else f"{value:.6f}"
                row = f"| 64 | ANIE | 300 | 7 | {' | '.join(map(number, values))} | {record['summary']['failed_count']}/180 |"
                after = after[:match.start()] + row + after[match.end():]
            sentence = "ANIE arranges states and spatiotemporal coordinates in space-major/time-minor order, with field features and (x,y,t) aligned within each token; training and evaluation use the same order."
            if sentence not in after:
                after = after.replace("ANIE uses MSE over all 76 points.", "ANIE uses MSE over all 76 points. " + sentence)
            after = refresh_curve_tables(after, inputs["reports"])
            manifests = [root / "original_style_figures/W1" / cohort / "figure_manifest.json" for cohort in COHORTS]
            if all(path.is_file() for path in manifests):
                maximum = max(read(path)["details"]["color_scales"]["selected_residual_maximum_absolute_value"] for path in manifests)
                after = after.replace("maximum residual 21.354568", f"maximum residual {maximum:.6f}")
        if relative == "docs/PERFORMANCE.md":
            after = re.sub(r"^\| ANIE, training for \d+ epochs, MSE-best epoch\d+ \| 20810\.388 \| 346\.84 \|$",
                           f"| ANIE, training for {epoch} epochs, MSE-best epoch{best} | {seconds:.3f} | {seconds/60:.2f} |", after, flags=re.M)
            reused_seconds = inputs.get("unchanged_training_seconds")
            total = (reused_seconds if reused_seconds is not None else 53591.249 - 20810.388) + seconds
            after = after.replace("53591.249 seconds (14.8865 hours)", f"{total:.3f} seconds ({total/3600:.4f} hours)")
            after = after.replace("The full scope is recorded in `FINAL_COST_CLOSURE.json` in the latter directory; ",
                f"ANIE training cost comes from `{(root/'JOB_EVENTS.jsonl').as_posix()}` (SHA-256 `{inputs['events_hash'].lower()}`). Cost scope for other methods is recorded in `FINAL_COST_CLOSURE.json` in the original directory; ")
        if relative == "docs/EXTERNAL_ADAPTATIONS.md":
            after = re.sub(r"^\| ANIE \|.*$", "| ANIE | Uses the official commit2071ad7 NS architecture and `aligned_ns_single_v1`; training c=0/y_init path; states and (x,y,t) share space-major/time-minor token order; functional dropout follows training mode and is disabled during evaluation. Shared 76 queries and micro6/effective32 adapt to the task/memory budget. W1 only. |", after, flags=re.M)
            after = after.replace("Four models reuse v5 directly; new U-NO/U-Net models are constructed from fixed public repository commits, without editing external on-disk source code.",
                "Each method is constructed from fixed public source code, with external on-disk sources kept read-only; ANIE coordinate order and individual modifications are detailed in [ANIE implementation notes](ANIE_ADAPTATION.md).")
        if relative == "docs/W1_W2_PROTOCOL.md":
            after = after.replace("Training configurations for GIFT, FNO2D, FNO3D and ANIE are in", "Training configurations for GIFT, FNO2D and FNO3D are in")
            if (root/'PROTOCOL_LOCK.json').as_posix() not in after:
                after = after.replace("Actual directory names locate sources; ", f"ANIE training and coordinate configurations are in `{(root/'PROTOCOL_LOCK.json').as_posix()}`. Actual directory names locate sources; ")
            if "States and spatiotemporal coordinates share space-major/time-minor order" not in after:
                after = after.replace("ANIE uses physical MSE, without changing it to relative L2. ", "ANIE uses physical MSE, without changing it to relative L2. States and spatiotemporal coordinates share space-major/time-minor order, with each token corresponding to the same spatial position and query time during training and evaluation.")
        if relative == "docs/RESULTS.md":
            anchor = "W1/W2 weights, complete raw HDF files and optimizer/random states remain in declared external experiment directories"
            if "ANIE training and numerical audits for both W1 populations are bound to" not in after:
                after = after.replace(anchor, f"ANIE training and numerical audits for both W1 populations are bound to `{root.as_posix()}`; sources for other models and W2 are in their result inventories.\n\n" + anchor)
        if relative == "docs/VALIDATION.md":
            if "The complete prediction recomputation entry point for the six existing models and W2" not in after:
                after = after.replace("The complete prediction recomputation entry point is `scripts.audit_w1_w2_single`", "The complete prediction recomputation entry point for the six existing models and W2 is `scripts.audit_w1_w2_single`")
            anchor = "Result evidence appears in the [W1 provenance manifest]"
            if "ANIE's tokenwise space-time correspondence and training configuration are declared in" not in after:
                after = after.replace(anchor, f"ANIE's tokenwise space-time correspondence and training configuration are declared in [ANIE implementation notes](ANIE_ADAPTATION.md); complete training/numerical audits come from `{root.as_posix()}` files `FINAL_TRAINING_AUDIT.json` and `FINAL_RESULT_AUDIT.json`.\n\n" + anchor)
        if relative == "docs/EXPERIMENT_EXECUTION.md":
            anchor = "The independent evaluation interface is `experiments.additional.w1_extended_evaluate`"
            if "ANIE_ADAPTATION.md" not in after:
                after = after.replace(anchor, f"ANIE configuration, model selection and W1 values are bound to `{root.as_posix()}`; its spatiotemporal token order is detailed in [ANIE implementation notes](ANIE_ADAPTATION.md). Other methods and W2 use their declared directories.\n\n" + anchor)
        if relative == "docs/SETUP.md":
            if "ANIE uses" not in after or root.as_posix() not in after:
                after = after.replace("Model files and SHAs are in", f"ANIE uses `{root.as_posix()}` declared coordinate configuration and training weights. Model files and SHAs are in")
        after = reconcile_record_wording(relative, after, root)
        if after != before:
            if not inputs["failed"] and inputs["status"].get("terminal_reason") != "native_early_stopping":
                after = after.replace(f"actually stopped early at epoch{epoch}", f"actually completed the {epoch}-epoch limit")
                after = after.replace(f"training stopped early at epoch{epoch}", f"training completed the {epoch}-epoch limit")
                after = after.replace(f"training terminated at epoch{epoch} under the original early-stopping rule", f"training completed the {epoch}-epoch limit")
                after = after.replace(f"terminated at epoch{epoch} under the patience1000 rule", f"terminated at the {epoch}-epoch limit")
                after = after.replace(f"trained through epoch{epoch} under the original early-stopping rule", f"trained through epoch{epoch} under the predefined limit")
            changed[relative] = (before, after)
    if inputs["failed"]:
        failure_description = "ANIE training encountered a declared scientific numerical or resource failure; the unique seed7 slot is retained, with 180/180 failures recorded for both W1 cohorts and no aggregate accuracy value, without replacing seeds or parameters."
        for relative, (before, after) in list(changed.items()):
            after = after.replace(f"epoch{best}", "no evaluable checkpoint").replace(f"epoch{best}", "no evaluable checkpoint")
            after = after.replace("all seven methods retain 180/180 finite trajectories", "the other six methods retain 180/180 finite trajectories; ANIE records 180/180 failures")
            after = after.replace("All seven methods retain 180/180 finite trajectories.", "Six methods retain 180/180 finite trajectories; ANIE retains its failed 180-trajectory slot.")
            after = after.replace("All methods have 0/180 non-finite predictions", "The other six methods have 0/180 non-finite predictions; ANIE records 180/180 failures")
            after = after.replace("All seven W1 methods and six W2 methods have 0/180 failures on their respective tasks in both populations.",
                                  "In W1, ANIE records 180/180 failures and the other six methods have 0/180 failures; all six W2 methods have 0/180 failures.")
            if relative == "EXPERIMENTS.md":
                after = re.sub(r"^\| ANIE \| 300 \| 7 \|.*$",
                    "| ANIE | 300 | 7 | limit5000; effective32/micro6 | NS Adam lr1e-4/wd1e-4, delayed CosineT101; scientific failure, no evaluable model | Original-unit MSE over all 76 points, including initial-state reconstruction; retain the unique failure slot |", after, flags=re.M)
                after = re.sub(r"ANIE uses no evaluable checkpoint validation-MSE-best weights, with training stopped early at epoch\d+\.", failure_description, after)
            if relative == "docs/RESULTS.md":
                after = re.sub(r"ANIE uses its best validation-MSE model from no evaluable checkpoint; training stopped at epoch\d+ under the original early-stopping rule\. ANIE participates only in W1\.",
                               failure_description + "ANIE participates only in W1.", after)
            if relative == "docs/CHECKPOINTS.md":
                after = re.sub(r"^\| ANIE \|.*$", "| ANIE | no evaluable checkpoint | none | seed7 scientific failure; W1 only |", after, flags=re.M)
                after = re.sub(r"ANIE selects no evaluable checkpoint under the original NS validation-MSE-best rule, training through epoch\d+ under the original early-stopping rule\.", failure_description, after)
            if relative == "docs/TRAINING_PROTOCOL.md":
                after = re.sub(r"^\| ANIE \|.*$",
                    "| ANIE | seed7, effective batch32/micro6, Adam lr1e-4/wd1e-4, delayed CosineT101, limit5000 epochs; scientific failure, no evaluable model | Official NS training c=0/y_init path, original-unit MSE over all 76 points, including initial-state reconstruction; W1 only. |", after, flags=re.M)
            if relative == "docs/PERFORMANCE.md":
                after = re.sub(r"^\| ANIE, training for \d+ epochs, MSE-best no evaluable checkpoint \|.*$",
                               f"| ANIE, scientific failure | {seconds:.3f} | {seconds/60:.2f} |", after, flags=re.M)
            if relative == "docs/W1_W2_PROTOCOL.md":
                after = re.sub(r"ANIE has a limit of 5000 epochs and 10 updates per epoch; actually stopped early at epoch\d+, with \d+ updates, using no evaluable checkpoint validation-MSE-best weights\.",
                    f"ANIE has a limit of 5000 epochs, completed {epoch} epochs before scientific failure and recorded {inputs['status'].get('updates', 0)} updates, with no evaluable checkpoint; the unique seed7 slot is retained.", after)
            if relative in {"EXPERIMENTS.md", "docs/RESULTS.md", "docs/TRAINING_PROTOCOL.md", "docs/CHECKPOINTS.md",
                            "docs/W1_W2_PROTOCOL.md"}:
                after += "\n" + failure_description + "\n"
            changed[relative] = (before, after)
    adaptation = f"""# ANIE: alignment of spatiotemporal coordinates and state tokens

ANIE uses the authors' public repository [emazap7/ANIE](https://github.com/emazap7/ANIE), pinned to commit `2071ad709b260a6c305450affcf7168fd825f692`, with source profile `aligned_ns_single_v1` and condition `native`. Users download the official source and adapt it outside this repository following [external adaptation notes](EXTERNAL_ADAPTATIONS.md); the GIFT repository retains only textual instructions, source references and generic external-module loading interfaces.

The state tensor is flattened in space-major/time-minor order. Spatial index s and time index j map to token `k=s*T+j`; state features, spatial coordinates and time coordinates use the same token index. Every internal solver update retains this order, which is inverted before decoding. Training, validation and W1 inference use the same configuration.

| External location or project interface | Adjustment and rationale |
| --- | --- |
| User's external ANIE runtime interface | Declares `aligned_ns_single_v1`, uses the native conditional path and retains training control from the official NS implementation. This interface file is kept outside the GIFT repository. |
| External adaptation record and [official solver](https://github.com/emazap7/ANIE/blob/2071ad709b260a6c305450affcf7168fd825f692/IE_source/Attentional_IE_solver.py) | Align spatial and temporal coordinates with state order according to the textual instructions; retain the original architecture, attention, integral iterations and output decoder. Full source patches are not distributed in the GIFT repository. |
| [Official attention implementation](https://github.com/emazap7/ANIE/blob/2071ad709b260a6c305450affcf7168fd825f692/IE_source/Galerkin_transformer.py) | Disable functional attention dropout during evaluation while retaining its original training probability; users apply the change in their own external code. |
| `training/w1_single_train.py` | Allow the new aligned profile through configuration validation and construct the original network using that profile; retain the existing controller, physical MSE over all 76 points, optimizer/scheduler, same-run validation-MSE-best selection and original early-stopping rule, without adding parameter selection or seeds. |

The registered functional attention dropout adjustment retains the original probability during training and disables dropout during evaluation, following the module training state. No normalization, spatial decoder modification, hard initial-state anchoring, auxiliary reconstruction or recursion is added.

Training uses the shared N300 list and 20 validation trajectories, with 76 points over t5–8 every 0.04 seconds. Each trajectory receives only its true t5 field; NS training and evaluation use the c=0/y_init conditional path. seed7, effective batch32, micro6, Adam lr1e-4/wd1e-4, delayed CosineT101, eta_min1e-9, limit5000 epochs and patience1000. {"Training encountered a scientific failure, with no evaluable checkpoint; all failed trajectories are retained. " if inputs['failed'] else f"Completed {epoch} epochs and {10*epoch} updates, selecting epoch{best}; "}Training process wall time was {seconds:.3f} seconds.

Checkpoint: `{path}`, SHA-256 `{digest}`. Complete training and numerical sources: `{root.as_posix()}`. Both 180-trajectory W1 cohorts are reported separately; the primary metric excludes the known initial state and uses future75 only. ANIE does not participate in W2. This experiment describes a fixed recipe on this task, rather than fully reproducing all data and settings of the original paper.
"""
    adaptation_path = PROJECT / "docs/ANIE_ADAPTATION.md"
    previous_adaptation = adaptation_path.read_text(encoding="utf-8") if adaptation_path.exists() else None
    if previous_adaptation != adaptation:
        changed["docs/ANIE_ADAPTATION.md"] = (previous_adaptation, adaptation)
    return changed


def prepare_document_plan(root):
    """Declare edit scope while training is pending without inventing any result."""
    path = root / "DOCUMENT_DELIVERY_PLAN.json"
    write_once(path, dict(status="prepared", current_documents=list(DOCS),
        implementation_record="docs/ANIE_ADAPTATION.md",
        package=PACKAGE_NAME, base_package=OLD_PACKAGE_NAME,
        prerequisite="completed seed7 STATUS and passed training/result audits; actual measurements only",
        plotting="scripts.plot_w1_original_style; unchanged structure, fixed IDs and signed residuals",
        untouched=["all historical packages", "six reused methods", "W2", "W3", "S1-S3"]))
    return dict(status="prepared", path=str(path))


def update_documents(root, inputs, package):
    """Snapshot current texts and activate only a verified completed publication."""
    verify_package(package, "W1")
    receipt_path = root / "DOCUMENT_ACTIVATION.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        for record in receipt["files"]:
            if sha256(PROJECT / record["path"]) != record["after_sha256"]:
                raise ValueError("activated current document was modified after receipt")
        return receipt["files"]
    changes = document_contents(root, inputs)
    snapshot = root / "document_snapshots_before_activation"
    records = []
    for relative, (before, after) in changes.items():
        path = PROJECT / relative
        if before is not None:
            text_once(snapshot / relative, before)
        if path.exists() and path.read_text(encoding="utf-8") == after:
            continue
        if (before is None and path.exists()) or (before is not None and path.read_text(encoding="utf-8") != before):
            raise ValueError(f"document changed since preparation: {path}")
        path.write_text(after, encoding="utf-8")
        records.append(dict(path=relative, before_sha256=sha256(snapshot / relative) if before is not None else None,
                            after_sha256=sha256(path)))
    write_once(root / "DOCUMENT_ACTIVATION.json", dict(status="complete", files=records,
        package_manifest_sha256=sha256(package / "published.json"),
        formal_record_style="final methods and measured results; no development chronology",
        figures="unchanged original manuscript renderer", W2_changed=False))
    return records


def build_document_audit(root, package):
    """Resolve current W1 references and verify their completed publication bytes.

    This checks evidence and Markdown pointers. It does not claim to inspect PNG
    appearance; actual visual inspection must be recorded separately by a reader.
    """
    activation = read(root / "DOCUMENT_ACTIVATION.json")
    integrity = verify_package(package, "W1")
    documents, references = [], []
    for record in activation["files"]:
        path = PROJECT / record["path"]
        if sha256(path) != record["after_sha256"]:
            raise ValueError(f"current document bytes differ after activation: {path}")
        documents.append(dict(path=record["path"], sha256=record["after_sha256"]))
    for relative in DOCS:
        path = PROJECT / relative
        text = path.read_text(encoding="utf-8")
        if OLD_PACKAGE_NAME in text:
            raise ValueError(f"current W1 record still points to the superseded package: {relative}")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if not any(key in target for key in (PACKAGE_NAME, "w1_aligned_anie_single76_n300_", "ANIE_COORDINATE_ADAPTATION_20261006.md")):
                continue
            target_path = (path.parent / target.split("#", 1)[0]).resolve(strict=True)
            references.append(dict(document=relative, target=target, resolved_path=str(target_path),
                                   sha256=sha256(target_path) if target_path.is_file() else None))
    for cohort in COHORTS:
        manifest = read(package / "figures" / cohort / "figure_manifest.json")
        for record in manifest["figure_contract"]["field_previews"]:
            asset = PROJECT / record["path"]
            if asset.stat().st_size != record["bytes"] or sha256(asset) != record["sha256"].upper():
                raise ValueError("linked field preview differs from figure source manifest")
    experiment_record = (PROJECT / "EXPERIMENTS.md").read_text(encoding="utf-8")
    cohort_tables = {}
    table_count = experiment_record.count("| Time | GIFT | GIFT-Lite | FNO-2D | U-NO | U-Net | ANIE | FNO-3D |")
    if table_count not in (1, 2):
        raise ValueError("the current W1 record must retain one or two declared cohort time tables")
    for cohort in COHORTS:
        numeric = read(root / "w1_evaluation" / cohort / "numeric_report.json")
        expected = measured_curve_table(numeric)
        expected_in_record = cohort == "confirmation" or table_count == 2
        if expected_in_record and expected not in experiment_record:
            raise ValueError(f"{cohort} seven-node experiment table differs from its actual measurements")
        csv_path = package / "figures" / cohort / "source_data.csv"
        with csv_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 49:
            raise ValueError("each published cohort needs all seven nodes for all seven methods")
        seen = set()
        indices = plot.nearest_times(np.asarray(numeric["absolute_times"]), plot.REQUESTED_TIMES)
        expected_slots = {(key, int(index)) for key in METHODS for index in indices}
        for row in rows:
            key, index = row["raw_method_key"], int(row["observation_index"])
            identity = key, index
            if identity in seen or identity not in expected_slots:
                raise ValueError("duplicate or unknown cohort curve measurement")
            seen.add(identity)
            summary = numeric["methods"][key]["summary"]
            means = summary.get("mean_relative_l2_by_time")
            value = means[index] if means is not None and summary["failed_count"] == 0 else None
            actual = float(row["mean_relative_l2"]) if row["mean_relative_l2"] else None
            if (row["cohort"] != cohort or float(row["absolute_time"]) != numeric["absolute_times"][index]
                    or (value is None) != (actual is None)
                    or (value is not None and abs(value-actual) > 2e-12)):
                raise ValueError("published cohort curve source CSV differs from its actual measurements")
        if seen != expected_slots:
            raise ValueError("published cohort curve source CSV omits an actual method/node slot")
        cohort_tables[cohort] = dict(status="passed", report_sha256=sha256(root / "w1_evaluation" / cohort / "numeric_report.json"),
            actual_node_count=7, method_count=7, published_curve_source_sha256=sha256(csv_path),
            source_csv_rows_verified=49, table_in_current_record=expected_in_record,
            precision="six decimal places for the record; full precision for curve source CSV")
    for relative in DOCS:
        content = (PROJECT / relative).read_text(encoding="utf-8")
        if "native token coordinates" in content or re.search(r"(?m)^- v[56]:", content):
            raise ValueError(f"current record retains a superseded coordinate statement or development label: {relative}")
    result = dict(schema="gift.anie-coordinate-refresh.document-audit.v1", status="passed",
        documents=documents, latest_result_references=references, publication_integrity=integrity,
        package_manifest_sha256=sha256(package / "published.json"),
        scientific_content_checks=dict(cohort_time_tables=cohort_tables,
            coordinate_statement="same state and (x,y,t) token order", development_chronology=False),
        original_manuscript_renderer=True, W2_measurements_and_package_unchanged=True,
        visual_inspection="pending; this audit verifies bytes and references only")
    return result


def audit_documents(root, package):
    """Commit the document audit after verifying both content and file pointers."""
    result = build_document_audit(root, package)
    write_once(root / "FINAL_DOCUMENT_AUDIT.json", result)
    return result


def repair_current_documents(root, base):
    """Reconcile document omissions while preserving all scientific measurements.

    Existing document receipts are copied byte-for-byte into a new evidence
    directory before their current pointers are replaced. Publication and the
    scientific process journal stay immutable. CPU engineering wall time is
    appended separately, including the preservation and independent checks.
    """
    started = datetime.now(timezone.utc)
    start_clock = time.perf_counter()
    command = ["python", "-m", "scripts.publish_anie_coordinate_refresh", "--root", str(root), "--repair-current-documents"]
    exit_code = 1
    try:
        inputs = completed_inputs(root, base)
        package = PROJECT / "results/formal" / PACKAGE_NAME
        verify_package(package, "W1")
        protected = {str(path): sha256(path) for path in (
            root / "PUBLICATION.json", root / "JOB_EVENTS.jsonl", root / "FINAL_TRAINING_AUDIT.json",
            root / "FINAL_RESULT_AUDIT.json", package / "published.json")}
        prior_activation = read(root / "DOCUMENT_ACTIVATION.json")
        for record in prior_activation["files"]:
            if sha256(PROJECT / record["path"]) != record["after_sha256"]:
                raise ValueError("a current document was modified outside its existing activation receipt")
        evidence = root / ("document_evidence_" + started.strftime("%Y%m%dT%H%M%S%fZ"))
        evidence.mkdir(exist_ok=False)
        for filename in ("DOCUMENT_ACTIVATION.json", "FINAL_DOCUMENT_AUDIT.json"):
            copy_once(root / filename, evidence / filename)
        changes = document_contents(root, inputs)
        changed = []
        for relative, (_, after) in changes.items():
            path = PROJECT / relative
            before = path.read_text(encoding="utf-8") if path.is_file() else None
            if before == after:
                continue
            if before is not None:
                copy_once(path, evidence / "documents" / relative)
            path.write_text(after, encoding="utf-8")
            changed.append(relative)
        root_report = root / "RESULTS.md"
        before_report = root_report.read_text(encoding="utf-8")
        after_report = before_report.replace(PENDING_VISUAL_STATEMENT, COMPLETED_VISUAL_STATEMENT)
        if before_report != after_report:
            copy_once(root_report, evidence / "RESULTS.md")
            root_report.write_text(after_report, encoding="utf-8")
        records = []
        for record in prior_activation["files"]:
            current = dict(record, after_sha256=sha256(PROJECT / record["path"]))
            if record["path"] in changed:
                current["before_sha256"] = sha256(evidence / "documents" / record["path"])
            records.append(current)
        activation = dict(prior_activation, files=records,
            prior_activation_evidence=str(evidence / "DOCUMENT_ACTIVATION.json"),
            prior_activation_sha256=sha256(evidence / "DOCUMENT_ACTIVATION.json"),
            updated_document_scope="actual measured tables, coordinate semantics and neutral source descriptions")
        atomic_json(root / "DOCUMENT_ACTIVATION.json", activation)
        result = build_document_audit(root, package)
        atomic_json(root / "FINAL_DOCUMENT_AUDIT.json", result)
        for path, digest in protected.items():
            if sha256(Path(path)) != digest:
                raise ValueError("scientific measurement or publication bytes changed during documentation work")
        receipt = dict(schema="gift.anie-coordinate-refresh.document-reconciliation.v1", status="complete",
            changed_documents=changed, evidence_directory=str(evidence),
            preserved_prior_receipts={name: sha256(evidence / name) for name in ("DOCUMENT_ACTIVATION.json", "FINAL_DOCUMENT_AUDIT.json")},
            current_document_activation_sha256=sha256(root / "DOCUMENT_ACTIVATION.json"),
            current_document_audit_sha256=sha256(root / "FINAL_DOCUMENT_AUDIT.json"),
            root_result_record_sha256=sha256(root_report), scientific_and_publication_files_unchanged=protected,
            no_training_or_model_inference=True)
        write_once(evidence / "RECONCILIATION.json", receipt)
        exit_code = 0
        return receipt
    finally:
        finished = datetime.now(timezone.utc)
        event = dict(label="reconcile_document_records", command=command, exit_code=exit_code,
            started_utc=started.isoformat(), finished_utc=finished.isoformat(),
            process_seconds=time.perf_counter()-start_clock,
            scope="CPU-only document reconciliation, evidence preservation and reference/content audit; no scientific ledger mutation")
        with (root / "CPU_EXECUTION_EVENTS.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--prepare-doc-plan", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--update-documents", action="store_true")
    parser.add_argument("--repair-current-documents", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    if args.prepare_doc_plan:
        print(json.dumps(prepare_document_plan(root), ensure_ascii=False))
    if args.repair_current_documents:
        if args.publish or args.update_documents or args.prepare_doc_plan:
            parser.error("--repair-current-documents is a separate document-only operation")
        print(json.dumps(repair_current_documents(root, args.base.resolve(strict=True)), ensure_ascii=False, indent=2))
        return
    default_delivery = not (args.prepare_doc_plan or args.publish or args.update_documents)
    if args.publish or args.update_documents or default_delivery:
        inputs, package = publish(root, args.base.resolve(strict=True))
        result = dict(status="published", package=str(package))
        if args.update_documents or default_delivery:
            result["documents"] = update_documents(root, inputs, package)
            result["document_audit"] = audit_documents(root, package)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
