"""Publish new compact W1/W2 evidence after the full population audit passes.

The existing packages and weights are never replaced. Curves are editable
vector SVGs; field previews are PNGs in assets, with their bounded selected
numeric arrays and hashes bound to the new package. Full raw HDF/journals stay
in the external run directory and these packages cannot resume training.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
from adapters import public_metadata

from experiments.formal._shared.figure_evidence import file_record, sha256, verify_svg
from scripts.run_w1_single import fixed_json, read_json
from scripts.verify_published_results import verify_package

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"GIFT": "GIFT", "GIFT-N50": "GIFT-N50", "fno2d": "FNO2D", "fno3d": "FNO3D",
          "anie_ns": "ANIE", "uno": "U-NO", "unet": "U-Net"}


def fixed_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError("published historical text differs: " + str(path))
    else:
        path.write_text(content, encoding="utf-8")


def fixed_copy(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(source) != sha256(target):
            raise ValueError("published historical bytes differ: " + str(target))
    else:
        shutil.copyfile(source, target)


def csv_text(rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def method_entries(report, experiment):
    """Expose each real grid/model slot once; never pool models or cohorts."""
    if experiment == "W1":
        return [(64, label, record) for label, record in report["methods"].items()]
    return [(int(grid), label, record) for grid, methods in report["methods"].items() for label, record in methods.items()]


def metric_value(record, key):
    value = record.get("summary", {}).get("metrics", {}).get(key)
    return value.get("mean") if isinstance(value, dict) else value


def summary_rows(reports, experiment):
    rows = []
    endpoint = "t8_relative_l2" if experiment == "W1" else "t6_relative_l2"
    for cohort, report in reports.items():
        for grid, label, record in method_entries(report, experiment):
            rows.append(dict(experiment=experiment, cohort=cohort, grid=grid, method=label,
                seed=record.get("seed"), training_count=50 if label == "GIFT-N50" else 300,
                status=record["status"], population=180, failed_count=record["summary"]["failed_count"],
                future_mean_relative_l2=metric_value(record, "future_mean_relative_l2"),
                endpoint_relative_l2=metric_value(record, endpoint), future_mse=metric_value(record, "future_mse"),
                comparison_role="reduced_data_and_update_budget" if label == "GIFT-N50" else "common_N300",
                checkpoint_sha256=record.get("checkpoint_sha256", "")))
    return rows


def portable(value):
    """Keep measurements and textual provenance, omitting code and local pointers."""
    def paths_removed(item):
        if isinstance(item, dict):
            return {key: paths_removed(child) for key, child in item.items() if key not in {
                "path", "file", "source_file", "window", "output", "root", "source_root", "data_file", "job_events_file",
                "checkpoint", "generator_checkpoint", "raw_prediction_file", "metric_file", "reuse_evaluation", "reused_measurement_from"}}
        if isinstance(item, list):
            return [paths_removed(child) for child in item]
        return item
    return paths_removed(public_metadata(value))


def selected_fields(directory, report, experiment):
    """Read only the predeclared trajectory/time field panels from completed raw."""
    selected = 100000 if report["cohort"] == "confirmation" else 1040
    row = report["trajectory_ids"].index(selected)
    times = np.asarray(report["query_times"], float) + 5
    wanted = [5.48, 6., 7., 8.] if experiment == "W1" else [5.24, 5.48, 5.76, 6.]
    columns = np.asarray([np.argmin(np.abs(times-time)) for time in wanted])
    if not np.allclose(times[columns], wanted, rtol=0, atol=1e-12):
        raise ValueError("selected field times are not exact saved nodes")
    arrays = {"trajectory_id": np.asarray([selected], np.int64), "absolute_time": np.asarray(wanted)}
    grids = (64,) if experiment == "W1" else (64, 96, 128)
    for grid in grids:
        truth = directory / "raw" / ("truth.h5" if experiment == "W1" else f"N{grid}_truth.h5")
        with h5py.File(truth, "r") as handle:
            arrays[f"N{grid}_truth"] = handle["vorticity"][row, columns]
        entries = report["methods"] if experiment == "W1" else report["methods"][str(grid)]
        for label, record in entries.items():
            if "raw_prediction_file" in record:
                with h5py.File(directory / record["raw_prediction_file"], "r") as handle:
                    arrays[f"N{grid}_{label.replace('-', '_')}"] = handle["prediction"][row, columns]
    return arrays


def fixed_npz(path, arrays):
    """Verify existing selected arrays exactly, including recorded nonfinite fields."""
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[key], value, equal_nan=True) for key, value in arrays.items()):
                raise ValueError("published selected field evidence differs")
    else:
        np.savez_compressed(path, **arrays)


def source_figure(directory, stem, suffix):
    """Use the actually committed export, including a preserved retry directory."""
    marker = read_json(directory / "FIGURES_COMPLETE.json")
    if not marker or marker.get("numeric_report_sha256", "").upper() != sha256(directory / "numeric_report.json"):
        raise ValueError("figure export belongs to a different numeric report")
    records = marker.get("file_sha256", marker.get("figure_sha256", {}))
    matches = [(relative, digest) for relative, digest in records.items() if Path(relative).name == stem + "." + suffix]
    if len(matches) != 1:
        raise ValueError("figure completion has no unique requested export")
    relative, digest = matches[0]
    path = directory / relative
    if sha256(path) != digest.upper():
        raise ValueError("completed figure bytes differ from its real receipt")
    return path


def publish_package(root, experiment):
    stage = "w1_evaluation" if experiment == "W1" else "w2_evaluation"
    reports = {cohort: read_json(root / stage / cohort / "numeric_report.json") for cohort in ("confirmation", "old_test")}
    if any(report is None for report in reports.values()):
        raise ValueError("both real cohort reports are required")
    package = ROOT / "results/formal" / f"{experiment}_native_single76_n300_20261005"
    measurement = dict(schema="gift.w1w2-native.published-measurement.v1", experiment=experiment,
        profile="single_native76_n300", status="numerical_complete", single_model_values=True,
        new_blind_test=False, anie_in_w2=False, cohorts=portable(reports),
        source_numeric_report_sha256={cohort: sha256(root / stage / cohort / "numeric_report.json") for cohort in reports},
        protocol_lock_sha256=sha256(root / "EXTENSION_LOCK.json"),
        result_audit_sha256=sha256(root / "FINAL_RESULT_AUDIT.json"))
    if (package / "published.json").is_file():
        if read_json(package / "summary/summary.json") != measurement:
            raise ValueError("existing compact publication belongs to a different measurement")
        return dict(path=str(package), integrity=verify_package(package, experiment))
    (package / "summary").mkdir(parents=True, exist_ok=True)
    fixed_json(package / "summary/summary.json", measurement)
    rows = summary_rows(reports, experiment)
    fixed_text(package / "summary/metrics.csv", csv_text(rows, list(rows[0])))
    report_hash = sha256(package / "summary/summary.json")
    figure_dirs = []
    for cohort, report in reports.items():
        source = root / stage / cohort
        if not (source / "FIGURES_COMPLETE.json").is_file():
            raise ValueError("numeric report has no completed real figure exports")
        relative = "figures/" + cohort
        destination = package / relative
        destination.mkdir(parents=True, exist_ok=True)
        stems = ["future_error"] if experiment == "W1" else ["future25_error", "cross_resolution_error", "outer_P31_t6"]
        for stem in stems:
            fixed_copy(source_figure(source, stem, "svg"), destination / (stem + ".svg"))
        curve_rows = []
        for grid, label, record in method_entries(report, experiment):
            curve = record["summary"].get("mean_relative_l2_by_time")
            if curve is not None:
                for local, value in zip(report["query_times"], curve):
                    curve_rows.append(dict(grid=grid, method=label, absolute_time=5+local, mean_relative_l2=value))
        fixed_text(destination / "source_data.csv", csv_text(curve_rows, ["grid", "method", "absolute_time", "mean_relative_l2"]))
        fixed_npz(destination / "source_fields.npz", selected_fields(source, report, experiment))
        previews = []
        selected = 100000 if cohort == "confirmation" else 1040
        png_stems = ["fixed_id_keyframes"] if experiment == "W1" else [f"fixed_id_keyframes_N{grid}" for grid in (64, 96, 128)]
        for stem in png_stems:
            filename = f"{experiment.lower()}_native_single76_n300_{cohort}_id{selected}_{stem}_20261005.png"
            preview = ROOT / "assets/figures" / filename
            fixed_copy(source_figure(source, stem, "png"), preview)
            previews.append(file_record(preview, ROOT))
        files = sorted(path for path in destination.iterdir() if path.is_file() and path.name != "figure_manifest.json")
        fixed_json(destination / "figure_manifest.json", dict(status="complete",
            evidence=dict(experiment=experiment, report_sha256=report_hash,
                source_numeric_report_sha256=measurement["source_numeric_report_sha256"][cohort],
                sources=[file_record(package / "summary/metrics.csv", package)],
                cohort=cohort, population=180, fixed_id=selected, single_model=True,
                interpretation="fixed recipe comparison with unequal supervision horizons and native budgets; no seed SD"),
            figure_contract=dict(backend="Python matplotlib", archetype="quantitative grid and field plate",
                claim="measured autonomous forecast errors for the fixed native recipes on the shared population",
                editable_vector_curves=True, field_previews=previews,
                selected_arrays="predeclared same ID/time panels, true original raw values; no cosmetic field editing"),
            files=[file_record(path, destination) for path in files],
            svg_checks={path.name: verify_svg(path) for path in files if path.suffix == ".svg"}))
        figure_dirs.append(relative)
    inventory = sorted(path for path in package.rglob("*") if path.is_file() and path.name != "published.json")
    fixed_json(package / "published.json", dict(schema="gift.published-results.v1", experiment=experiment,
        scope="summaries_and_selected_figures", full_raw_outputs_included=False, resumable_experiment=False,
        protocol_version="single_native76_n300_20261005", single_model_values=True,
        measurement=dict(report_sha256=report_hash, source_measurement=measurement),
        figure_directories=figure_dirs, files=[file_record(path, package) for path in inventory]))
    return dict(path=str(package), integrity=verify_package(package, experiment))


def cost_closure(root):
    """Sum process attempts once; distinguish reused sunk training from new work."""
    lock = read_json(root / "EXTENSION_LOCK.json")
    base = Path(lock["base_lock"]).parent
    def events(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    current, prior = events(root / "JOB_EVENTS.jsonl"), events(base / "JOB_EVENTS.jsonl")
    unique = [(row["label"], row["started_utc"]) for row in current]
    if len(unique) != len(set(unique)):
        raise ValueError("duplicate process attempt records")
    groups = {}
    for row in current:
        group = groups.setdefault(row["label"], dict(attempts=0, process_seconds=0., exit_codes=[]))
        group["attempts"] += 1
        group["process_seconds"] += row["process_seconds"]
        group["exit_codes"].append(row["exit_code"])
    reused = {label: sum(row["process_seconds"] for row in prior if row["label"] == "train_" + Path(record["checkpoint"]).parent.name)
              for label, record in lock["reused_models"].items() if label != "GIFT"}
    reused["GIFT"] = sum(row["process_seconds"] for row in prior if row["label"] == "train_GIFT_seed20260820")
    result = dict(schema="gift.w1w2-native.cost-closure.v1", status="complete", attempts=current,
        grouped_process_wall=groups, new_training_process_seconds=sum(value["process_seconds"] for label, value in groups.items() if label.startswith("train_")),
        current_all_process_seconds=sum(row["process_seconds"] for row in current),
        reused_model_original_training_process_seconds=reused, reused_training_new_process_seconds=0.,
        nested_inference_times_not_added_to_parent_wall=True,
        old_stopped_runs="v2/v3/v4 cost records remain separate historical lower bounds where attempts are unclosed",
        source_events_sha256=sha256(root / "JOB_EVENTS.jsonl"), reused_events_sha256=sha256(base / "JOB_EVENTS.jsonl"))
    fixed_json(root / "FINAL_COST_CLOSURE.json", result)
    return result


def render_final_report(root, cost, packages):
    lines = ["# W1/W2: N300, 76 points, a single initial state and one run per fixed recipe", "",
        "GIFT, FNO2D, FNO3D and ANIE reuse completed v5 weights and W1 measurements; only U-NO, U-Net and GIFT-N50 each receive one additional training run. There are no development candidates, parameter searches or cross-seed means/SDs.",
        "W1 includes seven methods; W2 includes six, excluding ANIE as specified. The following are measured single-model values, reported separately for the two cohorts.", "",
        "The confirmation cohort remains independent of training/validation, but the v5 results of four methods were known; this extension is not a new blind test. The original test180 is an exploratory reevaluation."]
    for experiment in ("W1", "W2"):
        for cohort in ("confirmation", "old_test"):
            directory = root / ("w1_evaluation" if experiment == "W1" else "w2_evaluation") / cohort
            report = read_json(directory / "numeric_report.json")
            lines += ["", f"## {experiment} / {cohort}", "", "| Grid | Method | seed | Ntrain | Future mean relative L2 | Endpoint relative L2 | future MSE | Failed/180 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
            for row in summary_rows({cohort: report}, experiment):
                percent = lambda key: "failed/no aggregate value" if row[key] is None else f"{100*row[key]:.4f}%"
                mse = "failed/no aggregate value" if row["future_mse"] is None else f"{row['future_mse']:.7g}"
                lines.append(f"| {row['grid']} | {LABELS[row['method']]} | {row['seed']} | {row['training_count']} | {percent('future_mean_relative_l2')} | {percent('endpoint_relative_l2')} | {mse} | {row['failed_count']}/180 |")
            lines += ["", f"[Numeric report]({directory.as_posix()}/numeric_report.json) | [Complete figures and interpretation]({directory.as_posix()}/RESULTS.md)"]
    lines += ["", "## Actual cost", "", "| Work in this run | Attempts | Process wall-clock hours |", "|---|---:|---:|"]
    for label, item in cost["grouped_process_wall"].items():
        lines.append(f"| {label} | {item['attempts']} | {item['process_seconds']/3600:.4f} |")
    lines += ["", "Total wall-clock hours for new training: " + f"{cost['new_training_process_seconds']/3600:.4f}. Internal inference/validation time is included in its process and is not counted twice.", "",
        "| Reused model | Original v5 training process hours | New training hours in this run |", "|---|---:|---:|"]
    for label, seconds in cost["reused_model_original_training_process_seconds"].items():
        lines.append(f"| {LABELS[label]} | {seconds/3600:.4f} | 0 |")
    lines += ["", "## Interpretation boundaries and adaptations", "",
        "- Shared N300/20 validation trajectories, t5–8/.04/76 points and one true initial state; native losses, target coverage and computation budgets differ. This experiment compares fixed training recipes and does not isolate architecture as a single factor.",
        "- U-NO uses 20 closed-loop supervised steps through t5.80; U-Net uses 4 through t5.16, both from a fixed t5 origin. Long-term performance must be interpreted with this difference in mind.",
        "- GIFT-N50 uses the first 50 trajectories in the shared fixed N300 sequence, retaining GIFT stage epochs/optimizer/batch while reducing total updates and computation. It is not the historical GIFT-Lite or a sample-size ablation with equal update counts.",
        "- The original GIFT .1/.5-second supervision targets become .08/.48 seconds; internal RK4 .02/150 steps, saved .04/76 nodes; coarse FD uses only the 76 observation points.",
        "- FNO2D v5 uses a fixed closed-loop curriculum; ANIE v5 uses the official NS training branch/original coordinates, disabled evaluation dropout and the original MSE-best selection. This is not a full reproduction of the original papers.",
        "- W2 reuses all N64-trained weights without target-grid fitting; FNO3D selects the first 25 of 75 joint queries, with statistics determined only by frozen N64 training data. The physical scale of fixed-pixel U-Net kernels changes with the grid.",
        "- All failed trajectories and scientific-failure slots are retained; weights/parameters are not selected using test results and failures are not removed. Historical W3/S1–S3 and earlier W1/W2 remain unchanged with their original protocols identified.", "",
        "## Publication and verification", "", "The new compact package contains only summaries and selected figure sources, without full raw outputs or resumable training directories."]
    lines += [f"- [{experiment} publication package]({item['path'].replace(chr(92), '/')}/published.json)" for experiment, item in packages.items()]
    lines += ["", "Numerical and training audits: FINAL_RESULT_AUDIT.json and FINAL_TRAINING_AUDIT.json; costs: FINAL_COST_CLOSURE.json. Figure appearance and main-document migration require separate verification before delivery is marked complete."]
    fixed_text(root / "RESULTS.md", "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    for name in ("FINAL_RESULT_AUDIT.json", "FINAL_TRAINING_AUDIT.json"):
        audit = read_json(root / name)
        if audit is None or audit.get("status") not in ("passed", "pass", "PASS", "complete", "PASS_NUMERICAL_AND_TRAINING", "PASS_NUMERICAL_RECOMPUTATION"):
            raise ValueError("full scientific audit required before publication: " + name)
    cost = cost_closure(root)
    packages = {experiment: publish_package(root, experiment) for experiment in ("W1", "W2")}
    fixed_json(root / "PUBLICATION_RECEIPT.json", dict(status="published_integrity_verified", packages=packages,
        visual_qa_pending=True, document_migration_pending=True))
    render_final_report(root, cost, packages)
    print(json.dumps(dict(packages=packages, new_training_hours=cost["new_training_process_seconds"]/3600), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
