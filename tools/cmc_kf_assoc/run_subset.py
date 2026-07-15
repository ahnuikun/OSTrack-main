"""Run and officially evaluate one frozen multi-dataset development subset."""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lib.test.analysis.extract_results import extract_results
from lib.test.analysis.plot_results import get_auc_curve, get_prec_curve
from lib.test.evaluation import get_dataset, trackerlist
from tools.cmc_kf_assoc.run_experiment import (
    RECORD_ROOT,
    git_output,
    load_failed_resume_record,
    sha256_file,
    source_fingerprint,
    validate_declared_identity,
    write_json,
)


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_logged_append(command, log_path):
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
        process = subprocess.Popen(
            command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            console_encoding = sys.stdout.encoding or "utf-8"
            safe = line.encode(console_encoding, errors="replace").decode(
                console_encoding, errors="replace")
            sys.stdout.write(safe)
            log.write(line)
        return process.wait()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--tracker-name", required=True)
    parser.add_argument("--parameter-name", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--association-backend", required=True)
    parser.add_argument("--adaptation-id", default=None)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--resume-from-experiment", default=None)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    return parser.parse_args()


def result_path(tracker_name, parameter_name, sequence_name):
    return (REPO / "output/test/tracking_results" / tracker_name /
            parameter_name / f"{sequence_name}.txt")


def diagnostic_path(parameter_name, dataset_name, sequence_name):
    return (REPO / "output/cmc_kf_candidate/runtime" / parameter_name /
            dataset_name / f"{sequence_name}.jsonl")


def validate_sequence_artifacts(tracker_name, parameter_name, dataset_name,
                                sequence, expect_diagnostics):
    result = result_path(tracker_name, parameter_name, sequence.name)
    boxes = np.loadtxt(result, delimiter="\t", ndmin=2)
    if boxes.shape != (len(sequence.frames), 4):
        raise ValueError(f"bad result shape for {dataset_name}/{sequence.name}: {boxes.shape}")
    if not np.isfinite(boxes).all() or not (boxes[:, 2:] > 0).all():
        raise ValueError(f"invalid result values for {dataset_name}/{sequence.name}")
    diagnostic = diagnostic_path(parameter_name, dataset_name, sequence.name)
    diagnostic_count = None
    if expect_diagnostics:
        lines = diagnostic.read_text(encoding="utf-8").splitlines()
        if any(not line for line in lines):
            raise ValueError(f"blank diagnostic line for {dataset_name}/{sequence.name}")
        records = [json.loads(line) for line in lines]
        if [item["frame_id"] for item in records] != list(range(len(sequence.frames))):
            raise ValueError(f"non-contiguous diagnostics for {dataset_name}/{sequence.name}")
        diagnostic_count = len(records)
    return {
        "result_path": str(result), "result_sha256": sha256_file(result),
        "diagnostic_path": str(diagnostic) if expect_diagnostics else None,
        "diagnostic_sha256": sha256_file(diagnostic) if expect_diagnostics else None,
        "frames": len(sequence.frames), "diagnostic_records": diagnostic_count,
    }


def evaluate_subset(experiment_id, tracker_name, parameter_name, split_data):
    metric_rows = []
    sequence_rows = []
    aggregate_overlap, aggregate_center, aggregate_center_norm = [], [], []
    for dataset_name, names in split_data["datasets"].items():
        lookup = {sequence.name: sequence for sequence in get_dataset(dataset_name)}
        subset = [lookup[name] for name in names]
        trackers = trackerlist(
            name=tracker_name, parameter_name=parameter_name,
            dataset_name=dataset_name, run_ids=None, display_name=parameter_name)
        report_name = f"cmc_kf_candidate/{experiment_id}/{dataset_name}"
        evaluation = extract_results(
            trackers, subset, report_name, skip_missing_seq=False)
        valid = torch.tensor(evaluation["valid_sequence"], dtype=torch.bool)
        overlap = torch.tensor(evaluation["ave_success_rate_plot_overlap"])
        center = torch.tensor(evaluation["ave_success_rate_plot_center"])
        center_norm = torch.tensor(evaluation["ave_success_rate_plot_center_norm"])
        auc_curve, auc = get_auc_curve(overlap, valid)
        _, precision = get_prec_curve(center, valid)
        _, norm_precision = get_prec_curve(center_norm, valid)
        thresholds = torch.tensor(evaluation["threshold_set_overlap"])
        metric_rows.append({
            "dataset": dataset_name, "sequences": len(subset),
            "frames": sum(len(sequence.frames) for sequence in subset),
            "auc": float(auc[0]),
            "op50": float(auc_curve[0, thresholds == 0.5][0]),
            "op75": float(auc_curve[0, thresholds == 0.75][0]),
            "precision": float(precision[0]),
            "norm_precision": float(norm_precision[0]),
        })
        for index, sequence in enumerate(subset):
            sequence_rows.append({
                "dataset": dataset_name, "sequence": sequence.name,
                "frames": len(sequence.frames),
                "auc": float(overlap[index, 0].mean() * 100.0),
                "precision": float(center[index, 0, 20] * 100.0),
                "norm_precision": float(center_norm[index, 0, 20] * 100.0),
            })
        aggregate_overlap.append(overlap[:, 0, :])
        aggregate_center.append(center[:, 0, :])
        aggregate_center_norm.append(center_norm[:, 0, :])

    overlap = torch.cat(aggregate_overlap, dim=0)
    center = torch.cat(aggregate_center, dim=0)
    center_norm = torch.cat(aggregate_center_norm, dim=0)
    metric_rows.append({
        "dataset": "development_v1_all", "sequences": overlap.shape[0],
        "frames": sum(row["frames"] for row in metric_rows),
        "auc": float(overlap.mean(dim=0).mean() * 100.0),
        "op50": float(overlap[:, 10].mean() * 100.0),
        "op75": float(overlap[:, 15].mean() * 100.0),
        "precision": float(center[:, 20].mean() * 100.0),
        "norm_precision": float(center_norm[:, 20].mean() * 100.0),
    })
    return metric_rows, sequence_rows


def main():
    args = parse_args()
    if args.num_gpus != 1:
        raise ValueError("only one GPU is registered for this experiment line")
    validate_declared_identity(
        args.variant, args.parameter_name, args.checkpoint_id,
        args.checkpoint_sha256)
    record_dir = RECORD_ROOT / args.experiment_id
    if record_dir.exists():
        raise FileExistsError(f"record exists: {record_dir}")
    split_path = (REPO / args.split_file).resolve()
    split_data = yaml.safe_load(split_path.read_text(encoding="utf-8"))
    split_sha256 = sha256_file(split_path)
    sequence_plan = []
    for dataset_name, names in split_data["datasets"].items():
        lookup = {sequence.name: sequence for sequence in get_dataset(dataset_name)}
        for name in names:
            if name not in lookup:
                raise ValueError(f"split sequence missing: {dataset_name}/{name}")
            sequence_plan.append((dataset_name, lookup[name]))
    sequence_names = [sequence.name for _, sequence in sequence_plan]
    if len(sequence_names) != len(set(sequence_names)):
        raise ValueError(
            "split contains cross-dataset sequence-name collisions; upstream result "
            "paths would overwrite each other")

    expect_diagnostics = args.tracker_name == "ostrack_cmc_kf_assoc"
    for dataset_name, sequence in sequence_plan:
        result_exists = result_path(
            args.tracker_name, args.parameter_name, sequence.name).exists()
        diagnostic_exists = diagnostic_path(
            args.parameter_name, dataset_name, sequence.name).exists()
        if args.resume_from_experiment is None and (result_exists or (expect_diagnostics and diagnostic_exists)):
            raise FileExistsError(f"artifact exists before registration: {dataset_name}/{sequence.name}")
        if args.resume_from_experiment is not None and result_exists:
            validate_sequence_artifacts(
                args.tracker_name, args.parameter_name, dataset_name, sequence,
                expect_diagnostics)

    if args.resume_from_experiment is not None:
        source_manifest = load_failed_resume_record(args.resume_from_experiment)
        for key, expected in {
                "tracker_name": args.tracker_name,
                "parameter_name": args.parameter_name,
                "checkpoint_id": args.checkpoint_id,
                "checkpoint_sha256": args.checkpoint_sha256}.items():
            if source_manifest["model"].get(key) != expected:
                raise ValueError(f"resume source model mismatch for {key}")
        if source_manifest["scope"].get("split_sha256") != split_sha256:
            raise ValueError("resume source split SHA256 does not match")

    source_hash, source_file_count = source_fingerprint()
    registered_at = now_iso()
    record_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "experiment": {
            "id": args.experiment_id, "stage": "S1", "variant": args.variant,
            "association_backend": args.association_backend,
            "adaptation_id": args.adaptation_id, "state": "registered",
            "hypothesis": args.hypothesis,
        },
        "scope": {
            "split_id": split_data["split_id"], "split_file": str(split_path),
            "split_sha256": split_sha256, "expected_sequences": len(sequence_plan),
            "expected_frames": sum(len(sequence.frames) for _, sequence in sequence_plan),
            "development_data": True, "holdout_opened": False,
            "datasets": split_data["datasets"],
        },
        "code": {
            "repo": str(REPO), "branch": git_output("branch", "--show-current"),
            "commit": git_output("rev-parse", "HEAD"),
            "dirty": bool(git_output("status", "--porcelain=v1", "-uall")),
            "source_tree_sha256": source_hash, "source_file_count": source_file_count,
        },
        "model": {
            "checkpoint_id": args.checkpoint_id,
            "checkpoint_sha256": args.checkpoint_sha256,
            "tracker_name": args.tracker_name,
            "parameter_name": args.parameter_name,
            "yaml": "vitb_256_mae_ce_32x4_ep300", "search_size": 256,
        },
        "runtime": {"python": sys.executable, "gpu_ids": [0], "threads": args.threads, "num_gpus": 1},
        "recovery": {"resume_from_experiment": args.resume_from_experiment},
        "timestamps": {"registered_at": registered_at},
    }
    (record_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (record_dir / "environment.txt").write_text(
        f"captured_at={registered_at}\npython={sys.version}\ntorch={torch.__version__}\n"
        f"cuda_available={torch.cuda.is_available()}\n"
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\n"
        f"source_tree_sha256={source_hash}\nsplit_sha256={split_sha256}\n",
        encoding="utf-8")

    commands = []
    for dataset_name, sequence in sequence_plan:
        commands.append([
            sys.executable, "-u", "tracking/test.py", args.tracker_name,
            args.parameter_name, "--dataset_name", dataset_name, "--sequence",
            sequence.name, "--threads", str(args.threads), "--num_gpus", "1",
        ])
    (record_dir / "commands.ps1").write_text(
        "\n".join("& '" + "' '".join(command) + "'" for command in commands) + "\n",
        encoding="utf-8")

    status = {
        "experiment_id": args.experiment_id, "state": "running",
        "started_at": now_iso(), "finished_at": None, "exit_code": None,
        "completed_sequences": 0, "expected_sequences": len(sequence_plan),
        "failure_reason": None, "last_sequence": None,
    }
    write_json(record_dir / "status.json", status)
    artifact_rows = []
    try:
        for index, ((dataset_name, sequence), command) in enumerate(zip(sequence_plan, commands), start=1):
            result = result_path(args.tracker_name, args.parameter_name, sequence.name)
            if not result.exists():
                exit_code = run_logged_append(command, record_dir / "stdout.log")
                if exit_code != 0:
                    raise RuntimeError(f"tracker exited {exit_code} on {dataset_name}/{sequence.name}")
            artifacts = validate_sequence_artifacts(
                args.tracker_name, args.parameter_name, dataset_name, sequence,
                expect_diagnostics)
            artifact_rows.append({"dataset": dataset_name, "sequence": sequence.name, **artifacts})
            status.update({"completed_sequences": index, "last_sequence": f"{dataset_name}/{sequence.name}"})
            write_json(record_dir / "status.json", status)

        metrics, sequence_metrics = evaluate_subset(
            args.experiment_id, args.tracker_name, args.parameter_name, split_data)
        for filename, rows in (("metrics.csv", metrics), ("sequence_metrics.csv", sequence_metrics),
                               ("artifacts.csv", artifact_rows)):
            fields = list(rows[0])
            with (record_dir / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        aggregate = metrics[-1]
        (record_dir / "report.md").write_text(
            f"# Experiment Report: `{args.experiment_id}`\n\n"
            f"- 状态：`completed`\n- Gate：`Not evaluated`（等待同批次对照与机制分析）\n"
            f"- Split：`{split_data['split_id']}`，{aggregate['sequences']} 序列，{aggregate['frames']} 帧\n"
            f"- Development aggregate AUC / Precision / Norm Precision："
            f"{aggregate['auc']:.2f} / {aggregate['precision']:.2f} / {aggregate['norm_precision']:.2f}\n"
            "- Holdout：未打开\n",
            encoding="utf-8")
        status.update({"state": "completed", "finished_at": now_iso(), "exit_code": 0})
    except Exception as error:
        status.update({
            "state": "failed", "finished_at": now_iso(),
            "exit_code": locals().get("exit_code"),
            "failure_reason": f"{type(error).__name__}: {error}",
        })
        write_json(record_dir / "status.json", status)
        raise
    write_json(record_dir / "status.json", status)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
