"""Register, run, validate, and summarize one CMC-KF experiment."""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


REPO = Path(__file__).resolve().parents[2]
RECORD_ROOT = REPO / "experiments" / "cmc_kf_candidate" / "records"
sys.path.insert(0, str(REPO))

from lib.test.parameter.ostrack_checkpoint_registry import CHECKPOINTS


def validate_declared_identity(variant, parameter_name, checkpoint_id,
                               checkpoint_sha256):
    """Reject a manifest identity that disagrees with its parameter/registry."""

    parts = parameter_name.lower().split("__")
    if len(parts) != 3:
        raise ValueError("parameter name must contain variant, checkpoint id, and version")
    parameter_variant, parameter_checkpoint, _ = parts
    if parameter_variant != str(variant).lower():
        raise ValueError(
            f"variant mismatch: CLI={variant!r}, parameter={parameter_variant!r}")
    if parameter_checkpoint != str(checkpoint_id).lower():
        raise ValueError(
            f"checkpoint mismatch: CLI={checkpoint_id!r}, parameter={parameter_checkpoint!r}")
    try:
        registered = CHECKPOINTS[parameter_checkpoint]
    except KeyError as error:
        raise ValueError(f"unregistered checkpoint: {parameter_checkpoint}") from error
    if registered.sha256 != str(checkpoint_sha256).lower():
        raise ValueError(
            f"checkpoint SHA256 mismatch: registry={registered.sha256}, "
            f"CLI={checkpoint_sha256}")


def load_failed_resume_record(record_id):
    record = RECORD_ROOT / record_id
    status_path = record / "status.json"
    manifest_path = record / "manifest.yaml"
    if not status_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"resume source record is incomplete: {record}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "failed":
        raise ValueError(f"resume source must be failed, got {status.get('state')!r}")
    return yaml.safe_load(manifest_path.read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint():
    roots = [
        REPO / "lib", REPO / "tracking", REPO / "tools" / "cmc_kf_assoc",
        REPO / "docs" / "research", REPO / "experiments" / "cmc_kf_candidate" / "manifests",
    ]
    digest = hashlib.sha256()
    file_count = 0
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            relative = path.relative_to(REPO).as_posix().encode("utf-8")
            digest.update(relative + b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
            file_count += 1
    return digest.hexdigest(), file_count


def git_output(*args):
    result = subprocess.run(
        ["git", *args], cwd=REPO, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_logged(command, log_path):
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            console_encoding = sys.stdout.encoding or "utf-8"
            safe_line = line.encode(console_encoding, errors="replace").decode(
                console_encoding, errors="replace")
            sys.stdout.write(safe_line)
            log.write(line)
        return process.wait()


def dataset_sequence(dataset_name, sequence_name):
    sys.path.insert(0, str(REPO))
    from lib.test.evaluation import get_dataset
    dataset = get_dataset(dataset_name)
    matches = [sequence for sequence in dataset if sequence.name == sequence_name]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {dataset_name}/{sequence_name} sequence, found {len(matches)}")
    return matches[0]


def parse_metrics(evaluator_text, parameter_name):
    for line in evaluator_text.splitlines():
        if line.strip().startswith(parameter_name) and "|" in line:
            values = [item.strip() for item in line.split("|")[1:] if item.strip()]
            if len(values) >= 5:
                return {
                    "auc": float(values[0]), "op50": float(values[1]),
                    "op75": float(values[2]), "precision": float(values[3]),
                    "norm_precision": float(values[4]),
                }
    raise ValueError("could not parse evaluator summary row")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--stage", required=True, choices=["S0", "S1", "S2", "S3"])
    parser.add_argument("--variant", required=True)
    parser.add_argument("--tracker-name", required=True)
    parser.add_argument("--parameter-name", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--association-backend", default="A0")
    parser.add_argument("--adaptation-id", default=None)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--resume-from-experiment", default=None,
                        help="Reuse already-written raw output from a retained failed record.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_gpus != 1:
        raise ValueError("this project is registered for one-GPU evaluation")
    validate_declared_identity(
        args.variant, args.parameter_name, args.checkpoint_id,
        args.checkpoint_sha256)
    record_dir = RECORD_ROOT / args.experiment_id
    if record_dir.exists():
        raise FileExistsError(f"record already exists and will not be overwritten: {record_dir}")

    sequence = dataset_sequence(args.dataset, args.sequence)
    expected_frames = len(sequence.frames)
    result_dir = (
        REPO / "output" / "test" / "tracking_results" /
        args.tracker_name / args.parameter_name)
    result_path = result_dir / f"{args.sequence}.txt"
    if result_path.exists() and args.resume_from_experiment is None:
        raise FileExistsError(f"raw result already exists and will not be reused: {result_path}")

    diagnostic_path = (
        REPO / "output" / "cmc_kf_candidate" / "runtime" /
        args.parameter_name / args.dataset / f"{args.sequence}.jsonl")
    if diagnostic_path.exists() and args.resume_from_experiment is None:
        raise FileExistsError(f"diagnostic already exists and will not be overwritten: {diagnostic_path}")
    if args.resume_from_experiment is not None:
        source_manifest = load_failed_resume_record(args.resume_from_experiment)
        expected_source = {
            "tracker_name": args.tracker_name,
            "parameter_name": args.parameter_name,
            "checkpoint_id": args.checkpoint_id,
            "checkpoint_sha256": args.checkpoint_sha256,
        }
        for key, expected in expected_source.items():
            if source_manifest["model"].get(key) != expected:
                raise ValueError(
                    f"resume source model mismatch for {key}: "
                    f"{source_manifest['model'].get(key)!r} != {expected!r}")
        source_scope = source_manifest["scope"]
        if (source_scope.get("dataset") != args.dataset or
                source_scope.get("sequence") != args.sequence):
            raise ValueError("resume source dataset/sequence does not match")
        if not result_path.is_file():
            raise FileNotFoundError(f"resume result is missing: {result_path}")

    source_hash, source_files = source_fingerprint()
    registered_at = now_iso()
    commit = git_output("rev-parse", "HEAD")
    branch = git_output("branch", "--show-current")
    dirty = bool(git_output("status", "--porcelain=v1", "-uall"))
    record_dir.mkdir(parents=True)

    manifest = {
        "schema_version": 1,
        "experiment": {
            "id": args.experiment_id, "title": args.hypothesis,
            "stage": args.stage, "variant": args.variant,
            "association_backend": args.association_backend,
            "adaptation_id": args.adaptation_id, "state": "registered",
            "hypothesis": args.hypothesis,
        },
        "scope": {
            "dataset_id": args.dataset_id, "dataset": args.dataset,
            "sequence": args.sequence, "expected_sequences": 1,
            "expected_frames": expected_frames, "development_data": False,
            "holdout_opened": False,
        },
        "code": {
            "repo": str(REPO), "branch": branch, "commit": commit,
            "dirty": dirty, "source_tree_sha256": source_hash,
            "source_file_count": source_files,
        },
        "model": {
            "checkpoint_id": args.checkpoint_id,
            "tracker_name": args.tracker_name,
            "parameter_name": args.parameter_name,
            "checkpoint_sha256": args.checkpoint_sha256,
            "yaml": "vitb_256_mae_ce_32x4_ep300", "search_size": 256,
        },
        "runtime": {
            "python": sys.executable, "gpu_ids": [0], "threads": args.threads,
            "num_gpus": args.num_gpus, "seed": 42,
        },
        "artifacts": {
            "record_dir": str(record_dir), "raw_result_path": str(result_path),
            "diagnostic_path": str(diagnostic_path),
        },
        "recovery": {
            "resume_from_experiment": args.resume_from_experiment,
            "reused_raw_result": args.resume_from_experiment is not None,
        },
        "timestamps": {"registered_at": registered_at},
    }
    (record_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")

    command = [
        sys.executable, "-u", "tracking/test.py", args.tracker_name,
        args.parameter_name, "--dataset_name", args.dataset, "--sequence",
        args.sequence, "--threads", str(args.threads), "--num_gpus",
        str(args.num_gpus),
    ]
    (record_dir / "command.ps1").write_text(
        "& '" + "' '".join(command) + "'\n", encoding="utf-8")
    (record_dir / "environment.txt").write_text(
        "\n".join([
            f"captured_at={registered_at}", f"platform={platform.platform()}",
            f"python={platform.python_version()}", f"python_executable={sys.executable}",
            f"torch={torch.__version__}", f"cuda_available={torch.cuda.is_available()}",
            f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}",
            f"git_commit={commit}", f"git_branch={branch}", f"git_dirty={str(dirty).lower()}",
            f"source_tree_sha256={source_hash}", f"source_file_count={source_files}",
        ]) + "\n", encoding="utf-8")

    started_at = now_iso()
    status = {
        "experiment_id": args.experiment_id, "state": "running",
        "started_at": started_at, "finished_at": None, "exit_code": None,
        "completed_sequences": 0, "expected_sequences": 1,
        "failure_reason": None,
    }
    write_json(record_dir / "status.json", status)

    try:
        if args.resume_from_experiment is None:
            exit_code = run_logged(command, record_dir / "stdout.log")
            if exit_code != 0:
                raise RuntimeError(f"tracker exited with code {exit_code}")
        else:
            exit_code = 0
            (record_dir / "stdout.log").write_text(
                f"Reused raw result from retained failed record: {args.resume_from_experiment}\n",
                encoding="utf-8")
        rows = np.loadtxt(result_path, delimiter="\t", ndmin=2)
        if rows.shape != (expected_frames, 4):
            raise ValueError(f"result shape mismatch: expected {(expected_frames, 4)}, got {rows.shape}")
        if not np.isfinite(rows).all() or not (rows[:, 2:] > 0).all():
            raise ValueError("result contains non-finite values or non-positive sizes")

        diagnostic_records = None
        if diagnostic_path.exists():
            lines = diagnostic_path.read_text(encoding="utf-8").splitlines()
            if any(not line for line in lines):
                raise ValueError("diagnostic JSONL contains blank records")
            records = [json.loads(line) for line in lines]
            if [item["frame_id"] for item in records] != list(range(expected_frames)):
                raise ValueError("diagnostic frame IDs are not contiguous")
            diagnostic_records = len(records)

        evaluator_command = [
            sys.executable, "-u", "tracking/analyze_uav_suite.py",
            "--tracker_name", args.tracker_name, "--tracker_param",
            args.parameter_name, "--dataset", args.dataset, "--force_evaluation",
            "--skip_missing_seq", "--per_sequence",
        ]
        evaluator_exit = run_logged(evaluator_command, record_dir / "evaluator.log")
        if evaluator_exit != 0:
            raise RuntimeError(f"evaluator exited with code {evaluator_exit}")
        evaluator_text = (record_dir / "evaluator.log").read_text(encoding="utf-8")
        metrics = parse_metrics(evaluator_text, args.parameter_name)
        metrics.update({
            "result_rows": int(rows.shape[0]),
            "diagnostic_records": diagnostic_records,
            "result_sha256": sha256_file(result_path),
            "diagnostic_sha256": sha256_file(diagnostic_path) if diagnostic_path.exists() else None,
        })
        with (record_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics))
            writer.writeheader()
            writer.writerow(metrics)
        (record_dir / "report.md").write_text(
            f"# Experiment Report: `{args.experiment_id}`\n\n"
            f"- 状态：`completed`\n"
            f"- Gate：`Not evaluated`（S0 单序列 smoke）\n"
            f"- 序列：`{args.dataset}/{args.sequence}`，{expected_frames} 帧\n"
            f"- AUC / Precision / Norm Precision：{metrics['auc']:.2f} / "
            f"{metrics['precision']:.2f} / {metrics['norm_precision']:.2f}\n"
            f"- 结果 SHA256：`{metrics['result_sha256']}`\n"
            f"- 诊断记录：{diagnostic_records}\n\n"
            "该数字只验证代码、数值、记录和评测通路，不进入正式主表。\n",
            encoding="utf-8")
        status.update({
            "state": "completed", "finished_at": now_iso(), "exit_code": 0,
            "completed_sequences": 1,
        })
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
