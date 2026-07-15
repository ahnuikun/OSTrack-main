"""Mechanism and tail analysis for one-sequence CMC-KF experiment matrices."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]


def iou_xywh(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    x1 = np.maximum(first[..., 0], second[..., 0])
    y1 = np.maximum(first[..., 1], second[..., 1])
    x2 = np.minimum(first[..., 0] + first[..., 2], second[..., 0] + second[..., 2])
    y2 = np.minimum(first[..., 1] + first[..., 3], second[..., 1] + second[..., 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    union = first[..., 2] * first[..., 3] + second[..., 2] * second[..., 3] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def center_error(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_center = first[..., :2] + 0.5 * first[..., 2:]
    second_center = second[..., :2] + 0.5 * second[..., 2:]
    return np.linalg.norm(first_center - second_center, axis=-1)


def percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def load_ground_truth(dataset_name, sequence_name):
    sys.path.insert(0, str(REPO))
    from lib.test.evaluation import get_dataset
    matches = [item for item in get_dataset(dataset_name) if item.name == sequence_name]
    if len(matches) != 1:
        raise ValueError(f"sequence lookup returned {len(matches)} matches")
    return np.asarray(matches[0].ground_truth_rect, dtype=np.float64)


def load_diagnostics(parameter_name, dataset_name, sequence_name):
    path = (
        REPO / "output" / "cmc_kf_candidate" / "runtime" /
        parameter_name / dataset_name / f"{sequence_name}.jsonl")
    return path, [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def analyze_variant(tracker_name, parameter_name, dataset_name, sequence_name, ground_truth):
    result_path = (
        REPO / "output" / "test" / "tracking_results" /
        tracker_name / parameter_name / f"{sequence_name}.txt")
    results = np.loadtxt(result_path, delimiter="\t", ndmin=2)
    if results.shape != ground_truth.shape:
        raise ValueError(f"shape mismatch for {parameter_name}: {results.shape} vs {ground_truth.shape}")
    result_iou = iou_xywh(results, ground_truth)
    row = {
        "parameter_name": parameter_name,
        "frames": len(results),
        "mean_iou": float(result_iou.mean()),
        "median_iou": float(np.median(result_iou)),
        "zero_iou_frames": int((result_iou <= 0.0).sum()),
        "result_center_error_median": percentile(center_error(results, ground_truth), 50),
        "result_center_error_p95": percentile(center_error(results, ground_truth), 95),
    }
    diagnostic_path = (
        REPO / "output" / "cmc_kf_candidate" / "runtime" /
        parameter_name / dataset_name / f"{sequence_name}.jsonl")
    if not diagnostic_path.exists():
        return row, []

    _, diagnostics = load_diagnostics(parameter_name, dataset_name, sequence_name)
    if len(diagnostics) != len(ground_truth):
        raise ValueError(f"diagnostic length mismatch for {parameter_name}")
    frames = []
    fallback_reasons = Counter()
    timing = []
    for frame_id, record in enumerate(diagnostics[1:], start=1):
        candidates = record["candidates"]
        candidate_boxes = np.asarray([item["image_box"] for item in candidates], dtype=np.float64)
        candidate_ious = iou_xywh(candidate_boxes, ground_truth[frame_id])
        top1_iou = float(candidate_ious[0])
        oracle_iou = float(candidate_ious.max())
        selected_rank = int(record["association"]["selected_rank"])
        selected_iou = float(candidate_ious[selected_rank - 1])
        cmc = record.get("cmc")
        if cmc is not None and not cmc["valid"]:
            fallback_reasons[cmc["fallback_reason"]] += 1
        predicted = record.get("predicted_box")
        search_reference = record.get("search_reference_box")
        previous = record.get("previous_output_box")
        frame = {
            "frame_id": frame_id,
            "top1_iou": top1_iou,
            "oracle_iou": oracle_iou,
            "selected_iou": selected_iou,
            "selected_rank": selected_rank,
            "recoverable_at_0_5": top1_iou < 0.5 <= oracle_iou,
            "rescue": selected_iou - top1_iou > 0.05,
            "harm": top1_iou - selected_iou > 0.05,
            "predicted_center_error": float(center_error(predicted, ground_truth[frame_id])),
            "search_shift": float(center_error(search_reference, previous)),
            "cmc_valid": None if cmc is None else bool(cmc["valid"]),
            "cmc_quality": None if cmc is None else float(cmc["quality"]),
            "kf_quality": None if record.get("kf") is None else float(record["kf"]["quality"]),
            "motion_weight": float(record["association"]["motion_weight"]),
            "measurement_accepted": record.get("measurement_accepted"),
        }
        frames.append(frame)
        timing.append(record["timing_ms"])

    row.update({
        "top1_recall_0_5": float(np.mean([item["top1_iou"] >= 0.5 for item in frames])),
        "top5_oracle_recall_0_5": float(np.mean([item["oracle_iou"] >= 0.5 for item in frames])),
        "recoverable_frames": sum(item["recoverable_at_0_5"] for item in frames),
        "selected_rank_gt1": sum(item["selected_rank"] > 1 for item in frames),
        "rescue_frames": sum(item["rescue"] for item in frames),
        "harm_frames": sum(item["harm"] for item in frames),
        "predicted_center_error_median": percentile([item["predicted_center_error"] for item in frames], 50),
        "predicted_center_error_p95": percentile([item["predicted_center_error"] for item in frames], 95),
        "search_shift_median": percentile([item["search_shift"] for item in frames], 50),
        "cmc_valid_fraction": (
            float(np.mean([item["cmc_valid"] for item in frames if item["cmc_valid"] is not None]))
            if any(item["cmc_valid"] is not None for item in frames) else None),
        "cmc_quality_median": percentile(
            [item["cmc_quality"] for item in frames if item["cmc_quality"] is not None], 50),
        "kf_quality_median": percentile(
            [item["kf_quality"] for item in frames if item["kf_quality"] is not None], 50),
        "motion_weight_mean": float(np.mean([item["motion_weight"] for item in frames])),
        "rejected_measurements": sum(item["measurement_accepted"] is False for item in frames),
        "total_time_median_ms": percentile([item["total"] for item in timing], 50),
        "total_time_p95_ms": percentile([item["total"] for item in timing], 95),
        "network_time_median_ms": percentile([item["network"] for item in timing], 50),
        "fallback_reasons": json.dumps(fallback_reasons, ensure_ascii=False, sort_keys=True),
    })
    return row, frames


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--tracker-name", default="ostrack_cmc_kf_assoc")
    parser.add_argument("--parameters", nargs="+", required=True)
    parser.add_argument("--baseline-tracker", default="ostrack")
    parser.add_argument("--baseline-parameter", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    ground_truth = load_ground_truth(args.dataset, args.sequence)
    rows = []
    baseline_row, _ = analyze_variant(
        args.baseline_tracker, args.baseline_parameter,
        args.dataset, args.sequence, ground_truth)
    rows.append(baseline_row)
    frame_rows = []
    for parameter in args.parameters:
        row, frames = analyze_variant(
            args.tracker_name, parameter, args.dataset, args.sequence, ground_truth)
        rows.append(row)
        for frame in frames:
            frame_rows.append({"parameter_name": parameter, **frame})

    fields = sorted({key for row in rows for key in row})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    frame_fields = sorted({key for row in frame_rows for key in row})
    with (output_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=frame_fields)
        writer.writeheader()
        writer.writerows(frame_rows)

    baseline_iou = rows[0]["mean_iou"]
    lines = [
        f"# S0 mechanism analysis: {args.dataset}/{args.sequence}", "",
        "This is a one-sequence smoke analysis and is not a promotion result.", "",
        "| Variant | Mean IoU | Delta vs E0 | Top-1 R@0.5 | Top-5 oracle R@0.5 | Recoverable | Rank>1 | Rescue | Harm | CMC valid | Rejected | Median ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['parameter_name']} | {row['mean_iou']:.4f} | "
            f"{row['mean_iou'] - baseline_iou:+.4f} | "
            f"{row.get('top1_recall_0_5', float('nan')):.4f} | "
            f"{row.get('top5_oracle_recall_0_5', float('nan')):.4f} | "
            f"{row.get('recoverable_frames', '')} | {row.get('selected_rank_gt1', '')} | "
            f"{row.get('rescue_frames', '')} | {row.get('harm_frames', '')} | "
            f"{row.get('cmc_valid_fraction', '')} | {row.get('rejected_measurements', '')} | "
            f"{row.get('total_time_median_ms', '')} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir / "report.md")


if __name__ == "__main__":
    main()
