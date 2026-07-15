"""Aggregate mechanism, sequence-tail, and promotion-gate analysis for S1."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lib.test.evaluation import get_dataset
from tools.cmc_kf_assoc.analyze_experiment import iou_xywh


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_result(tracker, parameter, sequence):
    path = (REPO / "output/test/tracking_results" / tracker / parameter /
            f"{sequence}.txt")
    return np.loadtxt(path, delimiter="\t", ndmin=2)


def load_diagnostics(parameter, dataset, sequence):
    path = (REPO / "output/cmc_kf_candidate/runtime" / parameter / dataset /
            f"{sequence}.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def longest_run(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def percentile(values, q):
    return float(np.percentile(values, q)) if values else float("nan")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--baseline-record", required=True)
    parser.add_argument("--baseline-tracker", default="ostrack")
    parser.add_argument("--baseline-parameter", required=True)
    parser.add_argument("--variant-record", action="append", nargs=3,
                        metavar=("LABEL", "RECORD", "PARAMETER"), required=True)
    parser.add_argument("--tracker-name", default="ostrack_cmc_kf_assoc")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    split = yaml.safe_load((REPO / args.split_file).read_text(encoding="utf-8"))
    baseline_record = REPO / "experiments/cmc_kf_candidate/records" / args.baseline_record
    baseline_metrics = load_csv(baseline_record / "metrics.csv")
    aggregate_dataset = baseline_metrics[-1]["dataset"]
    baseline_sequence_rows = load_csv(baseline_record / "sequence_metrics.csv")
    baseline_auc = {(row["dataset"], row["sequence"]): float(row["auc"])
                    for row in baseline_sequence_rows}
    ground_truth = {}
    sequence_order = []
    for dataset, names in split["datasets"].items():
        lookup = {sequence.name: sequence for sequence in get_dataset(dataset)}
        for name in names:
            ground_truth[(dataset, name)] = np.asarray(
                lookup[name].ground_truth_rect, dtype=np.float64)
            sequence_order.append((dataset, name))

    baseline_frame_iou = {}
    for dataset, name in sequence_order:
        baseline_frame_iou[(dataset, name)] = iou_xywh(
            load_result(args.baseline_tracker, args.baseline_parameter, name),
            ground_truth[(dataset, name)])

    metric_table = []
    for row in baseline_metrics:
        metric_table.append({
            "label": "E0", "dataset": row["dataset"],
            "auc": float(row["auc"]), "precision": float(row["precision"]),
            "norm_precision": float(row["norm_precision"]),
        })

    variant_summaries = []
    sequence_delta_rows = []
    sequence_auc_by_label = {"E0": baseline_auc}
    fallback_totals = Counter()
    for label, record_id, parameter in args.variant_record:
        record = REPO / "experiments/cmc_kf_candidate/records" / record_id
        metrics = load_csv(record / "metrics.csv")
        for row in metrics:
            metric_table.append({
                "label": label, "dataset": row["dataset"],
                "auc": float(row["auc"]), "precision": float(row["precision"]),
                "norm_precision": float(row["norm_precision"]),
            })
        sequence_rows = load_csv(record / "sequence_metrics.csv")
        sequence_auc = {(row["dataset"], row["sequence"]): float(row["auc"])
                        for row in sequence_rows}
        sequence_auc_by_label[label] = sequence_auc
        deltas = []
        regressions = []
        new_failure_max = 0
        top1_good = top5_good = recoverable = rank_gt1 = rescue = harm = 0
        diagnostic_frames = cmc_frames = cmc_valid = rejected = 0
        rejection_runs = []
        total_times = []
        network_times = []
        motion_weights = []
        switch_reasons = Counter()
        proposed_rank_gt1 = accepted_reranks = 0
        for dataset, name in sequence_order:
            delta = sequence_auc[(dataset, name)] - baseline_auc[(dataset, name)]
            deltas.append(delta)
            sequence_delta_rows.append({
                "label": label, "dataset": dataset, "sequence": name,
                "baseline_auc": baseline_auc[(dataset, name)],
                "variant_auc": sequence_auc[(dataset, name)], "delta_auc": delta,
            })
            gt = ground_truth[(dataset, name)]
            current_iou = iou_xywh(load_result(args.tracker_name, parameter, name), gt)
            base_iou = baseline_frame_iou[(dataset, name)]
            regressions.extend((base_iou - current_iou).tolist())
            new_failure_max = max(new_failure_max, longest_run(
                (current_iou < 0.1) & (base_iou >= 0.5)))
            diagnostics = load_diagnostics(parameter, dataset, name)
            if len(diagnostics) != len(gt):
                raise ValueError(f"diagnostic length mismatch: {label}/{dataset}/{name}")
            rejected_flags = []
            for frame_id, item in enumerate(diagnostics[1:], start=1):
                candidates = item["candidates"]
                candidate_iou = iou_xywh(
                    np.asarray([candidate["image_box"] for candidate in candidates]),
                    gt[frame_id])
                selected_rank = int(item["association"]["selected_rank"])
                selected_iou = float(candidate_iou[selected_rank - 1])
                top1_iou = float(candidate_iou[0])
                oracle_iou = float(candidate_iou.max())
                top1_good += top1_iou >= 0.5
                top5_good += oracle_iou >= 0.5
                recoverable += top1_iou < 0.5 <= oracle_iou
                rank_gt1 += selected_rank > 1
                rescue += selected_iou - top1_iou > 0.05
                harm += top1_iou - selected_iou > 0.05
                diagnostic_frames += 1
                cmc = item.get("cmc")
                if cmc is not None:
                    cmc_frames += 1
                    cmc_valid += bool(cmc["valid"])
                    if not cmc["valid"]:
                        fallback_totals[(label, cmc["fallback_reason"])] += 1
                is_rejected = item.get("measurement_accepted") is False
                rejected += is_rejected
                rejected_flags.append(is_rejected)
                total_times.append(float(item["timing_ms"]["total"]))
                network_times.append(float(item["timing_ms"]["network"]))
                motion_weights.append(float(item["association"]["motion_weight"]))
                switch = item.get("switch_control")
                if switch is not None:
                    switch_reasons[switch["reason"]] += 1
                    proposed_rank_gt1 += int(switch["proposed_rank"] > 1)
                    accepted_reranks += int(switch["accepted_rerank"])
            rejection_runs.append(longest_run(rejected_flags))
        aggregate = next(row for row in metrics if row["dataset"] == aggregate_dataset)
        worst_index = int(np.argmin(deltas))
        variant_summaries.append({
            "label": label, "parameter": parameter,
            "auc": float(aggregate["auc"]),
            "auc_delta_e0": float(aggregate["auc"]) - float(baseline_metrics[-1]["auc"]),
            "negative_sequence_fraction": float(np.mean(np.asarray(deltas) < 0.0)),
            "worst_sequence": "/".join(sequence_order[worst_index]),
            "worst_sequence_delta": float(deltas[worst_index]),
            "frame_regression_p95": percentile(regressions, 95),
            "frame_regression_p99": percentile(regressions, 99),
            "longest_new_failure_run": new_failure_max,
            "top1_recall_0_5": top1_good / diagnostic_frames,
            "top5_recall_0_5": top5_good / diagnostic_frames,
            "top5_gain_pp": 100.0 * (top5_good - top1_good) / diagnostic_frames,
            "recoverable_frames": recoverable, "selected_rank_gt1": rank_gt1,
            "rescue_frames": rescue, "harm_frames": harm,
            "cmc_valid_fraction": cmc_valid / cmc_frames if cmc_frames else None,
            "rejected_measurements": rejected,
            "longest_rejection_run": max(rejection_runs),
            "motion_weight_mean": float(np.mean(motion_weights)),
            "total_time_median_ms": percentile(total_times, 50),
            "total_time_p95_ms": percentile(total_times, 95),
            "network_time_median_ms": percentile(network_times, 50),
            "proposed_rank_gt1": proposed_rank_gt1,
            "accepted_reranks": accepted_reranks,
            "switch_margin_holds": switch_reasons["margin_hold"],
            "switch_pending_holds": switch_reasons["pending_confirmation"],
            "switch_consistency_holds": switch_reasons["consistency_hold"],
        })

    with (output_dir / "metric_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_table[0]))
        writer.writeheader(); writer.writerows(metric_table)
    with (output_dir / "sequence_deltas.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sequence_delta_rows[0]))
        writer.writeheader(); writer.writerows(sequence_delta_rows)
    with (output_dir / "mechanism_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(variant_summaries[0]))
        writer.writeheader(); writer.writerows(variant_summaries)
    (output_dir / "fallback_reasons.json").write_text(
        json.dumps({f"{label}:{reason}": count for (label, reason), count in fallback_totals.items()},
                   indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    baseline_aggregate = float(baseline_metrics[-1]["auc"])
    lines = [
        "# S1 frozen-development analysis", "",
        f"Split: `{split['split_id']}`; {len(sequence_order)} sequences. Holdout was not opened.", "",
        "## Official aggregate metrics", "",
        "| Variant | AUC | Delta vs E0 | Precision | Norm Precision |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ["E0"] + [item[0] for item in args.variant_record]:
        row = next(item for item in metric_table
                   if item["label"] == label and item["dataset"] == aggregate_dataset)
        lines.append(f"| {label} | {row['auc']:.2f} | {row['auc'] - baseline_aggregate:+.2f} | "
                     f"{row['precision']:.2f} | {row['norm_precision']:.2f} |")
    lines += ["", "## Per-dataset AUC", "",
              "| Dataset | " + " | ".join(["E0"] + [item[0] for item in args.variant_record]) + " |",
              "|---|" + "---:|" * (1 + len(args.variant_record))]
    for dataset in list(split["datasets"]) + [aggregate_dataset]:
        values = []
        for label in ["E0"] + [item[0] for item in args.variant_record]:
            row = next(item for item in metric_table
                       if item["label"] == label and item["dataset"] == dataset)
            values.append(f"{row['auc']:.2f}")
        lines.append(f"| {dataset} | " + " | ".join(values) + " |")
    lines += ["", "## Tail and mechanism evidence", "",
              "| Variant | Negative seq | Worst sequence delta | Frame regression P95/P99 | Top5 gain | Recoverable | Rank>1 | Rescue/Harm | CMC valid | Rejects (max run) | N2 proposed/accepted | N2 margin/pending/consistency holds | Median ms |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in variant_summaries:
        lines.append(
            f"| {row['label']} | {100*row['negative_sequence_fraction']:.1f}% | "
            f"{row['worst_sequence']} {row['worst_sequence_delta']:+.2f} | "
            f"{row['frame_regression_p95']:.3f}/{row['frame_regression_p99']:.3f} | "
            f"{row['top5_gain_pp']:.2f} pp | {row['recoverable_frames']} | "
            f"{row['selected_rank_gt1']} | {row['rescue_frames']}/{row['harm_frames']} | "
            f"{row['cmc_valid_fraction']:.3f} | {row['rejected_measurements']} "
            f"({row['longest_rejection_run']}) | "
            f"{row['proposed_rank_gt1']}/{row['accepted_reranks']} | "
            f"{row['switch_margin_holds']}/{row['switch_pending_holds']}/"
            f"{row['switch_consistency_holds']} | {row['total_time_median_ms']:.2f} |")
    e6 = next((row for row in variant_summaries if row["label"] == "E6"), None)
    if e6 is not None:
        candidate_gate = e6["top5_gain_pp"] >= 2.0 and e6["recoverable_frames"] >= 10
        tail_gate = (e6["negative_sequence_fraction"] <= 0.45 and
                     e6["worst_sequence_delta"] >= -5.0 and
                     e6["frame_regression_p95"] <= 0.03 and
                     e6["frame_regression_p99"] <= 0.08 and
                     e6["longest_new_failure_run"] <= 10)
        association_gate = (e6["rescue_frames"] > e6["harm_frames"] and
                            e6["longest_new_failure_run"] <= 10)
        e1_metric = next((item for item in metric_table
                          if item["label"] == "E1" and
                          item["dataset"] == aggregate_dataset), None)
        aggregate_gate = (e6["auc_delta_e0"] >= 0.30 and e1_metric is not None and
                          e6["auc"] >= e1_metric["auc"])
        promotion_gate = aggregate_gate and candidate_gate and association_gate and tail_gate
        lines += ["", "## Gate status", "",
                  f"- Candidate-value gate: `{'PASS' if candidate_gate else 'FAIL'}`.",
                  f"- Association-effect gate: `{'PASS' if association_gate else 'FAIL'}`.",
                  f"- E6 tail-safety gate: `{'PASS' if tail_gate else 'FAIL'}`.",
                  f"- E6 aggregate-vs-E0/E1 gate: `{'PASS' if aggregate_gate else 'FAIL'}`.",
                  f"- S1 promotion decision: `{'PASS' if promotion_gate else 'FAIL'}`."]
        if "M1" in sequence_auc_by_label:
            pairwise = [sequence_auc_by_label["E6"][key] - sequence_auc_by_label["M1"][key]
                        for key in sequence_order]
            worst = int(np.argmin(pairwise))
            lines += ["", "## E6 versus M1", "",
                      f"- Negative sequences: {100*np.mean(np.asarray(pairwise) < 0):.1f}%.",
                      f"- Worst sequence: {'/'.join(sequence_order[worst])} "
                      f"({pairwise[worst]:+.2f} AUC points)."]
    n2 = next((row for row in variant_summaries if row["label"] == "N2"), None)
    if n2 is not None and "M1" in sequence_auc_by_label and "E6" in sequence_auc_by_label:
        for reference in ("M1", "E6"):
            pairwise = [sequence_auc_by_label["N2"][key] -
                        sequence_auc_by_label[reference][key]
                        for key in sequence_order]
            worst = int(np.argmin(pairwise))
            aggregate_reference = next(
                item for item in metric_table
                if item["label"] == reference and item["dataset"] == aggregate_dataset)
            lines += ["", f"## N2 versus {reference}", "",
                      f"- Aggregate delta: {n2['auc'] - aggregate_reference['auc']:+.2f} AUC points.",
                      f"- Negative sequences: {100*np.mean(np.asarray(pairwise) < 0):.1f}%.",
                      f"- Worst sequence: {'/'.join(sequence_order[worst])} "
                      f"({pairwise[worst]:+.2f} AUC points)."]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir / "report.md")


if __name__ == "__main__":
    main()
