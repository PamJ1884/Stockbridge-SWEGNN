#!/usr/bin/env python3
"""Frozen M17K evaluation: validate first, then test exactly once per selected checkpoint.

Reuses the training code's teacher-forced and autoregressive evaluation
functions verbatim. Never trains or selects checkpoints using test events.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

import train_valid_stockbridge_m17k as m17k

EXPECTED = {
    "p4_k4_raw": {
        "K": 4,
        "checkpoint_suffix": "milestone17k_paperstyle_raw_pilot_v01/rainfall_aware/m17k_best_checkpoint.pt",
    },
    "p6b_k8_raw": {
        "K": 8,
        "checkpoint_suffix": "milestone17k_k8_raw_pilot_v01/rainfall_aware/m17k_best_checkpoint.pt",
    },
}
ROLLOUT_NUMERIC_KEYS = (
    "swegnn_val_loss", "H_RMSE_active_mean", "Qmag_RMSE_active_mean",
    "CSI_0p01_mean", "CSI_0p05_mean", "CSI_0p10_mean",
    "raw_negative_H_fraction_mean", "raw_negative_Qmag_fraction_mean",
    "final_step_pred_over_true_volume_mean",
)
ROLLOUT_INTEGER_KEYS = (
    "num_events", "num_steps", "steps_with_all_dry_prediction",
    "steps_with_zero_Qmag_prediction",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_defined(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else None


def event_summaries(rows: list[dict], events: list[dict], active: torch.Tensor, cell_area: float):
    by_event = {}
    for row in rows:
        by_event.setdefault(row["event_id"], []).append(row)
    result = []
    for event in events:
        event_id = event["event_id"]
        seq = by_event[event_id]
        if len(seq) != 24 or [x["target_time_index"] for x in seq] != list(range(1, 25)):
            raise ValueError(f"Expected 24 ordered rollout predictions for {event_id}")
        dry_rmse = [
            (event["H"][active, t].square().mean()).sqrt().item()
            for t in range(1, 25)
        ]
        h_rmse = sum(r["H_RMSE_active"] for r in seq) / len(seq)
        dry_mean = sum(dry_rmse) / len(dry_rmse)
        rec = {
            "event_id": event_id,
            "steps": len(seq),
            "H_RMSE_active_mean": h_rmse,
            "Qmag_RMSE_active_mean": sum(r["Qmag_RMSE_active"] for r in seq) / len(seq),
            "dry_H_RMSE_active_mean": dry_mean,
            "improvement_over_dry_percent": 100 * (dry_mean - h_rmse) / dry_mean if dry_mean else None,
            "CSI_0p01_mean": mean_defined([r["CSI_0p01"] for r in seq]),
            "CSI_0p05_mean": mean_defined([r["CSI_0p05"] for r in seq]),
            "CSI_0p10_mean": mean_defined([r["CSI_0p10"] for r in seq]),
            "steps_with_all_dry_prediction": sum(r["pred_positive_cells"] == 0 for r in seq),
            "steps_with_zero_Qmag_prediction": sum(r["pred_Qmag_max"] == 0 for r in seq),
            "raw_negative_Qmag_fraction_mean": sum(r["raw_negative_Qmag_fraction"] for r in seq) / 24,
            "true_final_volume_m3": seq[-1]["true_volume_m3"],
            "pred_final_volume_m3": seq[-1]["pred_volume_m3"],
            "final_volume_ratio": (
                seq[-1]["pred_volume_m3"] / seq[-1]["true_volume_m3"]
                if seq[-1]["true_volume_m3"] > 0 else None
            ),
        }
        result.append(rec)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", choices=tuple(EXPECTED), required=True)
    ap.add_argument("--split", choices=("valid", "test"), required=True)
    ap.add_argument("--audit-json", type=Path, default=None,
                    help="Required for test; must be successful validation evaluation of same checkpoint")
    args = ap.parse_args()

    dataset = args.dataset_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {output}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not str(checkpoint_path).endswith(EXPECTED[args.model]["checkpoint_suffix"]):
        raise ValueError("Checkpoint path does not match declared frozen experiment")
    if args.split == "test" and args.audit_json is None:
        raise ValueError("--split test requires --audit-json from the prior validation pass")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU CUDA required for this frozen test evaluation")
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)

    checkpoint_sha = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["m17k_config"]
    ck_args = checkpoint["args"]
    required = {
        "architecture": "rainfall_aware",
        "rainfall_alignment": "alternative",
        "rain_source_input": "trapezoid",
        "backbone_depth_mask": False,
        "output_constraint": "raw",
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Unexpected checkpoint {key}: {config.get(key)!r}, expected {expected!r}")
    if checkpoint.get("epoch") != 3 or checkpoint.get("best_epoch") != 3:
        raise ValueError("Expected validation-selected epoch 3 best checkpoint")
    if checkpoint["backbone_config"].get("K") != EXPECTED[args.model]["K"]:
        raise ValueError("Checkpoint K does not match experiment")
    if ck_args.get("K") != EXPECTED[args.model]["K"]:
        raise ValueError("Saved training K differs from requested model")
    if (ck_args.get("start_time_index"), ck_args.get("end_time_index")) != (0, 23):
        raise ValueError("Expected training and evaluation horizon T0->T24")
    if (ck_args.get("loss"), ck_args.get("loss_mask"), ck_args.get("select_metric")) != (
        "swegnn_rmse", "where_water", "rollout_swegnn_loss"
    ):
        raise ValueError("Unexpected loss or checkpoint selection method")
    for field in ("model_state_dict", "backbone_config", "train_event_ids", "valid_event_ids"):
        if field not in checkpoint:
            raise KeyError(f"Missing checkpoint field {field}")

    manifest = m17k.base.load_manifest(dataset)
    ids = {split: m17k.base.derive_split_event_ids(dataset, manifest, split)
           for split in ("train", "valid", "test")}
    assert len(ids["train"]) == 70 and len(ids["valid"]) == 15 and len(ids["test"]) == 15
    if set(ids["train"]) & set(ids["valid"]) or set(ids["train"]) & set(ids["test"]) or set(ids["valid"]) & set(ids["test"]):
        raise ValueError("Split overlap")
    if "R019" not in ids["test"]:
        raise ValueError("Expected test R019")
    if ids["train"] != sorted(checkpoint["train_event_ids"], key=m17k.base.natural_event_sort_key):
        raise ValueError("Checkpoint train IDs differ from manifest")
    if ids["valid"] != sorted(checkpoint["valid_event_ids"], key=m17k.base.natural_event_sort_key):
        raise ValueError("Checkpoint validation IDs differ from manifest")

    if args.split == "test":
        audit = json.loads(args.audit_json.expanduser().read_text())
        if audit.get("split") != "valid" or not audit.get("validation_audit_passed", False):
            raise ValueError("Validation audit must pass before opening test")
        if audit.get("checkpoint_sha256") != checkpoint_sha or audit.get("model") != args.model:
            raise ValueError("Validation audit does not match frozen checkpoint")
        if audit.get("manifest_sha256") != sha256(dataset / "manifest.json"):
            raise ValueError("Dataset manifest differs from audited validation run")
        if audit.get("script_sha256") != sha256(Path(__file__).resolve()):
            raise ValueError("Evaluation script changed since validation audit")

    static = m17k.base.load_static_graph(dataset)
    if float(static["cell_size"]) != 5.0:
        raise ValueError("Corrected 5m static graph required")
    static_dev = m17k.base.static_graph_to_device(static, device)
    active = m17k.active_mask_from_dem(static_dev["DEM"])
    if int(active.sum().item()) != 32249:
        raise ValueError("Expected 32249 active hydraulic nodes")
    ck_args = dict(ck_args)
    ck_args["swegnn_repo"] = Path(ck_args["swegnn_repo"]).expanduser()
    model_args = SimpleNamespace(**ck_args)
    model, backbone_config = m17k.build_model(model_args, static_dev, active, device)
    if backbone_config != checkpoint["backbone_config"]:
        raise ValueError("Built architecture differs from checkpoint")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    events = m17k.prepare_events(dataset, args.split, ids[args.split], device)
    time_indices = list(range(24))
    torch.cuda.reset_peak_memory_stats(device)
    loss_fn = m17k.make_loss_fn(model_args, active)
    tf = m17k.evaluate_teacher_forced(model, static_dev, events, time_indices, active, loss_fn, model_args)
    rollout, rows = m17k.evaluate_rollout(model, static_dev, events, time_indices, active, model_args)
    if rollout["num_events"] != 15 or rollout["num_steps"] != 360 or len(rows) != 360:
        raise ValueError("Incomplete 15-event, 24-step evaluation")
    by_event = event_summaries(rows, events, active, model_args.cell_area)
    dry_mean = sum(r["dry_H_RMSE_active_mean"] for r in by_event) / 15
    comparison = {
        "dry_H_RMSE_active_mean_m": dry_mean,
        "model_H_RMSE_active_mean_m": rollout["H_RMSE_active_mean"],
        "improvement_over_dry_percent": (
            100 * (dry_mean - rollout["H_RMSE_active_mean"]) / dry_mean if dry_mean else None
        ),
        "events_better_than_dry": sum(r["H_RMSE_active_mean"] < r["dry_H_RMSE_active_mean"] for r in by_event),
    }
    audit_passed = False
    if args.split == "valid":
        original = checkpoint["valid_summary"]
        if not isinstance(original, dict) or "rollout" not in original:
            raise ValueError("Checkpoint lacks original best-validation rollout summary")
        reference = original["rollout"]
        for key in ROLLOUT_NUMERIC_KEYS:
            a, b = rollout[key], reference[key]
            if not math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-5):
                raise AssertionError(f"Validation audit mismatch {key}: re-eval={a}, original={b}")
        for key in ROLLOUT_INTEGER_KEYS:
            if rollout[key] != reference[key]:
                raise AssertionError(f"Validation audit mismatch {key}: re-eval={rollout[key]}, original={reference[key]}")
        original_tf = original["teacher_forced"]
        for key in original_tf:
            a, b = tf[key], original_tf[key]
            if a is None or b is None:
                if a != b:
                    raise AssertionError(f"Validation audit mismatch TF {key}")
            elif not math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-5):
                raise AssertionError(f"Validation audit mismatch TF {key}: {a}, {b}")
        audit_passed = True

    script_path = Path(__file__).resolve()
    repo_dir = script_path.parent.parent
    git_revision = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
    summary = {
        "model": args.model, "split": args.split,
        "evaluation": "frozen checkpoint, no training or test-based selection",
        "validation_audit_passed": audit_passed,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_best_epoch": checkpoint["best_epoch"],
        "training_revision": config.get("git_commit"),
        "evaluation_revision": git_revision,
        "script_sha256": sha256(script_path),
        "manifest_sha256": sha256(dataset / "manifest.json"),
        "static_graph_sha256": sha256(dataset / "static_graph.pt"),
        "dataset_root": str(dataset),
        "architecture": checkpoint["backbone_config"],
        "model_configuration": {k: config[k] for k in required},
        "num_test_events_evaluated": 15 if args.split == "test" else 0,
        "time_range": "T0->T1 through T23->T24",
        "mask": "active = DEM != DEM.max(), 32249 cells",
        "aggregation": "mean of per-event/per-step active-cell RMSEs; CSI mean over defined steps",
        "autoregressive": rollout,
        "teacher_forced": tf,
        "dry_baseline": comparison,
        "max_memory_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }

    # Write outputs only when all checks and evaluation have succeeded.
    output.mkdir(parents=True)
    write_csv(output / f"{args.split}_rollout_steps.csv", m17k.ROLLOUT_FIELDNAMES, rows)
    write_csv(output / f"{args.split}_by_event.csv", list(by_event[0]), by_event)
    with (output / f"{args.split}_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(f"===== FROZEN {args.split.upper()} EVALUATION: {args.model} =====", flush=True)
    print(f"Training revision: {config.get('git_commit')}", flush=True)
    print(f"Checkpoint sha256: {checkpoint_sha}", flush=True)
    print(f"AR summary: {json.dumps(rollout, allow_nan=False)}", flush=True)
    print(f"TF summary: {json.dumps(tf, allow_nan=False)}", flush=True)
    print(f"Dry comparison: {json.dumps(comparison, allow_nan=False)}", flush=True)
    print(f"Validation audit passed: {audit_passed}", flush=True)
    print(f"Results saved in {output}", flush=True)


if __name__ == "__main__":
    main()
