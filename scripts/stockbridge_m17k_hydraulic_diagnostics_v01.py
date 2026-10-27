#!/usr/bin/env python3
"""Read-only M17K diagnostics: CityCAT wet-cell speeds (train), zero-H baseline (valid).

Uses the frozen Stockbridge Phase 2 SWE-GNN manifest and original CityCAT PTs.
No test events are opened. Does not train a model or modify any source data.

The speed is hypot(Vx, Vy) (m/s) from the canonical CityCAT PT tensor.
The characteristic proxy speed + sqrt(g*H) is NOT a flood-front velocity.
Exceedance fractions are reported per wet cell-time and weighted by the
specific discharge q = speed * H (share of the moving water that travels
faster than a threshold). Thresholds are the per-step reach of each model
divided by the 300 s step: SWE-GNN 2 layers x K hops x 5 m, and the measured
one-step reach of mSWE-GNN with 4 and 5 scales (proof of concept, 2026-10-27).
The dry baseline computes mean(per-event, per-step active-cell RMSE), exactly
matching M17K's rollout H_RMSE_active_mean aggregation (15*24 steps).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

NY, NX, NT = 305, 326, 25
NUM_NODES, ACTIVE_EXPECTED = NY * NX, 32249
G = 9.81
WET_H_M = 0.01
DT_S = 300.0
# label: per-step reach (m) of each model on the 5 m grid
REACH_M = {
    "SWE-GNN K=4 (2x4 hops)": 40.0,
    "SWE-GNN K=8 (2x8 hops)": 80.0,
    "reference 0.5 m/s": 150.0,
    "mSWE-GNN 4 scales (measured)": 309.0,
    "mSWE-GNN 5 scales (measured)": 487.0,
}
SPEED_THRESHOLDS = tuple(reach / DT_S for reach in REACH_M.values())
THRESHOLD_LABELS = tuple(REACH_M)
DEFAULT_ROOT = (
    Path.home() / "flood/02_data/CityCAT_Winchester/swegnn"
    / "stockbridge_phase2_100_5m_v01"
)
DEFAULT_OUT = (
    Path.home() / "flood/03_outputs/Stockbridge-SWEGNN"
    / "milestone17k_hydraulic_diagnostics_v01"
)


def load_pt(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Source does not exist: {path}")
    return torch.load(path, map_location="cpu")


def event_ids(manifest, root, split, count):
    split_ids = manifest.get("event_ids_by_split", {})
    if split_ids:
        ids = sorted(split_ids[split])
    else:
        ids = sorted(p.stem for p in (root / "events" / split).glob("*.pt"))
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError(f"{split}: expected {count} unique IDs, found {len(ids)}")
    return ids


def write_csv(path, fields, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_round(x, digits=9):
    return round(float(x), digits)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--mode", choices=("all", "dry", "velocities"), default="all")
    args = ap.parse_args()
    root, out = args.dataset_root.expanduser(), args.output_dir.expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {root}")
    if out.exists():
        raise FileExistsError(f"Output exists, refusing to overwrite: {out}")

    manifest_path = root / "manifest.json"
    manifest_csv_path = root / "manifest.csv"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    train_ids = event_ids(manifest, root, "train", 70)
    valid_ids = event_ids(manifest, root, "valid", 15)
    test_ids = event_ids(manifest, root, "test", 15)  # IDs only; no test PT files opened
    if (set(train_ids) & set(valid_ids) or set(train_ids) & set(test_ids)
            or set(valid_ids) & set(test_ids)):
        raise ValueError("Train/valid/test split leakage detected")
    if "R019" not in test_ids:
        raise ValueError("R019 is not in the reserved test split")

    static = load_pt(root / "static_graph.pt")
    if tuple(static["DEM"].shape) != (NUM_NODES,):
        raise ValueError("Unexpected static DEM size")
    if float(static["cell_size"]) != 5.0:
        raise ValueError("Expected 5-m graph")
    active = (static["DEM"] != static["DEM"].max()).bool()
    n_active = int(active.sum())
    if n_active != ACTIVE_EXPECTED:
        raise ValueError(f"Expected {ACTIVE_EXPECTED} active nodes, got {n_active}")
    active_grid = active.view(NY, NX, 1)
    print(f"Dataset: {root}", flush=True)
    print(f"Splits: train={len(train_ids)} valid={len(valid_ids)} test={len(test_ids)} [metadata only]", flush=True)
    print(f"Active cells: {n_active}/{NUM_NODES} | Time: T1..T24", flush=True)
    # Check raw PT path provenance before creating any outputs.
    raw_train_paths = {}
    if args.mode in ("all", "velocities"):
        with manifest_csv_path.open(newline="") as f:
            raw_train_rows = list(csv.DictReader(f))
        by_id = {r["event_id"]: r for r in raw_train_rows if r["split"] == "train"}
        if set(by_id) != set(train_ids):
            raise ValueError("Train IDs in manifest.csv do not agree with manifest.json")
        for event_id in train_ids:
            path = Path(by_id[event_id]["source_path"]).expanduser()
            if not path.is_absolute():
                path = (manifest_csv_path.parent / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Original CityCAT PT for train event {event_id} is missing: {path}")
            raw_train_paths[event_id] = path

    out.mkdir(parents=True)
    summary = {
        "purpose": "M17K hydraulic diagnostics; no training and no test data access",
        "dataset_root": str(root),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "train_ids": train_ids, "valid_ids": valid_ids,
        "test_ids_loaded": False, "active_mask": "DEM != DEM.max()",
        "n_active": n_active, "n_nodes": NUM_NODES,
        "time_indices": "T1..T24 inclusive (24 records, 5-min spacing)",
        "torch_version": torch.__version__,
    }
    try:
        result = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent.parent),
                                 "rev-parse", "HEAD"], text=True, capture_output=True, check=True)
        summary["git_commit_if_in_repo_scripts"] = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        summary["git_commit_if_in_repo_scripts"] = None

    if args.mode in ("all", "dry"):
        print("\n===== DRY BASELINE / VALIDATION =====", flush=True)
        dry_step_rows, dry_event_rows = [], []
        all_rmse = []
        sum_sq, n_sample = 0.0, 0
        for i, event_id in enumerate(valid_ids, 1):
            obj = load_pt(root / "events" / "valid" / f"{event_id}.pt")
            H = obj["H"].float()
            if tuple(H.shape) != (NUM_NODES, NT) or not torch.isfinite(H).all():
                raise ValueError(f"Invalid H for {event_id}: {tuple(H.shape)}")
            h = H[active, 1:NT]  # T1 through T24
            rmse_steps = torch.sqrt(h.square().mean(dim=0))
            for t, v in enumerate(rmse_steps.tolist(), start=1):
                dry_step_rows.append({"event_id": event_id, "target_time_index": t,
                                      "dry_H_RMSE_active_m": v})
            dry_event_rows.append({"event_id": event_id,
                                   "dry_H_RMSE_active_mean_m": rmse_steps.mean().item(),
                                   "wet_cell_times_H_ge_0p01": int((h >= WET_H_M).sum())})
            all_rmse.extend(rmse_steps.tolist())
            sum_sq += h.double().square().sum().item()
            n_sample += h.numel()
            print(f"  valid {i:02d}/15 {event_id}: mean dry RMSE {rmse_steps.mean().item():.6f} m", flush=True)
        mean_step_rmse = sum(all_rmse) / len(all_rmse)
        pooled_rmse = math.sqrt(sum_sq / n_sample)
        summary["dry_validation"] = {
            "num_events": 15, "num_steps": 360,
            "H_RMSE_active_mean_m__M17K_aggregation": mean_step_rmse,
            "H_RMSE_active_pooled_m__different_aggregation": pooled_rmse,
            "CSI_wet_thresholds": "0 if reference has any flooded cells; undefined when both dry",
            "baseline_prediction": "H=0 at all active nodes for all 24 predicted times",
        }
        write_csv(out / "dry_valid_by_event.csv", list(dry_event_rows[0]), dry_event_rows)
        write_csv(out / "dry_valid_by_step.csv", list(dry_step_rows[0]), dry_step_rows)
        print(f"Mean per-step active RMSE: {mean_step_rmse:.8f} m [comparable to M17K]", flush=True)
        print(f"Pooled active RMSE:        {pooled_rmse:.8f} m [different aggregation]", flush=True)

    if args.mode in ("all", "velocities"):
        print("\n===== VELOCITY DISTRIBUTION / TRAIN ONLY =====", flush=True)
        speed_arrays, characteristic_arrays = [], []
        event_rows = []
        total_wet = 0
        total_flux = 0.0
        exceed_v = [0] * len(SPEED_THRESHOLDS)
        exceed_c = [0] * len(SPEED_THRESHOLDS)
        exceed_q = [0.0] * len(SPEED_THRESHOLDS)
        for i, event_id in enumerate(train_ids, 1):
            path = raw_train_paths[event_id]
            raw = load_pt(path)
            if not isinstance(raw, torch.Tensor) or tuple(raw.shape) != (NY, NX, NT, 5):
                raise ValueError(f"Invalid source tensor for {event_id}: {path}")
            H = raw[..., 0]
            V = torch.hypot(raw[..., 1], raw[..., 2])
            if not torch.isfinite(H).all() or not torch.isfinite(V).all():
                raise ValueError(f"Nonfinite values in CityCAT tensor: {event_id}")
            wet = (H[:, :, 1:] >= WET_H_M) & active_grid
            v = V[:, :, 1:][wet].float()
            h = H[:, :, 1:][wet].float()
            c = v + (G * h.clamp_min(0)).sqrt()
            q = (v * h).double()  # specific discharge magnitude, m2/s
            event_flux = q.sum().item()
            del H, V, wet, raw
            if v.numel() == 0:
                raise ValueError(f"No wet active cells at T1..T24 for {event_id}")
            speed_arrays.append(v)
            characteristic_arrays.append(c)
            total_wet += v.numel()
            total_flux += event_flux
            row = {"event_id": event_id, "wet_cell_times": v.numel(),
                   "speed_mean_mps": v.mean().item(), "speed_max_mps": v.max().item(),
                   "characteristic_mean_mps": c.mean().item(),
                   "characteristic_max_mps": c.max().item()}
            for j, threshold in enumerate(SPEED_THRESHOLDS):
                fast = v > threshold
                nv, nc = int(fast.sum()), int((c > threshold).sum())
                fq = q[fast].sum().item()
                exceed_v[j] += nv
                exceed_c[j] += nc
                exceed_q[j] += fq
                row[f"speed_frac_gt_{threshold:.6f}_mps"] = nv / v.numel()
                row[f"flux_frac_gt_{threshold:.6f}_mps"] = (
                    fq / event_flux if event_flux > 0 else None
                )
                row[f"characteristic_frac_gt_{threshold:.6f}_mps"] = nc / c.numel()
            event_rows.append(row)
            print(f"  train {i:02d}/70 {event_id}: {v.numel():,} wet cell-times", flush=True)
        # numpy instead of torch.quantile: some PyTorch versions reject
        # torch.quantile on inputs above ~16 million elements.
        all_v = torch.cat(speed_arrays).numpy()
        all_c = torch.cat(characteristic_arrays).numpy()
        qs = [0.5, 0.9, 0.95, 0.99]
        quant_v = np.quantile(all_v, qs).tolist()
        quant_c = np.quantile(all_c, qs).tolist()
        velocity = {
            "events": len(train_ids), "wet_cell_times": total_wet,
            "wet_mask": "active & (CityCAT H >= 0.01 m), T1..T24",
            "speed": "hypot(Vx,Vy) from original CityCAT PT, m/s",
            "characteristic": "speed + sqrt(9.81 * H), m/s (approx. local wave characteristic, not flood-front speed)",
            "aggregation": (
                "speed/characteristic fractions: wet-cell-time weighted; flux "
                "fraction: weighted by q = speed*H; all 70 train events pooled"
            ),
            "speed_quantiles_mps": dict(zip(("p50", "p90", "p95", "p99"), quant_v)),
            "characteristic_quantiles_mps": dict(zip(("p50", "p90", "p95", "p99"), quant_c)),
            "thresholds": [
                {"label": THRESHOLD_LABELS[j], "reach_m_per_step": REACH_M[THRESHOLD_LABELS[j]],
                 "mps": threshold, "speed_fraction_gt": exceed_v[j] / total_wet,
                 "flux_fraction_gt": exceed_q[j] / total_flux if total_flux > 0 else None,
                 "characteristic_fraction_gt": exceed_c[j] / total_wet}
                for j, threshold in enumerate(SPEED_THRESHOLDS)
            ],
        }
        summary["train_velocities"] = velocity
        write_csv(out / "train_velocities_by_event.csv", list(event_rows[0]), event_rows)
        print("Speed quantiles:", velocity["speed_quantiles_mps"], flush=True)
        print("Characteristic quantiles:", velocity["characteristic_quantiles_mps"], flush=True)
        print(f"  {'model reach':32s} {'m/s':>6s} | cells faster | flux faster | characteristic", flush=True)
        for x in velocity["thresholds"]:
            flux = x["flux_fraction_gt"]
            flux_text = "n/a" if flux is None else f"{flux:.1%}"
            print(f"  {x['label']:32s} {x['mps']:6.3f} | {x['speed_fraction_gt']:12.1%} | "
                  f"{flux_text:>11s} | {x['characteristic_fraction_gt']:.1%}", flush=True)

    with (out / "m17k_diagnostics_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDONE: {out}", flush=True)


if __name__ == "__main__":
    main()
