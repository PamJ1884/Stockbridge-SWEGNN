import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from plot_stockbridge_validation_extent_confusion import (
    METRICS_FIELDNAMES as EXTENT_METRICS_FIELDNAMES,
    calculate_extent_metrics,
    metric_text,
    parse_thresholds,
    save_confusion_map,
    save_extent_comparison,
)
from plot_stockbridge_validation_rollout_diagnostics import (
    EDGE_FEATURES,
    EVENT_SUMMARY_FIELDNAMES,
    INPUT_FEATURES,
    OUTPUT_VARIABLES,
    TIMESTEP_METRICS_FIELDNAMES,
    create_event_figures,
    evaluate_event_mode,
    load_and_validate_checkpoint,
    load_dynamic_event,
    load_static_graph,
    mean_optional,
    mean_required,
    natural_event_sort_key,
    prepare_dynamic_events,
    select_device,
    static_graph_to_device,
    write_metrics_csv,
)


DATASET_ROOT = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/swegnn"
    / "stockbridge_phase2_swegnn_full_v01"
)
SWEGNN_REPO = Path.home() / "flood/01_repositories/SWE-GNN-paper-repository"
DEFAULT_CHECKPOINT = (
    Path.home()
    / "flood/03_outputs/Stockbridge-SWEGNN"
    / "milestone14b3_autoregressive_finetune_v01"
    / "milestone14b3_best_checkpoint.pt"
)
DEFAULT_OUTPUT_DIR = (
    Path.home()
    / "flood/03_outputs/Stockbridge-SWEGNN"
    / "milestone15a_test_r019_evaluation_v01"
)

NY = 305
NX = 326
NT = 25
NUM_NODES = 99430
NUM_EDGES = 396458
TIME_STEP_MINUTES = 5
EXPECTED_TEST_COUNT = 15
ALLOWED_MODES = ("teacher_forced", "autoregressive")
DEFAULT_CSI_THRESHOLDS = (
    ("CSI_0p05", 0.05),
    ("CSI_0p10", 0.10),
    ("CSI_0p20", 0.20),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for Milestone 15A."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the final SWE-GNN candidate on Stockbridge test events "
            "and create detailed R019 diagnostics."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--swegnn-repo", type=Path, default=SWEGNN_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-time-index", type=int, default=2)
    parser.add_argument("--end-time-index", type=int, default=16)
    parser.add_argument("--rainfall-scale", type=float, default=300000.0)
    parser.add_argument("--hid-features", type=int, default=16)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument(
        "--modes",
        default="teacher_forced,autoregressive",
    )
    parser.add_argument("--thresholds", default="0.05,0.10,0.20")
    parser.add_argument(
        "--r019-only-figures",
        action="store_true",
        help="Evaluate only R019 instead of all test events.",
    )
    parser.add_argument(
        "--clamp-output",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--clamp-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--max-depth-vmax", type=float, default=0.0)
    parser.add_argument("--error-vmax", type=float, default=0.0)
    return parser.parse_args()


def parse_modes(raw_modes: str) -> list[str]:
    """Parse ordered, duplicate-free SWE-GNN evaluation modes."""
    modes = [value.strip() for value in raw_modes.split(",") if value.strip()]
    if not modes:
        raise ValueError("--modes must contain at least one mode")
    if len(set(modes)) != len(modes):
        raise ValueError(f"--modes contains duplicates: {modes}")
    unsupported = [mode for mode in modes if mode not in ALLOWED_MODES]
    if unsupported:
        raise ValueError(
            f"Unsupported --modes values {unsupported}; allowed: {ALLOWED_MODES}"
        )
    return modes


def require_keys(mapping: dict, keys: tuple[str, ...], label: str) -> None:
    """Raise a contextual error for absent serialized fields."""
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"{label} is missing required keys: {missing}")


def validate_manifest_value(
    manifest: dict,
    key: str,
    expected_value,
) -> None:
    """Validate one required manifest value."""
    if manifest[key] != expected_value:
        raise ValueError(
            f"Expected manifest {key}={expected_value!r}, "
            f"got {manifest[key]!r}"
        )


def load_test_manifest(dataset_root: Path) -> dict:
    """Load and validate test-relevant fields in manifest.json."""
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest JSON not found: {manifest_path}")
    with manifest_path.open() as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, dict):
        raise TypeError("manifest.json must contain an object")
    require_keys(
        manifest,
        (
            "split_counts",
            "input_features",
            "output_variables",
            "num_nodes",
            "num_edges",
        ),
        "manifest JSON",
    )
    split_counts = manifest["split_counts"]
    if not isinstance(split_counts, dict) or split_counts.get("test") != 15:
        raise ValueError(
            "Expected manifest split_counts['test']=15, got "
            f"{split_counts!r}"
        )
    for key, expected_value in {
        "input_features": INPUT_FEATURES,
        "output_variables": OUTPUT_VARIABLES,
        "num_nodes": NUM_NODES,
        "num_edges": NUM_EDGES,
    }.items():
        validate_manifest_value(manifest, key, expected_value)
    if "edge_features" in manifest:
        validate_manifest_value(manifest, "edge_features", EDGE_FEATURES)
    if "total_events" in manifest:
        validate_manifest_value(manifest, "total_events", 100)
    if "contains_R019_in_test" in manifest:
        validate_manifest_value(manifest, "contains_R019_in_test", True)
    return manifest


def derive_test_event_ids(
    dataset_root: Path,
    manifest: dict,
    r019_only: bool,
) -> list[str]:
    """Resolve, naturally sort, and validate Stockbridge test event IDs."""
    event_ids_by_split = manifest.get("event_ids_by_split")
    if event_ids_by_split is not None:
        if not isinstance(event_ids_by_split, dict):
            raise TypeError("manifest event_ids_by_split must be a dictionary")
        if "test" not in event_ids_by_split:
            raise KeyError("manifest event_ids_by_split is missing 'test'")
        event_ids = event_ids_by_split["test"]
        if not isinstance(event_ids, list) or not all(
            isinstance(event_id, str) for event_id in event_ids
        ):
            raise TypeError(
                "manifest event_ids_by_split['test'] must be a string list"
            )
    else:
        test_dir = dataset_root / "events" / "test"
        if not test_dir.is_dir():
            raise FileNotFoundError(f"Test event directory not found: {test_dir}")
        event_ids = [path.stem for path in test_dir.glob("*.pt")]

    sorted_event_ids = sorted(event_ids, key=natural_event_sort_key)
    if len(set(sorted_event_ids)) != len(sorted_event_ids):
        raise ValueError("Duplicate event IDs in test split")
    if "R019" not in sorted_event_ids:
        raise ValueError("R019 must be present in the test split")
    r019_path = dataset_root / "events" / "test" / "R019.pt"
    if not r019_path.is_file():
        raise FileNotFoundError(f"R019 test event file not found: {r019_path}")
    if not r019_only and len(sorted_event_ids) != EXPECTED_TEST_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TEST_COUNT} test events, "
            f"found {len(sorted_event_ids)}"
        )
    event_ids_to_check = ["R019"] if r019_only else sorted_event_ids
    missing_paths = [
        dataset_root / "events" / "test" / f"{event_id}.pt"
        for event_id in event_ids_to_check
        if not (
            dataset_root / "events" / "test" / f"{event_id}.pt"
        ).is_file()
    ]
    if missing_paths:
        formatted = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing test event files:\n{formatted}")
    return sorted_event_ids


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate numerical evaluation and plotting arguments."""
    if args.start_time_index < 0:
        raise ValueError("--start-time-index must be nonnegative")
    if args.end_time_index >= NT - 1:
        raise ValueError(
            f"--end-time-index must be less than {NT - 1}; "
            f"got {args.end_time_index}"
        )
    if args.start_time_index > args.end_time_index:
        raise ValueError("--start-time-index must not exceed --end-time-index")
    if not math.isfinite(args.rainfall_scale):
        raise ValueError("--rainfall-scale must be finite")
    if args.hid_features <= 0:
        raise ValueError("--hid-features must be greater than zero")
    if args.K <= 0:
        raise ValueError("--K must be greater than zero")
    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than zero")
    for name, value in {
        "--max-depth-vmax": args.max_depth_vmax,
        "--error-vmax": args.error_vmax,
    }.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")


def validate_checkpoint_range(
    checkpoint_metadata: dict,
    start_time_index: int,
    end_time_index: int,
) -> None:
    """Ensure the checkpoint training range supports the evaluation range."""
    checkpoint_start = checkpoint_metadata["checkpoint_start_time_index"]
    checkpoint_end = checkpoint_metadata["checkpoint_end_time_index"]
    if start_time_index < checkpoint_start or end_time_index > checkpoint_end:
        raise ValueError(
            "Checkpoint time range does not support requested evaluation: "
            f"checkpoint T{checkpoint_start}..T{checkpoint_end}, requested "
            f"T{start_time_index}..T{end_time_index}"
        )


def prepare_output_dir(
    output_dir: Path,
    dataset_root: Path,
    swegnn_repo: Path,
    checkpoint_path: Path,
    overwrite: bool,
) -> None:
    """Create or safely replace only the requested Milestone 15A output."""
    output_exists = output_dir.exists() or output_dir.is_symlink()
    if output_exists and output_dir.is_symlink():
        raise ValueError(f"Refusing symlink --output-dir: {output_dir}")
    if output_exists and not output_dir.is_dir():
        raise ValueError(f"--output-dir is not a directory: {output_dir}")
    if output_exists and not overwrite and any(output_dir.iterdir()):
        raise FileExistsError(
            "Refusing to overwrite nonempty Milestone 15A output without "
            f"--overwrite: {output_dir}"
        )
    if output_exists and overwrite:
        resolved_output = output_dir.resolve()
        protected_paths = {
            Path("/").resolve(),
            Path.home().resolve(),
            dataset_root.resolve(),
            swegnn_repo.resolve(),
            Path(__file__).resolve().parents[1],
            checkpoint_path.resolve().parent,
            Path.cwd().resolve(),
        }
        if resolved_output in protected_paths:
            raise ValueError(
                f"Refusing unsafe --overwrite target: {resolved_output}"
            )
        if any(
            resolved_output in protected_path.parents
            for protected_path in protected_paths
        ):
            raise ValueError(
                f"Refusing to delete parent of protected path: {resolved_output}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def validate_r019_prediction_bundle(prediction: dict) -> None:
    """Validate all required R019 bundle fields before serialization."""
    required_keys = (
        "event_id",
        "mode",
        "split",
        "start_time_index",
        "end_time_index",
        "target_time_indices",
        "target_minutes",
        "pred_H",
        "pred_Qmag",
        "true_H",
        "true_Qmag",
        "rainfall_global",
        "rainfall_scale",
        "clamp_output",
        "clamp_feedback",
        "input_features",
        "edge_features",
        "output_variables",
        "checkpoint_path",
        "checkpoint_type",
        "checkpoint_best_epoch",
    )
    require_keys(prediction, required_keys, "R019 prediction bundle")
    if prediction["event_id"] != "R019" or prediction["split"] != "test":
        raise ValueError("Saved R019 prediction must identify test/R019")
    num_steps = len(prediction["target_time_indices"])
    expected_shape = (num_steps, NUM_NODES)
    for field in ("pred_H", "pred_Qmag", "true_H", "true_Qmag"):
        tensor = prediction[field]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"R019 {field} must be a tensor")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Expected R019 {field} shape {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.device.type != "cpu":
            raise ValueError(f"Saved R019 {field} is not on CPU")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Saved R019 {field} contains NaN/inf")
    rainfall = prediction["rainfall_global"]
    if not isinstance(rainfall, torch.Tensor):
        raise TypeError("R019 rainfall_global must be a tensor")
    if tuple(rainfall.shape) != (NT,) or rainfall.device.type != "cpu":
        raise ValueError("R019 rainfall_global must be a length-25 CPU tensor")
    if not torch.isfinite(rainfall).all().item():
        raise ValueError("R019 rainfall_global contains NaN/inf")
    if prediction["input_features"] != INPUT_FEATURES:
        raise ValueError("R019 input_features do not match expected order")
    if prediction["edge_features"] != EDGE_FEATURES:
        raise ValueError("R019 edge_features do not match expected order")
    if prediction["output_variables"] != OUTPUT_VARIABLES:
        raise ValueError("R019 output_variables do not match expected order")


def save_r019_prediction(
    prediction: dict,
    output_path: Path,
    checkpoint_path: Path,
    checkpoint_metadata: dict,
) -> None:
    """Add Milestone 15A provenance and save one R019 CPU bundle."""
    prediction.update(
        {
            "split": "test",
            "edge_features": EDGE_FEATURES,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_type": checkpoint_metadata["checkpoint_type"],
            "checkpoint_best_epoch": checkpoint_metadata[
                "checkpoint_best_epoch"
            ],
        }
    )
    validate_r019_prediction_bundle(prediction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prediction, output_path)
    if not output_path.is_file():
        raise RuntimeError(f"R019 prediction was not saved: {output_path}")


def write_extent_metrics_csv(output_path: Path, rows: list[dict]) -> None:
    """Write the complete ordered R019 extent-confusion table."""
    if not rows:
        raise ValueError("Cannot write an empty R019 extent metrics CSV")
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=EXTENT_METRICS_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)


def calculate_aggregate_test_metrics(
    event_summaries: list[dict],
    modes: list[str],
) -> list[dict]:
    """Average event summary metrics separately for each evaluation mode."""
    required_metrics = (
        "mean_combined_MSE",
        "mean_H_RMSE",
        "mean_H_MAE",
        "mean_Qmag_RMSE",
        "mean_Qmag_MAE",
    )
    optional_metrics = (
        "H_peak_ratio",
        "mean_CSI_0p05",
        "mean_CSI_0p10",
        "mean_CSI_0p20",
    )
    aggregates = []
    for mode in modes:
        selected = [row for row in event_summaries if row["mode"] == mode]
        if not selected:
            raise ValueError(f"No event summaries for mode {mode}")
        aggregate = {"mode": mode, "num_events": len(selected)}
        for field in required_metrics:
            aggregate[field] = mean_required(
                [float(row[field]) for row in selected],
                f"test {mode} {field}",
            )
        aggregate["mean_H_peak_ratio"] = mean_optional(
            [row["H_peak_ratio"] for row in selected]
        )
        for field in optional_metrics[1:]:
            aggregate[field] = mean_optional(
                [row[field] for row in selected]
            )
        aggregates.append(aggregate)
    return aggregates


def values_by_mode(
    rows: list[dict],
    modes: list[str],
    metric: str,
) -> list[float]:
    """Resolve one optional metric into plotting values in mode order."""
    lookup = {row["mode"]: row for row in rows}
    return [
        math.nan if lookup[mode][metric] is None else float(lookup[mode][metric])
        for mode in modes
    ]


def plot_mode_metric(
    rows: list[dict],
    modes: list[str],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot one aggregate metric as a bar for each evaluation mode."""
    figure, axis = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    axis.bar(modes, values_by_mode(rows, modes, metric))
    axis.set(title=title, xlabel="Evaluation mode", ylabel=ylabel)
    axis.grid(axis="y", alpha=0.3)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_test_csi_by_mode_threshold(
    aggregates: list[dict],
    modes: list[str],
    output_path: Path,
    dpi: int,
) -> None:
    """Plot aggregate test mean CSI for the three standard thresholds."""
    lookup = {row["mode"]: row for row in aggregates}
    positions = list(range(len(modes)))
    width = 0.8 / len(DEFAULT_CSI_THRESHOLDS)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for index, (field, threshold) in enumerate(DEFAULT_CSI_THRESHOLDS):
        offset = (index - (len(DEFAULT_CSI_THRESHOLDS) - 1) / 2.0) * width
        values = [
            math.nan
            if lookup[mode][f"mean_{field}"] is None
            else float(lookup[mode][f"mean_{field}"])
            for mode in modes
        ]
        axis.bar(
            [position + offset for position in positions],
            values,
            width=width,
            label=f"H >= {threshold:.2f} m",
        )
    axis.set(
        title="Stockbridge test mean CSI by mode and threshold",
        xlabel="Evaluation mode",
        ylabel="Mean CSI",
        ylim=(0.0, 1.0),
        xticks=positions,
        xticklabels=modes,
    )
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_r019_peak_ratio(
    r019_summaries: list[dict],
    modes: list[str],
    output_path: Path,
    dpi: int,
) -> None:
    """Plot R019 evaluated-interval water-depth peak ratio by mode."""
    figure, axis = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    axis.bar(
        modes,
        values_by_mode(r019_summaries, modes, "H_peak_ratio"),
    )
    axis.axhline(1.0, color="black", linestyle="--", label="Ratio = 1")
    axis.set(
        title="R019 H peak ratio by mode",
        xlabel="Evaluation mode",
        ylabel="Predicted / true peak H",
    )
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_r019_extent_metric(
    extent_rows: list[dict],
    modes: list[str],
    thresholds: list[tuple[float, str]],
    metric: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot R019 maximum/final extent metrics by mode and threshold."""
    series = [
        (mode, stage)
        for mode in modes
        for stage in ("maximum", "final")
    ]
    lookup = {
        (row["mode"], row["stage"], row["threshold_m"]): row[metric]
        for row in extent_rows
    }
    positions = list(range(len(thresholds)))
    width = 0.82 / len(series)
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for index, (mode, stage) in enumerate(series):
        offset = (index - (len(series) - 1) / 2.0) * width
        values = []
        for threshold, _ in thresholds:
            value = lookup[(mode, stage, threshold)]
            values.append(math.nan if value is None else float(value))
        axis.bar(
            [position + offset for position in positions],
            values,
            width=width,
            label=f"{mode} / {stage}",
        )
    axis.set(
        title=f"R019 extent {metric} by threshold",
        xlabel="Flood-depth threshold (m)",
        ylabel=metric,
        xticks=positions,
        xticklabels=[f"{value:.2f}" for value, _ in thresholds],
    )
    if metric == "CSI":
        axis.set_ylim(0.0, 1.0)
    else:
        finite_values = [
            float(value)
            for value in lookup.values()
            if value is not None
        ]
        upper = max([1.25, *(value * 1.15 for value in finite_values)])
        axis.set_ylim(0.0, upper)
        axis.axhline(1.0, color="black", linestyle="--", label="Bias = 1")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def create_summary_figures(
    aggregates: list[dict],
    event_summaries: list[dict],
    extent_rows: list[dict],
    modes: list[str],
    thresholds: list[tuple[float, str]],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    """Create all six required test and R019 summary figures."""
    r019_summaries = [
        row for row in event_summaries if row["event_id"] == "R019"
    ]
    paths = [
        output_dir / "test_mean_H_RMSE_by_mode.png",
        output_dir / "test_mean_combined_MSE_by_mode.png",
        output_dir / "test_mean_CSI_by_mode_threshold.png",
        output_dir / "r019_H_peak_ratio_by_mode.png",
        output_dir / "r019_extent_CSI_by_threshold.png",
        output_dir / "r019_extent_Bias_by_threshold.png",
    ]
    plot_mode_metric(
        aggregates,
        modes,
        "mean_H_RMSE",
        "Mean H RMSE",
        "Stockbridge test mean H RMSE by mode",
        paths[0],
        dpi,
    )
    plot_mode_metric(
        aggregates,
        modes,
        "mean_combined_MSE",
        "Mean combined MSE",
        "Stockbridge test mean combined MSE by mode",
        paths[1],
        dpi,
    )
    plot_test_csi_by_mode_threshold(aggregates, modes, paths[2], dpi)
    plot_r019_peak_ratio(r019_summaries, modes, paths[3], dpi)
    plot_r019_extent_metric(
        extent_rows,
        modes,
        thresholds,
        "CSI",
        paths[4],
        dpi,
    )
    plot_r019_extent_metric(
        extent_rows,
        modes,
        thresholds,
        "Bias",
        paths[5],
        dpi,
    )
    return paths


def print_event_summary(summary: dict) -> None:
    """Print the requested concise per-event evaluation diagnostics."""
    print(
        f"{summary['event_id']} | {summary['mode']} | "
        f"mean_H_RMSE={summary['mean_H_RMSE']:.8g} | "
        f"mean_combined_MSE={summary['mean_combined_MSE']:.8g} | "
        f"H_peak_ratio={metric_text(summary['H_peak_ratio'])} | "
        f"mean_CSI="
        f"{metric_text(summary['mean_CSI_0p05'])}/"
        f"{metric_text(summary['mean_CSI_0p10'])}/"
        f"{metric_text(summary['mean_CSI_0p20'])}"
    )


def print_aggregate_metrics(aggregates: list[dict]) -> None:
    """Print aggregate test metrics separately for each selected mode."""
    print("Aggregate test metrics by mode:")
    for aggregate in aggregates:
        print(
            f"  {aggregate['mode']} | "
            f"mean_H_RMSE={aggregate['mean_H_RMSE']:.8g} | "
            f"mean_combined_MSE={aggregate['mean_combined_MSE']:.8g} | "
            f"mean_H_peak_ratio="
            f"{metric_text(aggregate['mean_H_peak_ratio'])} | "
            f"mean_CSI="
            f"{metric_text(aggregate['mean_CSI_0p05'])}/"
            f"{metric_text(aggregate['mean_CSI_0p10'])}/"
            f"{metric_text(aggregate['mean_CSI_0p20'])}"
        )


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    validate_arguments(args)
    dataset_root = args.dataset_root.expanduser()
    swegnn_repo = args.swegnn_repo.expanduser()
    checkpoint_path = args.checkpoint.expanduser()
    output_dir = args.output_dir.expanduser()
    modes = parse_modes(args.modes)
    thresholds = parse_thresholds(args.thresholds)
    device = select_device(args.device, args.allow_cpu)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not swegnn_repo.is_dir():
        raise FileNotFoundError(f"Original SWE-GNN repo not found: {swegnn_repo}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    manifest = load_test_manifest(dataset_root)
    test_event_ids = derive_test_event_ids(
        dataset_root,
        manifest,
        args.r019_only_figures,
    )
    evaluated_event_ids = (
        ["R019"] if args.r019_only_figures else test_event_ids
    )
    static_graph = load_static_graph(dataset_root)
    checkpoint, checkpoint_metadata = load_and_validate_checkpoint(
        checkpoint_path,
        device,
        args,
    )
    validate_checkpoint_range(
        checkpoint_metadata,
        args.start_time_index,
        args.end_time_index,
    )
    if checkpoint_metadata["checkpoint_type"] != "14B-3":
        print(
            "WARNING: using fallback checkpoint type "
            f"{checkpoint_metadata['checkpoint_type']} instead of 14B-3"
        )
    checkpoint_rainfall_scale = checkpoint_metadata[
        "checkpoint_rainfall_scale"
    ]
    if args.rainfall_scale != checkpoint_rainfall_scale:
        print(
            "WARNING: evaluation rainfall scale differs from checkpoint: "
            f"{args.rainfall_scale} vs {checkpoint_rainfall_scale}"
        )

    time_indices = list(
        range(args.start_time_index, args.end_time_index + 1)
    )
    target_time_indices = [time_index + 1 for time_index in time_indices]
    target_minutes = [
        time_index * TIME_STEP_MINUTES for time_index in target_time_indices
    ]
    print("=== Stockbridge SWE-GNN milestone 15A test/R019 evaluation ===")
    print("Dataset root:", dataset_root)
    print("Checkpoint path:", checkpoint_path)
    print("Output directory:", output_dir)
    print("Device:", device)
    print("Test events:", test_event_ids)
    print("Modes:", modes)
    print("Thresholds:", [value for value, _ in thresholds])
    print(
        "Evaluation range: "
        f"T{args.start_time_index}->T{args.start_time_index + 1} through "
        f"T{args.end_time_index}->T{args.end_time_index + 1}"
    )
    print("Target minutes:", target_minutes)
    print("R019 present:", "R019" in test_event_ids)

    prepare_output_dir(
        output_dir,
        dataset_root,
        swegnn_repo,
        checkpoint_path,
        args.overwrite,
    )
    predictions_dir = output_dir / "predictions"
    figures_dir = output_dir / "figures"
    r019_maps_dir = figures_dir / "r019_maps"
    r019_timeseries_dir = figures_dir / "r019_timeseries"
    r019_extent_dir = figures_dir / "r019_extent_confusion"
    summary_figures_dir = figures_dir / "summary"
    for directory in (
        predictions_dir,
        r019_maps_dir,
        r019_timeseries_dir,
        r019_extent_dir,
        summary_figures_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    static_device_graph = static_graph_to_device(static_graph, device)
    prepared_events = prepare_dynamic_events(
        dataset_root,
        "test",
        evaluated_event_ids,
        device,
    )

    swegnn_repo_path = str(swegnn_repo.resolve())
    if swegnn_repo_path not in sys.path:
        sys.path.insert(0, swegnn_repo_path)
    from models.gnn import GNN

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = GNN(
        node_features=7,
        edge_features=3,
        type_GNN="SWEGNN",
        hid_features=args.hid_features,
        K=args.K,
        gnn_activation="tanh",
        dropout=0.0,
        mlp_layers=2,
        mlp_activation="prelu",
        seed=args.seed,
        with_filter_matrix=True,
        with_gradient=True,
        with_WL=True,
        previous_t=1,
        device=device,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.eval()

    all_timestep_metrics = []
    all_event_summaries = []
    prediction_paths = []
    map_paths = []
    timeseries_paths = []
    extent_figure_paths = []
    r019_extent_rows = []
    for event in prepared_events:
        for mode in modes:
            timestep_metrics, event_summary, prediction = evaluate_event_mode(
                model,
                static_device_graph,
                event,
                time_indices,
                args.rainfall_scale,
                mode,
                args.clamp_output,
                args.clamp_feedback,
            )
            all_timestep_metrics.extend(timestep_metrics)
            all_event_summaries.append(event_summary)
            print_event_summary(event_summary)

            if event["event_id"] == "R019":
                prediction_path = (
                    predictions_dir / mode / "R019.pt"
                )
                save_r019_prediction(
                    prediction,
                    prediction_path,
                    checkpoint_path,
                    checkpoint_metadata,
                )
                prediction_paths.append(prediction_path)
                event_map_paths, event_timeseries_paths = create_event_figures(
                    prediction,
                    timestep_metrics,
                    checkpoint_path,
                    r019_maps_dir,
                    r019_timeseries_dir,
                    args.max_depth_vmax,
                    args.error_vmax,
                    args.dpi,
                )
                map_paths.extend(event_map_paths)
                timeseries_paths.extend(event_timeseries_paths)

                for stage in ("maximum", "final"):
                    for threshold, label in thresholds:
                        row, confusion = calculate_extent_metrics(
                            prediction,
                            "R019",
                            mode,
                            stage,
                            threshold,
                            label,
                        )
                        confusion_path = r019_extent_dir / (
                            f"R019_{mode}_{stage}_extent_confusion_"
                            f"Hgte_{label}.png"
                        )
                        comparison_path = r019_extent_dir / (
                            f"R019_{mode}_{stage}_extent_true_pred_"
                            f"Hgte_{label}.png"
                        )
                        save_confusion_map(
                            row,
                            confusion,
                            confusion_path,
                            args.dpi,
                        )
                        save_extent_comparison(
                            prediction,
                            row,
                            confusion,
                            comparison_path,
                            args.dpi,
                        )
                        r019_extent_rows.append(row)
                        extent_figure_paths.extend(
                            (confusion_path, comparison_path)
                        )
            del prediction

    timestep_metrics_csv = (
        output_dir / "milestone15a_test_timestep_metrics.csv"
    )
    event_summary_csv = output_dir / "milestone15a_test_event_summary.csv"
    r019_extent_csv = (
        output_dir / "milestone15a_r019_extent_confusion_metrics.csv"
    )
    write_metrics_csv(
        timestep_metrics_csv,
        TIMESTEP_METRICS_FIELDNAMES,
        all_timestep_metrics,
    )
    write_metrics_csv(
        event_summary_csv,
        EVENT_SUMMARY_FIELDNAMES,
        all_event_summaries,
    )
    write_extent_metrics_csv(r019_extent_csv, r019_extent_rows)
    aggregate_test_metrics = calculate_aggregate_test_metrics(
        all_event_summaries,
        modes,
    )
    summary_figure_paths = create_summary_figures(
        aggregate_test_metrics,
        all_event_summaries,
        r019_extent_rows,
        modes,
        thresholds,
        summary_figures_dir,
        args.dpi,
    )

    expected_evaluations = len(evaluated_event_ids) * len(modes)
    expected_extent_rows = len(modes) * 2 * len(thresholds)
    if len(all_event_summaries) != expected_evaluations:
        raise RuntimeError("Unexpected number of test event summary rows")
    if len(prediction_paths) != len(modes):
        raise RuntimeError("Expected one saved R019 prediction per mode")
    if len(map_paths) != 2 * len(modes):
        raise RuntimeError("Unexpected number of R019 depth maps")
    if len(timeseries_paths) != 3 * len(modes):
        raise RuntimeError("Unexpected number of R019 time-series figures")
    if len(r019_extent_rows) != expected_extent_rows:
        raise RuntimeError("Unexpected number of R019 extent rows")
    if len(extent_figure_paths) != 2 * expected_extent_rows:
        raise RuntimeError("Unexpected number of R019 extent figures")
    if len(summary_figure_paths) != 6:
        raise RuntimeError("Expected exactly six summary figures")
    all_output_paths = (
        [timestep_metrics_csv, event_summary_csv, r019_extent_csv]
        + prediction_paths
        + map_paths
        + timeseries_paths
        + extent_figure_paths
        + summary_figure_paths
    )
    for output_path in all_output_paths:
        if not output_path.is_file():
            raise RuntimeError(f"Expected output was not saved: {output_path}")

    cuda_device_name = None
    max_memory_allocated_mb = None
    if device.type == "cuda":
        cuda_device_name = torch.cuda.get_device_name(device)
        max_memory_allocated_mb = torch.cuda.max_memory_allocated(device) / (
            1024**2
        )
    summary_json = output_dir / "milestone15a_summary.json"
    total_elapsed_seconds = time.perf_counter() - run_start
    summary = {
        "script_name": Path(__file__).name,
        "dataset_root": str(dataset_root),
        "swegnn_repo": str(swegnn_repo),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "test_event_ids": test_event_ids,
        "r019_present": "R019" in test_event_ids,
        "evaluated_event_ids": evaluated_event_ids,
        "modes": modes,
        "thresholds": [value for value, _ in thresholds],
        "input_features": INPUT_FEATURES,
        "edge_features": EDGE_FEATURES,
        "output_variables": OUTPUT_VARIABLES,
        "rainfall_scale": args.rainfall_scale,
        "checkpoint_rainfall_scale": checkpoint_rainfall_scale,
        "checkpoint_type": checkpoint_metadata["checkpoint_type"],
        "checkpoint_best_epoch": checkpoint_metadata[
            "checkpoint_best_epoch"
        ],
        "checkpoint_best_loss": checkpoint_metadata[
            "checkpoint_best_loss"
        ],
        "start_time_index": args.start_time_index,
        "end_time_index": args.end_time_index,
        "target_time_indices": target_time_indices,
        "target_minutes": target_minutes,
        "clamp_output": args.clamp_output,
        "clamp_feedback": args.clamp_feedback,
        "hid_features": args.hid_features,
        "K": args.K,
        "seed": args.seed,
        "num_prediction_files_saved": len(prediction_paths),
        "num_map_figures": len(map_paths),
        "num_timeseries_figures": len(timeseries_paths),
        "num_extent_confusion_figures": len(extent_figure_paths),
        "num_summary_figures": len(summary_figure_paths),
        "timestep_metrics_csv": str(timestep_metrics_csv),
        "event_summary_csv": str(event_summary_csv),
        "r019_extent_metrics_csv": str(r019_extent_csv),
        "r019_prediction_paths": [str(path) for path in prediction_paths],
        "figures_dir": str(figures_dir),
        "device": str(device),
        "total_elapsed_seconds": total_elapsed_seconds,
        "aggregate_test_metrics": aggregate_test_metrics,
    }
    if device.type == "cuda":
        summary["cuda_device_name"] = cuda_device_name
        summary["max_memory_allocated_mb"] = max_memory_allocated_mb
    with summary_json.open("w") as summary_file:
        json.dump(summary, summary_file, indent=2, allow_nan=False)

    print("\nOutput paths:")
    print("Timestep metrics:", timestep_metrics_csv)
    print("Event summary:", event_summary_csv)
    print("R019 extent metrics:", r019_extent_csv)
    print("Summary JSON:", summary_json)
    print("Predictions directory:", predictions_dir)
    print("Figures directory:", figures_dir)
    print(
        "Number of figures:",
        len(map_paths)
        + len(timeseries_paths)
        + len(extent_figure_paths)
        + len(summary_figure_paths),
    )
    print_aggregate_metrics(aggregate_test_metrics)
    print("R019 metrics by mode:")
    for summary_row in all_event_summaries:
        if summary_row["event_id"] == "R019":
            print_event_summary(summary_row)
    print()
    print("Stockbridge SWE-GNN milestone 15A test/R019 evaluation passed.")


if __name__ == "__main__":
    main()
