import argparse
import csv
import json
import math
import re
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


DEFAULT_14B4_OUTPUT_DIR = (
    Path.home()
    / "flood/03_outputs/Stockbridge-SWEGNN"
    / "milestone14b4_validation_visual_diagnostic_v01"
)
DEFAULT_PREDICTIONS_DIR = DEFAULT_14B4_OUTPUT_DIR / "predictions"
DEFAULT_EVENT_SUMMARY_CSV = (
    DEFAULT_14B4_OUTPUT_DIR / "milestone14b4_event_summary.csv"
)
DEFAULT_TIMESTEP_METRICS_CSV = (
    DEFAULT_14B4_OUTPUT_DIR / "milestone14b4_timestep_metrics.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path.home()
    / "flood/03_outputs/Stockbridge-SWEGNN"
    / "milestone14b5_validation_extent_confusion_v01"
)

NY = 305
NX = 326
NT = 25
NUM_NODES = 99430
TIME_STEP_MINUTES = 5

ALLOWED_MODES = ("teacher_forced", "autoregressive")
ALLOWED_STAGES = ("maximum", "final")
REQUIRED_PREDICTION_KEYS = (
    "event_id",
    "mode",
    "start_time_index",
    "end_time_index",
    "target_time_indices",
    "target_minutes",
    "pred_H",
    "true_H",
    "pred_Qmag",
    "true_Qmag",
    "rainfall_global",
    "rainfall_scale",
    "clamp_output",
    "clamp_feedback",
    "input_features",
    "output_variables",
)
METRICS_FIELDNAMES = (
    "event_id",
    "mode",
    "stage",
    "threshold_m",
    "threshold_label",
    "num_steps",
    "start_time_index",
    "end_time_index",
    "target_time_indices",
    "target_minutes",
    "tp_count",
    "fp_count",
    "fn_count",
    "tn_count",
    "true_wet_count",
    "pred_wet_count",
    "union_wet_count",
    "CSI",
    "POD",
    "FAR",
    "Precision",
    "F1",
    "Bias",
    "Jaccard",
    "wet_area_error_count",
    "wet_area_error_ratio",
    "max_true_H",
    "max_pred_H",
    "H_peak_ratio",
)

CONFUSION_COLOURS = ("#eeeeee", "#2878b5", "#e87525", "#7b3294")
CONFUSION_LABELS = (
    "Dry correct / TN",
    "True positive / TP",
    "False positive / FP",
    "False negative / FN",
)
MODE_COLOURS = {
    "teacher_forced": "#2878b5",
    "autoregressive": "#e87525",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for Milestone 14B-5."""
    parser = argparse.ArgumentParser(
        description=(
            "Create threshold-based flood-extent diagnostics from saved "
            "Milestone 14B-4 validation predictions."
        )
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
    )
    parser.add_argument(
        "--event-summary-csv",
        type=Path,
        default=DEFAULT_EVENT_SUMMARY_CSV,
    )
    parser.add_argument(
        "--timestep-metrics-csv",
        type=Path,
        default=DEFAULT_TIMESTEP_METRICS_CSV,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--event-ids", default="auto")
    parser.add_argument(
        "--modes",
        default="teacher_forced,autoregressive",
    )
    parser.add_argument("--thresholds", default="0.05,0.10,0.20")
    parser.add_argument("--stages", default="maximum,final")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def parse_modes(raw_modes: str) -> list[str]:
    """Return an ordered, duplicate-free list of allowed modes."""
    modes = [value.strip() for value in raw_modes.split(",") if value.strip()]
    if not modes:
        raise ValueError("--modes must contain at least one mode")
    if len(set(modes)) != len(modes):
        raise ValueError(f"--modes contains duplicates: {modes}")
    invalid = [mode for mode in modes if mode not in ALLOWED_MODES]
    if invalid:
        raise ValueError(
            f"Unsupported --modes values {invalid}; allowed: {ALLOWED_MODES}"
        )
    return modes


def parse_stages(raw_stages: str) -> list[str]:
    """Return an ordered, duplicate-free list of allowed map stages."""
    stages = [
        value.strip() for value in raw_stages.split(",") if value.strip()
    ]
    if not stages:
        raise ValueError("--stages must contain at least one stage")
    if len(set(stages)) != len(stages):
        raise ValueError(f"--stages contains duplicates: {stages}")
    invalid = [stage for stage in stages if stage not in ALLOWED_STAGES]
    if invalid:
        raise ValueError(
            f"Unsupported --stages values {invalid}; allowed: {ALLOWED_STAGES}"
        )
    return stages


def threshold_label(threshold: float) -> str:
    """Create an unambiguous filesystem label for one positive threshold."""
    text = format(threshold, ".12g")
    if 0.0 < threshold < 1.0:
        decimal_places = len(text.partition(".")[2])
        if decimal_places == 1:
            text += "0"
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def parse_thresholds(raw_thresholds: str) -> list[tuple[float, str]]:
    """Parse finite, positive, unique thresholds and their labels."""
    raw_values = [
        value.strip()
        for value in raw_thresholds.split(",")
        if value.strip()
    ]
    if not raw_values:
        raise ValueError("--thresholds must contain at least one value")
    thresholds = []
    for raw_value in raw_values:
        try:
            threshold = float(raw_value)
        except ValueError as error:
            raise ValueError(
                f"Invalid flood-depth threshold: {raw_value!r}"
            ) from error
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                "Every --thresholds value must be finite and greater than zero; "
                f"got {raw_value!r}"
            )
        thresholds.append(threshold)
    if len(set(thresholds)) != len(thresholds):
        raise ValueError(f"--thresholds contains duplicates: {thresholds}")
    labelled = [(value, threshold_label(value)) for value in thresholds]
    labels = [label for _, label in labelled]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Threshold labels are not unique: {labels}")
    return labelled


def validate_event_id(event_id: str) -> None:
    """Reject empty, unsafe, or explicitly prohibited event identifiers."""
    if not event_id or event_id in {".", ".."}:
        raise ValueError(f"Invalid event ID: {event_id!r}")
    if Path(event_id).name != event_id or "/" in event_id or "\\" in event_id:
        raise ValueError(f"Unsafe event ID: {event_id!r}")
    if event_id == "R019":
        raise ValueError("R019 must not be used by Milestone 14B-5")


def natural_event_sort_key(event_id: str) -> tuple:
    """Sort event identifiers naturally without restricting allowed IDs."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", event_id)
    )


def parse_explicit_event_ids(raw_event_ids: str) -> list[str]:
    """Parse explicit event IDs exactly in their requested order."""
    event_ids = [
        value.strip() for value in raw_event_ids.split(",") if value.strip()
    ]
    if not event_ids:
        raise ValueError("--event-ids must be 'auto' or a nonempty list")
    for event_id in event_ids:
        validate_event_id(event_id)
    if len(set(event_ids)) != len(event_ids):
        raise ValueError(f"--event-ids contains duplicates: {event_ids}")
    return event_ids


def event_ids_from_summary(event_summary_csv: Path) -> list[str]:
    """Read and de-duplicate event IDs in 14B-4 summary row order."""
    with event_summary_csv.open(newline="") as summary_file:
        reader = csv.DictReader(summary_file)
        if "event_id" not in (reader.fieldnames or []):
            raise ValueError(
                f"Event summary CSV lacks event_id: {event_summary_csv}"
            )
        ordered_event_ids = []
        seen = set()
        for row in reader:
            event_id = row.get("event_id", "").strip()
            if not event_id:
                raise ValueError(
                    f"Blank event_id in event summary: {event_summary_csv}"
                )
            if event_id == "R019":
                continue
            validate_event_id(event_id)
            if event_id not in seen:
                seen.add(event_id)
                ordered_event_ids.append(event_id)
    return ordered_event_ids


def available_event_ids(predictions_dir: Path, mode: str) -> set[str]:
    """Find safe event identifiers represented by one mode directory."""
    mode_dir = predictions_dir / mode
    if not mode_dir.is_dir():
        raise FileNotFoundError(f"Prediction mode directory not found: {mode_dir}")
    event_ids = set()
    for prediction_path in mode_dir.glob("*.pt"):
        event_id = prediction_path.stem
        if event_id == "R019":
            continue
        validate_event_id(event_id)
        event_ids.add(event_id)
    return event_ids


def resolve_event_ids(
    raw_event_ids: str,
    predictions_dir: Path,
    event_summary_csv: Path,
    modes: list[str],
) -> list[str]:
    """Resolve automatic or explicit events with complete mode coverage."""
    available_by_mode = {
        mode: available_event_ids(predictions_dir, mode) for mode in modes
    }
    complete_event_ids = set.intersection(
        *(available_by_mode[mode] for mode in modes)
    )
    if raw_event_ids.strip().lower() == "auto":
        if event_summary_csv.is_file():
            candidates = event_ids_from_summary(event_summary_csv)
            selected = [
                event_id
                for event_id in candidates
                if event_id in complete_event_ids
            ]
        else:
            selected = sorted(complete_event_ids, key=natural_event_sort_key)
        if not selected:
            raise ValueError(
                "Automatic event selection found no events with prediction "
                f"files for every requested mode: {modes}"
            )
        return selected

    selected = parse_explicit_event_ids(raw_event_ids)
    missing = {
        mode: [
            event_id
            for event_id in selected
            if event_id not in available_by_mode[mode]
        ]
        for mode in modes
    }
    missing = {mode: values for mode, values in missing.items() if values}
    if missing:
        raise FileNotFoundError(
            f"Missing requested prediction files by mode: {missing}"
        )
    return selected


def require_keys(mapping: dict, path: Path) -> None:
    """Validate that a prediction bundle contains every required field."""
    missing = [key for key in REQUIRED_PREDICTION_KEYS if key not in mapping]
    if missing:
        raise KeyError(f"Prediction file {path} is missing keys: {missing}")


def validate_integer(value, label: str) -> int:
    """Validate a metadata integer while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer, got {value!r}")
    return value


def validate_numeric_sequence(value, label: str) -> list[float | int]:
    """Validate a finite list or tuple of numeric metadata values."""
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{label} must be a list or tuple")
    validated = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{label} contains nonnumeric value {item!r}")
        if not math.isfinite(float(item)):
            raise ValueError(f"{label} contains non-finite value {item!r}")
        validated.append(item)
    return validated


def validate_string_sequence(value, label: str) -> list[str]:
    """Validate a nonempty list or tuple containing only strings."""
    if not isinstance(value, (list, tuple)) or not value:
        raise TypeError(f"{label} must be a nonempty list or tuple")
    if any(not isinstance(item, str) or not item for item in value):
        raise TypeError(f"{label} must contain nonempty strings")
    return list(value)


def assert_cpu_finite_tensor(
    value,
    expected_shape: tuple[int, ...],
    label: str,
) -> torch.Tensor:
    """Validate tensor type, shape, CPU residence, and finiteness."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if tuple(value.shape) != expected_shape:
        raise ValueError(
            f"Expected {label} shape {expected_shape}, got {tuple(value.shape)}"
        )
    if value.device.type != "cpu":
        raise ValueError(f"{label} must be stored on CPU, got {value.device}")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{label} contains NaN or infinite values")
    return value


def load_prediction_bundle(
    prediction_path: Path,
    expected_event_id: str,
    expected_mode: str,
) -> dict:
    """Load and fully validate one saved Milestone 14B-4 prediction bundle."""
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")
    prediction = torch.load(prediction_path, weights_only=True)
    if not isinstance(prediction, dict):
        raise TypeError(
            f"Prediction file must contain a dict: {prediction_path}"
        )
    require_keys(prediction, prediction_path)

    if prediction["event_id"] != expected_event_id:
        raise ValueError(
            f"Prediction event_id {prediction['event_id']!r} does not match "
            f"filename {expected_event_id!r}: {prediction_path}"
        )
    validate_event_id(prediction["event_id"])
    if prediction["mode"] != expected_mode:
        raise ValueError(
            f"Prediction mode {prediction['mode']!r} does not match parent "
            f"directory {expected_mode!r}: {prediction_path}"
        )

    start_time_index = validate_integer(
        prediction["start_time_index"],
        f"{prediction_path} start_time_index",
    )
    end_time_index = validate_integer(
        prediction["end_time_index"],
        f"{prediction_path} end_time_index",
    )
    if start_time_index < 0 or end_time_index < start_time_index:
        raise ValueError(
            f"Invalid evaluation indices in {prediction_path}: "
            f"{start_time_index}..{end_time_index}"
        )
    target_time_indices = validate_numeric_sequence(
        prediction["target_time_indices"],
        f"{prediction_path} target_time_indices",
    )
    if any(isinstance(value, float) for value in target_time_indices):
        raise TypeError(f"target_time_indices must contain integers: {prediction_path}")
    target_minutes = validate_numeric_sequence(
        prediction["target_minutes"],
        f"{prediction_path} target_minutes",
    )
    num_steps = len(target_time_indices)
    if num_steps == 0:
        raise ValueError(f"Prediction contains no evaluated steps: {prediction_path}")
    if end_time_index - start_time_index + 1 != num_steps:
        raise ValueError(f"Evaluation index count is inconsistent: {prediction_path}")
    expected_target_indices = list(
        range(start_time_index + 1, end_time_index + 2)
    )
    if target_time_indices != expected_target_indices:
        raise ValueError(
            f"Unexpected target_time_indices in {prediction_path}: "
            f"{target_time_indices}"
        )
    expected_target_minutes = [
        index * TIME_STEP_MINUTES for index in expected_target_indices
    ]
    if target_minutes != expected_target_minutes:
        raise ValueError(
            f"Unexpected target_minutes in {prediction_path}: {target_minutes}"
        )

    field_shape = (num_steps, NUM_NODES)
    for field in ("pred_H", "true_H", "pred_Qmag", "true_Qmag"):
        assert_cpu_finite_tensor(
            prediction[field],
            field_shape,
            f"{prediction_path} {field}",
        )
    assert_cpu_finite_tensor(
        prediction["rainfall_global"],
        (NT,),
        f"{prediction_path} rainfall_global",
    )
    if (prediction["pred_H"] < 0.0).any().item():
        raise ValueError(f"pred_H contains negative values: {prediction_path}")
    if (prediction["true_H"] < 0.0).any().item():
        raise ValueError(f"true_H contains negative values: {prediction_path}")

    rainfall_scale = prediction["rainfall_scale"]
    if isinstance(rainfall_scale, bool) or not isinstance(
        rainfall_scale, (int, float)
    ):
        raise TypeError(f"rainfall_scale must be numeric: {prediction_path}")
    if not math.isfinite(float(rainfall_scale)):
        raise ValueError(f"rainfall_scale must be finite: {prediction_path}")
    for field in ("clamp_output", "clamp_feedback"):
        if not isinstance(prediction[field], bool):
            raise TypeError(f"{field} must be boolean: {prediction_path}")
    validate_string_sequence(
        prediction["input_features"],
        f"{prediction_path} input_features",
    )
    output_variables = validate_string_sequence(
        prediction["output_variables"],
        f"{prediction_path} output_variables",
    )
    if output_variables != ["H", "Qmag"]:
        raise ValueError(
            f"Unexpected output_variables in {prediction_path}: {output_variables}"
        )
    return prediction


def metadata_signature(prediction: dict) -> dict:
    """Return timing metadata that must agree across all selected bundles."""
    return {
        "start_time_index": prediction["start_time_index"],
        "end_time_index": prediction["end_time_index"],
        "target_time_indices": list(prediction["target_time_indices"]),
        "target_minutes": list(prediction["target_minutes"]),
    }


def stage_depths(prediction: dict, stage: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the true and predicted node depths for one diagnostic stage."""
    if stage == "maximum":
        return (
            prediction["true_H"].max(dim=0).values,
            prediction["pred_H"].max(dim=0).values,
        )
    if stage == "final":
        return prediction["true_H"][-1], prediction["pred_H"][-1]
    raise ValueError(f"Unsupported stage: {stage}")


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    """Return a finite ratio or None when its denominator is zero."""
    if denominator == 0:
        return None
    value = float(numerator) / float(denominator)
    if not math.isfinite(value):
        raise ValueError("Calculated a non-finite diagnostic ratio")
    return value


def calculate_extent_metrics(
    prediction: dict,
    event_id: str,
    mode: str,
    stage: str,
    threshold: float,
    label: str,
) -> tuple[dict, torch.Tensor]:
    """Calculate confusion classes and all requested extent metrics."""
    true_depth, pred_depth = stage_depths(prediction, stage)
    true_wet = true_depth >= threshold
    pred_wet = pred_depth >= threshold
    tp = pred_wet & true_wet
    fp = pred_wet & ~true_wet
    fn = ~pred_wet & true_wet
    tn = ~pred_wet & ~true_wet

    tp_count = int(tp.sum().item())
    fp_count = int(fp.sum().item())
    fn_count = int(fn.sum().item())
    tn_count = int(tn.sum().item())
    true_wet_count = tp_count + fn_count
    pred_wet_count = tp_count + fp_count
    union_wet_count = tp_count + fp_count + fn_count
    if tp_count + fp_count + fn_count + tn_count != NUM_NODES:
        raise RuntimeError("Extent confusion counts do not sum to NUM_NODES")

    csi = safe_ratio(tp_count, union_wet_count)
    precision = safe_ratio(tp_count, tp_count + fp_count)
    f1 = safe_ratio(2 * tp_count, 2 * tp_count + fp_count + fn_count)
    max_true_h = float(true_depth.max().item())
    max_pred_h = float(pred_depth.max().item())
    wet_area_error_count = pred_wet_count - true_wet_count
    metrics = {
        "event_id": event_id,
        "mode": mode,
        "stage": stage,
        "threshold_m": threshold,
        "threshold_label": label,
        "num_steps": int(prediction["pred_H"].shape[0]),
        "start_time_index": prediction["start_time_index"],
        "end_time_index": prediction["end_time_index"],
        "target_time_indices": json.dumps(prediction["target_time_indices"]),
        "target_minutes": json.dumps(prediction["target_minutes"]),
        "tp_count": tp_count,
        "fp_count": fp_count,
        "fn_count": fn_count,
        "tn_count": tn_count,
        "true_wet_count": true_wet_count,
        "pred_wet_count": pred_wet_count,
        "union_wet_count": union_wet_count,
        "CSI": csi,
        "POD": safe_ratio(tp_count, tp_count + fn_count),
        "FAR": safe_ratio(fp_count, tp_count + fp_count),
        "Precision": precision,
        "F1": f1,
        "Bias": safe_ratio(pred_wet_count, true_wet_count),
        "Jaccard": csi,
        "wet_area_error_count": wet_area_error_count,
        "wet_area_error_ratio": safe_ratio(
            wet_area_error_count,
            true_wet_count,
        ),
        "max_true_H": max_true_h,
        "max_pred_H": max_pred_h,
        "H_peak_ratio": safe_ratio(max_pred_h, max_true_h),
    }
    confusion = torch.zeros(NUM_NODES, dtype=torch.uint8)
    confusion[tp] = 1
    confusion[fp] = 2
    confusion[fn] = 3
    return metrics, confusion


def metric_text(value: float | None) -> str:
    """Format an optional metric for figure titles and console output."""
    return "undefined" if value is None else f"{value:.3f}"


def confusion_legend_handles() -> list[Patch]:
    """Build a consistent legend for four confusion classes."""
    return [
        Patch(facecolor=colour, edgecolor="0.4", label=label)
        for colour, label in zip(
            CONFUSION_COLOURS,
            CONFUSION_LABELS,
            strict=True,
        )
    ]


def apply_map_style(axis: plt.Axes) -> None:
    """Use consistent raster axes for the Stockbridge grid."""
    axis.set_aspect("equal")
    axis.set_xlabel("Grid x index")
    axis.set_ylabel("Grid y index")


def save_confusion_map(
    metrics: dict,
    confusion: torch.Tensor,
    output_path: Path,
    dpi: int,
) -> None:
    """Save one discrete four-class flood-extent confusion map."""
    figure, axis = plt.subplots(figsize=(8.2, 7.6))
    axis.imshow(
        confusion.reshape(NY, NX).numpy(),
        origin="upper",
        cmap=ListedColormap(CONFUSION_COLOURS),
        vmin=-0.5,
        vmax=3.5,
        interpolation="nearest",
    )
    apply_map_style(axis)
    axis.set_title(
        f"{metrics['event_id']} | {metrics['mode']} | {metrics['stage']} | "
        f"H >= {metrics['threshold_m']:g} m\n"
        f"CSI {metric_text(metrics['CSI'])} | "
        f"F1 {metric_text(metrics['F1'])} | "
        f"Bias {metric_text(metrics['Bias'])}"
    )
    axis.legend(
        handles=confusion_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=True,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_extent_comparison(
    prediction: dict,
    metrics: dict,
    confusion: torch.Tensor,
    output_path: Path,
    dpi: int,
) -> None:
    """Save true, predicted, and difference-class extent panels."""
    true_depth, pred_depth = stage_depths(prediction, metrics["stage"])
    threshold = metrics["threshold_m"]
    true_wet = (true_depth >= threshold).reshape(NY, NX).numpy()
    pred_wet = (pred_depth >= threshold).reshape(NY, NX).numpy()
    binary_cmap = ListedColormap(("#eeeeee", "#2878b5"))
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))
    axes[0].imshow(
        true_wet,
        origin="upper",
        cmap=binary_cmap,
        vmin=-0.5,
        vmax=1.5,
        interpolation="nearest",
    )
    axes[1].imshow(
        pred_wet,
        origin="upper",
        cmap=binary_cmap,
        vmin=-0.5,
        vmax=1.5,
        interpolation="nearest",
    )
    axes[2].imshow(
        confusion.reshape(NY, NX).numpy(),
        origin="upper",
        cmap=ListedColormap(CONFUSION_COLOURS),
        vmin=-0.5,
        vmax=3.5,
        interpolation="nearest",
    )
    axes[0].set_title(f"True wet extent\n{metrics['true_wet_count']} nodes")
    axes[1].set_title(
        f"Predicted wet extent\n{metrics['pred_wet_count']} nodes"
    )
    axes[2].set_title(
        f"Difference classes\nTP {metrics['tp_count']} | "
        f"FP {metrics['fp_count']} | FN {metrics['fn_count']}"
    )
    for axis in axes:
        apply_map_style(axis)
    axes[0].legend(
        handles=(
            Patch(facecolor="#eeeeee", edgecolor="0.4", label="Dry"),
            Patch(facecolor="#2878b5", edgecolor="0.4", label="Wet"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    axes[2].legend(
        handles=confusion_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    figure.suptitle(
        f"{metrics['event_id']} | {metrics['mode']} | {metrics['stage']} | "
        f"H >= {threshold:g} m",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def rows_for_setting(
    rows: list[dict],
    stage: str,
    threshold: float,
) -> list[dict]:
    """Select metric rows for one stage and numerical threshold."""
    return [
        row
        for row in rows
        if row["stage"] == stage and row["threshold_m"] == threshold
    ]


def save_grouped_metric_plot(
    rows: list[dict],
    event_ids: list[str],
    modes: list[str],
    stage: str,
    threshold: float,
    label: str,
    metric: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Save one event-wise grouped mode comparison chart."""
    lookup = {
        (row["event_id"], row["mode"]): row
        for row in rows_for_setting(rows, stage, threshold)
    }
    x_positions = list(range(len(event_ids)))
    group_width = 0.8
    bar_width = group_width / len(modes)
    figure, axis = plt.subplots(figsize=(max(8.0, len(event_ids) * 1.3), 5.4))
    for mode_index, mode in enumerate(modes):
        offset = (mode_index - (len(modes) - 1) / 2.0) * bar_width
        values = []
        for event_id in event_ids:
            value = lookup[(event_id, mode)][metric]
            values.append(math.nan if value is None else value)
        axis.bar(
            [position + offset for position in x_positions],
            values,
            width=bar_width * 0.9,
            label=mode,
            color=MODE_COLOURS[mode],
        )
    axis.set_xticks(x_positions, event_ids)
    axis.set_xlabel("Validation event")
    axis.set_ylabel(metric)
    axis.set_title(
        f"{stage.capitalize()} extent {metric} by event | "
        f"H >= {threshold:g} m"
    )
    axis.grid(axis="y", alpha=0.3)
    if metric in {"CSI", "F1"}:
        axis.set_ylim(0.0, 1.0)
    else:
        finite_values = [
            row[metric]
            for row in lookup.values()
            if row[metric] is not None
        ]
        upper = max([1.25, *(value * 1.15 for value in finite_values)])
        axis.set_ylim(0.0, upper)
        axis.axhline(1.0, color="0.25", linestyle="--", label="Bias = 1")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def save_autoregressive_csi_summary(
    rows: list[dict],
    event_ids: list[str],
    thresholds: list[tuple[float, str]],
    output_path: Path,
    dpi: int,
) -> None:
    """Save maximum-stage autoregressive CSI grouped by threshold."""
    lookup = {
        (row["event_id"], row["threshold_m"]): row["CSI"]
        for row in rows
        if row["mode"] == "autoregressive" and row["stage"] == "maximum"
    }
    x_positions = list(range(len(event_ids)))
    group_width = 0.82
    bar_width = group_width / len(thresholds)
    figure, axis = plt.subplots(figsize=(max(8.0, len(event_ids) * 1.4), 5.5))
    colours = plt.get_cmap("viridis")(
        torch.linspace(0.2, 0.8, len(thresholds)).numpy()
    )
    for threshold_index, ((threshold, label), colour) in enumerate(
        zip(thresholds, colours, strict=True)
    ):
        offset = (
            threshold_index - (len(thresholds) - 1) / 2.0
        ) * bar_width
        values = [
            math.nan
            if lookup[(event_id, threshold)] is None
            else lookup[(event_id, threshold)]
            for event_id in event_ids
        ]
        axis.bar(
            [position + offset for position in x_positions],
            values,
            width=bar_width * 0.9,
            label=f"H >= {threshold:g} m ({label})",
            color=colour,
        )
    axis.set_xticks(x_positions, event_ids)
    axis.set_xlabel("Validation event")
    axis.set_ylabel("CSI")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Autoregressive maximum-extent CSI by threshold")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def create_summary_figures(
    rows: list[dict],
    event_ids: list[str],
    modes: list[str],
    stages: list[str],
    thresholds: list[tuple[float, str]],
    summary_dir: Path,
    dpi: int,
) -> list[Path]:
    """Create all per-setting summary plots and the optional combined plot."""
    output_paths = []
    for stage in stages:
        for threshold, label in thresholds:
            for metric in ("CSI", "F1", "Bias"):
                output_path = (
                    summary_dir
                    / f"{stage}_{metric}_by_event_Hgte_{label}.png"
                )
                save_grouped_metric_plot(
                    rows,
                    event_ids,
                    modes,
                    stage,
                    threshold,
                    label,
                    metric,
                    output_path,
                    dpi,
                )
                output_paths.append(output_path)
    if "autoregressive" in modes and "maximum" in stages:
        output_path = (
            summary_dir / "summary_autoregressive_CSI_by_threshold.png"
        )
        save_autoregressive_csi_summary(
            rows,
            event_ids,
            thresholds,
            output_path,
            dpi,
        )
        output_paths.append(output_path)
    return output_paths


def mean_optional(values: list[float | None]) -> float | None:
    """Calculate a finite mean using only defined metric values."""
    defined_values = [value for value in values if value is not None]
    if not defined_values:
        return None
    return math.fsum(defined_values) / len(defined_values)


def aggregate_metrics(
    rows: list[dict],
    modes: list[str],
    stages: list[str],
    thresholds: list[tuple[float, str]],
) -> list[dict]:
    """Aggregate event metrics for every selected setting."""
    aggregates = []
    for mode in modes:
        for stage in stages:
            for threshold, label in thresholds:
                selected = [
                    row
                    for row in rows
                    if row["mode"] == mode
                    and row["stage"] == stage
                    and row["threshold_m"] == threshold
                ]
                if not selected:
                    raise RuntimeError(
                        f"No rows to aggregate for {mode}/{stage}/{threshold}"
                    )
                totals = {
                    field: sum(row[field] for row in selected)
                    for field in ("tp_count", "fp_count", "fn_count", "tn_count")
                }
                total_tp = totals["tp_count"]
                total_fp = totals["fp_count"]
                total_fn = totals["fn_count"]
                total_tn = totals["tn_count"]
                aggregates.append(
                    {
                        "mode": mode,
                        "stage": stage,
                        "threshold_m": threshold,
                        "threshold_label": label,
                        "num_events": len(selected),
                        "mean_CSI": mean_optional(
                            [row["CSI"] for row in selected]
                        ),
                        "mean_F1": mean_optional(
                            [row["F1"] for row in selected]
                        ),
                        "mean_Bias": mean_optional(
                            [row["Bias"] for row in selected]
                        ),
                        "mean_POD": mean_optional(
                            [row["POD"] for row in selected]
                        ),
                        "mean_FAR": mean_optional(
                            [row["FAR"] for row in selected]
                        ),
                        "mean_Precision": mean_optional(
                            [row["Precision"] for row in selected]
                        ),
                        "total_TP": total_tp,
                        "total_FP": total_fp,
                        "total_FN": total_fn,
                        "total_TN": total_tn,
                        "pooled_CSI": safe_ratio(
                            total_tp,
                            total_tp + total_fp + total_fn,
                        ),
                        "pooled_F1": safe_ratio(
                            2 * total_tp,
                            2 * total_tp + total_fp + total_fn,
                        ),
                        "pooled_Bias": safe_ratio(
                            total_tp + total_fp,
                            total_tp + total_fn,
                        ),
                    }
                )
    return aggregates


def prepare_output_dir(
    output_dir: Path,
    predictions_dir: Path,
    event_summary_csv: Path,
    timestep_metrics_csv: Path,
    overwrite: bool,
) -> None:
    """Create or safely replace only the requested 14B-5 output directory."""
    output_exists = output_dir.exists() or output_dir.is_symlink()
    if output_exists and output_dir.is_symlink():
        raise ValueError(f"Refusing symlink --output-dir: {output_dir}")
    if output_exists and not output_dir.is_dir():
        raise ValueError(f"--output-dir is not a directory: {output_dir}")
    if output_exists and not overwrite and any(output_dir.iterdir()):
        raise FileExistsError(
            "Refusing to overwrite nonempty Milestone 14B-5 output "
            f"without --overwrite: {output_dir}"
        )
    if output_exists and overwrite:
        resolved_output = output_dir.resolve()
        repo_root = Path(__file__).resolve().parents[1]
        protected_paths = {
            Path("/").resolve(),
            Path.home().resolve(),
            predictions_dir.resolve(),
            DEFAULT_14B4_OUTPUT_DIR.resolve(),
            repo_root,
            Path.cwd().resolve(),
            event_summary_csv.resolve(),
            timestep_metrics_csv.resolve(),
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


def write_metrics_csv(metrics_path: Path, rows: list[dict]) -> None:
    """Write the complete ordered extent-confusion metrics table."""
    if not rows:
        raise ValueError("Cannot write an empty extent-confusion metrics CSV")
    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=METRICS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_row_diagnostics(row: dict) -> None:
    """Print one concise event/mode/stage/threshold diagnostic line."""
    print(
        f"{row['event_id']} | {row['mode']} | {row['stage']} | "
        f"H >= {row['threshold_m']:g} m | "
        f"CSI={metric_text(row['CSI'])} "
        f"F1={metric_text(row['F1'])} "
        f"Bias={metric_text(row['Bias'])} | "
        f"true_wet={row['true_wet_count']} "
        f"pred_wet={row['pred_wet_count']} | "
        f"TP/FP/FN={row['tp_count']}/{row['fp_count']}/{row['fn_count']}"
    )


def validate_input_path(path: Path, label: str, required: bool) -> None:
    """Validate one file input, optionally allowing it to be absent."""
    if path.exists() and not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    if required and not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    predictions_dir = args.predictions_dir.expanduser()
    event_summary_csv = args.event_summary_csv.expanduser()
    timestep_metrics_csv = args.timestep_metrics_csv.expanduser()
    output_dir = args.output_dir.expanduser()
    modes = parse_modes(args.modes)
    stages = parse_stages(args.stages)
    thresholds = parse_thresholds(args.thresholds)
    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than zero")
    if not predictions_dir.is_dir():
        raise FileNotFoundError(
            f"Milestone 14B-4 predictions directory not found: {predictions_dir}"
        )
    validate_input_path(
        event_summary_csv,
        "Milestone 14B-4 event summary CSV",
        required=False,
    )
    validate_input_path(
        timestep_metrics_csv,
        "Milestone 14B-4 timestep metrics CSV",
        required=False,
    )
    selected_event_ids = resolve_event_ids(
        args.event_ids,
        predictions_dir,
        event_summary_csv,
        modes,
    )

    print("=== Stockbridge SWE-GNN milestone 14B-5 diagnostics ===")
    print("Predictions directory:", predictions_dir)
    print("Output directory:", output_dir)
    print("Selected events:", selected_event_ids)
    print("Modes:", modes)
    print("Thresholds:", [value for value, _ in thresholds])
    print("Stages:", stages)

    prepare_output_dir(
        output_dir,
        predictions_dir,
        event_summary_csv,
        timestep_metrics_csv,
        args.overwrite,
    )
    figures_dir = output_dir / "figures"
    confusion_dir = figures_dir / "confusion_maps"
    extent_dir = figures_dir / "extent_maps"
    summary_dir = figures_dir / "summary"
    for directory in (confusion_dir, extent_dir, summary_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    confusion_paths = []
    extent_paths = []
    timing_metadata = None
    prediction_files_read = 0
    for event_id in selected_event_ids:
        for mode in modes:
            prediction_path = predictions_dir / mode / f"{event_id}.pt"
            prediction = load_prediction_bundle(
                prediction_path,
                event_id,
                mode,
            )
            prediction_files_read += 1
            current_timing = metadata_signature(prediction)
            if timing_metadata is None:
                timing_metadata = current_timing
            elif current_timing != timing_metadata:
                raise ValueError(
                    "Selected prediction files have inconsistent timing metadata: "
                    f"{prediction_path}"
                )

            for stage in stages:
                for threshold, label in thresholds:
                    row, confusion = calculate_extent_metrics(
                        prediction,
                        event_id,
                        mode,
                        stage,
                        threshold,
                        label,
                    )
                    confusion_path = confusion_dir / (
                        f"{event_id}_{mode}_{stage}_extent_confusion_"
                        f"Hgte_{label}.png"
                    )
                    extent_path = extent_dir / (
                        f"{event_id}_{mode}_{stage}_extent_true_pred_"
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
                        extent_path,
                        args.dpi,
                    )
                    metrics_rows.append(row)
                    confusion_paths.append(confusion_path)
                    extent_paths.append(extent_path)
                    print_row_diagnostics(row)
            del prediction

    if timing_metadata is None:
        raise RuntimeError("No prediction files were read")
    metrics_path = output_dir / "milestone14b5_extent_confusion_metrics.csv"
    write_metrics_csv(metrics_path, metrics_rows)
    summary_paths = create_summary_figures(
        metrics_rows,
        selected_event_ids,
        modes,
        stages,
        thresholds,
        summary_dir,
        args.dpi,
    )
    expected_rows = (
        len(selected_event_ids)
        * len(modes)
        * len(stages)
        * len(thresholds)
    )
    if len(metrics_rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} metrics rows, got {len(metrics_rows)}"
        )
    for output_path in (
        [metrics_path] + confusion_paths + extent_paths + summary_paths
    ):
        if not output_path.is_file():
            raise RuntimeError(f"Expected output was not saved: {output_path}")

    aggregates = aggregate_metrics(
        metrics_rows,
        modes,
        stages,
        thresholds,
    )
    summary_path = output_dir / "milestone14b5_summary.json"
    total_elapsed_seconds = time.perf_counter() - run_start
    summary = {
        "script_name": Path(__file__).name,
        "predictions_dir": str(predictions_dir),
        "event_summary_csv": str(event_summary_csv),
        "timestep_metrics_csv": str(timestep_metrics_csv),
        "output_dir": str(output_dir),
        "selected_event_ids": selected_event_ids,
        "modes": modes,
        "thresholds": [value for value, _ in thresholds],
        "stages": stages,
        "num_prediction_files_read": prediction_files_read,
        "num_metrics_rows": len(metrics_rows),
        "num_confusion_map_figures": len(confusion_paths),
        "num_extent_map_figures": len(extent_paths),
        "num_summary_figures": len(summary_paths),
        "metrics_csv": str(metrics_path),
        "figures_dir": str(figures_dir),
        **timing_metadata,
        "total_elapsed_seconds": total_elapsed_seconds,
        "aggregate_metrics": aggregates,
    }
    with summary_path.open("w") as summary_file:
        json.dump(summary, summary_file, indent=2, allow_nan=False)

    print("\nOutput paths:")
    print("Metrics CSV:", metrics_path)
    print("Summary JSON:", summary_path)
    print("Figures directory:", figures_dir)
    print("Confusion maps:", len(confusion_paths))
    print("Extent maps:", len(extent_paths))
    print("Summary figures:", len(summary_paths))
    print("\nAggregate autoregressive CSI/F1/Bias by threshold and stage:")
    autoregressive_aggregates = [
        aggregate
        for aggregate in aggregates
        if aggregate["mode"] == "autoregressive"
    ]
    if not autoregressive_aggregates:
        print("  autoregressive mode not selected")
    for aggregate in autoregressive_aggregates:
        print(
            f"  {aggregate['stage']} | "
            f"H >= {aggregate['threshold_m']:g} m | "
            f"CSI={metric_text(aggregate['mean_CSI'])} "
            f"F1={metric_text(aggregate['mean_F1'])} "
            f"Bias={metric_text(aggregate['mean_Bias'])}"
        )
    print()
    print(
        "Stockbridge SWE-GNN milestone 14B-5 validation extent confusion "
        "diagnostic passed."
    )


if __name__ == "__main__":
    main()
