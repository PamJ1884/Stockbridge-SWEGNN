"""M17K: controlled supervised training for SWE-GNN and rainfall-aware SWE-GNN.

One-step teacher-forced training with validation early stopping, followed by an
autoregressive rollout from the start index on the validation events using the
best checkpoint.

The validated M14B-1 trainer (train_valid_stockbridge_full_onestep_earlystop.py)
is imported for loading and validation and is NOT modified. Every M17K choice
is a command-line option and is stored in the checkpoints and summary:

  --architecture         swegnn | rainfall_aware
  --rainfall-alignment   original | alternative
                         original:    rain_t = PT[t],   rain_next = PT[t+1]
                         alternative: rain_t = PT[t+1], rain_next = PT[t+2]
                         With rainfall_alignment="future" in the PT files
                         (PT[k] = TXT at (k-1)*300 s), the alternative gives the
                         TXT records at the start and end of the predicted
                         interval t -> t+1. PT is extended with one zero
                         (TXT at 7200 s, verified zero for Phase 2).
  --rain-source-input    next | trapezoid (rainfall_aware only)
  --backbone-depth-mask  keep the original SWE-GNN 1 mm mask inside the backbone
  --loss-mask            all | active | where_water (loss over all nodes, hydraulic
                         cells, or active cells where prediction or target is
                         nonzero, as only_where_water in the original code)

Multi-step training (M17K-B):

  --rollout-schedule     curriculum 'steps:epochs,...', e.g. '2:3,4:3,8:6,16:10'.
                         Each sample starts from the CityCAT state at a start
                         time and is rolled out on the model's own H and Qmag;
                         the loss averages all steps. Empty = one-step TF.
  --max-starts-per-event random start times per event and epoch (0 = all)
  --detach-rollout       stop gradients between steps (less memory)
  --select-metric        tf_loss | rollout_CSI_0p05 | rollout_H_RMSE
                         rollout metrics come from a full T0 -> T24 validation
                         rollout every epoch.
  --init-checkpoint      start from a previous M17K checkpoint

Original SWE-GNN training strategy (M17K-P; Bentivoglio et al. 2023, Sect. 3.3,
Eq. 14, Algorithm 1, as implemented in SWE-GNN-paper-repository):

  --loss swegnn_rmse --loss-q-weight 3 --loss-mask where_water
  --curriculum-epochs 15 --max-rollout-steps 8 --epochs 150
  --lr 0.007 --lr-step-size 7 --lr-gamma 0.9 --accumulate-samples 8
  --grad-clip-value 0.5 --grad-clip-norm 0
  --select-metric rollout_swegnn_loss --early-stopping consecutive_increase
  --patience 30

With the default options the script reproduces the one-step M17K runs exactly.
A persistence reference (H(t+1) = H(t)) is reported for the teacher-forced
metrics, which are dominated by persistence because of the residual update.

Validation and rollout metrics are always computed on active hydraulic cells.
Test events are never loaded.
"""

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch_geometric.data import Data

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
for path in (SCRIPTS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_valid_stockbridge_full_onestep_earlystop as base  # noqa: E402
from experimental.rainfall_aware_swegnn_v2 import (  # noqa: E402
    RainfallAwareSWEGNNV2,
    active_mask_from_dem,
)

NUM_NODES = base.NUM_NODES
NT = base.NT
EXPECTED_ACTIVE_CELLS = 32249
CSI_THRESHOLDS = (0.01, 0.05, 0.10)
DEFAULT_DATASET_ROOT = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/swegnn"
    / "stockbridge_phase2_100_5m_v01"
)
DEFAULT_OUTPUT_DIR = (
    Path.home() / "flood/03_outputs/Stockbridge-SWEGNN" / "milestone17k_smoke_v01"
)

HISTORY_FIELDNAMES = [
    "epoch",
    "train_loss_mean",
    "train_loss_min",
    "train_loss_max",
    "mean_grad_norm",
    "valid_loss_mean",
    "valid_H_RMSE_active_mean",
    "valid_Qmag_RMSE_active_mean",
    "valid_H_peak_ratio_mean",
    "valid_CSI_0p01_mean",
    "valid_CSI_0p05_mean",
    "best_valid_loss_so_far",
    "best_epoch_so_far",
    "is_best",
    "epoch_elapsed_seconds",
    "rollout_steps",
    "train_starts",
    "rollout_H_RMSE_active_mean",
    "rollout_CSI_0p01_mean",
    "rollout_CSI_0p05_mean",
    "rollout_steps_all_dry",
    "rollout_steps_zero_Qmag",
    "selection_value",
    "best_selection_so_far",
    "rollout_swegnn_val_loss",
    "learning_rate",
]
SELECT_METRICS = {
    # name: (summary source, key, higher_is_better)
    "tf_loss": ("tf", "valid_loss_mean", False),
    "rollout_CSI_0p05": ("rollout", "CSI_0p05_mean", True),
    "rollout_H_RMSE": ("rollout", "H_RMSE_active_mean", False),
    # Original SWE-GNN selection: validation rollout RMSE where there is water.
    "rollout_swegnn_loss": ("rollout", "swegnn_val_loss", False),
}
ROLLOUT_FIELDNAMES = [
    "event_id",
    "input_time_index",
    "target_time_index",
    "H_RMSE_active",
    "Qmag_RMSE_active",
    "true_H_max",
    "pred_H_max",
    "true_Qmag_max",
    "pred_Qmag_max",
    "true_wet_cells_0p01",
    "pred_wet_cells_0p01",
    "pred_positive_cells",
    "CSI_0p01",
    "CSI_0p05",
    "CSI_0p10",
    "true_volume_m3",
    "pred_volume_m3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--swegnn-repo", type=Path, default=base.SWEGNN_REPO)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--architecture", choices=("swegnn", "rainfall_aware"), default="rainfall_aware"
    )
    parser.add_argument(
        "--rainfall-alignment", choices=("original", "alternative"), default="alternative"
    )
    parser.add_argument(
        "--rain-source-input", choices=("next", "trapezoid"), default="trapezoid"
    )
    parser.add_argument(
        "--backbone-depth-mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Keep the original 1 mm mask inside the SWE-GNN backbone. "
            "Default: on for swegnn, off for rainfall_aware."
        ),
    )
    parser.add_argument(
        "--loss-mask", choices=("all", "active", "where_water"), default="active"
    )
    parser.add_argument(
        "--rollout-eval", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--start-time-index", type=int, default=0)
    parser.add_argument("--end-time-index", type=int, default=23)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--rainfall-scale", type=float, default=300000.0)
    parser.add_argument("--dt", type=float, default=300.0)
    parser.add_argument("--cell-area", type=float, default=25.0)
    parser.add_argument("--hid-features", type=int, default=16)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-events", type=int, default=0)
    parser.add_argument("--max-valid-events", type=int, default=0)

    # Multi-step (autoregressive) training. Defaults reproduce one-step
    # teacher forcing selected on the teacher-forced validation loss.
    parser.add_argument(
        "--rollout-schedule",
        default="",
        help=(
            "Curriculum as 'steps:epochs,...', e.g. '1:5,2:5,4:5,8:15'. "
            "Each training sample starts from the CityCAT state at a random "
            "start time and is rolled out 'steps' times on the model's own "
            "predictions; the loss averages all steps. Overrides --epochs. "
            "Empty: one-step teacher forcing for --epochs epochs."
        ),
    )
    parser.add_argument(
        "--max-starts-per-event",
        type=int,
        default=0,
        help="Random start times per training event and epoch (0 = all).",
    )
    parser.add_argument(
        "--detach-rollout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop gradients between rollout steps (cheaper, shorter credit).",
    )
    parser.add_argument(
        "--select-metric", choices=tuple(SELECT_METRICS), default="tf_loss"
    )
    parser.add_argument(
        "--epoch-rollout-eval",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run the validation rollout every epoch (default: on if selecting on it).",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Start from the weights of a previous M17K checkpoint (same architecture).",
    )

    # Options that reproduce the original SWE-GNN training (Bentivoglio et al.
    # 2023, Sect. 3.3, Eq. 14, Algorithm 1; config.yaml and training/ of the
    # SWE-GNN-paper-repository). Defaults keep the earlier M17K behaviour.
    parser.add_argument(
        "--loss",
        choices=("mse", "swegnn_rmse"),
        default="mse",
        help=(
            "mse: mean squared error over H and Qmag. swegnn_rmse: per step, "
            "RMSE over nodes for each variable, weighted (1, --loss-q-weight) "
            "and summed, then averaged over rollout steps (training/loss.py)."
        ),
    )
    parser.add_argument("--loss-q-weight", type=float, default=3.0)
    parser.add_argument(
        "--curriculum-epochs",
        type=int,
        default=0,
        help=(
            "Original curriculum: rollout steps = min(epoch // N + 1, "
            "--max-rollout-steps) (training/train.py). 0 = off. "
            "Mutually exclusive with --rollout-schedule."
        ),
    )
    parser.add_argument("--max-rollout-steps", type=int, default=8)
    parser.add_argument(
        "--grad-clip-value",
        type=float,
        default=0.0,
        help="Element-wise gradient clipping (original: 0.5). 0 = off.",
    )
    parser.add_argument(
        "--accumulate-samples",
        type=int,
        default=1,
        help="Samples per optimizer step (gradient accumulation; original batch: 8).",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=0,
        help="StepLR step in epochs (original: 7). 0 = constant learning rate.",
    )
    parser.add_argument("--lr-gamma", type=float, default=0.9)
    parser.add_argument(
        "--early-stopping",
        choices=("since_best", "consecutive_increase"),
        default="since_best",
        help=(
            "since_best: epochs without improving the best selection value "
            "(only counted in the last curriculum stage). consecutive_increase: "
            "original rule, consecutive epochs whose validation value does not "
            "improve on the previous epoch."
        ),
    )
    args = parser.parse_args()
    if args.curriculum_epochs and args.rollout_schedule:
        parser.error("--curriculum-epochs and --rollout-schedule are mutually exclusive")
    if args.epoch_rollout_eval is None:
        args.epoch_rollout_eval = SELECT_METRICS[args.select_metric][0] == "rollout"

    if args.backbone_depth_mask is None:
        args.backbone_depth_mask = args.architecture == "swegnn"
    if args.architecture == "swegnn":
        args.rain_source_input = None
    return args


def parse_rollout_schedule(
    text: str, epochs: int, curriculum_epochs: int = 0, max_rollout_steps: int = 8
) -> list[int]:
    """Return the number of rollout steps for every epoch."""
    if curriculum_epochs > 0:
        # Original SWE-GNN: self.epoch is incremented before each training
        # step, rollout_steps = epoch // curriculum_epoch + 1, capped.
        return [
            min(epoch // curriculum_epochs + 1, max_rollout_steps)
            for epoch in range(1, epochs + 1)
        ]
    if not text.strip():
        return [1] * epochs
    per_epoch = []
    for item in text.split(","):
        try:
            steps, stage_epochs = (int(part) for part in item.strip().split(":"))
        except ValueError as error:
            raise ValueError(f"Invalid --rollout-schedule item {item!r}") from error
        if steps <= 0 or stage_epochs <= 0:
            raise ValueError(f"--rollout-schedule values must be positive: {item!r}")
        per_epoch.extend([steps] * stage_epochs)
    return per_epoch


def validate_args(args: argparse.Namespace) -> None:
    if args.start_time_index < 0:
        raise ValueError("--start-time-index must be nonnegative")
    if args.end_time_index > NT - 2:
        raise ValueError(f"--end-time-index must be at most {NT - 2}")
    if args.start_time_index > args.end_time_index:
        raise ValueError("--start-time-index must not exceed --end-time-index")
    for name in ("epochs", "patience", "hid_features", "K"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("lr", "rainfall_scale", "dt", "cell_area"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("min_delta", "weight_decay", "grad_clip_norm", "grad_clip_value",
                 "loss_q_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    for name in ("accumulate_samples", "max_rollout_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("curriculum_epochs", "lr_step_size"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    if not 0.0 < args.lr_gamma <= 1.0:
        raise ValueError("--lr-gamma must be in (0, 1]")


def git_commit() -> str | None:
    revision = os.environ.get("M17K_GIT_REVISION")
    if revision:
        return revision
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return result.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_events(dataset_root, split, event_ids, device):
    """Load events with the M14B-1 loader and add the zero-extended rainfall."""
    events = base.prepare_dynamic_events(dataset_root, split, event_ids, device)
    for event in events:
        rain = event["rainfall_global"]
        event["rainfall_extended"] = torch.cat([rain, rain.new_zeros(1)])
    return events


def interval_rainfall(event: dict, t: int, alignment: str):
    rain = event["rainfall_extended"]
    if alignment == "original":
        return rain[t], rain[t + 1]
    if alignment == "alternative":
        return rain[t + 1], rain[t + 2]
    raise ValueError(f"Unknown rainfall alignment {alignment!r}")


def build_sample(
    static_graph: dict,
    event: dict,
    t: int,
    args: argparse.Namespace,
    H_t: torch.Tensor | None = None,
    Qmag_t: torch.Tensor | None = None,
) -> Data:
    """Build one sample. H_t/Qmag_t default to CityCAT (teacher forcing)."""
    if not 0 <= t <= NT - 2:
        raise ValueError(f"time index must satisfy 0 <= t <= {NT - 2}; got {t}")
    if H_t is None:
        H_t = event["H"][:, t]
    if Qmag_t is None:
        Qmag_t = event["Qmag"][:, t]
    rain_a, rain_b = interval_rainfall(event, t, args.rainfall_alignment)
    x = torch.stack(
        [
            static_graph["slope_x"],
            static_graph["slope_y"],
            (rain_a * args.rainfall_scale).expand(NUM_NODES),
            (rain_b * args.rainfall_scale).expand(NUM_NODES),
            static_graph["DEM"],
            H_t,
            Qmag_t,
        ],
        dim=1,
    )
    y = torch.stack([event["H"][:, t + 1], event["Qmag"][:, t + 1]], dim=1)
    sample = Data(
        x=x,
        y=y,
        edge_index=static_graph["edge_index"],
        edge_attr=static_graph["edge_attr"],
    )
    sample.num_nodes = NUM_NODES
    return sample


def build_model(args, static_graph, active, device):
    swegnn_repo = str(args.swegnn_repo.expanduser())
    if swegnn_repo not in sys.path:
        sys.path.insert(0, swegnn_repo)
    from models.gnn import GNN

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    backbone_config = {
        "node_features": 7,
        "edge_features": 3,
        "type_GNN": "SWEGNN",
        "hid_features": args.hid_features,
        "K": args.K,
        "gnn_activation": "tanh",
        "dropout": 0.0,
        "mlp_layers": 2,
        "mlp_activation": "prelu",
        "seed": args.seed,
        "with_filter_matrix": True,
        "with_gradient": True,
        "with_WL": True,
        "previous_t": 1,
    }
    backbone = GNN(**backbone_config, device=device).to(device)
    if args.architecture == "rainfall_aware":
        model = RainfallAwareSWEGNNV2(
            backbone,
            static_graph["DEM"],
            active,
            rain_source_input=args.rain_source_input,
            backbone_depth_mask=args.backbone_depth_mask,
            rain_scale=args.rainfall_scale,
            dt=args.dt,
        ).to(device)
    else:
        if not args.backbone_depth_mask:
            from experimental.rainfall_aware_swegnn_v2 import _identity_depth_mask

            backbone._mask_small_WD = _identity_depth_mask
        model = backbone
    return model, backbone_config


def masked_mse(pred, target, mask):
    error = (pred - target).square()
    if mask is not None:
        error = error[mask]
    return error.mean()


def csi(pred_H, true_H, threshold):
    pred_wet = pred_H >= threshold
    true_wet = true_H >= threshold
    hits = (pred_wet & true_wet).sum().item()
    misses = (~pred_wet & true_wet).sum().item()
    false_alarms = (pred_wet & ~true_wet).sum().item()
    denominator = hits + misses + false_alarms
    return None if denominator == 0 else hits / denominator


def threshold_label(threshold: float) -> str:
    return f"CSI_{threshold:.2f}".replace(".", "p")


def mean_defined(values):
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


RMSE_EPS = 1e-12  # keeps sqrt differentiable when a step is predicted exactly


def where_water(pred, target, active):
    """Active cells where prediction or target is nonzero (original only_where_water)."""
    wet = ((pred != 0) | (target != 0)).any(dim=1)
    return active & wet


def make_loss_fn(args, active):
    """Per-step training/validation loss selected by --loss and --loss-mask."""

    def region(pred, target):
        if args.loss_mask == "all":
            return None
        if args.loss_mask == "active":
            return active
        return where_water(pred.detach(), target, active)

    def loss_fn(pred, target):
        mask = region(pred, target)
        if mask is not None and not bool(mask.any()):
            return pred.sum() * 0.0  # nothing wet yet: zero loss, graph kept
        if args.loss == "mse":
            return masked_mse(pred, target, mask)
        diff = pred - target
        if mask is not None:
            diff = diff[mask]
        rmse = torch.sqrt(diff.square().mean(dim=0) + RMSE_EPS)
        return rmse[0] + args.loss_q_weight * rmse[1]

    return loss_fn


def training_starts(num_events, time_indices, rollout_steps, max_starts_per_event):
    """(event_index, start_t) pairs whose rollout stays inside the time range."""
    last_start = time_indices[-1] - rollout_steps + 1
    starts = [t for t in time_indices if t <= last_start]
    if not starts:
        raise ValueError(
            f"{rollout_steps} rollout steps do not fit in "
            f"T{time_indices[0]}..T{time_indices[-1] + 1}"
        )
    pairs = []
    for event_index in range(num_events):
        event_starts = starts
        if 0 < max_starts_per_event < len(starts):
            event_starts = random.sample(starts, max_starts_per_event)
        pairs.extend((event_index, t) for t in event_starts)
    return pairs


def train_one_epoch(
    model, optimizer, static_graph, events, pairs, loss_fn, args, rollout_steps=1
):
    """One epoch of k-step training. k=1 is one-step teacher forcing.

    Each pair starts from the CityCAT state at start_t. For k>1 the model is fed
    its own H and Qmag for the following steps and the loss averages all steps.
    """
    epoch_pairs = pairs.copy()
    if args.shuffle:
        random.shuffle(epoch_pairs)
    model.train()
    losses, grad_norms = [], []
    accumulate = args.accumulate_samples
    optimizer.zero_grad(set_to_none=True)
    for sample_number, (event_index, start_t) in enumerate(epoch_pairs, start=1):
        event = events[event_index]
        t = start_t
        H, Q = None, None
        step_losses = []
        for step in range(rollout_steps):
            t = start_t + step
            sample = build_sample(static_graph, event, t, args, H_t=H, Qmag_t=Q)
            pred = base.extract_prediction(
                model(sample), f"training {event['event_id']} T{t}"
            )
            if tuple(pred.shape) != (NUM_NODES, 2):
                raise ValueError(f"Unexpected prediction shape {tuple(pred.shape)}")
            step_losses.append(loss_fn(pred, sample.y))
            H = pred[:, 0].clamp_min(0.0)
            Q = pred[:, 1].clamp_min(0.0)
            if args.detach_rollout:
                H, Q = H.detach(), Q.detach()
        loss = torch.stack(step_losses).mean()
        base.assert_finite(f"{event['event_id']} T{start_t} training loss", loss)
        losses.append(loss.item())
        # With accumulation, the update uses the mean loss of the group,
        # like a mini-batch of `accumulate` samples.
        group_size = min(accumulate, len(epoch_pairs) - (sample_number - 1) // accumulate * accumulate)
        (loss / group_size).backward()
        if sample_number % accumulate != 0 and sample_number != len(epoch_pairs):
            continue
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError(f"Non-finite gradient in {name} at {event['event_id']} T{t}")
        if args.grad_clip_value > 0.0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip_value)
        if args.grad_clip_norm > 0.0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            if not torch.isfinite(norm).item():
                raise ValueError(
                    f"Non-finite gradient norm at {event['event_id']} T{t}: "
                    f"{norm.item()}"
                )
            grad_norms.append(norm.item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return (
        sum(losses) / len(losses),
        min(losses),
        max(losses),
        sum(grad_norms) / len(grad_norms) if grad_norms else None,
    )


def evaluate_teacher_forced(model, static_graph, events, time_indices, active, loss_fn, args):
    losses, H_rmse, Q_rmse, csi01, csi05, peak_ratios = [], [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for event in events:
            max_pred, max_true = -math.inf, -math.inf
            for t in time_indices:
                sample = build_sample(static_graph, event, t, args)
                pred = base.extract_prediction(model(sample), f"validation {event['event_id']} T{t}")
                base.assert_finite("validation prediction", pred)
                losses.append(loss_fn(pred, sample.y).item())
                H_rmse.append(masked_mse(pred[:, 0], sample.y[:, 0], active).sqrt().item())
                Q_rmse.append(masked_mse(pred[:, 1], sample.y[:, 1], active).sqrt().item())
                csi01.append(csi(pred[active, 0], sample.y[active, 0], 0.01))
                csi05.append(csi(pred[active, 0], sample.y[active, 0], 0.05))
                max_pred = max(max_pred, pred[active, 0].max().item())
                max_true = max(max_true, sample.y[active, 0].max().item())
            peak_ratios.append(base.safe_peak_ratio(max_pred, max_true))
    return {
        "valid_loss_mean": sum(losses) / len(losses),
        "valid_H_RMSE_active_mean": sum(H_rmse) / len(H_rmse),
        "valid_Qmag_RMSE_active_mean": sum(Q_rmse) / len(Q_rmse),
        "valid_H_peak_ratio_mean": mean_defined(peak_ratios),
        "valid_CSI_0p01_mean": mean_defined(csi01),
        "valid_CSI_0p05_mean": mean_defined(csi05),
    }


def evaluate_rollout(model, static_graph, events, time_indices, active, args):
    """Autoregressive rollout from the first time index, fed with own predictions."""
    rows = []
    event_val_losses = []
    model.eval()
    with torch.inference_mode():
        for event in events:
            # Original SWE-GNN validation loss: RMSE per variable over all
            # wet (pred or target nonzero) node-times of the event, then the
            # mean over variables (utils/miscellaneous.get_rollout_loss).
            wet_sq = torch.zeros(2, dtype=torch.float64)
            wet_count = 0
            t0 = time_indices[0]
            H = event["H"][:, t0].clone()
            Q = event["Qmag"][:, t0].clone()
            for t in time_indices:
                sample = build_sample(static_graph, event, t, args, H_t=H, Qmag_t=Q)
                pred = base.extract_prediction(model(sample), f"rollout {event['event_id']} T{t}")
                base.assert_finite("rollout prediction", pred)
                H = pred[:, 0].clamp_min(0.0)
                Q = pred[:, 1].clamp_min(0.0)
                true_H = sample.y[:, 0]
                true_Q = sample.y[:, 1]
                prediction = torch.stack([H, Q], dim=1)
                wet = where_water(prediction, sample.y, active)
                wet_sq += (prediction - sample.y)[wet].double().square().sum(dim=0).cpu()
                wet_count += int(wet.sum().item())
                row = {
                    "event_id": event["event_id"],
                    "input_time_index": t,
                    "target_time_index": t + 1,
                    "H_RMSE_active": masked_mse(H, true_H, active).sqrt().item(),
                    "Qmag_RMSE_active": masked_mse(Q, true_Q, active).sqrt().item(),
                    "true_H_max": true_H[active].max().item(),
                    "pred_H_max": H[active].max().item(),
                    "true_Qmag_max": true_Q[active].max().item(),
                    "pred_Qmag_max": Q[active].max().item(),
                    "true_wet_cells_0p01": int((true_H[active] >= 0.01).sum().item()),
                    "pred_wet_cells_0p01": int((H[active] >= 0.01).sum().item()),
                    "pred_positive_cells": int((H[active] > 0).sum().item()),
                    "true_volume_m3": true_H[active].sum().item() * args.cell_area,
                    "pred_volume_m3": H[active].sum().item() * args.cell_area,
                }
                for threshold in CSI_THRESHOLDS:
                    row[threshold_label(threshold)] = csi(H[active], true_H[active], threshold)
                rows.append(row)
            if wet_count > 0:
                event_val_losses.append((wet_sq / wet_count).sqrt().mean().item())

    summary = {
        "swegnn_val_loss": mean_defined(event_val_losses),
        "num_events": len(events),
        "num_steps": len(rows),
        "H_RMSE_active_mean": sum(r["H_RMSE_active"] for r in rows) / len(rows),
        "Qmag_RMSE_active_mean": sum(r["Qmag_RMSE_active"] for r in rows) / len(rows),
        "steps_with_all_dry_prediction": sum(r["pred_positive_cells"] == 0 for r in rows),
        "steps_with_zero_Qmag_prediction": sum(r["pred_Qmag_max"] == 0.0 for r in rows),
    }
    for threshold in CSI_THRESHOLDS:
        label = threshold_label(threshold)
        summary[f"{label}_mean"] = mean_defined([r[label] for r in rows])
    final_rows = [r for r in rows if r["input_time_index"] == time_indices[-1]]
    summary["final_step_pred_over_true_volume_mean"] = mean_defined(
        [
            r["pred_volume_m3"] / r["true_volume_m3"] if r["true_volume_m3"] > 0 else None
            for r in final_rows
        ]
    )
    return summary, rows


class Persistence(torch.nn.Module):
    """Reference model: H(t+1) = H(t), Qmag(t+1) = Qmag(t)."""

    def forward(self, graph):
        return graph.x[:, 5:7].clone()


def load_init_checkpoint(model, path, m17k_config, backbone_config, device):
    checkpoint = torch.load(path, map_location=device)
    for key in ("architecture", "rain_source_input", "backbone_depth_mask"):
        if checkpoint["m17k_config"].get(key) != m17k_config.get(key):
            raise ValueError(
                f"--init-checkpoint {key}={checkpoint['m17k_config'].get(key)!r} "
                f"does not match this run ({m17k_config.get(key)!r})"
            )
    if checkpoint["backbone_config"] != backbone_config:
        raise ValueError("--init-checkpoint backbone_config does not match this run")
    model.load_state_dict(checkpoint["model_state_dict"])
    return {
        "path": str(path),
        "epoch": checkpoint.get("epoch"),
        "git_commit": checkpoint["m17k_config"].get("git_commit"),
    }


def parameter_snapshot(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def parameter_change_by_module(model, before):
    """Total absolute change per top-level module (backbone/rainfall_source)."""
    changes = {}
    for name, parameter in model.named_parameters():
        module = name.split(".")[0]
        change = (parameter.detach() - before[name]).abs().sum().item()
        changes[module] = changes.get(module, 0.0) + change
    return changes


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    validate_args(args)
    dataset_root = args.dataset_root.expanduser()
    output_dir = args.output_dir.expanduser()
    device = base.select_device(args.device, args.allow_cpu)

    paths = {
        "history": output_dir / "m17k_history.csv",
        "best_checkpoint": output_dir / "m17k_best_checkpoint.pt",
        "final_checkpoint": output_dir / "m17k_final_checkpoint.pt",
        "rollout_csv": output_dir / "m17k_valid_rollout_steps.csv",
        "summary": output_dir / "m17k_summary.json",
    }
    existing = [p for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    static_graph = base.load_static_graph(dataset_root)
    if float(static_graph["cell_size"]) != 5.0:
        raise ValueError(
            f"Expected the corrected 5 m graph, got cell_size={static_graph['cell_size']}"
        )
    manifest = base.load_manifest(dataset_root)
    ids = {
        split: base.derive_split_event_ids(dataset_root, manifest, split)
        for split in ("train", "valid", "test")
    }
    if "R019" not in ids["test"]:
        raise ValueError("R019 must be in the test split")
    if set(ids["train"]) & set(ids["valid"]) or set(ids["train"] + ids["valid"]) & set(ids["test"]):
        raise ValueError("Split overlap detected")
    train_ids = base.apply_event_limit(ids["train"], args.max_train_events, "train")
    valid_ids = base.apply_event_limit(ids["valid"], args.max_valid_events, "valid")

    time_indices = list(range(args.start_time_index, args.end_time_index + 1))
    schedule = parse_rollout_schedule(
        args.rollout_schedule, args.epochs, args.curriculum_epochs, args.max_rollout_steps
    )
    if max(schedule) > len(time_indices):
        raise ValueError(
            f"Rollout of {max(schedule)} steps exceeds the {len(time_indices)} "
            "transitions in the time range"
        )
    if args.max_starts_per_event < 0:
        raise ValueError("--max-starts-per-event must be nonnegative")
    if args.init_checkpoint is not None and not args.init_checkpoint.is_file():
        raise FileNotFoundError(f"--init-checkpoint not found: {args.init_checkpoint}")
    select_source, select_key, higher_is_better = SELECT_METRICS[args.select_metric]
    if select_source == "rollout" and not args.epoch_rollout_eval:
        raise ValueError("Selecting on a rollout metric requires --epoch-rollout-eval")

    static_dev = base.static_graph_to_device(static_graph, device)
    active = active_mask_from_dem(static_dev["DEM"])
    active_count = int(active.sum().item())
    if active_count != EXPECTED_ACTIVE_CELLS:
        raise ValueError(f"Expected {EXPECTED_ACTIVE_CELLS} active cells, got {active_count}")
    loss_fn = make_loss_fn(args, active)

    m17k_config = {
        "architecture": args.architecture,
        "rainfall_alignment": args.rainfall_alignment,
        "rain_source_input": args.rain_source_input,
        "backbone_depth_mask": args.backbone_depth_mask,
        "loss_mask": args.loss_mask,
        "metrics_mask": "active",
        "active_mask_rule": "DEM != DEM.max()",
        "active_cell_count": active_count,
        "rainfall_extension": "PT rainfall_global + one zero (TXT 7200 s)",
        "rollout_schedule": (
            f"original: min(epoch // {args.curriculum_epochs} + 1, {args.max_rollout_steps})"
            if args.curriculum_epochs
            else args.rollout_schedule or f"1:{args.epochs}"
        ),
        "loss": args.loss,
        "loss_q_weight": args.loss_q_weight if args.loss == "swegnn_rmse" else None,
        "grad_clip_value": args.grad_clip_value,
        "grad_clip_norm": args.grad_clip_norm,
        "accumulate_samples": args.accumulate_samples,
        "lr": args.lr,
        "lr_step_size": args.lr_step_size,
        "lr_gamma": args.lr_gamma if args.lr_step_size else None,
        "early_stopping": args.early_stopping,
        "max_starts_per_event": args.max_starts_per_event,
        "detach_rollout": args.detach_rollout,
        "select_metric": args.select_metric,
        "git_commit": git_commit(),
    }

    print("=== Stockbridge SWE-GNN M17K controlled training ===")
    print("Dataset root:", dataset_root)
    print("Output directory:", output_dir)
    print("Device:", device)
    print("M17K config:", json.dumps(m17k_config))
    print(f"Train events: {len(train_ids)} | Valid events: {len(valid_ids)}")
    print(f"Time range: T{time_indices[0]}->T{time_indices[0] + 1} ... "
          f"T{time_indices[-1]}->T{time_indices[-1] + 1}")
    if args.max_train_events or args.max_valid_events:
        print("WARNING: debugging subset, not the full 70/15 run.")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_events = prepare_events(dataset_root, "train", train_ids, device)
    valid_events = prepare_events(dataset_root, "valid", valid_ids, device)

    model, backbone_config = build_model(args, static_dev, active, device)
    init_info = None
    if args.init_checkpoint is not None:
        init_info = load_init_checkpoint(
            model, args.init_checkpoint, m17k_config, backbone_config, device
        )
        m17k_config["init_checkpoint"] = init_info
        print("Initialised from:", init_info)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_step_size:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    parameter_count = sum(p.numel() for p in model.parameters())
    before = parameter_snapshot(model)
    print("Backbone config:", backbone_config)
    print("Parameter count:", parameter_count)
    print("Rollout schedule (steps per epoch):", schedule)
    print("Selection metric:", args.select_metric)

    persistence_valid = evaluate_teacher_forced(
        Persistence(), static_dev, valid_events, time_indices, active, loss_fn, args
    )
    print("Persistence reference (teacher forced):", persistence_valid)
    initial_valid = evaluate_teacher_forced(
        model, static_dev, valid_events, time_indices, active, loss_fn, args
    )
    print("Initial validation:", initial_valid)
    initial_rollout = None
    if args.epoch_rollout_eval:
        initial_rollout, _ = evaluate_rollout(
            model, static_dev, valid_events, time_indices, active, args
        )
        print("Initial rollout:", initial_rollout)

    base.prepare_output_dir(output_dir, dataset_root, args.overwrite)

    def checkpoint(epoch, summary, best_epoch, best_loss, early_stopped):
        return {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_epoch": best_epoch,
            "best_valid_loss": best_loss,
            "early_stopped": early_stopped,
            "args": {k: json_safe(v) for k, v in vars(args).items()},
            "m17k_config": m17k_config,
            "backbone_config": backbone_config,
            "train_event_ids": train_ids,
            "valid_event_ids": valid_ids,
            "input_features": base.INPUT_FEATURES,
            "edge_features": base.EDGE_FEATURES,
            "output_variables": base.OUTPUT_VARIABLES,
            "start_time_index": args.start_time_index,
            "end_time_index": args.end_time_index,
            "valid_summary": summary,
        }

    def selection_value(tf_summary, rollout_summary):
        source = tf_summary if select_source == "tf" else rollout_summary
        value = source[select_key]
        if value is None:
            return -math.inf if higher_is_better else math.inf
        return value

    def improves(value, best):
        if higher_is_better:
            return value > best + args.min_delta
        return value < best - args.min_delta

    best_loss = math.inf  # best teacher-forced validation loss seen (diagnostic)
    best_selection = -math.inf if higher_is_better else math.inf
    best_epoch, best_summary = 0, None
    since_improvement, epochs_completed, early_stopped = 0, 0, False
    consecutive_without_gain, previous_value = 0, None
    total_epochs = len(schedule)
    with paths["history"].open("w", newline="") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDNAMES)
        writer.writeheader()
        for epoch, rollout_steps in enumerate(schedule, start=1):
            if epoch > 1 and rollout_steps != schedule[epoch - 2]:
                # New curriculum stage: the since_best patience counter restarts.
                since_improvement = 0
                print(f"--- Curriculum stage: {rollout_steps} rollout steps ---")
            final_stage = rollout_steps == schedule[-1] and all(
                steps == schedule[-1] for steps in schedule[epoch - 1 :]
            )
            epoch_start = time.perf_counter()
            pairs = training_starts(
                len(train_events), time_indices, rollout_steps, args.max_starts_per_event
            )
            train_summary = train_one_epoch(
                model, optimizer, static_dev, train_events, pairs, loss_fn, args,
                rollout_steps=rollout_steps,
            )
            valid = evaluate_teacher_forced(
                model, static_dev, valid_events, time_indices, active, loss_fn, args
            )
            rollout = None
            if args.epoch_rollout_eval:
                rollout, _ = evaluate_rollout(
                    model, static_dev, valid_events, time_indices, active, args
                )
            best_loss = min(best_loss, valid["valid_loss_mean"])
            value = selection_value(valid, rollout)
            is_best = improves(value, best_selection)
            if is_best:
                best_selection, best_epoch = value, epoch
                best_summary = {"teacher_forced": valid, "rollout": rollout}
                since_improvement = 0
                torch.save(
                    checkpoint(epoch, best_summary, best_epoch, best_loss, False),
                    paths["best_checkpoint"],
                )
            else:
                since_improvement += 1
            # Original rule (training/train.py): from the third epoch, count
            # consecutive epochs whose value is not better than the previous one.
            if previous_value is not None and epoch >= 3:
                if improves(value, previous_value):
                    consecutive_without_gain = 0
                else:
                    consecutive_without_gain += 1
            previous_value = value
            learning_rate = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()
            epochs_completed = epoch
            elapsed = time.perf_counter() - epoch_start
            rollout = rollout or {}
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss_mean": train_summary[0],
                    "train_loss_min": train_summary[1],
                    "train_loss_max": train_summary[2],
                    "mean_grad_norm": train_summary[3],
                    **valid,
                    "best_valid_loss_so_far": best_loss,
                    "best_epoch_so_far": best_epoch,
                    "is_best": is_best,
                    "epoch_elapsed_seconds": elapsed,
                    "rollout_steps": rollout_steps,
                    "train_starts": len(pairs),
                    "rollout_H_RMSE_active_mean": rollout.get("H_RMSE_active_mean"),
                    "rollout_CSI_0p01_mean": rollout.get("CSI_0p01_mean"),
                    "rollout_CSI_0p05_mean": rollout.get("CSI_0p05_mean"),
                    "rollout_steps_all_dry": rollout.get("steps_with_all_dry_prediction"),
                    "rollout_steps_zero_Qmag": rollout.get("steps_with_zero_Qmag_prediction"),
                    "selection_value": value,
                    "best_selection_so_far": best_selection,
                    "rollout_swegnn_val_loss": rollout.get("swegnn_val_loss"),
                    "learning_rate": learning_rate,
                }
            )
            history_file.flush()
            rollout_text = ""
            if rollout:
                rollout_text = (
                    f" | AR CSI0.05 {rollout['CSI_0p05_mean']} "
                    f"H_RMSE {rollout['H_RMSE_active_mean']:.4g} "
                    f"zeroQ {rollout['steps_with_zero_Qmag_prediction']}"
                )
            print(
                f"Epoch {epoch}/{total_epochs} k={rollout_steps} | "
                f"train {train_summary[0]:.6g} | TF valid {valid['valid_loss_mean']:.6g}"
                f"{rollout_text} | best {best_epoch} | {elapsed:.1f} s",
                flush=True,
            )
            if args.early_stopping == "since_best":
                stop = final_stage and since_improvement >= args.patience
            else:
                stop = consecutive_without_gain >= args.patience
            if stop:
                early_stopped = True
                break

    parameter_changes = parameter_change_by_module(model, before)
    print("Absolute parameter change by module:", parameter_changes)
    unchanged = [name for name, change in parameter_changes.items() if not change > 0.0]
    if unchanged:
        raise RuntimeError(f"Parameters did not change in: {unchanged}")

    final_valid = evaluate_teacher_forced(
        model, static_dev, valid_events, time_indices, active, loss_fn, args
    )
    torch.save(
        checkpoint(epochs_completed, final_valid, best_epoch, best_loss, early_stopped),
        paths["final_checkpoint"],
    )

    rollout_summary = None
    if args.rollout_eval:
        best_state = torch.load(paths["best_checkpoint"], map_location=device)
        model.load_state_dict(best_state["model_state_dict"])
        rollout_summary, rollout_rows = evaluate_rollout(
            model, static_dev, valid_events, time_indices, active, args
        )
        with paths["rollout_csv"].open("w", newline="") as rollout_file:
            writer = csv.DictWriter(rollout_file, fieldnames=ROLLOUT_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rollout_rows)
        print("Autoregressive rollout (best checkpoint):", rollout_summary)
        for row in rollout_rows:
            if row["event_id"] == valid_ids[0] and row["target_time_index"] in (1, 2, 4, 8, 12, 24):
                print(
                    f"  {row['event_id']} T{row['target_time_index']:02d} | "
                    f"wet>=1cm true {row['true_wet_cells_0p01']:5d} pred {row['pred_wet_cells_0p01']:5d} | "
                    f"Hmax true {row['true_H_max']:.3f} pred {row['pred_H_max']:.3f} | "
                    f"Qmax true {row['true_Qmag_max']:.3f} pred {row['pred_Qmag_max']:.3f}"
                )

    summary = {
        "script_name": Path(__file__).name,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "m17k_config": m17k_config,
        "backbone_config": backbone_config,
        "args": {k: json_safe(v) for k, v in vars(args).items()},
        "train_event_ids": train_ids,
        "valid_event_ids": valid_ids,
        "parameter_count": parameter_count,
        "persistence_valid_summary": persistence_valid,
        "initial_valid_summary": initial_valid,
        "initial_rollout_summary": initial_rollout,
        "rollout_schedule_per_epoch": schedule,
        "select_metric": args.select_metric,
        "best_selection_value": best_selection,
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "best_valid_summary": best_summary,
        "final_valid_summary": final_valid,
        "epochs_completed": epochs_completed,
        "early_stopped": early_stopped,
        "parameter_change_by_module": parameter_changes,
        "rollout_summary_best_checkpoint": rollout_summary,
        "device": str(device),
        "total_elapsed_seconds": time.perf_counter() - run_start,
    }
    if device.type == "cuda":
        summary["cuda_device_name"] = torch.cuda.get_device_name(device)
        summary["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    with paths["summary"].open("w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print("Outputs:", {k: str(v) for k, v in paths.items()})
    print("Stockbridge SWE-GNN M17K controlled training passed.")


if __name__ == "__main__":
    main()
