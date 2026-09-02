import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path

import torch


PT_ROOT = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/pt/stockbridge_phase2_core_v01"
)
DEFAULT_OUTPUT_ROOT = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/swegnn"
    / "stockbridge_phase2_swegnn_full_v01"
)

NY = 305
NX = 326
NT = 25
CHANNELS = 5
NUM_NODES = 99430
EXPECTED_EDGE_COUNT = 396458
SPLITS = {"train": 70, "valid": 15, "test": 15}
EXPECTED_TENSOR_SHAPE = (NY, NX, NT, CHANNELS)

CHANNEL_MEANING = {
    0: "H",
    1: "Vx",
    2: "Vy",
    3: "rainfall",
    4: "DEM",
}
INPUT_FEATURES = [
    "slope_x",
    "slope_y",
    "rainfall_t",
    "rainfall_t_plus_1",
    "DEM",
    "H_t",
    "Qmag_t",
]
EDGE_FEATURES = ["dx", "dy", "edge_slope"]
OUTPUT_VARIABLES = ["H", "Qmag"]

RAINFALL_UNIFORMITY_TOLERANCE = 1e-8
NONNEGATIVE_TOLERANCE = 1e-8
DEM_ATOL = 1e-6
DEM_RTOL = 1e-6

MANIFEST_FIELDNAMES = [
    "event_id",
    "split",
    "source_path",
    "dynamic_path",
    "H_max",
    "Qmag_max",
    "rainfall_raw_max",
    "rainfall_raw_sum",
    "rainfall_scaled_reference_max",
    "rainfall_scaled_reference_sum",
    "rainfall_spatial_max_abs_diff",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for milestone 13A."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare shared-static and per-event dynamic SWE-GNN files for "
            "the full Stockbridge Phase 2 split."
        )
    )
    parser.add_argument("--pt-root", type=Path, default=PT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--cell-size", type=float, default=1.0)
    parser.add_argument(
        "--rainfall-scale-reference",
        type=float,
        default=300000.0,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def event_id_from_path(path: Path) -> str:
    """Extract a Stockbridge event ID such as R019 from a file path."""
    match = re.search(r"(R\d{3})", path.name)
    if match is None:
        raise ValueError(
            f"Could not find an event ID matching R followed by three digits "
            f"in filename: {path.name}"
        )
    return match.group(1)


def find_split_files(
    pt_root: Path,
    split: str,
    expected_count: int,
) -> list[Path]:
    """Find one split's source tensors sorted by parsed event ID."""
    split_dir = pt_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Expected split directory not found: {split_dir}")

    paths = list(split_dir.glob("*.pt"))
    paths.sort(key=lambda path: event_id_from_path(path))
    if len(paths) != expected_count:
        raise ValueError(
            f"Expected exactly {expected_count} .pt files in {split_dir}, "
            f"found {len(paths)}"
        )
    return paths


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    """Assert that a tensor contains neither NaN nor infinite values."""
    assert torch.isfinite(tensor).all().item(), f"{name} contains NaN/inf values"


def load_event_tensor(path: Path) -> torch.Tensor:
    """Load and validate one preprocessed Stockbridge event tensor."""
    if not path.is_file():
        raise FileNotFoundError(f"Event tensor not found: {path}")
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor in {path}, got {type(tensor).__name__}"
        )
    if tuple(tensor.shape) != EXPECTED_TENSOR_SHAPE:
        raise ValueError(
            f"Expected tensor shape {EXPECTED_TENSOR_SHAPE} in {path}, "
            f"got {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError(
            f"Expected floating-point tensor in {path}, got {tensor.dtype}"
        )
    assert_finite(str(path), tensor)
    return tensor


def calculate_dem_slopes(
    DEM: torch.Tensor,
    cell_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate finite-difference DEM slopes in physical grid units."""
    dem_grid = DEM.reshape(NY, NX)
    slope_x = torch.empty_like(dem_grid)
    slope_y = torch.empty_like(dem_grid)

    slope_x[:, 1:-1] = (
        dem_grid[:, 2:] - dem_grid[:, :-2]
    ) / (2.0 * cell_size)
    slope_x[:, 0] = (dem_grid[:, 1] - dem_grid[:, 0]) / cell_size
    slope_x[:, -1] = (dem_grid[:, -1] - dem_grid[:, -2]) / cell_size

    slope_y[1:-1, :] = (
        dem_grid[2:, :] - dem_grid[:-2, :]
    ) / (2.0 * cell_size)
    slope_y[0, :] = (dem_grid[1, :] - dem_grid[0, :]) / cell_size
    slope_y[-1, :] = (dem_grid[-1, :] - dem_grid[-2, :]) / cell_size
    return slope_x.reshape(-1), slope_y.reshape(-1)


def build_static_graph_from_reference(
    reference_tensor: torch.Tensor,
    cell_size: float,
) -> dict:
    """Build the shared static directed four-neighbour grid graph."""
    DEM = (
        reference_tensor[:, :, 0, 4]
        .reshape(NUM_NODES)
        .contiguous()
        .float()
    )
    node_ids = torch.arange(NUM_NODES, dtype=torch.long).reshape(NY, NX)

    left = node_ids[:, :-1].reshape(-1)
    right = node_ids[:, 1:].reshape(-1)
    top = node_ids[:-1, :].reshape(-1)
    bottom = node_ids[1:, :].reshape(-1)
    row = torch.cat([left, right, top, bottom])
    col = torch.cat([right, left, bottom, top])
    edge_index = torch.stack([row, col], dim=0)

    yy, xx = torch.meshgrid(
        torch.arange(NY, dtype=torch.float32),
        torch.arange(NX, dtype=torch.float32),
        indexing="ij",
    )
    pos = torch.stack(
        [xx.reshape(-1) * cell_size, yy.reshape(-1) * cell_size],
        dim=1,
    )
    edge_relative_distance = pos[col] - pos[row]
    edge_distance = torch.linalg.norm(edge_relative_distance, dim=1)
    edge_slope = (DEM[col] - DEM[row]) / edge_distance
    edge_attr = torch.cat(
        [edge_relative_distance, edge_slope.view(-1, 1)],
        dim=1,
    )
    slope_x, slope_y = calculate_dem_slopes(DEM, cell_size)

    expected_shapes = {
        "DEM": (NUM_NODES,),
        "slope_x": (NUM_NODES,),
        "slope_y": (NUM_NODES,),
        "pos": (NUM_NODES, 2),
        "edge_index": (2, EXPECTED_EDGE_COUNT),
        "edge_relative_distance": (EXPECTED_EDGE_COUNT, 2),
        "edge_distance": (EXPECTED_EDGE_COUNT,),
        "edge_slope": (EXPECTED_EDGE_COUNT,),
        "edge_attr": (EXPECTED_EDGE_COUNT, 3),
    }
    tensors = {
        "DEM": DEM,
        "slope_x": slope_x,
        "slope_y": slope_y,
        "pos": pos,
        "edge_index": edge_index,
        "edge_relative_distance": edge_relative_distance,
        "edge_distance": edge_distance,
        "edge_slope": edge_slope,
        "edge_attr": edge_attr,
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(tensors[name].shape)
        assert actual_shape == expected_shape, (
            f"Expected {name} shape {expected_shape}, got {actual_shape}"
        )
        assert_finite(name, tensors[name])
    assert not (row == col).any().item(), "Static graph contains self-loops"
    assert (edge_distance > 0).all().item(), "All edge distances must be positive"

    return {
        "ny": NY,
        "nx": NX,
        "nt": NT,
        "num_nodes": NUM_NODES,
        "num_edges": EXPECTED_EDGE_COUNT,
        "cell_size": cell_size,
        **tensors,
        "input_features": INPUT_FEATURES,
        "edge_features": EDGE_FEATURES,
        "output_variables": OUTPUT_VARIABLES,
    }


def convert_event(
    tensor: torch.Tensor,
    event_id: str,
    split: str,
    source_path: Path,
    dynamic_path: Path,
    static_DEM: torch.Tensor,
    rainfall_scale_reference: float,
    warnings: list[str],
) -> dict[str, object]:
    """Convert and save one dynamic event, returning its manifest row."""
    H = tensor[:, :, :, 0].reshape(NUM_NODES, NT).contiguous().float()
    Vx = tensor[:, :, :, 1].reshape(NUM_NODES, NT).contiguous().float()
    Vy = tensor[:, :, :, 2].reshape(NUM_NODES, NT).contiguous().float()
    rainfall = tensor[:, :, :, 3].contiguous().float()
    DEM_event = (
        tensor[:, :, 0, 4].reshape(NUM_NODES).contiguous().float()
    )

    if not torch.allclose(
        DEM_event,
        static_DEM,
        atol=DEM_ATOL,
        rtol=DEM_RTOL,
    ):
        max_abs_difference = torch.max(torch.abs(DEM_event - static_DEM)).item()
        raise ValueError(
            f"{event_id} DEM does not match the shared static DEM; "
            f"max_abs_difference={max_abs_difference:.8g}, "
            f"atol={DEM_ATOL}, rtol={DEM_RTOL}"
        )
    DEM_max_abs_difference = torch.max(
        torch.abs(DEM_event - static_DEM)
    ).item()
    if DEM_max_abs_difference > 0.0:
        warnings.append(
            f"{event_id}: DEM differs from the reference within tolerance "
            f"(max_abs_difference={DEM_max_abs_difference:.8g})."
        )

    rainfall_global = rainfall[0, 0, :].clone().contiguous().float()
    rainfall_spatial_max_abs_diff = torch.max(
        torch.abs(rainfall - rainfall_global.view(1, 1, NT))
    )
    rainfall_difference = rainfall_spatial_max_abs_diff.item()
    if rainfall_difference > RAINFALL_UNIFORMITY_TOLERANCE:
        raise ValueError(
            f"{event_id} rainfall is spatially non-uniform: "
            f"max_abs_difference={rainfall_difference:.8g} exceeds "
            f"{RAINFALL_UNIFORMITY_TOLERANCE}"
        )
    if rainfall_difference > 0.0:
        warnings.append(
            f"{event_id}: rainfall has a tiny spatial difference within "
            f"tolerance (max_abs_difference={rainfall_difference:.8g})."
        )

    Qx = Vx * H
    Qy = Vy * H
    Qmag = torch.sqrt(Qx.square() + Qy.square())
    for name, value in {
        "H": H,
        "Qmag": Qmag,
        "rainfall_global": rainfall_global,
    }.items():
        assert_finite(f"{event_id} {name}", value)
    assert torch.isfinite(rainfall_spatial_max_abs_diff).item(), (
        f"{event_id} rainfall spatial difference is NaN/inf"
    )

    H_min = H.min().item()
    Qmag_min = Qmag.min().item()
    if H_min < -NONNEGATIVE_TOLERANCE:
        raise ValueError(
            f"{event_id} H contains values below the allowed tolerance: "
            f"minimum={H_min:.8g}, tolerance={NONNEGATIVE_TOLERANCE}"
        )
    if Qmag_min < -NONNEGATIVE_TOLERANCE:
        raise ValueError(
            f"{event_id} Qmag contains values below the allowed tolerance: "
            f"minimum={Qmag_min:.8g}, tolerance={NONNEGATIVE_TOLERANCE}"
        )
    if H_min < 0.0:
        warnings.append(
            f"{event_id}: H has a tiny negative minimum within tolerance "
            f"({H_min:.8g})."
        )
    if Qmag_min < 0.0:
        warnings.append(
            f"{event_id}: Qmag has a tiny negative minimum within tolerance "
            f"({Qmag_min:.8g})."
        )

    H_max = H.max()
    Qmag_max = Qmag.max()
    rainfall_raw_max = rainfall_global.max()
    rainfall_raw_sum = rainfall_global.sum()
    rainfall_scaled_reference_max = (
        rainfall_raw_max * rainfall_scale_reference
    )
    rainfall_scaled_reference_sum = (
        rainfall_raw_sum * rainfall_scale_reference
    )
    for name, value in {
        "H_max": H_max,
        "Qmag_max": Qmag_max,
        "rainfall_raw_max": rainfall_raw_max,
        "rainfall_raw_sum": rainfall_raw_sum,
        "rainfall_scaled_reference_max": rainfall_scaled_reference_max,
        "rainfall_scaled_reference_sum": rainfall_scaled_reference_sum,
    }.items():
        assert_finite(f"{event_id} {name}", value)
    dynamic_event = {
        "event_id": event_id,
        "split": split,
        "source_path": str(source_path),
        "ny": NY,
        "nx": NX,
        "nt": NT,
        "num_nodes": NUM_NODES,
        "H": H,
        "Qmag": Qmag,
        "rainfall_global": rainfall_global,
        "rainfall_spatial_max_abs_diff": rainfall_spatial_max_abs_diff,
        "H_max": H_max,
        "Qmag_max": Qmag_max,
        "rainfall_raw_max": rainfall_raw_max,
        "rainfall_raw_sum": rainfall_raw_sum,
        "rainfall_scaled_reference_max": rainfall_scaled_reference_max,
        "rainfall_scaled_reference_sum": rainfall_scaled_reference_sum,
    }

    dynamic_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dynamic_event, dynamic_path)
    if not dynamic_path.is_file():
        raise RuntimeError(f"Dynamic event file was not created: {dynamic_path}")

    return {
        "event_id": event_id,
        "split": split,
        "source_path": str(source_path),
        "dynamic_path": str(dynamic_path),
        "H_max": H_max.item(),
        "Qmag_max": Qmag_max.item(),
        "rainfall_raw_max": rainfall_raw_max.item(),
        "rainfall_raw_sum": rainfall_raw_sum.item(),
        "rainfall_scaled_reference_max": (
            rainfall_scaled_reference_max.item()
        ),
        "rainfall_scaled_reference_sum": (
            rainfall_scaled_reference_sum.item()
        ),
        "rainfall_spatial_max_abs_diff": rainfall_difference,
    }


def recreate_output_root(
    output_root: Path,
    pt_root: Path,
    overwrite: bool,
) -> None:
    """Safely recreate only the requested milestone output directory."""
    output_exists = output_root.exists() or output_root.is_symlink()
    if output_exists and not overwrite:
        raise FileExistsError(
            f"Output root already exists; pass --overwrite to replace it: "
            f"{output_root}"
        )
    if output_exists:
        if output_root.is_symlink():
            raise ValueError(
                f"Refusing to delete a symlink used as --output-root: {output_root}"
            )
        if not output_root.is_dir():
            raise ValueError(
                f"Refusing to delete non-directory --output-root: {output_root}"
            )

        resolved_output = output_root.resolve()
        resolved_pt_root = pt_root.resolve()
        prohibited = {
            Path("/").resolve(),
            Path.home().resolve(),
            Path.cwd().resolve(),
            resolved_pt_root,
        }
        if resolved_output in prohibited:
            raise ValueError(
                f"Refusing unsafe --overwrite target: {resolved_output}"
            )
        resolved_cwd = Path.cwd().resolve()
        resolved_default_output = DEFAULT_OUTPUT_ROOT.resolve()
        if (
            resolved_output in resolved_cwd.parents
            or resolved_output in resolved_default_output.parents
        ):
            raise ValueError(
                "Refusing to delete a broad parent directory: "
                f"{resolved_output}"
            )
        if resolved_output in resolved_pt_root.parents:
            raise ValueError(
                "Refusing to delete an output root that contains the input root: "
                f"{resolved_output}"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)


def output_size_bytes(output_root: Path) -> int:
    """Return the total size of regular output files."""
    return sum(
        path.stat().st_size
        for path in output_root.rglob("*")
        if path.is_file()
    )


def main() -> None:
    args = parse_args()
    pt_root = args.pt_root.expanduser()
    output_root = args.output_root.expanduser()

    if not math.isfinite(args.cell_size) or args.cell_size <= 0.0:
        raise ValueError("--cell-size must be finite and greater than zero")
    if (
        not math.isfinite(args.rainfall_scale_reference)
        or args.rainfall_scale_reference <= 0.0
    ):
        raise ValueError(
            "--rainfall-scale-reference must be finite and greater than zero"
        )
    if not pt_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {pt_root}")
    if (
        output_root.exists() or output_root.is_symlink()
    ) and not args.overwrite:
        raise FileExistsError(
            f"Output root already exists; pass --overwrite to replace it: "
            f"{output_root}"
        )

    print("=== Stockbridge SWE-GNN milestone 13A full dataset preparation ===")
    print("Input root:", pt_root)
    print("Output root:", output_root)

    split_files = {
        split: find_split_files(pt_root, split, expected_count)
        for split, expected_count in SPLITS.items()
    }
    event_ids_by_split = {
        split: [event_id_from_path(path) for path in paths]
        for split, paths in split_files.items()
    }
    split_counts = {
        split: len(paths) for split, paths in split_files.items()
    }
    print("Split counts:", split_counts)
    for split in SPLITS:
        event_ids = event_ids_by_split[split]
        print(
            f"{split}: first 5 event IDs={event_ids[:5]} "
            f"count={len(event_ids)}"
        )

    all_event_ids = [
        event_id
        for split in SPLITS
        for event_id in event_ids_by_split[split]
    ]
    total_events = len(all_event_ids)
    if total_events != 100:
        raise ValueError(f"Expected 100 total events, found {total_events}")
    if len(set(all_event_ids)) != total_events:
        duplicates = sorted(
            {
                event_id
                for event_id in all_event_ids
                if all_event_ids.count(event_id) > 1
            }
        )
        raise ValueError(f"Event IDs are not unique across splits: {duplicates}")
    contains_R019_in_test = "R019" in event_ids_by_split["test"]
    if not contains_R019_in_test:
        raise ValueError("Expected R019 to be present in the test split")

    reference_path = split_files["train"][0]
    print("Reference file for static graph:", reference_path)
    reference_tensor = load_event_tensor(reference_path)
    static_graph = build_static_graph_from_reference(
        reference_tensor,
        args.cell_size,
    )
    if static_graph["num_edges"] != EXPECTED_EDGE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_EDGE_COUNT} static edges, "
            f"got {static_graph['num_edges']}"
        )
    print("Static graph shapes:")
    print("  DEM:", tuple(static_graph["DEM"].shape))
    print("  pos:", tuple(static_graph["pos"].shape))
    print("  edge_index:", tuple(static_graph["edge_index"].shape))
    print("  edge_attr:", tuple(static_graph["edge_attr"].shape))

    recreate_output_root(output_root, pt_root, args.overwrite)
    static_graph_path = output_root / "static_graph.pt"
    manifest_csv_path = output_root / "manifest.csv"
    manifest_json_path = output_root / "manifest.json"
    torch.save(static_graph, static_graph_path)
    if not static_graph_path.is_file():
        raise RuntimeError(f"Static graph file was not created: {static_graph_path}")

    warnings: list[str] = []
    manifest_rows = []
    dynamic_paths = []
    for split, paths in split_files.items():
        split_count = len(paths)
        print(f"Processing {split} split: 0/{split_count}")
        for event_number, source_path in enumerate(paths, start=1):
            event_id = event_id_from_path(source_path)
            tensor = load_event_tensor(source_path)
            dynamic_path = (
                output_root / "events" / split / f"{event_id}.pt"
            )
            manifest_row = convert_event(
                tensor=tensor,
                event_id=event_id,
                split=split,
                source_path=source_path,
                dynamic_path=dynamic_path,
                static_DEM=static_graph["DEM"],
                rainfall_scale_reference=args.rainfall_scale_reference,
                warnings=warnings,
            )
            manifest_rows.append(manifest_row)
            dynamic_paths.append(dynamic_path)
            if event_number % 10 == 0 or event_number == split_count:
                print(
                    f"Processing {split} split: "
                    f"{event_number}/{split_count}"
                )
            del tensor

    missing_dynamic_paths = [
        path for path in dynamic_paths if not path.is_file()
    ]
    if missing_dynamic_paths:
        formatted = "\n".join(f"  - {path}" for path in missing_dynamic_paths)
        raise RuntimeError(f"Dynamic event files are missing:\n{formatted}")
    if len(dynamic_paths) != total_events:
        raise RuntimeError(
            f"Expected {total_events} dynamic files, created {len(dynamic_paths)}"
        )

    with manifest_csv_path.open("w", newline="") as manifest_csv_file:
        writer = csv.DictWriter(
            manifest_csv_file,
            fieldnames=MANIFEST_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    created_files_summary = {
        "static_graph_files": 1,
        "dynamic_event_files": len(dynamic_paths),
        "dynamic_event_files_by_split": split_counts,
        "manifest_files": 2,
        "total_files": 1 + len(dynamic_paths) + 2,
    }
    manifest = {
        "script_name": Path(__file__).name,
        "pt_root": str(pt_root),
        "output_root": str(output_root),
        "static_graph_path": str(static_graph_path),
        "manifest_csv_path": str(manifest_csv_path),
        "manifest_json_path": str(manifest_json_path),
        "cell_size": args.cell_size,
        "rainfall_scale_reference": args.rainfall_scale_reference,
        "split_counts": split_counts,
        "expected_split_counts": SPLITS,
        "total_events": total_events,
        "input_tensor_shape": list(EXPECTED_TENSOR_SHAPE),
        "channel_meaning": CHANNEL_MEANING,
        "input_features": INPUT_FEATURES,
        "edge_features": EDGE_FEATURES,
        "output_variables": OUTPUT_VARIABLES,
        "num_nodes": NUM_NODES,
        "num_edges": EXPECTED_EDGE_COUNT,
        "event_ids_by_split": event_ids_by_split,
        "contains_R019_in_test": contains_R019_in_test,
        "created_files_summary": created_files_summary,
        "warnings": warnings,
    }
    with manifest_json_path.open("w") as manifest_json_file:
        json.dump(manifest, manifest_json_file, indent=2, allow_nan=False)

    for required_path in (
        static_graph_path,
        manifest_csv_path,
        manifest_json_path,
        *dynamic_paths,
    ):
        if not required_path.is_file():
            raise RuntimeError(f"Required output file is missing: {required_path}")

    total_size = output_size_bytes(output_root)
    print("Manifest CSV:", manifest_csv_path)
    print("Manifest JSON:", manifest_json_path)
    print("R019 present in test:", contains_R019_in_test)
    print(
        f"Total output size: {total_size} bytes "
        f"({total_size / (1024**2):.2f} MiB)"
    )
    print()
    print("Stockbridge SWE-GNN milestone 13A full dataset preparation passed.")


if __name__ == "__main__":
    main()
