#!/bin/bash
#SBATCH --job-name=swegnn100_5m_qc
#SBATCH --account=comet_smhfs
#SBATCH --partition=default_free
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.out

set -euo pipefail

module load apptainer/1.4.1

SIF="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"
BASE="$HOME/flood/02_data/CityCAT_Winchester"

OLD="$BASE/swegnn/stockbridge_phase2_swegnn_full_v01"
NEW="$BASE/swegnn/stockbridge_phase2_100_5m_v01"
MASTER="$BASE/pt/stockbridge_phase2plus3_300_v01/stockbridge_phase2plus3_300_manifest.csv"

echo "===== SWE-GNN 100 DYNAMIC QC ====="
date

apptainer exec "$SIF" python - "$OLD" "$NEW" "$MASTER" <<'PY'
import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

old_root, new_root, master_path = map(Path, sys.argv[1:])

expected_sha = (
    "6a687bcd240292e895758679f8a2084149242bdcb60fecebc82937f43c4b5092"
)

actual_sha = hashlib.sha256(master_path.read_bytes()).hexdigest()
assert actual_sha == expected_sha, "Master manifest checksum mismatch"
print("Master manifest SHA-256: PASS")

with master_path.open(newline="") as f:
    master = list(csv.DictReader(f))

expected = {}
for row in master:
    if row["phase"] != "phase2":
        continue
    event = row["event_id"].split("_")[0]
    assert event not in expected
    expected[event] = row["split"]

assert len(expected) == 100
assert Counter(expected.values()) == {
    "train": 70, "valid": 15, "test": 15
}
print("Frozen Phase 2 split: PASS")


def discover(root):
    result = {}
    for split in ("train", "valid", "test"):
        directory = root / "events" / split
        assert directory.is_dir(), f"Missing: {directory}"

        for path in directory.rglob("*.pt"):
            match = re.search(r"R\d{3}", path.name)
            assert match, f"Invalid filename: {path}"
            event = match.group()

            assert event not in result, f"Duplicate event: {event}"
            assert expected.get(event) == split, (
                f"Split mismatch: {event}, {split}"
            )

            result[event] = path

    assert set(result) == set(expected), (
        f"Missing/extra IDs: {set(expected) ^ set(result)}"
    )
    return result


old_files = discover(old_root)
new_files = discover(new_root)

print("Historical event IDs and splits: PASS")
print("Corrected event IDs and splits: PASS")


def compare(a, b, location):
    assert type(a) is type(b), f"{location}: type mismatch"

    if isinstance(a, torch.Tensor):
        assert a.shape == b.shape, f"{location}: shape mismatch"
        assert a.dtype == b.dtype, f"{location}: dtype mismatch"
        assert torch.equal(a, b), f"{location}: tensor values differ"

    elif isinstance(a, np.ndarray):
        assert a.shape == b.shape, f"{location}: shape mismatch"
        assert a.dtype == b.dtype, f"{location}: dtype mismatch"
        assert np.array_equal(a, b, equal_nan=True), (
            f"{location}: array values differ"
        )

    elif isinstance(a, dict):
        assert a.keys() == b.keys(), f"{location}: keys differ"
        for key in a:
            compare(a[key], b[key], f"{location}.{key}")

    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), f"{location}: length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{location}[{i}]")

    elif hasattr(a, "to_dict") and callable(a.to_dict):
        compare(a.to_dict(), b.to_dict(), location)

    else:
        assert a == b, f"{location}: values differ"


print("\n===== DYNAMIC EVENT COMPARISON =====")

for index, event in enumerate(sorted(expected), start=1):
    old_data = torch.load(old_files[event], map_location="cpu")
    new_data = torch.load(new_files[event], map_location="cpu")

    compare(old_data, new_data, event)

    if index % 10 == 0:
        print(f"Compared {index}/100 events: PASS", flush=True)

print("\n===== FINAL RESULT =====")
print("100/100 event contents identical: PASS")
print("Frozen splits: PASS")
print("DYNAMIC EQUIVALENCE QC PASSED")
PY

echo "===== JOB COMPLETED ====="
date
