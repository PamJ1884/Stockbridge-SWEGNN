#!/usr/bin/env python3
"""Independent QC: canonical CityCAT PT vs SWE-GNN 100 at 5 m."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

BASE = Path.home() / "flood/02_data/CityCAT_Winchester"

ROOT = BASE / "swegnn/stockbridge_phase2_100_5m_v01"

MASTER = (
    BASE / "pt/stockbridge_phase2plus3_300_v01"
    / "stockbridge_phase2plus3_300_manifest.csv"
)

EXPECTED_SHA = (
    "6a687bcd240292e895758679f8a2084149242bdcb60fecebc82937f43c4b5092"
)


def check_close(name, actual, expected, atol=1e-6, rtol=1e-5):
    assert actual.shape == expected.shape, f"{name}: shape mismatch"
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol
    )


def main():
    print("===== CANONICAL PT -> SWE-GNN QC =====", flush=True)

    sha = hashlib.sha256(MASTER.read_bytes()).hexdigest()
    assert sha == EXPECTED_SHA
    print("Master manifest checksum: PASS", flush=True)

    with MASTER.open(newline="") as f:
        rows = [
            r for r in csv.DictReader(f)
            if r["phase"] == "phase2"
        ]

    assert len(rows) == 100
    assert Counter(r["split"] for r in rows) == {
        "train": 70, "valid": 15, "test": 15
    }

    graph = torch.load(ROOT / "static_graph.pt", map_location="cpu")
    assert float(graph["cell_size"]) == 5.0
    assert graph["num_nodes"] == 99430
    assert graph["num_edges"] == 396458
    print("Graph metadata: PASS", flush=True)

    for index, row in enumerate(rows, 1):
        event = row["event_id"].split("_")[0]
        split = row["split"]
        source_path = Path(row["source_path"])

        source = torch.load(source_path, map_location="cpu")
        derived = torch.load(
            ROOT / "events" / split / f"{event}.pt",
            map_location="cpu",
        )

        assert source.shape == (305, 326, 25, 5)
        assert source.dtype == torch.float32
        assert derived["event_id"] == event
        assert derived["split"] == split
        assert derived["num_nodes"] == 99430
        assert derived["nt"] == 25

        H = source[..., 0].reshape(99430, 25)
        Vx = source[..., 1].reshape(99430, 25)
        Vy = source[..., 2].reshape(99430, 25)
        rain = source[..., 3]
        DEM = source[:, :, 0, 4].reshape(-1)

        Qx = Vx * H
        Qy = Vy * H
        Qmag = torch.sqrt(Qx**2 + Qy**2)

        check_close(event + " H", derived["H"], H, 0, 0)
        check_close(event + " Qmag", derived["Qmag"], Qmag)
        check_close(event + " DEM", graph["DEM"], DEM, 0, 0)

        expected_rain = rain[0, 0, :]
        check_close(
            event + " rainfall",
            derived["rainfall_global"],
            expected_rain,
            1e-9, 1e-5,
        )
        check_close(
            event + " spatial rainfall",
            rain,
            expected_rain.reshape(1, 1, 25).expand_as(rain),
            1e-9, 1e-5,
        )

        sidecar = json.loads(
            Path(str(source_path) + ".json").read_text()
        )
        assert sidecar["raster_resolution"] == [5.0, 5.0]
        assert sidecar["crs"] == "EPSG:27700"
        assert sidecar["rainfall_alignment"] == "future"

        if index % 10 == 0:
            print(f"Compared {index}/100 events: PASS", flush=True)

    print("\n===== FINAL RESULT =====")
    print("H, Qmag, rainfall, DEM: PASS")
    print("Event IDs and frozen splits: PASS")
    print("Geometry metadata: PASS")
    print("CANONICAL-TO-GRAPH QC PASSED", flush=True)


if __name__ == "__main__":
    main()
