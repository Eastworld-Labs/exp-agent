#!/usr/bin/env python3
"""Project an aligned 3D Gaussian scene into a conservative 2D occupancy grid."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bounds", type=float, nargs=4, default=(-1.4, -1.4, 1.4, 1.4))
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--min-height", type=float, default=0.18)
    parser.add_argument("--max-height", type=float, default=1.4)
    parser.add_argument("--min-points", type=int, default=50)
    args = parser.parse_args()

    scene_path = args.scene.expanduser().resolve()
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    ply_path = Path(scene["gaussian_splat"]).expanduser()
    if not ply_path.is_absolute():
        ply_path = (scene_path.parent / ply_path).resolve()
    world_t_gs = np.asarray(scene["gaussian_alignment"], dtype=np.float64)
    gs_t_world = np.linalg.inv(world_t_gs)

    vertex = PlyData.read(ply_path)["vertex"].data
    gs_xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")])
    homogeneous = np.column_stack([gs_xyz, np.ones(len(gs_xyz))])
    world_xyz = (gs_t_world @ homogeneous.T).T[:, :3]

    min_x, min_y, max_x, max_y = args.bounds
    width = math.ceil((max_x - min_x) / args.resolution)
    height = math.ceil((max_y - min_y) / args.resolution)
    keep = (
        (world_xyz[:, 2] >= args.min_height)
        & (world_xyz[:, 2] <= args.max_height)
        & (world_xyz[:, 0] >= min_x)
        & (world_xyz[:, 0] < max_x)
        & (world_xyz[:, 1] >= min_y)
        & (world_xyz[:, 1] < max_y)
    )
    cells = np.floor((world_xyz[keep, :2] - (min_x, min_y)) / args.resolution).astype(int)
    counts = np.zeros((height, width), dtype=np.int32)
    np.add.at(counts, (cells[:, 1], cells[:, 0]), 1)
    occupied_yx = np.argwhere(counts >= args.min_points)
    occupied = [[int(x), int(y)] for y, x in occupied_yx]

    output = {
        "width": width,
        "height": height,
        "resolution": args.resolution,
        "origin": [min_x, min_y],
        "occupied": occupied,
        "provenance": {
            "type": "gaussian_splat_height_projection",
            "source": str(ply_path),
            "gaussian_count": int(len(gs_xyz)),
            "height_band_m": [args.min_height, args.max_height],
            "minimum_gaussians_per_cell": args.min_points,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "gaussians": len(gs_xyz),
                "projected": int(keep.sum()),
                "occupied_cells": len(occupied),
                "grid": [width, height],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
