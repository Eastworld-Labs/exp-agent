#!/usr/bin/env python3
"""Create a bounded MuJoCo/3DGS bundle from a coherent Habitat-GS scene."""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import deque
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--max-gaussians", type=int, default=300_000)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--minimum-points", type=int, default=8)
    parser.add_argument(
        "--initial-pose",
        type=float,
        nargs=3,
        metavar=("X", "Y", "YAW"),
        help=(
            "Visually verified initial pose in the generated world frame. "
            "Defaults to the geometric high-clearance origin."
        ),
    )
    args = parser.parse_args()
    source = args.scene_directory.resolve()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = source.name
    gs = source / f"{stem}.gs.ply"
    navmesh = source / f"{stem}.navmesh"
    if not gs.exists() or not navmesh.exists():
        raise FileNotFoundError("Habitat-GS scene needs matching .gs.ply and .navmesh files")

    with navmesh.open("rb") as handle:
        header = handle.read(40)
    magic, version, _ = struct.unpack_from("<3i", header)
    if magic != 0x4D534554 or version != 2:
        raise ValueError("unsupported Habitat navmesh header")
    nav_origin = np.asarray(struct.unpack_from("<3f", header, 12), dtype=np.float64)
    tile_x, tile_z = struct.unpack_from("<2f", header, 24)
    center_x = nav_origin[0] + tile_x / 2
    center_z = nav_origin[2] + tile_z / 2
    # Habitat is Y-up. MuJoCo is Z-up; center the horizontal navmesh at spawn.
    world_to_gs = np.asarray(
        [[1, 0, 0, center_x], [0, 0, 1, 0], [0, -1, 0, center_z], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    gs_to_world = np.linalg.inv(world_to_gs)

    ply = PlyData.read(str(gs))
    vertex = ply["vertex"].data
    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float64)
    homogeneous = np.column_stack([xyz, np.ones(len(xyz))])
    world = (gs_to_world @ homogeneous.T).T[:, :3]
    half_x, half_y = tile_x / 2, tile_z / 2
    width, height = math.ceil(tile_x / args.resolution), math.ceil(tile_z / args.resolution)
    keep = (
        (world[:, 2] >= 0.15) & (world[:, 2] <= 1.8)
        & (np.abs(world[:, 0]) < half_x) & (np.abs(world[:, 1]) < half_y)
    )
    cells = np.floor((world[keep, :2] + (half_x, half_y)) / args.resolution).astype(int)
    counts = np.zeros((height, width), dtype=np.int32)
    valid = (cells[:, 0] >= 0) & (cells[:, 0] < width) & (cells[:, 1] >= 0) & (cells[:, 1] < height)
    np.add.at(counts, (cells[valid, 1], cells[valid, 0]), 1)
    occupied = {(int(x), int(y)) for y, x in np.argwhere(counts >= args.minimum_points)}
    occupied.update((x, y) for x in range(width) for y in (0, height - 1))
    occupied.update((x, y) for y in range(height) for x in (0, width - 1))

    radius = math.ceil(0.32 / args.resolution)
    inflated = set(occupied)
    for x, y in occupied:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    inflated.add((x + dx, y + dy))
    free = {(x, y) for x in range(width) for y in range(height) if (x, y) not in inflated}
    components = []
    while free:
        seed = free.pop(); component = {seed}; queue = deque([seed])
        while queue:
            x, y = queue.popleft()
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in free:
                    free.remove(neighbor); component.add(neighbor); queue.append(neighbor)
        components.append(component)
    if not components:
        raise ValueError("Gaussian projection produced no navigable component")
    navigable = max(components, key=len)
    # Put the immutable scene transform—not the route—at a high-clearance spawn.
    distance = {cell: 0 for cell in occupied}
    queue = deque(occupied)
    while queue:
        x, y = queue.popleft()
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (0 <= neighbor[0] < width and 0 <= neighbor[1] < height
                    and neighbor not in distance):
                distance[neighbor] = distance[(x, y)] + 1
                queue.append(neighbor)
    spawn = max(navigable, key=lambda cell: distance.get(cell, 0))
    safe = {
        cell for cell in navigable
        # Manhattan distance overestimates diagonal clearance; retain a wider
        # conservative band so the planner's Euclidean 0.15 m margin also fits.
        if distance.get(cell, 0) * args.resolution >= 0.8
    }
    reachable_safe = {spawn}
    queue = deque([spawn])
    while queue:
        x, y = queue.popleft()
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if neighbor in safe and neighbor not in reachable_safe:
                reachable_safe.add(neighbor); queue.append(neighbor)
    if len(reachable_safe) < 3:
        reachable_safe = navigable
    spawn_x = -half_x + (spawn[0] + 0.5) * args.resolution
    spawn_y = -half_y + (spawn[1] + 0.5) * args.resolution
    world_to_gs[0, 3] += spawn_x
    world_to_gs[2, 3] -= spawn_y

    initial_pose = tuple(args.initial_pose or (0.0, 0.0, 0.0))
    initial_cell = (
        math.floor((initial_pose[0] + half_x + spawn_x) / args.resolution),
        math.floor((initial_pose[1] + half_y + spawn_y) / args.resolution),
    )
    if initial_cell not in navigable:
        raise ValueError(
            f"initial pose {initial_pose[:2]} is outside the largest navigable component"
        )
    initial_clearance = distance.get(initial_cell, 0) * args.resolution
    if initial_clearance < 0.47:
        raise ValueError(
            f"initial pose clearance {initial_clearance:.3f} m is below the 0.47 m minimum"
        )

    grid_path = output / "navigation_grid.json"
    grid_path.write_text(json.dumps({
        "width": width, "height": height, "resolution": args.resolution,
        "origin": [-half_x - spawn_x, -half_y - spawn_y],
        "occupied": [list(cell) for cell in sorted(occupied, key=lambda c: (c[1], c[0]))],
        "provenance": {"type": "coherent_habitat_gs_projection", "navmesh": str(navmesh),
                       "gaussian": str(gs), "height_band_m": [0.15, 1.8]},
    }, indent=2) + "\n")

    lod = output / f"{stem}.lod{args.max_gaussians}.gs.ply"
    if len(vertex) <= args.max_gaussians:
        selected = vertex
    elif "opacity" in vertex.dtype.names:
        index = np.argpartition(vertex["opacity"], -args.max_gaussians)[-args.max_gaussians:]
        selected = vertex[index]
    else:
        index = np.linspace(0, len(vertex) - 1, args.max_gaussians, dtype=int)
        selected = vertex[index]
    PlyData([PlyElement.describe(selected, "vertex")], text=False).write(str(lod))

    # Generate conservative wall proxies. The source GS/navmesh remains immutable.
    mjcf = output / "habitat_gs_scene.xml"
    lines = ['<mujoco model="Habitat-GS indoor scene">', '  <worldbody>',
             '    <geom name="floor" type="plane" size="20 20 0.05" friction="1 0.005 0.0001"/>']
    for index, (x, y) in enumerate(sorted(occupied)):
        wx = -half_x + (x + 0.5) * args.resolution - spawn_x
        wy = -half_y + (y + 0.5) * args.resolution - spawn_y
        lines.append(f'    <geom name="scene_collision_{index}" type="box" pos="{wx:.4f} {wy:.4f} 1" size="{args.resolution/2:.4f} {args.resolution/2:.4f} 1" rgba="0 0 0 0"/>')
    lines.extend(['  </worldbody>', '</mujoco>'])
    mjcf.write_text("\n".join(lines) + "\n")

    # Widely separated free landmarks let System-2 plan across the fixed scene.
    landmarks = {}
    for name, score in (
        ("west_room", lambda c: c[0]),
        ("north_room", lambda c: c[1]),
        ("east_room", lambda c: -c[0]),
    ):
        cell = min(reachable_safe, key=score)
        landmarks[name] = {"x": -half_x + (cell[0] + .5) * args.resolution - spawn_x,
                           "y": -half_y + (cell[1] + .5) * args.resolution - spawn_y,
                           "z": 0, "yaw": 0}
    semantic = output / "semantic_locations.json"
    semantic.write_text(json.dumps({"locations": landmarks}, indent=2) + "\n")
    manifest = output / "scene_bundle.json"
    manifest.write_text(json.dumps({
        "external_mjcf": mjcf.name, "navigation_grid": grid_path.name,
        "semantic_map": semantic.name, "gaussian_splat": lod.name,
        "gaussian_alignment": world_to_gs.tolist(), "navigation_footprint_radius_m": 0.32,
        "initial_pose": {
            "x": initial_pose[0], "y": initial_pose[1], "yaw": initial_pose[2],
            "selection": (
                "operator_visually_verified" if args.initial_pose
                else "geometric_clearance_only_visual_validation_required"
            ),
        },
        "source": {"format": "Habitat-GS", "gaussian": str(gs), "navmesh": str(navmesh),
                   "coherent": True, "max_gaussians": args.max_gaussians},
    }, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest), "source_gaussians": len(vertex),
                      "lod_gaussians": len(selected), "occupied_cells": len(occupied),
                      "extent_m": [tile_x, tile_z], "spawn_cell": spawn,
                      "navigable_cells": len(navigable)}, indent=2))


if __name__ == "__main__":
    main()
