from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..modules.semantic_map import Pose3D


def select_chase_camera_eye(grid: Any, pose: Pose3D) -> np.ndarray:
    """Choose a close, collision-aware rear-quarter camera position.

    Indoor routes cannot use a fixed boom: at doors and near walls it places
    the camera inside scene geometry. Candidate sight lines are checked against
    the same immutable occupancy map used by navigation. If no horizontal view
    is available, a local overhead view remains inside the robot's free cell.
    """
    candidates: list[tuple[float, np.ndarray]] = []
    for distance in (2.6, 2.2, 1.8, 1.4):
        for offset in (math.radians(28.0), -math.radians(28.0), 0.0):
            angle = pose.yaw + math.pi + offset
            eye = np.asarray(
                (
                    pose.x + distance * math.cos(angle),
                    pose.y + distance * math.sin(angle),
                    1.55 + 0.18 * distance,
                ),
                dtype=np.float64,
            )
            if not grid.segment_is_free((pose.x, pose.y), (float(eye[0]), float(eye[1]))):
                continue
            # Prefer an uncluttered view, then a moderately wide composition.
            midpoint = ((pose.x + float(eye[0])) / 2, (pose.y + float(eye[1])) / 2)
            clearance = min(
                grid.clearance(float(eye[0]), float(eye[1])),
                grid.clearance(*midpoint),
            )
            score = clearance + 0.08 * distance - 0.05 * abs(offset)
            candidates.append((score, eye))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return np.asarray((pose.x, pose.y, pose.z + 2.6), dtype=np.float64)


def merged_occupancy_rectangles(
    navigation_grid: str | Path, *, coarsen: int = 2
) -> tuple[list[tuple[float, float, float, float]], dict[str, Any]]:
    """Merge occupied grid cells into static collision rectangles.

    Coarsening is conservative: a coarse cell is occupied if any source cell is
    occupied. The planner still uses the original-resolution grid.
    """
    data = json.loads(Path(navigation_grid).read_text(encoding="utf-8"))
    resolution = float(data["resolution"]) * coarsen
    origin_x, origin_y = (float(value) for value in data["origin"])
    occupied = {(int(x) // coarsen, int(y) // coarsen) for x, y in data["occupied"]}
    height = math.ceil(int(data["height"]) / coarsen)
    rows: dict[int, list[int]] = {}
    for x, y in occupied:
        rows.setdefault(y, []).append(x)
    active: dict[tuple[int, int], list[int]] = {}
    rectangles: list[tuple[int, int, int, int]] = []
    for y in range(height):
        runs: list[list[int]] = []
        for x in sorted(rows.get(y, ())):
            if not runs or x > runs[-1][1] + 1:
                runs.append([x, x])
            else:
                runs[-1][1] = x
        spans = {tuple(run) for run in runs}
        for span, (start_y, previous_y) in list(active.items()):
            if span not in spans:
                rectangles.append((span[0], span[1], start_y, previous_y))
                del active[span]
        for span in spans:
            if span in active:
                active[span][1] = y
            else:
                active[span] = [y, y]
    rectangles.extend((x0, x1, y0, y1) for (x0, x1), (y0, y1) in active.items())
    world = []
    for x0, x1, y0, y1 in rectangles:
        width = (x1 - x0 + 1) * resolution
        depth = (y1 - y0 + 1) * resolution
        center_x = origin_x + (x0 + x1 + 1) * resolution / 2.0
        center_y = origin_y + (y0 + y1 + 1) * resolution / 2.0
        world.append((center_x, center_y, width, depth))
    return world, data


def configure_nurec_rendering() -> None:
    """Configure registered compositing before a NuRec volume is loaded."""
    import carb

    settings = carb.settings.get_settings()
    for key, value in (
        ("/rtx/post/histogram/enabled", False),
        ("/rtx/post/registeredCompositing/invertToneMap", True),
        ("/rtx/post/registeredCompositing/invertColorCorrection", True),
        ("/rtx/matteObject/visibility/secondaryRays", True),
        ("/rtx/post/tonemap/op", 2),
        ("/rtx/material/enableRefraction", False),
        ("/rtx/raytracing/fractionalCutoutOpacity", False),
    ):
        settings.set(key, value)


def compose_multiroom_stage(
    *,
    nurec_usdz: str | Path,
    gaussian_alignment: Sequence[Sequence[float]],
    navigation_grid: str | Path,
    simfoundry_assets: str | Path,
) -> dict[str, Any]:
    """Compose visual, collision and rigid-object layers into the live USD stage."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    # Keep this prim untyped: native Gaussian USDs commonly author their
    # ParticleField3DGaussianSplat as the default prim. A stronger local Xform
    # type would mask that schema and make Hydra treat it as an empty node.
    gaussian = stage.OverridePrim("/World/GaussianScene")
    gaussian.GetReferences().AddReference(str(Path(nurec_usdz).resolve()))
    world_to_gaussian = np.asarray(gaussian_alignment, dtype=np.float64).reshape(4, 4)
    gaussian_to_world = np.linalg.inv(world_to_gaussian)
    # Bundle matrices use column-vector convention; Gf/USD matrices transform
    # row vectors, so transpose when authoring the Xform op.
    transform = Gf.Matrix4d(*(float(value) for value in gaussian_to_world.T.reshape(-1)))
    UsdGeom.Xformable(gaussian).AddTransformOp().Set(transform)

    rectangles, grid = merged_occupancy_rectangles(navigation_grid, coarsen=2)
    collision_root = stage.DefinePrim("/World/SceneCollisions", "Xform")
    for index, (x, y, width, depth) in enumerate(rectangles):
        cube = UsdGeom.Cube.Define(stage, f"{collision_root.GetPath()}/c{index:04d}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, 1.0))
        xform.AddScaleOp().Set(Gf.Vec3f(width, depth, 2.0))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        cube.MakeInvisible()

    # A real rigid-object island near the living area. The table is static;
    # the three SimFoundry dishware assets retain authored rigid bodies/colliders.
    table = UsdGeom.Cube.Define(stage, "/World/Interactables/Table")
    table.CreateSizeAttr(1.0)
    table_xform = UsdGeom.Xformable(table)
    table_xform.AddTranslateOp().Set(Gf.Vec3d(9.15, -14.15, 0.42))
    table_xform.AddScaleOp().Set(Gf.Vec3f(1.2, 0.75, 0.84))
    table.CreateDisplayColorAttr([Gf.Vec3f(0.24, 0.16, 0.10)])
    UsdPhysics.CollisionAPI.Apply(table.GetPrim())
    assets = Path(simfoundry_assets)
    objects = (
        ("BlueBowl", "scenes/YAM/stack_dishware/objects/blue_bowl/jpkksu/usd/jpkksu.usd", (8.85, -14.15, 0.89)),
        ("CreamMug", "scenes/YAM/stack_dishware/objects/cream_mug/bpupsc/usd/bpupsc.usd", (9.15, -14.15, 0.89)),
        ("GreenPlate", "scenes/YAM/stack_dishware/objects/green_plate/zritke/usd/zritke.usd", (9.45, -14.15, 0.89)),
    )
    for name, relative, position in objects:
        prim = stage.DefinePrim(f"/World/Interactables/{name}", "Xform")
        prim.GetReferences().AddReference(str((assets / relative).resolve()))
        placement = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*position))
        UsdGeom.Xformable(prim).AddTransformOp(opSuffix="scenePlacement").Set(placement)

    for camera_path in ("/World/ChaseCamera", "/World/HeadCamera"):
        camera = UsdGeom.Camera.Define(stage, camera_path)
        camera.CreateFocalLengthAttr(22.0 if "Chase" in camera_path else 18.0)
    return {"collision_rectangles": len(rectangles), "grid": grid}


class IsaacRouteRecorder:
    """Two-camera MP4 recorder with a route/map overlay."""

    def __init__(
        self,
        output: str | Path,
        *,
        grid: Any,
        path: Sequence[Pose3D],
        cameras: tuple[Any, Any],
        width: int = 960,
        height: int = 540,
        fps: float = 10.0,
    ) -> None:
        import cv2

        self.cv2 = cv2
        self.cameras = cameras
        self.grid = grid
        self.path = tuple(path)
        self.width = width
        self.height = height
        self.frames = 0
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"could not open video writer: {output}")
    def prepare(self, pose: Pose3D) -> None:
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        chase_eye = select_chase_camera_eye(self.grid, pose)
        target = np.asarray((pose.x, pose.y, 0.85))
        self.cameras[0].set_world_poses_from_view(chase_eye[None, :], target[None, :])
        head_eye = np.asarray((pose.x + 0.04 * c, pose.y + 0.04 * s, pose.z + 0.92))
        head_target = np.asarray((pose.x + 2.5 * c, pose.y + 2.5 * s, pose.z + 0.82))
        self.cameras[1].set_world_poses_from_view(head_eye[None, :], head_target[None, :])

    def write(self, pose: Pose3D, label: str) -> bool:
        for camera in self.cameras:
            camera.update(0.005)
        images = [camera.data.output["rgb"][0].detach().cpu().numpy() for camera in self.cameras]
        if any(image is None or not getattr(image, "size", 0) for image in images):
            return False
        main = self.cv2.cvtColor(np.asarray(images[0])[..., :3], self.cv2.COLOR_RGB2BGR)
        head = self.cv2.cvtColor(np.asarray(images[1])[..., :3], self.cv2.COLOR_RGB2BGR)
        head = self.cv2.resize(head, (320, 180), interpolation=self.cv2.INTER_AREA)
        main[10:190, self.width - 330 : self.width - 10] = head
        self.cv2.rectangle(main, (self.width - 330, 10), (self.width - 10, 190), (255, 255, 255), 2)
        self.cv2.putText(main, "G1 head camera", (self.width - 320, 34), self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, self.cv2.LINE_AA)
        self._draw_map(main, pose)
        self.cv2.putText(main, label, (16, 28), self.cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, self.cv2.LINE_AA)
        self.writer.write(main)
        self.frames += 1
        if self.frames == 1:
            print("ISAAC_RECORDING_FIRST_FRAME", flush=True)
        return True

    def _draw_map(self, image: np.ndarray, pose: Pose3D) -> None:
        side = 190
        x0, y0 = 12, self.height - side - 12
        overlay = np.full((side, side, 3), 30, dtype=np.uint8)
        sx, sy = side / self.grid.width, side / self.grid.height
        for x, y in self.grid.occupied:
            px = min(side - 1, int(x * sx))
            py = min(side - 1, int((self.grid.height - 1 - y) * sy))
            overlay[max(0, py - 1) : py + 1, max(0, px - 1) : px + 1] = (75, 75, 75)
        points = []
        for waypoint in self.path:
            gx, gy = self.grid.world_to_cell(waypoint.x, waypoint.y)
            points.append((int(gx * sx), int((self.grid.height - 1 - gy) * sy)))
        if len(points) > 1:
            self.cv2.polylines(overlay, [np.asarray(points, dtype=np.int32)], False, (0, 200, 255), 2)
        gx, gy = self.grid.world_to_cell(pose.x, pose.y)
        self.cv2.circle(overlay, (int(gx * sx), int((self.grid.height - 1 - gy) * sy)), 5, (40, 40, 255), -1)
        image[y0 : y0 + side, x0 : x0 + side] = overlay
        self.cv2.rectangle(image, (x0, y0), (x0 + side, y0 + side), (255, 255, 255), 2)

    def close(self) -> None:
        self.writer.release()
