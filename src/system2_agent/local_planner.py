"""Walk the last few metres to something the robot can SEE, not to a label.

##### `plan()` MOVES NOTHING. It returns a pose somebody else may drive to. #####

`navigate_to` answers "where is the pantry" from a map somebody surveyed. This
answers "where is the sink" from the picture in front of the robot right now,
and it deliberately never opens the semantic map. The chain:

    a box in the colour frame        grounding.VisionGrounder (a model call)
      -> a metric point               the depth image's own intrinsics
      -> a standoff pose              short of it, on the line from the robot
      -> a feasibility check          A* on the Nav2 LOCAL costmap
      -> one /goal_pose               Nav2 plans and tracks the actual walk

## The frames, and why there are three of them

  local   the robot's pose AT THE INSTANT THE TOOL RAN. Origin at
          base_footprint, +x forward, +y left. Everything reported to the model
          is in this frame, because "1.9 m ahead and 0.4 m left" is a claim a
          reader can check against a photograph. It is not a frame anything
          else on this stack knows about, so it never leaves the result.
  odom    what the local costmap is published in (`global_frame: odom`,
          `rolling_window: true`). ALL THE ARITHMETIC HAPPENS HERE. Rotating a
          0.05 m grid into the local frame would alias cells for no gain: the
          two frames differ by a rigid transform, so every answer is identical
          and the lethal cells stay exactly where Nav2 put them.
  map     what a /goal_pose must be in. Reached from local via the map-frame
          pose read at the same instant as the odom one.

⚠️ THE TWO POSES MUST BE READ TOGETHER. `odom -> map` is recovered from the pair
(odom pose, map pose) sampled at one moment; if they are seconds apart the
composed transform carries the drift between them as a position error. The
robot is standing still between tool calls, which is what makes this legal --
and `InitPoseProvider` reads both through one freshness gate for the same
reason.

## What is pure here

Everything in this module is a function of its arguments: no MQTT, no model, no
clock. The link adapters live in `g1/local_costmap.py` and `g1/depth.py`, and an
in-process simulator can drive the same planner by implementing `DepthSource`
and `LocalGridSource` over its own renderer. That split is what lets the sign
conventions below be tested with nothing installed -- and a sign error here does
not look like a bug, it looks like a planner fault: a goal on the far side of
the object, or a heading 180 degrees out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Protocol, Sequence

from .grounding import Box, Grounding
from .modules.camera import CameraBackend, CameraFrame
from .modules.semantic_map import Pose3D
from .navigation_core import AStarPlanner, GridMap
from .types import Json

# Nav2 costmap cell semantics (nav2_costmap_2d/cost_values.hpp). Mirrored rather
# than imported: this host has no ROS, and these four numbers are a wire format.
COST_UNKNOWN = -1
COST_FREE = 0
COST_INSCRIBED = 99
COST_LETHAL = 100

#: What counts as "there is something solid here" when ranging by ray-cast.
#: LETHAL only: 99 (inscribed) means the ROBOT'S CENTRE cannot go there, which is
#: a statement about the footprint, not about where the obstacle's surface is.
RANGE_LETHAL_FROM = COST_LETHAL

#: What counts as "the robot may not stand here" when planning. 99 and 100 both
#: do; the 1..98 inflation gradient deliberately does NOT -- it is a soft cost
#: Nav2's own planners route through, and treating it as a wall would refuse
#: every doorway. The footprint radius is re-inflated on top of this.
PLAN_OCCUPIED_FROM = COST_INSCRIBED


# --------------------------------------------------------------------- maths --
def wrap_angle(radians: float) -> float:
    """Fold an angle into (-pi, pi]."""
    return math.atan2(math.sin(radians), math.cos(radians))


def _percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of nothing")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (pct / 100.0)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _median(values: Sequence[float]) -> float:
    return _percentile(values, 50.0)


# ------------------------------------------------------------------- frames --
def odom_to_local(x: float, y: float, init: Pose3D) -> tuple[float, float]:
    """An odom-frame point in the frame of the robot's pose at call time."""
    dx, dy = x - init.x, y - init.y
    c, s = math.cos(init.yaw), math.sin(init.yaw)
    return (dx * c + dy * s, -dx * s + dy * c)


def local_to_odom(x: float, y: float, init: Pose3D) -> tuple[float, float]:
    c, s = math.cos(init.yaw), math.sin(init.yaw)
    return (init.x + x * c - y * s, init.y + x * s + y * c)


def local_to_map(x: float, y: float, yaw: float, init_map: Pose3D) -> Pose3D:
    """A local-frame pose as a map-frame goal, via the map pose read at call time."""
    c, s = math.cos(init_map.yaw), math.sin(init_map.yaw)
    return Pose3D(
        x=init_map.x + x * c - y * s,
        y=init_map.y + x * s + y * c,
        yaw=wrap_angle(init_map.yaw + yaw),
        frame=init_map.frame or "map",
    )


# ------------------------------------------------------------------- camera --
class CameraGeometry(Protocol):
    """The half of `sim.head_camera.HeadCameraSpec` this module needs.

    A Protocol rather than the class itself so nothing here depends on the
    simulator package, and so a caller with only a pitch and a field of view can
    stand one up. `intrinsics` is the FALLBACK: when a depth image arrives it
    carries its own, measured by the driver for the size actually published.
    """

    width: int
    height: int
    mount_xyz: tuple[float, float, float]

    @property
    def intrinsics(self) -> tuple[float, float, float, float]: ...

    def optical_rotation_in_mount(self) -> tuple[tuple[float, float, float], ...]: ...


def optical_to_base(
    geometry: CameraGeometry, vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    """An optical-frame vector (+X right, +Y down, +Z forward) in the robot's.

    The robot frame here is +X forward, +Y LEFT, +Z up -- so a pixel to the
    RIGHT of centre yields a NEGATIVE y and therefore a negative (clockwise)
    bearing. That is the single most invertible sign in this file.
    """
    rotation = geometry.optical_rotation_in_mount()
    return tuple(  # type: ignore[return-value]
        rotation[axis][0] * vector[0]
        + rotation[axis][1] * vector[1]
        + rotation[axis][2] * vector[2]
        for axis in range(3)
    )


def pixel_ray_base(
    geometry: CameraGeometry,
    intrinsics: tuple[float, float, float, float],
    u: float,
    v: float,
) -> tuple[float, float, float]:
    """Robot-frame direction through pixel (u, v). Not normalised."""
    fx, fy, cx, cy = intrinsics
    if fx <= 0 or fy <= 0:
        raise ValueError("camera focal lengths must be positive")
    return optical_to_base(geometry, ((u - cx) / fx, (v - cy) / fy, 1.0))


def pixel_azimuth(
    geometry: CameraGeometry,
    intrinsics: tuple[float, float, float, float],
    u: float,
    v: float,
) -> float:
    """Bearing of pixel (u, v) off the robot's nose. Positive is LEFT.

    ⚠️ ON A PITCHED CAMERA THIS DEPENDS ON `v`. A horizontal image row is not a
    constant-bearing line once the sensor is tipped towards the floor, so a fan
    of rays across a box must share one `v` (its vertical centre) rather than
    taking each column at whatever row happened to be convenient.
    """
    x, y, _ = pixel_ray_base(geometry, intrinsics, u, v)
    return math.atan2(y, x)


# -------------------------------------------------------------------- depth --
@dataclass(frozen=True)
class DepthImage:
    """Metric depth, with the intrinsics of the grid it is actually published on.

    ⚠️ THE INTRINSICS TRAVEL WITH THE PIXELS, and that is the whole reason this
    is not just an array. The publisher downsamples, and a field of view read off
    a datasheet describes neither the sensor's real optics nor the size that
    arrived. Whoever resized the image is who knows what fx became.

    `depth_mm` is row-major, `width * height` entries, 0 meaning NO RETURN --
    never "zero metres". Aligned depth is resampled into the COLOUR camera's
    pixel grid, which is what makes "the box is at (u, v) so the range is
    depth[v][u]" true at all; `frame_id` should therefore be a colour optical
    frame.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_mm: Sequence[int]
    scale: float = 0.001
    frame_id: str = ""
    source: str = ""
    age_s: float = 0.0
    info_age_s: float = 0.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("depth image size must be positive")
        if len(self.depth_mm) != self.width * self.height:
            raise ValueError(
                f"depth image is {len(self.depth_mm)} pixels, not "
                f"{self.width}x{self.height}={self.width * self.height}"
            )

    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        return (self.fx, self.fy, self.cx, self.cy)

    def at(self, col: int, row: int) -> float:
        """Metres at a pixel. 0.0 means no return."""
        if not (0 <= col < self.width and 0 <= row < self.height):
            return 0.0
        return float(self.depth_mm[row * self.width + col]) * self.scale

    def as_json(self) -> Json:
        return {
            "size": [self.width, self.height],
            "frame_id": self.frame_id,
            "source": self.source,
            "age_s": round(self.age_s, 2),
            "info_age_s": round(self.info_age_s, 2),
        }


class DepthSource(Protocol):
    """Where a metric depth frame comes from. The seam for a second backend.

    ⚠️ `capture()` RETURNS None RATHER THAN RAISING when there is no usable
    frame. Depth is an upgrade, not a precondition: without it the planner falls
    back to ranging on the costmap and says so. `status()` carries the reason so
    a snapshot can show "no depth, because ..." without a refusal.
    """

    def capture(self) -> DepthImage | None: ...

    def status(self) -> Json: ...


class NoDepthReturn(Exception):
    """The depth image had nothing measurable over the box. Not a refusal yet."""


# ------------------------------------------------------------------ costmap --
@dataclass(frozen=True)
class RayHit:
    range_m: float | None
    end: str  # "lethal" | "edge" | "max_range"
    x: float
    y: float
    unknown_crossed: int


@dataclass(frozen=True)
class GridUpdate:
    """A map_msgs/OccupancyGridUpdate patch: a rectangle of new costs."""

    x: int
    y: int
    width: int
    height: int
    cost: tuple[int, ...]


@dataclass(frozen=True)
class CostGrid:
    """One Nav2 costmap, decoded. Row-major from the origin corner.

    `age_s` is measured from ARRIVAL, not from the message stamp: this grid is
    retained by the broker on the real robot, so a subscriber that just
    connected holds a snapshot that may be hours old and looks perfectly formed.
    """

    frame: str
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    cost: Sequence[int]
    age_s: float = 0.0
    full_grid_age_s: float = 0.0
    patches_applied: int = 0
    live: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("costmap size must be positive")
        if self.resolution <= 0:
            raise ValueError("costmap resolution must be positive")
        if len(self.cost) != self.width * self.height:
            raise ValueError(
                f"costmap carries {len(self.cost)} cells, not "
                f"{self.width}x{self.height}={self.width * self.height}"
            )

    # Same convention as navigation_core.GridMap, deliberately: floor division
    # from the origin corner, so a cell index means the same thing in both.
    def cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(math.floor((x - self.origin_x) / self.resolution)),
            int(math.floor((y - self.origin_y) / self.resolution)),
        )

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def cost_at(self, x: float, y: float) -> int | None:
        """The cost under a world point, or None when it is off the window."""
        col, row = self.cell(x, y)
        if not self.in_bounds((col, row)):
            return None
        return int(self.cost[row * self.width + col])

    @property
    def size_m(self) -> tuple[float, float]:
        return (self.width * self.resolution, self.height * self.resolution)

    @property
    def half_window_m(self) -> float:
        """Shortest distance from the window centre to an edge."""
        return min(self.size_m) / 2.0

    def counts(self) -> Json:
        lethal = unknown = inscribed = 0
        for value in self.cost:
            if value >= COST_LETHAL:
                lethal += 1
            elif value >= COST_INSCRIBED:
                inscribed += 1
            elif value < COST_FREE:
                unknown += 1
        return {"lethal_cells": lethal, "inscribed_cells": inscribed, "unknown_cells": unknown}

    def raycast(
        self,
        x0: float,
        y0: float,
        bearing_rad: float,
        *,
        max_range_m: float,
        lethal_from: int = RANGE_LETHAL_FROM,
    ) -> RayHit:
        """March until something solid, the window edge, or `max_range_m`.

        Unknown cells are passed THROUGH and counted. On this stack the local
        costmap does not track unknown space, so a run of them means the ray left
        the region the sensors have described -- worth reporting, not worth
        stopping on.
        """
        step = self.resolution / 2.0
        dx, dy = math.cos(bearing_rad), math.sin(bearing_rad)
        unknown = 0
        distance = 0.0
        x, y = x0, y0
        while distance <= max_range_m:
            cost = self.cost_at(x, y)
            if cost is None:
                return RayHit(None, "edge", x, y, unknown)
            if cost < COST_FREE:
                unknown += 1
            elif cost >= lethal_from:
                return RayHit(distance, "lethal", x, y, unknown)
            distance += step
            x, y = x0 + dx * distance, y0 + dy * distance
        return RayHit(None, "max_range", x, y, unknown)

    def to_gridmap(
        self,
        *,
        occupied_from: int = PLAN_OCCUPIED_FROM,
        unknown_is_occupied: bool = True,
    ) -> GridMap:
        """The planner's grid. Unknown is occupied: not-described is not free."""
        occupied = set()
        for row in range(self.height):
            base = row * self.width
            for col in range(self.width):
                value = self.cost[base + col]
                if value >= occupied_from or (unknown_is_occupied and value < COST_FREE):
                    occupied.add((col, row))
        return GridMap(
            width=self.width,
            height=self.height,
            resolution=self.resolution,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            occupied=frozenset(occupied),
        )

    def with_patch(self, patch: GridUpdate) -> "CostGrid | None":
        """This grid with a rectangle overwritten, or None if it does not fit.

        A patch that does not fit is DROPPED rather than clipped: it means this
        grid and that patch describe different windows (the rolling costmap
        moved), and half-applying one produces a plausible map of nowhere.
        """
        if patch.width <= 0 or patch.height <= 0:
            return None
        if patch.x < 0 or patch.y < 0:
            return None
        if patch.x + patch.width > self.width or patch.y + patch.height > self.height:
            return None
        if len(patch.cost) != patch.width * patch.height:
            return None
        cells = list(self.cost)
        for row in range(patch.height):
            start = (patch.y + row) * self.width + patch.x
            cells[start:start + patch.width] = patch.cost[
                row * patch.width:(row + 1) * patch.width
            ]
        return CostGrid(
            frame=self.frame,
            width=self.width,
            height=self.height,
            resolution=self.resolution,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            cost=cells,
            age_s=self.age_s,
            full_grid_age_s=self.full_grid_age_s,
            patches_applied=self.patches_applied + 1,
            live=self.live,
        )

    def as_json(self) -> Json:
        return {
            "frame": self.frame,
            "size_m": [round(self.size_m[0], 2), round(self.size_m[1], 2)],
            "resolution": self.resolution,
            "age_s": round(self.age_s, 2),
            "full_grid_age_s": round(self.full_grid_age_s, 2),
            "patches_applied": self.patches_applied,
            **self.counts(),
        }


class LocalGridSource(Protocol):
    """Where the local costmap comes from. `grid()` refuses with ValueError."""

    def grid(self) -> CostGrid: ...

    def status(self) -> Json: ...


# ------------------------------------------------------------------- ranging --
@dataclass(frozen=True)
class RangeEstimate:
    """Where the target is, in the robot's own frame at call time."""

    method: str  # "depth" | "costmap_raycast"
    target_base: tuple[float, float]
    detail: str = ""
    valid_frac: float | None = None
    pixels: int | None = None
    z_percentile_m: float | None = None
    rays: int | None = None
    hits: int | None = None
    spread_m: float | None = None
    unknown_crossed: int | None = None
    costmap_check_m: float | None = None
    agrees: bool | None = None

    @property
    def range_m(self) -> float:
        return math.hypot(*self.target_base)

    @property
    def azimuth_rad(self) -> float:
        return math.atan2(self.target_base[1], self.target_base[0])

    def as_json(self) -> Json:
        out: Json = {
            "method": self.method,
            "range_m": round(self.range_m, 3),
            "azimuth_deg": round(math.degrees(self.azimuth_rad), 1),
            "detail": self.detail,
        }
        for key in (
            "valid_frac", "pixels", "z_percentile_m", "rays", "hits",
            "spread_m", "unknown_crossed", "costmap_check_m", "agrees",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = round(value, 3) if isinstance(value, float) else value
        return out


class DepthRanger:
    """Turn a box plus a depth image into a point in the robot's frame."""

    def __init__(
        self,
        *,
        keep: float = 0.6,
        percentile: float = 30.0,
        min_range_m: float = 0.3,
        max_range_m: float = 6.0,
        min_valid_frac: float = 0.10,
        min_valid_px: int = 20,
        centroid_band_m: float = 0.15,
    ) -> None:
        self.keep = keep
        # ⚠️ THE 30th PERCENTILE, NOT THE MEDIAN, AND NOT THE MINIMUM. A box over
        # a real object is bimodal: the object's face, and whatever shows around
        # its silhouette. The minimum is a speckle; the median lands between the
        # two modes when the box is more than about half background. g1_vision
        # measured 30 on this hardware and this mirrors it rather than
        # re-deriving it.
        self.percentile = percentile
        self.min_range_m = min_range_m
        self.max_range_m = max_range_m
        self.min_valid_frac = min_valid_frac
        self.min_valid_px = min_valid_px
        self.centroid_band_m = centroid_band_m

    def range(
        self, box: Box, geometry: CameraGeometry, depth: DepthImage
    ) -> RangeEstimate:
        inner = box.shrunk(self.keep)
        col0 = max(0, int(math.floor(inner.x0 * depth.width)))
        col1 = min(depth.width, max(col0 + 1, int(math.ceil(inner.x1 * depth.width))))
        row0 = max(0, int(math.floor(inner.y0 * depth.height)))
        row1 = min(depth.height, max(row0 + 1, int(math.ceil(inner.y1 * depth.height))))

        samples: list[tuple[int, int, float]] = []
        total = (col1 - col0) * (row1 - row0)
        for row in range(row0, row1):
            base = row * depth.width
            for col in range(col0, col1):
                metres = float(depth.depth_mm[base + col]) * depth.scale
                if self.min_range_m <= metres <= self.max_range_m:
                    samples.append((col, row, metres))
        valid_frac = (len(samples) / total) if total else 0.0
        if len(samples) < self.min_valid_px or valid_frac < self.min_valid_frac:
            raise NoDepthReturn(
                f"{len(samples)} of {total} pixels in the box carry a depth return "
                f"between {self.min_range_m:g} and {self.max_range_m:g} m "
                f"({valid_frac * 100:.0f}%)"
            )

        z = _percentile([sample[2] for sample in samples], self.percentile)
        near = [s for s in samples if abs(s[2] - z) <= self.centroid_band_m] or samples
        # Pixel CENTRES: a pixel spans [col, col+1), and using its corner biases
        # every bearing by half a pixel in the same direction.
        u = sum(s[0] for s in near) / len(near) + 0.5
        v = sum(s[1] for s in near) / len(near) + 0.5

        fx, fy, cx, cy = depth.intrinsics
        # Depth is distance along the optical axis, not radial distance, so the
        # ray is scaled by z rather than normalised to it.
        point = optical_to_base(geometry, ((u - cx) / fx * z, (v - cy) / fy * z, z))
        mount_x, mount_y = geometry.mount_xyz[0], geometry.mount_xyz[1]
        return RangeEstimate(
            method="depth",
            target_base=(point[0] + mount_x, point[1] + mount_y),
            valid_frac=valid_frac,
            pixels=len(samples),
            z_percentile_m=z,
            detail=(
                f"{self.percentile:g}th percentile of {len(samples)} depth pixels "
                f"in the middle {self.keep * 100:.0f}% of the box"
            ),
        )


class CostmapRaycastRanger:
    """Range by asking the costmap what the camera is pointed at.

    The fallback, and the cross-check. It cannot tell the target from anything
    else along the same bearing, so a chair between the robot and the sink
    ranges the chair -- which yields a shorter, safer standoff rather than a
    wrong one. It also cannot see a target the LiDAR and depth layers never
    marked (glass, a thin rail, something above the scan band); that is a
    refusal, not a guess.
    """

    def __init__(
        self,
        *,
        rays: int = 9,
        keep: float = 0.6,
        min_hit_fraction: float = 0.5,
        max_range_m: float = 4.0,
    ) -> None:
        self.rays = max(1, rays)
        self.keep = keep
        self.min_hit_fraction = min_hit_fraction
        self.max_range_m = max_range_m

    def range(
        self,
        box: Box,
        geometry: CameraGeometry,
        grid: CostGrid,
        robot: Pose3D,
        *,
        intrinsics: tuple[float, float, float, float] | None = None,
        size: tuple[int, int] | None = None,
    ) -> RangeEstimate:
        hits, bearings = self._cast(box, geometry, grid, robot, intrinsics, size)
        needed = max(1, math.ceil(self.min_hit_fraction * self.rays))
        if len(hits) < needed:
            raise ValueError(
                f"the local costmap has nothing solid along that bearing within "
                f"{self._reach(grid):.1f} m ({len(hits)} of {self.rays} rays hit "
                "something). The object may be glass, too thin or too high for the "
                "robot's obstacle sensors, or beyond the local window. Get closer "
                "with navigate_to, or ask a human."
            )
        distance = _median([hit for hit, _ in hits])
        azimuth = bearings[len(bearings) // 2]
        camera = self._camera_xy(geometry, robot)
        # Back into the robot's frame from the hit point, so `range_m` means the
        # same thing here as it does for depth: distance from base_footprint.
        hit_x = camera[0] + math.cos(robot.yaw + azimuth) * distance
        hit_y = camera[1] + math.sin(robot.yaw + azimuth) * distance
        return RangeEstimate(
            method="costmap_raycast",
            target_base=odom_to_local(hit_x, hit_y, robot),
            rays=self.rays,
            hits=len(hits),
            spread_m=max(h for h, _ in hits) - min(h for h, _ in hits),
            unknown_crossed=sum(unknown for _, unknown in hits),
            detail=f"median of {len(hits)} costmap ray-casts across the box",
        )

    def check(
        self,
        geometry: CameraGeometry,
        grid: CostGrid,
        robot: Pose3D,
        azimuth_rad: float,
    ) -> float | None:
        """Distance from base_footprint to the first solid cell on one bearing."""
        camera = self._camera_xy(geometry, robot)
        hit = grid.raycast(
            camera[0], camera[1], robot.yaw + azimuth_rad, max_range_m=self._reach(grid)
        )
        if hit.range_m is None:
            return None
        return math.hypot(*odom_to_local(hit.x, hit.y, robot))

    # ------------------------------------------------------------ internal --
    def _reach(self, grid: CostGrid) -> float:
        # Never ask past the window: a ray that leaves it reports "edge", and
        # calling that a miss is honest, but calling a far corner a hit is not.
        return max(grid.resolution, min(self.max_range_m, grid.half_window_m - 0.25))

    def _camera_xy(self, geometry: CameraGeometry, robot: Pose3D) -> tuple[float, float]:
        c, s = math.cos(robot.yaw), math.sin(robot.yaw)
        mount_x, mount_y = geometry.mount_xyz[0], geometry.mount_xyz[1]
        return (robot.x + mount_x * c - mount_y * s, robot.y + mount_x * s + mount_y * c)

    def _cast(
        self,
        box: Box,
        geometry: CameraGeometry,
        grid: CostGrid,
        robot: Pose3D,
        intrinsics: tuple[float, float, float, float] | None,
        size: tuple[int, int] | None,
    ) -> tuple[list[tuple[float, int]], list[float]]:
        intr = intrinsics or geometry.intrinsics
        width, height = size or (geometry.width, geometry.height)
        inner = box.shrunk(self.keep)
        _, v_norm = inner.centre
        v = v_norm * height
        camera = self._camera_xy(geometry, robot)
        reach = self._reach(grid)
        hits: list[tuple[float, int]] = []
        bearings: list[float] = []
        for index in range(self.rays):
            ratio = 0.5 if self.rays == 1 else index / (self.rays - 1)
            u = (inner.x0 + ratio * (inner.x1 - inner.x0)) * width
            azimuth = pixel_azimuth(geometry, intr, u, v)
            bearings.append(azimuth)
            hit = grid.raycast(
                camera[0], camera[1], robot.yaw + azimuth, max_range_m=reach
            )
            if hit.range_m is not None:
                hits.append((hit.range_m, hit.unknown_crossed))
        return hits, bearings


# ------------------------------------------------------------------ standoff --
def standoff_pose(
    robot_xy: tuple[float, float],
    target_xy: tuple[float, float],
    standoff_m: float,
) -> tuple[float, float, float, float]:
    """(goal_x, goal_y, goal_yaw, robot-to-target distance).

    The goal sits `standoff_m` short of the target ON THE LINE FROM THE ROBOT,
    and faces the target.

    ⚠️ FROM THE ROBOT'S SIDE, NOT AN ABSOLUTE BEARING, AND THAT IS THE WHOLE
    CHOICE. "Go to the sink" means approach it from where you are; picking any
    other side means walking PAST the object to get behind it, through space
    nothing has told the robot is clear. It also makes the answer stable -- the
    approach direction is whatever the camera was looking along. Ported from
    g1_bridge/object_goal_geom.py, which the robot's own object-goal path uses.
    """
    dx = target_xy[0] - robot_xy[0]
    dy = target_xy[1] - robot_xy[1]
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        return (target_xy[0], target_xy[1], 0.0, 0.0)
    yaw = math.atan2(dy, dx)
    # Never longer than the distance itself, or the "standoff" is a goal BEHIND
    # the robot and it walks backwards away from the thing it was sent to.
    back = min(float(standoff_m), distance)
    return (
        target_xy[0] - math.cos(yaw) * back,
        target_xy[1] - math.sin(yaw) * back,
        yaw,
        distance,
    )


def approach_goal(
    robot_xy: tuple[float, float],
    target_xy: tuple[float, float],
    *,
    standoff_m: float,
    max_leg_m: float,
) -> tuple[float, float, float, float, bool]:
    """One leg towards the target: (x, y, yaw, distance_to_target, is_final).

    ##### A LOCAL PLAN IS OFTEN NOT THE WHOLE WALK, AND PRETENDING OTHERWISE IS
    THE BUG. ##### The costmap is a rolling window a few metres across and the
    camera can see well past its edge, so a sink across a room is routinely
    visible, correctly ranged, and further away than anything this planner is
    allowed to commit to. Refusing there would be useless: the robot can see it
    and could obviously walk at it.

    So the goal is clamped to the longest leg that is both inside the window and
    inside the walk budget, and the caller is told `is_final=False`. The model
    looks again from the new spot and calls the tool again -- each leg grounded
    on a fresh picture and a fresh costmap, which is also what makes the
    approach robust to the target being partly occluded at the start.

    `is_final` means this leg ends at the standoff, so there is nothing left to
    do but arrive.
    """
    # The final leg IS the standoff pose, so there is one implementation of
    # "short of it, on the line from the robot, facing it" and this clamps it.
    goal_x, goal_y, yaw, distance = standoff_pose(robot_xy, target_xy, standoff_m)
    desired = math.hypot(goal_x - robot_xy[0], goal_y - robot_xy[1])
    travel = min(desired, max(0.0, float(max_leg_m)))
    if travel >= desired - 1e-6:
        return (goal_x, goal_y, yaw, distance, True)
    return (
        robot_xy[0] + math.cos(yaw) * travel,
        robot_xy[1] + math.sin(yaw) * travel,
        yaw,
        distance,
        False,
    )


def back_off_until_free(
    free: GridMap,
    robot_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    *,
    step_m: float | None = None,
) -> tuple[float, float, float]:
    """Pull a goal back towards the robot until it is somewhere to stand.

    Returns (x, y, metres given up). The thing being approached IS an obstacle,
    so a goal placed a nominal standoff short of it lands inside its inflation
    routinely -- this is the ordinary case, not the error case. Stepping towards
    the robot is safe by construction: the robot is standing at the far end of
    that line, so the space has just been occupied by the robot itself.
    """
    step = step_m or free.resolution
    dx = goal_xy[0] - robot_xy[0]
    dy = goal_xy[1] - robot_xy[1]
    travel = math.hypot(dx, dy)
    if travel < 1e-9:
        return (robot_xy[0], robot_xy[1], 0.0)
    ux, uy = dx / travel, dy / travel
    given_up = 0.0
    while given_up <= travel + 1e-9:
        x = goal_xy[0] - ux * given_up
        y = goal_xy[1] - uy * given_up
        cell = free.world_to_cell(x, y)
        if free.in_bounds(cell) and cell not in free.occupied:
            return (x, y, given_up)
        given_up += step
    return (robot_xy[0], robot_xy[1], travel)


# --------------------------------------------------------------------- poses --
@dataclass(frozen=True)
class InitPose:
    """Where the robot was when the tool ran, in both frames it needs.

    Sampled together. See the module docstring: the composed odom->map transform
    is only as good as the simultaneity of these two reads.
    """

    odom: Pose3D
    map: Pose3D
    odom_age_s: float = 0.0
    map_age_s: float = 0.0

    def as_json(self) -> Json:
        return {
            "odom": {
                "x": round(self.odom.x, 3),
                "y": round(self.odom.y, 3),
                "yaw_deg": round(math.degrees(self.odom.yaw), 1),
                "frame": self.odom.frame,
                "age_ms": round(self.odom_age_s * 1000),
            },
            "map": {
                "x": round(self.map.x, 3),
                "y": round(self.map.y, 3),
                "yaw_deg": round(math.degrees(self.map.yaw), 1),
                "frame": self.map.frame,
                "age_ms": round(self.map_age_s * 1000),
            },
        }


class InitPoseProvider(Protocol):
    def init_pose(self, expected_odom_frame: str) -> InitPose: ...


class TargetGrounder(Protocol):
    def ground(self, target: str, frame: CameraFrame) -> Grounding: ...


# ---------------------------------------------------------------- the result --
@dataclass(frozen=True)
class LocalPlan:
    target: str
    grounding: Grounding
    camera_label: str
    intrinsics_source: str
    range: RangeEstimate
    init: InitPose
    target_local: tuple[float, float]
    standoff_local: Pose3D
    standoff_odom: Pose3D
    standoff_map: Pose3D
    standoff_m: float
    #: True when this leg ends AT the standoff. False means it is one hop of a
    #: longer approach and the model should look again and call again.
    final: bool
    #: Distance from the goal of this leg to the target itself.
    remaining_m: float
    leg_m: float
    given_up_m: float
    turn_only: bool
    trajectory_local: tuple[tuple[float, float], ...]
    length_m: float
    min_clearance_m: float
    start_adjusted_m: float
    footprint_radius_m: float
    grid: CostGrid
    depth: DepthImage | None = None
    #: Carried so the plan can report its own body clearance without a caller
    #: re-deriving it from the planner. See `body_clearance`.
    nose_reach_m: float = 0.3079
    arrival_box_m: float = 0.10
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def achieved_standoff_m(self) -> float:
        """Centre-to-object distance of the goal actually being sent.

        ⚠️ NOT `standoff_m`, WHICH IS ONLY WHAT WAS ASKED FOR. The costmap gets
        the last word: `back_off_until_free` pulls the goal towards the robot
        until the body fits, so a target against a wall, or with a chair in
        front of it, ends up FURTHER away than requested by `given_up_m`.
        """
        return self.standoff_m + self.given_up_m

    @property
    def body_clearance(self) -> tuple[float, float]:
        """(nominal, worst-case) metres between the body's front and THE TARGET.

        The worst case is the one that matters: the goal checker may declare
        arrival anywhere inside `arrival_box_m`, so that radius comes straight
        off the gap. Both are reported because a reader who sees only the
        nominal figure will not believe the floor is where it is.

        ⚠️ MEASURED TO THE RANGED TARGET, NOT TO WHATEVER IS NEAREST. When
        `given_up_m` is non-zero the goal was pulled back by the costmap -- so
        something OTHER than the target is the near thing, this number is the gap
        to the target BEYOND it, and it will look generous precisely when the
        robot is closest to something. What bounds the body then is the inflated
        costmap, which is why `back_off_until_free` searches an inflation of
        `footprint_radius_m`: the guarantee is that the body disc holds no lethal
        cell, and `trajectory.min_clearance_m` is that margin. A leg like this
        also reports `final: False`, so it is never read as an arrival.
        """
        nominal = self.achieved_standoff_m - self.nose_reach_m
        return (nominal, nominal - self.arrival_box_m)

    @property
    def next_step(self) -> str:
        """What the mission controller should do after this leg. Read by a model."""
        if self.final:
            return (
                "this leg ends at the standoff: check the fresh camera frame to "
                "confirm the robot is in front of the target, then move on"
            )
        return (
            f"NOT THERE YET -- about {self.remaining_m:.1f} m of approach is left "
            "beyond this leg. Look at the fresh camera frame and call local_planner "
            "again with the same target to walk the next leg."
        )

    def as_json(self) -> Json:
        return {
            "target": self.target,
            "reached_standoff": self.final,
            "remaining_m": round(self.remaining_m, 2),
            "next_step": self.next_step,
            "grounding": self.grounding.as_json(),
            "camera": {
                "label": self.camera_label,
                "intrinsics_source": self.intrinsics_source,
                "frame_id": None if self.depth is None else self.depth.frame_id,
            },
            "range": self.range.as_json(),
            "target_local": {
                "x": round(self.target_local[0], 3),
                "y": round(self.target_local[1], 3),
                "frame": "robot pose when this tool ran: +x forward, +y left",
            },
            "goal": {
                "local": _pose_json(self.standoff_local),
                "map": _pose_json(self.standoff_map),
                "leg_m": round(self.leg_m, 3),
                "standoff_m": round(self.standoff_m, 3),
                "achieved_standoff_m": round(self.achieved_standoff_m, 3),
                "pulled_back_m": round(self.given_up_m, 3),
                "turn_only": self.turn_only,
                # ⚠️ THE ONLY NUMBERS HERE AN OPERATOR CAN CHECK BY LOOKING. Every
                # other distance is centre-to-object, which is ~0.31 m more than
                # the gap they will see. `worst_case_m` is what to judge: the goal
                # checker may latch at the near edge of its box.
                "body_clearance_m": {
                    "nominal": round(self.body_clearance[0], 3),
                    "worst_case": round(self.body_clearance[1], 3),
                    "note": (
                        "gap between the FRONT OF THE BODY and the TARGET, not the "
                        "centre-to-object standoff above; worst_case assumes the goal "
                        f"checker latches at the near edge of its {self.arrival_box_m:g} m box"
                        + (
                            ". ⚠️ pulled_back_m is non-zero, so the costmap moved this "
                            "goal and something nearer than the target is what stopped "
                            "it: THIS gap is to the target beyond that thing, and the "
                            "margin actually holding the body is trajectory."
                            "min_clearance_m. reached_standoff is false for the same "
                            "reason -- look again and call again"
                            if self.given_up_m > 1e-9
                            else ""
                        )
                    ),
                },
            },
            "trajectory": {
                "frame": "local",
                "points": [[round(x, 3), round(y, 3)] for x, y in self.trajectory_local],
                "length_m": round(self.length_m, 3),
                "min_clearance_m": round(self.min_clearance_m, 3),
                "footprint_radius_m": self.footprint_radius_m,
                "start_adjusted_m": round(self.start_adjusted_m, 3),
                "note": (
                    "a feasibility check on the local costmap, not the route walked: "
                    "Nav2 plans and tracks its own path to the goal pose"
                ),
            },
            "costmap": self.grid.as_json(),
            "depth": None if self.depth is None else self.depth.as_json(),
            "init_pose": self.init.as_json(),
            "notes": list(self.notes),
        }


def _pose_json(pose: Pose3D) -> Json:
    return {
        "x": round(pose.x, 3),
        "y": round(pose.y, 3),
        "yaw_deg": round(math.degrees(pose.yaw), 1),
    }


# -------------------------------------------------------------- the planner --
class LocalPlanner:
    """Ground, range, place a goal, prove a path. Publishes nothing."""

    def __init__(
        self,
        *,
        camera: CameraBackend,
        grounder: TargetGrounder,
        grid_source: LocalGridSource,
        init_pose: InitPoseProvider,
        geometry: CameraGeometry,
        depth: DepthSource | None = None,
        colour_label: str = "head_colour",
        expected_grid_frame: str = "odom",
        footprint_radius_m: float = 0.35,
        nose_reach_m: float = 0.3079,
        stop_zone_m: float = 0.40,
        arrival_box_m: float = 0.10,
        stop_zone_margin_m: float = 0.10,
        standoff_m: float = 0.60,
        standoff_min_m: float | None = None,
        standoff_max_m: float = 3.0,
        max_leg_m: float = 4.0,
        min_leg_m: float = 0.25,
        path_slack_m: float = 1.0,
        max_range_m: float = 6.0,
        start_adjust_m: float = 0.4,
        window_margin_m: float = 0.35,
        depth_ranger: DepthRanger | None = None,
        costmap_ranger: CostmapRaycastRanger | None = None,
    ) -> None:
        self.camera = camera
        self.grounder = grounder
        self.grid_source = grid_source
        self.init_pose_source = init_pose
        self.geometry = geometry
        self.depth_source = depth
        self.colour_label = colour_label
        self.expected_grid_frame = expected_grid_frame
        self.footprint_radius_m = footprint_radius_m
        # ⚠️ EVERY STANDOFF HERE IS base_footprint-CENTRE TO OBJECT, AND THAT IS
        # NOT WHAT AN OPERATOR MEANS BY "GET 0.2 m FROM IT". The measured G1
        # footprint reaches x = +0.3079 in front of the origin
        # (g1_auto_navigation nav2_g1_robot.yaml `footprint`, from
        # scripts/measure_footprint.py), so the gap a person actually sees is
        # `standoff_m - nose_reach_m`. Asking for a 0.30 m standoff parks the
        # object 0.01 m INSIDE the body: the robot standing on the thing it was
        # sent to look at. Callers should ask in clearance and let
        # `standoff_for_clearance` do the arithmetic once, here.
        self.nose_reach_m = nose_reach_m
        #: collision_monitor's `StopZone`: a hard stop that reads /scan directly
        #: and logs NOTHING when it fires. g1-016 has no hardware E-stop, so this
        #: is the backstop rather than a second opinion -- see the fleet notes.
        self.stop_zone_m = stop_zone_m
        # ⚠️ THIS MUST MATCH NAV2'S `xy_goal_tolerance`, AND IT LIVES IN ANOTHER
        # REPO (g1_auto_navigation src/g1_bridge/config/nav2_g1_robot.yaml).
        # `SimpleGoalChecker` latches ANYWHERE inside this radius, so it comes
        # straight off the clearance: a 0.25 m box against a 0.60 m standoff
        # allows a worst-case arrival of 0.35 m, i.e. the object 0.04 m inside
        # the footprint AND inside the stop zone. The floor below is derived from
        # this value rather than hardcoded precisely so the two cannot drift
        # apart silently: raise this and close approaches start being refused,
        # with a message naming the YAML to change.
        self.arrival_box_m = arrival_box_m
        #: How much clear air to keep over `stop_zone_m` at worst-case arrival.
        #: Without it the floor sits exactly ON the stop zone, where a single
        #: overshoot onto a LiDAR-visible object reads as "arrived slightly
        #: short" with no zone hit in any log.
        self.stop_zone_margin_m = stop_zone_margin_m
        self.standoff_m = standoff_m
        # ⚠️ A FLOOR, NOT A PREFERENCE -- BUT A DERIVED ONE. The binding
        # constraint is not the body (0.31 m); it is the stop zone plus the
        # arrival box plus a margin, because the checker may latch anywhere in
        # the box and anything inside the zone halts the walk unlogged:
        #     0.40 + 0.10 + 0.10 = 0.60 m centre-to-object
        #                        = 0.29 m of nose clearance nominal
        #                        = 0.19 m at worst-case arrival
        # which is as close as this robot can be asked to stand and still be
        # guaranteed not to stop itself. Closer than that is not a tuning
        # question, it is a different robot: a smaller stop zone, and that is
        # the only stop this machine has.
        self.standoff_min_m = (
            stop_zone_m + arrival_box_m + stop_zone_margin_m
            if standoff_min_m is None
            else standoff_min_m
        )
        self.standoff_max_m = standoff_max_m
        #: The longest single commitment. Not a limit on how far the target may
        #: be -- see approach_goal: further just means more legs.
        self.max_leg_m = max_leg_m
        #: Below this a "leg" is not a walk, it is a shuffle Nav2's goal
        #: tolerance would swallow whole.
        self.min_leg_m = min_leg_m
        self.path_slack_m = path_slack_m
        self.max_range_m = max_range_m
        self.start_adjust_m = start_adjust_m
        #: Keep the goal this far inside the costmap window. A goal on the rim is
        #: a goal the rolling window may have left behind by the time Nav2 plans.
        self.window_margin_m = window_margin_m
        self.depth_ranger = depth_ranger or DepthRanger(max_range_m=max_range_m)
        self.costmap_ranger = costmap_ranger or CostmapRaycastRanger()

    # ------------------------------------------------------------------ API --
    def plan(
        self,
        target: str,
        *,
        standoff_m: float | None = None,
        clearance_m: float | None = None,
    ) -> LocalPlan:
        """One leg towards `target`, or ValueError saying why not.

        `clearance_m` is the operator-facing spelling: metres of clear air to
        leave between the FRONT OF THE BODY and the object. `standoff_m` is the
        same request measured centre-to-object, which is what a goal pose needs.
        Pass one or neither -- both together is a caller that has not decided
        which it means, and the two disagree by `nose_reach_m`.
        """
        if clearance_m is not None and standoff_m is not None:
            raise ValueError(
                "pass clearance_m or standoff_m, not both: they measure the same "
                f"approach from different origins ({self.nose_reach_m:.2f} m apart), "
                "so supplying both cannot be honoured as written"
            )
        if clearance_m is not None:
            standoff_m = self.standoff_for_clearance(clearance_m)
        try:
            return self._plan(target, standoff_m)
        except ValueError:
            raise
        except NoDepthReturn as exc:
            raise ValueError(f"could not measure a distance to {target!r}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            # ⚠️ Tool.run ONLY CATCHES (KeyError, TypeError, ValueError). A
            # decoder error or a model transport failure raised from here would
            # otherwise leave the agent loop with a traceback rather than a step
            # the model can react to -- and would do it AFTER grounding spent a
            # call but BEFORE anything moved, which is a confusing place to die.
            raise ValueError(
                f"local planning failed before anything moved: {type(exc).__name__}: {exc}"
            ) from exc

    # ------------------------------------------------------------- internal --
    def _plan(self, target: str, standoff_override: float | None) -> LocalPlan:
        standoff = self._standoff(standoff_override)
        notes: list[str] = []

        frame = self._colour_frame()
        init = self.init_pose_source.init_pose(self.expected_grid_frame)
        grid = self.grid_source.grid()
        if grid.frame != init.odom.frame:
            raise ValueError(
                f"the local costmap is in frame {grid.frame!r} but the robot's "
                f"odometry is in {init.odom.frame!r}; they must match to place the "
                "robot on the grid"
            )
        depth = self._depth()

        grounding = self.grounder.ground(target, frame)
        assert grounding.box is not None  # VisionGrounder refuses otherwise

        estimate, intrinsics_source = self._range(grounding.box, grid, init.odom, depth, notes)
        if estimate.range_m > self.max_range_m:
            raise ValueError(
                f"{target!r} measures {estimate.range_m:.1f} m away, past the "
                f"{self.max_range_m:g} m this camera's ranging is trusted to. Use "
                "navigate_to to get into the same part of the room first, then look "
                "again."
            )

        target_odom = local_to_odom(*estimate.target_base, init.odom)
        planning_grid = grid.to_gridmap()
        free = planning_grid.inflated(self.footprint_radius_m)

        # How far this leg may commit: the walk budget, and the window the
        # costmap actually describes. Whichever is smaller.
        reach = max(0.0, grid.half_window_m - self.window_margin_m)
        goal_x, goal_y, goal_yaw, distance, final = approach_goal(
            (init.odom.x, init.odom.y),
            target_odom,
            standoff_m=standoff,
            max_leg_m=min(self.max_leg_m, reach),
        )
        goal_x, goal_y, given_up = back_off_until_free(
            free, (init.odom.x, init.odom.y), (goal_x, goal_y)
        )
        leg = math.hypot(goal_x - init.odom.x, goal_y - init.odom.y)
        remaining = max(0.0, distance - leg - standoff)
        final = final and given_up <= 1e-9

        turn_only = leg < self.min_leg_m
        if turn_only:
            if distance > standoff + self.min_leg_m:
                # There IS ground to cover and no free pose along the line: the
                # approach is blocked, and turning on the spot would report
                # progress that did not happen.
                raise ValueError(
                    f"there is no free place to stand between the robot and {target!r}: "
                    f"it is {distance:.1f} m away but every pose along the approach is "
                    "inside an obstacle or its safety margin on the local costmap. Try "
                    "approaching from somewhere else with navigate_to."
                )
            # Already inside the standoff. Still a goal, still gated, still
            # published -- Nav2's goal checker delivers the heading -- but say
            # so, because "arrived" after a pure turn is a different claim.
            goal_x, goal_y = init.odom.x, init.odom.y
            leg = 0.0
            final = True
            remaining = 0.0
            notes.append(
                f"already within {standoff:.2f} m of it: this turns to face it "
                "rather than walking"
            )
        if not final:
            notes.append(
                f"this is one leg of a longer approach: about {remaining:.1f} m will "
                "still be left when it ends"
            )

        trajectory, length, clearance, adjusted = self._verify(
            planning_grid, free, init.odom,
            Pose3D(goal_x, goal_y, yaw=goal_yaw, frame=grid.frame),
        )
        budget = min(self.max_leg_m, reach) + self.path_slack_m
        if length > budget:
            raise ValueError(
                f"the only way to that goal on the local costmap is a {length:.1f} m "
                f"detour, more than the {budget:.1f} m one leg may commit to. Use "
                "navigate_to to get past whatever is in the way, then look again."
            )

        goal_local_xy = odom_to_local(goal_x, goal_y, init.odom)
        goal_local_yaw = wrap_angle(goal_yaw - init.odom.yaw)
        return LocalPlan(
            target=target,
            grounding=grounding,
            camera_label=frame.label,
            intrinsics_source=intrinsics_source,
            range=estimate,
            init=init,
            target_local=estimate.target_base,
            standoff_local=Pose3D(*goal_local_xy, yaw=goal_local_yaw, frame="local"),
            standoff_odom=Pose3D(goal_x, goal_y, yaw=goal_yaw, frame=grid.frame),
            standoff_map=local_to_map(
                goal_local_xy[0], goal_local_xy[1], goal_local_yaw, init.map
            ),
            standoff_m=standoff,
            final=final,
            remaining_m=remaining,
            leg_m=leg,
            given_up_m=given_up,
            turn_only=turn_only,
            trajectory_local=tuple(
                odom_to_local(pose.x, pose.y, init.odom) for pose in trajectory
            ),
            length_m=length,
            min_clearance_m=clearance,
            start_adjusted_m=adjusted,
            footprint_radius_m=self.footprint_radius_m,
            nose_reach_m=self.nose_reach_m,
            arrival_box_m=self.arrival_box_m,
            grid=grid,
            depth=depth,
            notes=tuple(notes),
        )

    # ------------------------------------------------------- clearance <-> --
    # One place converts between what an operator means ("stop 0.2 m from it")
    # and what a goal pose is ("put base_footprint's centre here"). Two places
    # doing this arithmetic is how a 0.31 m body becomes a collision.
    def standoff_for_clearance(self, clearance_m: float) -> float:
        """Centre-to-object standoff that leaves `clearance_m` in front."""
        return float(clearance_m) + self.nose_reach_m

    def clearance_for_standoff(self, standoff_m: float) -> float:
        """Nominal gap between the front of the body and the object."""
        return float(standoff_m) - self.nose_reach_m

    def worst_case_clearance_for_standoff(self, standoff_m: float) -> float:
        """The gap if the goal checker latches at the near edge of its box.

        This is the number that has to stay positive and outside the stop zone,
        not the nominal one -- `SimpleGoalChecker` is free to declare arrival
        anywhere inside `arrival_box_m`.
        """
        return float(standoff_m) - self.arrival_box_m - self.nose_reach_m

    @property
    def min_clearance_m(self) -> float:
        """Closest the front of the body may be asked to come to an object."""
        return self.clearance_for_standoff(self.standoff_min_m)

    @property
    def max_clearance_m(self) -> float:
        return self.clearance_for_standoff(self.standoff_max_m)

    def _standoff(self, override: float | None) -> float:
        if override is None:
            return self.standoff_m
        value = float(override)
        if not self.standoff_min_m <= value <= self.standoff_max_m:
            # ⚠️ THE REFUSAL EXPLAINS ITSELF IN CLEARANCE, because the number the
            # caller chose is centre-to-object and the number they were thinking
            # of is not. The old message said "inside the collision stop zone",
            # which is true and reads as a tunable safety margin -- so a model
            # would retry with 0.31 rather than understand that it had asked the
            # robot to stand on the object.
            raise ValueError(
                f"standoff_m must be between {self.standoff_min_m:g} and "
                f"{self.standoff_max_m:g} metres; {value:g} is outside that. This is "
                f"measured from the robot's CENTRE, and its body reaches "
                f"{self.nose_reach_m:.2f} m forward, so {value:g} m leaves "
                f"{self.clearance_for_standoff(value):.2f} m in front of it -- and the "
                f"{self.arrival_box_m:g} m arrival box takes that down to "
                f"{self.worst_case_clearance_for_standoff(value):.2f} m. The floor of "
                f"{self.standoff_min_m:g} m is the stop zone ({self.stop_zone_m:g} m, "
                f"which halts the robot with nothing in any log) plus that box plus "
                f"{self.stop_zone_margin_m:g} m of margin. If what you wanted was to get "
                f"close enough to reach the object, ask for clearance_m instead: the "
                f"closest this robot can stand is {self.min_clearance_m:.2f} m of clear "
                f"air in front of its body, and the arm reaches past that."
            )
        return value

    def _colour_frame(self) -> CameraFrame:
        frames = list(self.camera.capture())
        for frame in frames:
            if frame.label == self.colour_label:
                return frame
        seen = ", ".join(frame.label for frame in frames) or "(nothing)"
        raise ValueError(
            f"no fresh {self.colour_label} frame to look at; the camera is offering: "
            f"{seen}. Without a current picture there is nothing to ground."
        )

    def _depth(self) -> DepthImage | None:
        if self.depth_source is None:
            return None
        try:
            return self.depth_source.capture()
        except Exception:  # noqa: BLE001
            # Depth is an upgrade. A decoder that fell over must degrade to the
            # costmap, not end the mission -- the reason is in status().
            return None

    def _range(
        self,
        box: Box,
        grid: CostGrid,
        robot: Pose3D,
        depth: DepthImage | None,
        notes: list[str],
    ) -> tuple[RangeEstimate, str]:
        if depth is not None:
            try:
                estimate = self.depth_ranger.range(box, self.geometry, depth)
            except NoDepthReturn as exc:
                notes.append(f"no usable depth over the box ({exc}); ranged on the costmap")
            else:
                check = self.costmap_ranger.check(
                    self.geometry, grid, robot, estimate.azimuth_rad
                )
                agrees = None if check is None else abs(check - estimate.range_m) <= 0.5
                if agrees is False:
                    notes.append(
                        f"the costmap puts the first solid thing on that bearing at "
                        f"{check:.2f} m against the camera's {estimate.range_m:.2f} m"
                    )
                return (
                    replace(estimate, costmap_check_m=check, agrees=agrees),
                    "depth_info",
                )
        else:
            notes.append("no metric depth on this robot's link; ranged on the costmap")
        return (
            self.costmap_ranger.range(box, self.geometry, grid, robot),
            "spec_fov",
        )

    def _verify(
        self, grid: GridMap, free: GridMap, start: Pose3D, goal: Pose3D
    ) -> tuple[list[Pose3D], float, float, float]:
        """A* to the goal: proof a path exists, and how tight it is.

        ⚠️ THE FOOTPRINT IS ALREADY IN `free`, so the planner is handed a radius
        of ZERO rather than inflating the same grid a second time. Inflating a
        160x160 costmap by a 7-cell radius is the most expensive thing in a
        plan; doing it twice made every approach visibly slower for nothing.
        `grid` stays the un-inflated one, because the clearance REPORTED to an
        operator should be the distance to the real obstacle, not to the edge of
        the robot's own safety margin.
        """
        planner = AStarPlanner(free, footprint_radius_m=0.0)
        origin, adjusted = self._free_start(planner.grid, start)
        try:
            path = planner.plan(origin, goal)
        except ValueError as exc:
            raise ValueError(
                f"the goal pose is not usable on the local costmap: {exc}"
            ) from exc
        except RuntimeError:
            raise ValueError(
                "the local costmap has no collision-free path from here to a pose "
                "facing the target. Something is in the way that the robot cannot "
                "walk around within the local window. Try approaching from somewhere "
                "else with navigate_to."
            ) from None
        length = sum(
            math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(path, path[1:])
        )
        clearance = min(grid.clearance(pose.x, pose.y) for pose in path)
        return (path, length, clearance, adjusted)

    def _free_start(self, inflated: GridMap, start: Pose3D) -> tuple[Pose3D, float]:
        """A* refuses to start inside inflation; standing near a wall is normal.

        Nudging the SEARCH start is honest because the robot is already there and
        Nav2 gets the same job with its own recovery behaviours. Nudging the GOAL
        would not be, which is why only this end moves.
        """
        cell = inflated.world_to_cell(start.x, start.y)
        if inflated.in_bounds(cell) and cell not in inflated.occupied:
            return (start, 0.0)
        rings = max(1, int(math.ceil(self.start_adjust_m / inflated.resolution)))
        best: tuple[float, Pose3D] | None = None
        for radius in range(1, rings + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    candidate = (cell[0] + dx, cell[1] + dy)
                    if not inflated.in_bounds(candidate) or candidate in inflated.occupied:
                        continue
                    wx, wy = inflated.cell_to_world(candidate)
                    distance = math.hypot(wx - start.x, wy - start.y)
                    if distance <= self.start_adjust_m and (best is None or distance < best[0]):
                        best = (distance, Pose3D(wx, wy, yaw=start.yaw, frame=start.frame))
            if best is not None:
                return (best[1], best[0])
        raise ValueError(
            "the robot is standing inside the local costmap's obstacle inflation and "
            f"there is no clear cell within {self.start_adjust_m:g} m of it. Back away "
            "with navigate_to before approaching anything."
        )


__all__ = [
    "COST_INSCRIBED",
    "COST_LETHAL",
    "COST_UNKNOWN",
    "CameraGeometry",
    "CostGrid",
    "CostmapRaycastRanger",
    "DepthImage",
    "DepthRanger",
    "DepthSource",
    "GridUpdate",
    "InitPose",
    "InitPoseProvider",
    "LocalGridSource",
    "LocalPlan",
    "LocalPlanner",
    "NoDepthReturn",
    "RangeEstimate",
    "RayHit",
    "TargetGrounder",
    "approach_goal",
    "back_off_until_free",
    "local_to_map",
    "local_to_odom",
    "odom_to_local",
    "optical_to_base",
    "pixel_azimuth",
    "pixel_ray_base",
    "standoff_pose",
    "wrap_angle",
]
