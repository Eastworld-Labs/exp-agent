from __future__ import annotations

import heapq
import json
import math
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .modules.semantic_map import Pose3D


@dataclass(frozen=True)
class VelocityCommand:
    vx: float
    vy: float
    yaw_rate: float
    facing_yaw: float | None = None


@dataclass(frozen=True)
class LocalObstacle:
    """A sensor-derived circular obstacle in the navigation map frame."""

    x: float
    y: float
    radius_m: float = 0.0
    label: str = "obstacle"

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.x, self.y, self.radius_m)):
            raise ValueError("local obstacle coordinates and radius must be finite")
        if self.radius_m < 0:
            raise ValueError("local obstacle radius must be non-negative")


@dataclass(frozen=True)
class LocalNavigationObservation:
    """Control-rate perception consumed inside a blocking navigation skill.

    RGB/depth/LiDAR adapters are responsible for localization and projection
    into ``frame``. The mission model never turns pixels into velocity commands.
    """

    source: str
    obstacles: tuple[LocalObstacle, ...] = ()
    frame: str = "map"
    healthy: bool = True
    emergency_stop: bool = False
    detail: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "frame": self.frame,
            "healthy": self.healthy,
            "emergency_stop": self.emergency_stop,
            "obstacle_count": len(self.obstacles),
            "detail": self.detail,
        }


class LocalNavigationObserver(Protocol):
    """Depth/LiDAR/vision-costmap boundary for short-horizon navigation."""

    def observe(self, pose: Pose3D) -> LocalNavigationObservation: ...


class MobileBase(Protocol):
    """The only boundary a navigation controller needs from locomotion."""

    @property
    def name(self) -> str: ...

    def pose(self) -> Pose3D: ...

    def command_velocity(self, command: VelocityCommand, dt: float) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True)
class GridMap:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    occupied: frozenset[tuple[int, int]]

    @classmethod
    def from_json(cls, path: str | Path) -> "GridMap":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            width=int(raw["width"]),
            height=int(raw["height"]),
            resolution=float(raw["resolution"]),
            origin_x=float(raw.get("origin", [0.0, 0.0])[0]),
            origin_y=float(raw.get("origin", [0.0, 0.0])[1]),
            occupied=frozenset((int(cell[0]), int(cell[1])) for cell in raw.get("occupied", [])),
        )

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(math.floor((x - self.origin_x) / self.resolution)),
            int(math.floor((y - self.origin_y) / self.resolution)),
        )

    def cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (
            self.origin_x + (cell[0] + 0.5) * self.resolution,
            self.origin_y + (cell[1] + 0.5) * self.resolution,
        )

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def inflated(self, radius_m: float) -> "GridMap":
        radius = max(0, int(math.ceil(radius_m / self.resolution)))
        cells = set(self.occupied)
        for x, y in self.occupied:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:
                        candidate = (x + dx, y + dy)
                        if self.in_bounds(candidate):
                            cells.add(candidate)
        return GridMap(
            self.width,
            self.height,
            self.resolution,
            self.origin_x,
            self.origin_y,
            frozenset(cells),
        )

    def segment_is_free(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> bool:
        """Conservatively collision-check a world-frame line segment."""
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        samples = max(1, int(math.ceil(distance / (self.resolution * 0.25))))
        for index in range(samples + 1):
            ratio = index / samples
            cell = self.world_to_cell(
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
            )
            if not self.in_bounds(cell) or cell in self.occupied:
                return False
        return True

    def segment_has_clearance(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        minimum_clearance_m: float,
    ) -> bool:
        """Check occupancy and a continuous clearance margin along a segment."""
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        samples = max(1, int(math.ceil(distance / (self.resolution * 0.25))))
        return all(
            self.segment_is_free(
                (
                    start[0] + index / samples * (end[0] - start[0]),
                    start[1] + index / samples * (end[1] - start[1]),
                ),
                (
                    start[0] + index / samples * (end[0] - start[0]),
                    start[1] + index / samples * (end[1] - start[1]),
                ),
            )
            and self.clearance(
                start[0] + index / samples * (end[0] - start[0]),
                start[1] + index / samples * (end[1] - start[1]),
            ) >= minimum_clearance_m
            for index in range(samples + 1)
        )

    def clearance(self, x: float, y: float) -> float:
        """Approximate distance to an occupied cell or the map boundary."""
        boundary = min(
            x - self.origin_x,
            y - self.origin_y,
            self.origin_x + self.width * self.resolution - x,
            self.origin_y + self.height * self.resolution - y,
        )
        if not self.occupied:
            return max(0.0, boundary)
        cell = self.world_to_cell(x, y)
        if not self.in_bounds(cell):
            return 0.0
        obstacle = self._clearance_field[cell[1]][cell[0]] - self.resolution * math.sqrt(0.5)
        return max(0.0, min(boundary, obstacle))

    @cached_property
    def _clearance_field(self) -> tuple[tuple[float, ...], ...]:
        """Precompute an 8-neighbor distance transform for room-scale maps."""
        distance = [[math.inf] * self.width for _ in range(self.height)]
        frontier: list[tuple[float, tuple[int, int]]] = []
        for cell in self.occupied:
            if self.in_bounds(cell):
                distance[cell[1]][cell[0]] = 0.0
                heapq.heappush(frontier, (0.0, cell))
        moves = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
                 (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)))
        while frontier:
            value, (x, y) = heapq.heappop(frontier)
            if value != distance[y][x]:
                continue
            for dx, dy, cost in moves:
                nx, ny = x + dx, y + dy
                candidate = value + cost * self.resolution
                if 0 <= nx < self.width and 0 <= ny < self.height and candidate < distance[ny][nx]:
                    distance[ny][nx] = candidate
                    heapq.heappush(frontier, (candidate, (nx, ny)))
        return tuple(tuple(row) for row in distance)


class AStarPlanner:
    def __init__(
        self,
        grid: GridMap,
        *,
        footprint_radius_m: float = 0.32,
        path_clearance_m: float = 0.0,
    ) -> None:
        self.grid = grid.inflated(footprint_radius_m)
        self.path_clearance_m = path_clearance_m

    def plan(self, start: Pose3D, goal: Pose3D) -> list[Pose3D]:
        start_cell = self.grid.world_to_cell(start.x, start.y)
        goal_cell = self.grid.world_to_cell(goal.x, goal.y)
        for label, cell in (("start", start_cell), ("goal", goal_cell)):
            if not self.grid.in_bounds(cell):
                raise ValueError(f"{label} is outside the navigation grid: {cell}")
            if cell in self.grid.occupied:
                raise ValueError(f"{label} lies in an occupied/inflated cell: {cell}")

        frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start_cell)]
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}
        cost = {start_cell: 0.0}
        moves = (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
        )
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_cell:
                break
            for dx, dy, step_cost in moves:
                nxt = (current[0] + dx, current[1] + dy)
                if not self.grid.in_bounds(nxt) or nxt in self.grid.occupied:
                    continue
                # Do not cut diagonally through the corner of two obstacles.
                if dx and dy and (
                    (current[0] + dx, current[1]) in self.grid.occupied
                    or (current[0], current[1] + dy) in self.grid.occupied
                ):
                    continue
                wx, wy = self.grid.cell_to_world(nxt)
                clearance = self.grid.clearance(wx, wy)
                if nxt != goal_cell and clearance < self.path_clearance_m:
                    continue
                # Prefer the center of doors/corridors instead of merely
                # avoiding occupied cells. The footprint inflation is a hard
                # constraint; this is a soft buffer for the humanoid torso.
                clearance_penalty = 0.12 / max(clearance, self.grid.resolution * 0.25)
                new_cost = cost[current] + step_cost + clearance_penalty
                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt] = new_cost
                    heuristic = math.hypot(goal_cell[0] - nxt[0], goal_cell[1] - nxt[1])
                    heapq.heappush(frontier, (new_cost + heuristic, nxt))
                    came_from[nxt] = current
        if goal_cell not in came_from:
            raise RuntimeError("no collision-free path exists on the current grid")

        cells: list[tuple[int, int]] = []
        current: tuple[int, int] | None = goal_cell
        while current is not None:
            cells.append(current)
            current = came_from[current]
        cells.reverse()
        cells = self._line_of_sight_simplify(cells)
        poses = [Pose3D(*self.grid.cell_to_world(cell), frame=goal.frame) for cell in cells]
        poses[0] = Pose3D(start.x, start.y, start.z, start.yaw, start.frame)
        poses[-1] = goal
        return poses

    def _line_of_sight_simplify(self, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(cells) < 3:
            return cells
        result = [cells[0]]
        anchor = 0
        while anchor < len(cells) - 1:
            candidate = len(cells) - 1
            while candidate > anchor + 1 and not self._clear_segment(cells[anchor], cells[candidate]):
                candidate -= 1
            result.append(cells[candidate])
            anchor = candidate
        return result

    def _clear_segment(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        return self.grid.segment_has_clearance(
            self.grid.cell_to_world(start),
            self.grid.cell_to_world(end),
            self.path_clearance_m,
        )


class SmoothTrajectoryPlanner:
    """Collision-safe global planning followed by geometric trajectory optimization.

    A* remains the completeness layer. Its sparse polyline is resampled and an
    elastic-band style optimizer removes grid corners while every update is
    checked against the already footprint-inflated collision map.
    """

    def __init__(
        self,
        grid: GridMap,
        *,
        footprint_radius_m: float = 0.32,
        sample_spacing_m: float = 0.18,
        smoothing_iterations: int = 40,
        smoothing_gain: float = 0.35,
        path_clearance_m: float = 0.15,
    ) -> None:
        self.seed_planner = AStarPlanner(
            grid,
            footprint_radius_m=footprint_radius_m,
            path_clearance_m=path_clearance_m,
        )
        self.grid = self.seed_planner.grid
        self.sample_spacing_m = sample_spacing_m
        self.smoothing_iterations = smoothing_iterations
        self.smoothing_gain = smoothing_gain
        self.path_clearance_m = path_clearance_m
        self.last_plan_metrics: dict[str, Any] = {}

    def plan(self, start: Pose3D, goal: Pose3D) -> list[Pose3D]:
        seed = self.seed_planner.plan(start, goal)
        points = self._resample([(pose.x, pose.y) for pose in seed])
        points = self._smooth(points)
        poses: list[Pose3D] = []
        for index, (x, y) in enumerate(points):
            if index + 1 < len(points):
                tangent = math.atan2(points[index + 1][1] - y, points[index + 1][0] - x)
            elif poses:
                tangent = poses[-1].yaw
            else:
                tangent = start.yaw
            poses.append(Pose3D(x, y, start.z, tangent, goal.frame))
        poses[0] = start
        poses[-1] = Pose3D(goal.x, goal.y, goal.z, goal.yaw, goal.frame)
        length = sum(
            math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(poses, poses[1:])
        )
        self.last_plan_metrics = {
            "global_planner": "footprint_inflated_a_star",
            "trajectory_optimizer": "collision_checked_elastic_band",
            "trajectory_points": len(poses),
            "length_m": round(length, 3),
            "minimum_clearance_m": round(
                min(self.grid.clearance(pose.x, pose.y) for pose in poses), 3
            ),
        }
        return poses

    def _resample(self, polyline: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(polyline) < 2:
            return list(polyline)
        result = [polyline[0]]
        for start, end in zip(polyline, polyline[1:]):
            length = math.hypot(end[0] - start[0], end[1] - start[1])
            count = max(1, int(math.ceil(length / self.sample_spacing_m)))
            result.extend(
                (
                    start[0] + index / count * (end[0] - start[0]),
                    start[1] + index / count * (end[1] - start[1]),
                )
                for index in range(1, count + 1)
            )
        return result

    def _smooth(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        result = list(points)
        for _ in range(self.smoothing_iterations):
            changed = False
            for index in range(1, len(result) - 1):
                previous, current, following = result[index - 1 : index + 2]
                candidate = (
                    current[0] + self.smoothing_gain * (
                        0.5 * (previous[0] + following[0]) - current[0]
                    ),
                    current[1] + self.smoothing_gain * (
                        0.5 * (previous[1] + following[1]) - current[1]
                    ),
                )
                if (
                    self.grid.segment_has_clearance(
                        previous, candidate, self.path_clearance_m
                    )
                    and self.grid.segment_has_clearance(
                        candidate, following, self.path_clearance_m
                    )
                ):
                    result[index] = candidate
                    changed = changed or math.hypot(
                        candidate[0] - current[0], candidate[1] - current[1]
                    ) > 1e-5
            if not changed:
                break
        return result


class PathFollower:
    """Small holonomic path tracker that emits bounded body-frame commands."""

    def __init__(
        self,
        *,
        max_speed: float = 0.45,
        max_yaw_rate: float = 0.7,
        position_tolerance: float = 0.12,
        yaw_tolerance: float = 0.15,
        k_position: float = 1.2,
        k_yaw: float = 1.8,
        holonomic: bool = True,
        turn_in_place_threshold: float = 0.3,
        turn_in_place: bool = True,
        align_final_yaw: bool = True,
    ) -> None:
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.position_tolerance = position_tolerance
        self.yaw_tolerance = yaw_tolerance
        self.k_position = k_position
        self.k_yaw = k_yaw
        self.holonomic = holonomic
        self.turn_in_place_threshold = turn_in_place_threshold
        self.turn_in_place = turn_in_place
        self.align_final_yaw = align_final_yaw

    def command(self, pose: Pose3D, target: Pose3D, *, final: bool) -> tuple[VelocityCommand, bool]:
        dx, dy = target.x - pose.x, target.y - pose.y
        distance = math.hypot(dx, dy)
        yaw_target = (
            target.yaw
            if final and self.align_final_yaw and distance < self.position_tolerance
            else math.atan2(dy, dx)
        )
        yaw_error = _wrap(yaw_target - pose.yaw)
        if distance < self.position_tolerance:
            if not final or not self.align_final_yaw or abs(yaw_error) < self.yaw_tolerance:
                return VelocityCommand(0.0, 0.0, 0.0), True
            return VelocityCommand(
                0.0,
                0.0,
                _clip(self.k_yaw * yaw_error, self.max_yaw_rate),
                yaw_target,
            ), False

        if not self.holonomic:
            if self.turn_in_place and abs(yaw_error) > self.turn_in_place_threshold:
                return VelocityCommand(
                    0.0,
                    0.0,
                    _clip(self.k_yaw * yaw_error, self.max_yaw_rate),
                    yaw_target,
                ), False
            forward = min(self.k_position * distance, self.max_speed)
            return VelocityCommand(
                forward,
                0.0,
                _clip(self.k_yaw * yaw_error, self.max_yaw_rate),
                yaw_target,
            ), False

        world_vx = _clip(self.k_position * dx, self.max_speed)
        world_vy = _clip(self.k_position * dy, self.max_speed)
        magnitude = math.hypot(world_vx, world_vy)
        if magnitude > self.max_speed:
            world_vx *= self.max_speed / magnitude
            world_vy *= self.max_speed / magnitude
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        body_vx = c * world_vx + s * world_vy
        body_vy = -s * world_vx + c * world_vy
        yaw_rate = _clip(self.k_yaw * yaw_error, self.max_yaw_rate)
        return VelocityCommand(body_vx, body_vy, yaw_rate), False


class RegulatedTrajectoryFollower(PathFollower):
    """Look-ahead trajectory tracking with curvature and clearance regulation."""

    def __init__(
        self,
        grid: GridMap,
        *,
        lookahead_m: float = 0.35,
        prediction_horizon_s: float = 1.0,
        minimum_speed: float = 0.10,
        local_safety_margin_m: float = 0.12,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.grid = grid
        self.lookahead_m = lookahead_m
        self.prediction_horizon_s = prediction_horizon_s
        self.minimum_speed = minimum_speed
        self.local_safety_margin_m = local_safety_margin_m
        self.local_observation: LocalNavigationObservation | None = None

    def update_local_observation(self, observation: LocalNavigationObservation) -> None:
        self.local_observation = observation

    def command_path(
        self, pose: Pose3D, path: Sequence[Pose3D], waypoint: int
    ) -> tuple[VelocityCommand, bool, int]:
        final_distance = math.hypot(path[-1].x - pose.x, path[-1].y - pose.y)
        if final_distance < self.position_tolerance:
            command, reached = self.command(pose, path[-1], final=True)
            return command, reached, len(path) if reached else len(path) - 1

        nearest = min(
            range(max(0, waypoint - 1), len(path)),
            key=lambda index: math.hypot(path[index].x - pose.x, path[index].y - pose.y),
        )
        target_index = nearest
        arc = 0.0
        while target_index + 1 < len(path) and arc < self.lookahead_m:
            current, following = path[target_index], path[target_index + 1]
            arc += math.hypot(following.x - current.x, following.y - current.y)
            target_index += 1
        target = path[target_index]
        command, _ = self.command(pose, target, final=target_index == len(path) - 1)

        desired_yaw = math.atan2(target.y - pose.y, target.x - pose.x)
        heading_error = abs(_wrap(desired_yaw - pose.yaw))
        heading_scale = max(0.35, math.cos(min(heading_error, math.pi / 2)))
        clearance = self.grid.clearance(pose.x, pose.y)
        clearance_scale = min(1.0, max(0.35, clearance / max(self.lookahead_m, 1e-6)))
        requested_speed = math.hypot(command.vx, command.vy)
        speed = min(requested_speed, max(self.minimum_speed, self.max_speed * min(heading_scale, clearance_scale)))

        # A short forward rollout is a final local collision guard. The global
        # route is static; live perception can update this grid/ESDF later.
        prediction_distance = min(
            speed * self.prediction_horizon_s,
            math.hypot(target.x - pose.x, target.y - pose.y),
        )
        prediction = (
            pose.x + prediction_distance * math.cos(desired_yaw),
            pose.y + prediction_distance * math.sin(desired_yaw),
        )
        if not self.grid.segment_is_free((pose.x, pose.y), prediction):
            speed = 0.0
        if self.local_observation is not None and any(
            _point_segment_distance(
                obstacle.x,
                obstacle.y,
                pose.x,
                pose.y,
                prediction[0],
                prediction[1],
            ) <= obstacle.radius_m + self.local_safety_margin_m
            for obstacle in self.local_observation.obstacles
        ):
            speed = 0.0
        if requested_speed > 1e-6:
            scale = speed / requested_speed
            command = VelocityCommand(
                command.vx * scale,
                command.vy * scale,
                command.yaw_rate,
                command.facing_yaw,
            )
        # `waypoint` tracks measured progress, not the temporary look-ahead
        # target. Advancing it to `target_index` on every control tick skips
        # through an entire path while the robot is still turning, eventually
        # asking the local guard for a wall-crossing shortcut to the goal.
        return command, False, max(waypoint, nearest)


class PlannedNavigationBackend:
    """Blocking skill: global planning, trajectory tracking, measured completion."""

    def __init__(
        self,
        planner: AStarPlanner | SmoothTrajectoryPlanner,
        follower: PathFollower,
        base: MobileBase,
        *,
        control_hz: float = 20.0,
        timeout_s: float = 90.0,
        realtime: bool = False,
        stuck_timeout_s: float = 8.0,
        minimum_progress_m: float = 0.04,
        minimum_upright_height_m: float = 0.5,
        local_observer: LocalNavigationObserver | None = None,
    ) -> None:
        self.planner = planner
        self.follower = follower
        self.base = base
        self.dt = 1.0 / control_hz
        self.timeout_s = timeout_s
        self.realtime = realtime
        self.stuck_timeout_s = stuck_timeout_s
        self.minimum_progress_m = minimum_progress_m
        self.minimum_upright_height_m = minimum_upright_height_m
        self.local_observer = local_observer
        if local_observer is not None and not hasattr(follower, "update_local_observation"):
            raise ValueError(
                "local_observer requires a follower with update_local_observation()"
            )
        self._status: dict[str, Any] = {"state": "idle", "locomotion": base.name}

    def navigate(self, goal: Pose3D) -> Mapping[str, Any]:
        start = self.base.pose()
        path = self.planner.plan(start, goal)
        return self.navigate_path(path)

    def navigate_path(self, path: Sequence[Pose3D]) -> Mapping[str, Any]:
        """Execute an already planned collision-free route without replanning it."""
        if not path:
            raise ValueError("navigation path must contain at least one pose")
        goal = path[-1]
        self._status = {
            "state": "running",
            "locomotion": self.base.name,
            "goal": goal.as_json(),
            "waypoints": [pose.as_json() for pose in path],
        }
        metrics = getattr(self.planner, "last_plan_metrics", None)
        if metrics:
            self._status["planning"] = dict(metrics)
        started = time.monotonic()
        last_progress_at = started
        progress_pose = self.base.pose()
        commands = 0
        try:
            waypoint = 1 if len(path) > 1 else 0
            while waypoint < len(path):
                if time.monotonic() - started > self.timeout_s:
                    raise TimeoutError(f"navigation exceeded {self.timeout_s:.1f}s")
                before = time.monotonic()
                pose = self.base.pose()
                if pose.z and pose.z < self.minimum_upright_height_m:
                    raise RuntimeError(
                        f"robot is not upright (base height {pose.z:.3f} m)"
                    )
                if self.local_observer is not None:
                    local = self.local_observer.observe(pose)
                    if local.frame != pose.frame:
                        raise RuntimeError(
                            f"local observation frame {local.frame!r} does not match "
                            f"robot pose frame {pose.frame!r}"
                        )
                    if not local.healthy:
                        raise RuntimeError(
                            "local navigation perception is unhealthy"
                            + (f": {local.detail}" if local.detail else "")
                        )
                    if local.emergency_stop:
                        raise RuntimeError(
                            "local navigation perception requested an emergency stop"
                            + (f": {local.detail}" if local.detail else "")
                        )
                    self.follower.update_local_observation(local)  # type: ignore[attr-defined]
                    self._status["local_observation"] = local.as_json()
                translated = math.hypot(
                    pose.x - progress_pose.x, pose.y - progress_pose.y
                )
                rotated_equivalent = 0.3 * abs(_wrap(pose.yaw - progress_pose.yaw))
                if translated + rotated_equivalent >= self.minimum_progress_m:
                    progress_pose = pose
                    last_progress_at = before
                elif before - last_progress_at > self.stuck_timeout_s:
                    raise RuntimeError(
                        f"navigation made no measurable progress for "
                        f"{self.stuck_timeout_s:.1f}s"
                    )
                command_path = getattr(self.follower, "command_path", None)
                if command_path is not None:
                    command, reached, waypoint = command_path(pose, path, waypoint)
                else:
                    command, reached = self.follower.command(
                        pose, path[waypoint], final=waypoint == len(path) - 1
                    )
                if reached:
                    waypoint += 1
                    continue
                self.base.command_velocity(command, self.dt)
                commands += 1
                if self.realtime:
                    remaining = self.dt - (time.monotonic() - before)
                    if remaining > 0:
                        time.sleep(remaining)
            self.base.stop()
            measured = self.base.pose()
            self._status = {
                "state": "succeeded",
                "locomotion": self.base.name,
                "pose": measured.as_json(),
                "goal": goal.as_json(),
                "command_count": commands,
                "waypoint_count": len(path),
            }
            metrics = getattr(self.planner, "last_plan_metrics", None)
            if metrics:
                self._status["planning"] = dict(metrics)
            local = getattr(self.follower, "local_observation", None)
            if local is not None:
                self._status["local_observation"] = local.as_json()
            return dict(self._status)
        except Exception as exc:
            self.base.stop()
            self._status = {
                "state": "failed",
                "locomotion": self.base.name,
                "pose": self.base.pose().as_json(),
                "goal": goal.as_json(),
                "command_count": commands,
                "error": str(exc),
            }
            local = getattr(self.follower, "local_observation", None)
            if local is not None:
                self._status["local_observation"] = local.as_json()
            return dict(self._status)

    def status(self) -> Mapping[str, Any]:
        return {**self._status, "pose": self.base.pose().as_json()}


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _point_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return math.hypot(px - ax, py - ay)
    ratio = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_squared))
    return math.hypot(px - (ax + ratio * dx), py - (ay + ratio * dy))


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
