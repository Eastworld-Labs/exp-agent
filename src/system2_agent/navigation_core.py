from __future__ import annotations

import heapq
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .modules.semantic_map import Pose3D


@dataclass(frozen=True)
class VelocityCommand:
    vx: float
    vy: float
    yaw_rate: float


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


class AStarPlanner:
    def __init__(self, grid: GridMap, *, footprint_radius_m: float = 0.32) -> None:
        self.grid = grid.inflated(footprint_radius_m)

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
                new_cost = cost[current] + step_cost
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
        distance = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
        for i in range(distance + 1):
            t = i / max(distance, 1)
            cell = (round(start[0] + t * (end[0] - start[0])), round(start[1] + t * (end[1] - start[1])))
            if cell in self.grid.occupied:
                return False
        return True


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
    ) -> None:
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.position_tolerance = position_tolerance
        self.yaw_tolerance = yaw_tolerance
        self.k_position = k_position
        self.k_yaw = k_yaw

    def command(self, pose: Pose3D, target: Pose3D, *, final: bool) -> tuple[VelocityCommand, bool]:
        dx, dy = target.x - pose.x, target.y - pose.y
        distance = math.hypot(dx, dy)
        yaw_target = target.yaw if final and distance < self.position_tolerance else math.atan2(dy, dx)
        yaw_error = _wrap(yaw_target - pose.yaw)
        if distance < self.position_tolerance:
            if not final or abs(yaw_error) < self.yaw_tolerance:
                return VelocityCommand(0.0, 0.0, 0.0), True
            return VelocityCommand(0.0, 0.0, _clip(self.k_yaw * yaw_error, self.max_yaw_rate)), False

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


class PlannedNavigationBackend:
    """Blocking skill: A* global plan, local tracking, measured completion."""

    def __init__(
        self,
        planner: AStarPlanner,
        follower: PathFollower,
        base: MobileBase,
        *,
        control_hz: float = 20.0,
        timeout_s: float = 90.0,
        realtime: bool = False,
    ) -> None:
        self.planner = planner
        self.follower = follower
        self.base = base
        self.dt = 1.0 / control_hz
        self.timeout_s = timeout_s
        self.realtime = realtime
        self._status: dict[str, Any] = {"state": "idle", "locomotion": base.name}

    def navigate(self, goal: Pose3D) -> Mapping[str, Any]:
        start = self.base.pose()
        path = self.planner.plan(start, goal)
        self._status = {
            "state": "running",
            "locomotion": self.base.name,
            "goal": goal.as_json(),
            "waypoints": [pose.as_json() for pose in path],
        }
        started = time.monotonic()
        commands = 0
        try:
            waypoint = 1 if len(path) > 1 else 0
            while waypoint < len(path):
                if time.monotonic() - started > self.timeout_s:
                    raise TimeoutError(f"navigation exceeded {self.timeout_s:.1f}s")
                before = time.monotonic()
                pose = self.base.pose()
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
            return dict(self._status)
        except Exception as exc:
            self.base.stop()
            self._status = {
                "state": "failed",
                "locomotion": self.base.name,
                "pose": self.base.pose().as_json(),
                "goal": goal.as_json(),
                "error": str(exc),
            }
            return dict(self._status)

    def status(self) -> Mapping[str, Any]:
        return {**self._status, "pose": self.base.pose().as_json()}


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
