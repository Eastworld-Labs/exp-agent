"""Where to stand to see what the robot has NOT seen yet.

##### THIS MOVES NOTHING. It turns "I cannot see it" into "stand there". #####

`local_planner(target)` needs the thing in the current camera frame. When it is
not there, the robot has to go and look -- and the whole question is WHERE.
This module answers it with geometry, and it exists because the obvious answer
is wrong.

## Why the costmap alone cannot answer it

The robot arrives at a labelled place and faces a counter. The object is on the
floor behind it. On the local costmap that strip of floor reads FREE: nothing
has ever sensed it, and Nav2's local costmap does not track unknown space
(`nav2_g1_robot.yaml` sets `track_unknown_space` on the GLOBAL costmap only, so
the local one defaults false). A candidate generator reading only "is this cell
free" therefore proposes standing INSIDE the region nothing has looked at, which
is both useless -- it reveals nothing about how it got there -- and the one
place a biped should not walk.

So this module keeps its OWN memory: a :class:`VisibilityMap` of what the camera
has actually looked at, built by ray-casting the camera cone through the costmap
so a ray STOPS at the counter. The strip behind it stays unobserved, and the
cells that are observed-and-free next to it -- the frontier, which in that scene
lines the counter's two ends -- are where a robot goes to see more.

⚠️ TWO GRIDS, TWO JOBS, AND THEY MUST NOT BE CONFUSED. The costmap says what is
SOLID (can I stand there, does a ray stop here). The visibility map says what
has been LOOKED AT. A cell can be free and unobserved -- that is the counter
case and the entire reason for this file -- and it can be observed and lethal,
which is a wall the robot has seen and should not walk into.

## What bounds it

Ray-casting stops at the costmap window's edge, not at the camera's range. The
rolling window is about 8 m across, the camera is trusted to 6 m, so the window
is usually the binding limit -- and stopping there is the honest choice: beyond
it nothing describes whether anything occludes the view.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .local_planner import COST_FREE, COST_LETHAL, CostGrid, wrap_angle
from .types import Json

#: A ray stops here.
#:
#: ⚠️ THIS IS THE 0-100 OccupancyGrid SCALE, NOT costmap_2d's RAW 0-255. The
#: fleet link carries `nav_msgs/OccupancyGrid`, where 100 is lethal, 0 is free
#: and -1 is unknown -- so the 253 that means "lethal" inside Nav2's own C++ is
#: a value no cell here can ever hold. A sight threshold of 253 does not make
#: the search cautious, it makes every ray pass straight through every wall and
#: the visibility map claim the robot has seen the next room.
SIGHT_LETHAL_FROM = COST_LETHAL

#: How finely the visibility map remembers. Deliberately COARSER than the
#: costmap's 0.05 m: this grid is about "has anyone looked over there", a
#: question at the scale of a piece of furniture, and a 0.05 m version of an
#: 8 m square is 25,600 cells to ray-cast over on every candidate.
DEFAULT_RESOLUTION_M = 0.10

#: Rays per camera cone. 41 across an 87-degree field is one ray every ~2.2
#: degrees, which at 4 m is a 15 cm gap -- narrower than anything this grid can
#: represent, so nothing slips between two rays unseen.
DEFAULT_RAYS = 41


@dataclass
class VisibilityMap:
    """What the camera has looked at, in the odometry frame.

    ##### ANCHORED, NOT ROLLING. ##### The costmap scrolls with the robot; this
    does not. A search walks a few metres and must remember the corner it looked
    round two legs ago, which a window centred on the robot cannot do.

    Odometry rather than map frame because the costmap is in odometry and every
    ray is cast through it: mixing the two would put the rays a localisation
    correction away from the obstacles that should stop them. Over one search --
    a few metres, a couple of minutes -- odometry drift is far below this grid's
    own 0.10 m cell.
    """

    origin_x: float
    origin_y: float
    resolution: float
    width: int
    height: int
    seen: bytearray
    frame: str = "odom"

    @classmethod
    def around(
        cls,
        x: float,
        y: float,
        radius_m: float,
        *,
        resolution: float = DEFAULT_RESOLUTION_M,
        frame: str = "odom",
    ) -> "VisibilityMap":
        """A square map centred on (x, y), covering the search radius each way."""
        if radius_m <= 0:
            raise ValueError("search radius must be positive")
        if resolution <= 0:
            raise ValueError("visibility resolution must be positive")
        span = int(math.ceil(2 * radius_m / resolution)) + 2
        return cls(
            origin_x=x - span * resolution / 2.0,
            origin_y=y - span * resolution / 2.0,
            resolution=resolution,
            width=span,
            height=span,
            seen=bytearray(span * span),
            frame=frame,
        )

    # ------------------------------------------------------------- cells --
    def cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(math.floor((x - self.origin_x) / self.resolution)),
            int(math.floor((y - self.origin_y) / self.resolution)),
        )

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def cell_centre(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (
            self.origin_x + (cell[0] + 0.5) * self.resolution,
            self.origin_y + (cell[1] + 0.5) * self.resolution,
        )

    def is_seen(self, x: float, y: float) -> bool:
        col, row = self.cell(x, y)
        if not self.in_bounds((col, row)):
            return False
        return bool(self.seen[row * self.width + col])

    def mark(self, x: float, y: float) -> bool:
        """Record that this point has been looked at. True if it is news."""
        col, row = self.cell(x, y)
        if not self.in_bounds((col, row)):
            return False
        index = row * self.width + col
        if self.seen[index]:
            return False
        self.seen[index] = 1
        return True

    @property
    def seen_cells(self) -> int:
        return sum(self.seen)

    def area_seen_m2(self) -> float:
        return self.seen_cells * self.resolution * self.resolution

    # -------------------------------------------------------------- rays --
    def _cone_rays(self, yaw: float, hfov_rad: float, rays: int) -> list[float]:
        if rays < 1:
            raise ValueError("a cone needs at least one ray")
        if rays == 1:
            return [yaw]
        half = hfov_rad / 2.0
        step = hfov_rad / (rays - 1)
        return [yaw - half + step * index for index in range(rays)]

    def _march(
        self,
        grid: CostGrid,
        x: float,
        y: float,
        bearing: float,
        max_range_m: float,
    ) -> Iterable[tuple[float, float]]:
        """Points along one ray, stopping at the first solid cell or the window.

        ⚠️ THE FIRST SOLID CELL IS INCLUDED AND THEN THE RAY ENDS. The counter's
        own face has been looked at; the floor behind it has not. Marking the
        obstacle but not what is behind it is the whole mechanism.
        """
        step = self.resolution / 2.0
        dx, dy = math.cos(bearing), math.sin(bearing)
        distance = 0.0
        while distance <= max_range_m:
            px, py = x + dx * distance, y + dy * distance
            cost = grid.cost_at(px, py)
            if cost is None:
                # Off the costmap window. Nothing describes whether the view is
                # blocked out here, so claiming to have seen it would be a lie
                # that later suppresses a perfectly good candidate.
                return
            yield (px, py)
            if cost >= SIGHT_LETHAL_FROM:
                return
            distance += step

    def observe(
        self,
        grid: CostGrid,
        x: float,
        y: float,
        yaw: float,
        *,
        hfov_rad: float,
        max_range_m: float,
        rays: int = DEFAULT_RAYS,
    ) -> int:
        """Record one camera cone. Returns how many cells that was news for."""
        news = 0
        for bearing in self._cone_rays(yaw, hfov_rad, rays):
            for px, py in self._march(grid, x, y, bearing, max_range_m):
                if self.mark(px, py):
                    news += 1
        return news

    def predict_gain(
        self,
        grid: CostGrid,
        x: float,
        y: float,
        yaw: float,
        *,
        hfov_rad: float,
        max_range_m: float,
        rays: int = DEFAULT_RAYS,
    ) -> int:
        """Cells a cone from here WOULD newly reveal. Marks nothing.

        This is the whole score: a candidate is worth walking to exactly insofar
        as standing there shows the robot floor it has not looked at.
        """
        fresh: set[int] = set()
        for bearing in self._cone_rays(yaw, hfov_rad, rays):
            for px, py in self._march(grid, x, y, bearing, max_range_m):
                col, row = self.cell(px, py)
                if not self.in_bounds((col, row)):
                    continue
                index = row * self.width + col
                if not self.seen[index]:
                    fresh.add(index)
        return len(fresh)

    # ---------------------------------------------------------- frontier --
    def frontier(self) -> list[tuple[int, int]]:
        """Observed cells with an unobserved 4-neighbour.

        The boundary of what is known -- and the only place worth standing,
        because a candidate deeper in requires walking through the unknown and
        one further back reveals nothing new.
        """
        edge: list[tuple[int, int]] = []
        for row in range(self.height):
            base = row * self.width
            for col in range(self.width):
                if not self.seen[base + col]:
                    continue
                for dcol, drow in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ncol, nrow = col + dcol, row + drow
                    if not self.in_bounds((ncol, nrow)):
                        continue
                    if not self.seen[nrow * self.width + ncol]:
                        edge.append((col, row))
                        break
        return edge

    def unseen_bearing(self, cell: tuple[int, int], reach: int = 6) -> float | None:
        """Which way the unobserved space lies from a frontier cell.

        Averaged over every unobserved cell within `reach`, so a standpoint at
        the end of a counter faces ALONG the hidden strip rather than at the
        single nearest unknown cell, which is often straight into the counter.
        """
        col, row = cell
        sum_x = sum_y = 0.0
        count = 0
        for drow in range(-reach, reach + 1):
            for dcol in range(-reach, reach + 1):
                if dcol == 0 and drow == 0:
                    continue
                ncol, nrow = col + dcol, row + drow
                if not self.in_bounds((ncol, nrow)):
                    continue
                if self.seen[nrow * self.width + ncol]:
                    continue
                norm = math.hypot(dcol, drow)
                sum_x += dcol / norm
                sum_y += drow / norm
                count += 1
        if not count or (abs(sum_x) < 1e-9 and abs(sum_y) < 1e-9):
            return None
        return math.atan2(sum_y, sum_x)

    def as_json(self) -> Json:
        return {
            "frame": self.frame,
            "resolution_m": self.resolution,
            "size_m": round(self.width * self.resolution, 2),
            "observed_cells": self.seen_cells,
            "observed_m2": round(self.area_seen_m2(), 1),
            "frontier_cells": len(self.frontier()),
        }


@dataclass(frozen=True)
class Standpoint:
    """One place the robot could stand, and what it would buy.

    `bearing_deg` and `image_fraction` describe the candidate FROM THE ROBOT'S
    CURRENT VIEW, which is how the picker is asked about it: a model looking at
    the camera frame can be told "candidate 2 is 30 degrees left, about a third
    of the way across the picture" and reason about what is there.
    """

    x: float
    y: float
    yaw: float
    gain_cells: int
    distance_m: float
    bearing_rad: float
    image_fraction: float | None

    @property
    def gain_m2_at(self) -> float:
        return self.gain_cells

    def as_json(self, resolution: float = DEFAULT_RESOLUTION_M) -> Json:
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "yaw_deg": round(math.degrees(self.yaw), 1),
            "reveals_m2": round(self.gain_cells * resolution * resolution, 1),
            "walk_m": round(self.distance_m, 2),
            "bearing_deg": round(math.degrees(self.bearing_rad), 1),
            "image_fraction": None if self.image_fraction is None else round(self.image_fraction, 3),
        }


def image_fraction_for(bearing_rad: float, robot_yaw: float, hfov_rad: float) -> float | None:
    """Where a bearing falls across the current frame, 0 at the left edge.

    None when it is outside the frame -- which is the common case for a
    candidate behind the robot, and must stay distinguishable from "at the edge"
    so a picker is never told to look at something that is not in the picture.
    """
    offset = wrap_angle(bearing_rad - robot_yaw)
    half = hfov_rad / 2.0
    if abs(offset) > half:
        return None
    # +bearing is LEFT (see local_planner.optical_to_base), and a picture's
    # column 0 is the left edge, so the fraction runs the other way.
    return float(0.5 - offset / hfov_rad)


def standpoints(
    visibility: VisibilityMap,
    grid: CostGrid,
    free: "object",
    robot: tuple[float, float, float],
    *,
    anchor: tuple[float, float],
    radius_m: float,
    hfov_rad: float,
    max_range_m: float,
    visited: Sequence[tuple[float, float]] = (),
    min_separation_m: float = 0.8,
    min_gain_cells: int = 12,
    max_candidates: int = 6,
    rays: int = DEFAULT_RAYS,
) -> list[Standpoint]:
    """Places worth walking to, best first. Empty means the area is exhausted.

    `free` is the INFLATED planning grid (`CostGrid.to_gridmap().inflated(r)`),
    the same object `local_planner` verifies its approach against, so a
    standpoint this returns is one the robot's body fits at and can reach in a
    straight line. Nav2 still does the real planning; this only avoids proposing
    goals that are obviously impossible.

    ⚠️ EVERY FILTER HERE IS A REFUSAL, NOT A PREFERENCE. A candidate outside the
    radius leaves the area the operator approved. One too near a visited
    standpoint re-photographs a place the robot has already been. One under
    `min_gain_cells` reveals nothing, and walking to it is how a search burns
    its budget looking busy.
    """
    robot_x, robot_y, robot_yaw = robot
    clusters: list[Standpoint] = []
    chosen: list[tuple[float, float]] = list(visited)

    scored: list[tuple[int, float, tuple[float, float], float]] = []
    for cell in visibility.frontier():
        cx, cy = visibility.cell_centre(cell)
        if math.hypot(cx - anchor[0], cy - anchor[1]) > radius_m:
            continue
        cost = grid.cost_at(cx, cy)
        if cost is None or cost >= SIGHT_LETHAL_FROM or cost < COST_FREE:
            continue
        if not free.segment_is_free((cx, cy), (cx, cy)):
            continue
        bearing = visibility.unseen_bearing(cell)
        if bearing is None:
            continue
        gain = visibility.predict_gain(
            grid, cx, cy, bearing, hfov_rad=hfov_rad, max_range_m=max_range_m, rays=rays
        )
        if gain < min_gain_cells:
            continue
        walk = math.hypot(cx - robot_x, cy - robot_y)
        scored.append((gain, walk, (cx, cy), bearing))

    # Most revealing first; a shorter walk breaks a tie. Sorting before the
    # separation filter is what makes that filter keep the BEST member of each
    # cluster rather than whichever one the row-major scan reached first.
    scored.sort(key=lambda row: (-row[0], row[1]))

    for gain, walk, (cx, cy), bearing in scored:
        if any(math.hypot(cx - px, cy - py) < min_separation_m for px, py in chosen):
            continue
        if not free.segment_is_free((robot_x, robot_y), (cx, cy)):
            continue
        chosen.append((cx, cy))
        to_candidate = math.atan2(cy - robot_y, cx - robot_x)
        clusters.append(
            Standpoint(
                x=cx,
                y=cy,
                yaw=bearing,
                gain_cells=gain,
                distance_m=walk,
                bearing_rad=to_candidate,
                image_fraction=image_fraction_for(to_candidate, robot_yaw, hfov_rad),
            )
        )
        if len(clusters) >= max_candidates:
            break
    return clusters


def sweep_yaws(yaw: float, quarters: int = 4) -> list[float]:
    """Headings for a look-around, starting one step from where the robot faces.

    The current heading is NOT repeated: the caller has just grounded on that
    frame, and turning 360 degrees to photograph it again is one wasted goal at
    the start of every search.
    """
    if quarters < 1:
        raise ValueError("a sweep needs at least one quarter")
    step = 2 * math.pi / quarters
    return [wrap_angle(yaw + step * index) for index in range(1, quarters + 1)]


__all__ = [
    "DEFAULT_RAYS",
    "DEFAULT_RESOLUTION_M",
    "SIGHT_LETHAL_FROM",
    "Standpoint",
    "VisibilityMap",
    "image_fraction_for",
    "standpoints",
    "sweep_yaws",
]
