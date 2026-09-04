"""The robot's Nav2 local costmap and its pose, off the fleet link.

##### READ-ONLY. Nothing here publishes anything. #####

`/local_costmap/costmap` is a `nav_msgs/OccupancyGrid` the robot already
forwards (`g1_fleet/config/topics.json`), so the host can see the same rolling
window Nav2's controller sees. Two properties of that topic drive everything in
this file:

  RETAINED.  The broker replays the last grid to a late subscriber, instantly
             and looking perfectly fresh. A costmap from before the robot walked
             into the next room describes a room it is not in.
  PATCHED.   On the real robot `always_send_full_costmap` is false: the whole
             grid is re-sent only when the rolling window's origin moves, and in
             between the changes arrive as `map_msgs/OccupancyGridUpdate`
             rectangles at ~2 Hz. Standing still -- which is exactly when this
             tool runs -- the full grid ages and the PATCHES carry the news.

So this subscribes with `Link.watch` and applies patches as they land, rather
than reading `latest()` twice and hoping. A link without `watch` still works and
applies the newest patch it can see; the difference is recorded in `status()`
rather than hidden.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from ..local_planner import CostGrid, GridUpdate, InitPose
from ..modules.semantic_map import Pose3D
from ..types import Json
from .link import Link
from .wire import decode_grid_update, decode_occupancy_grid, decode_pose

COSTMAP_TOPIC = "/local_costmap/costmap"
COSTMAP_UPDATES_TOPIC = "/local_costmap/costmap_updates"

#: Nav2's local costmap is published in the ODOM frame with `rolling_window:
#: true` (nav2_g1_robot.yaml, nav2_sonic_sim.yaml). Not `map`: the local
#: controller must keep working across a localization jump, so its window is
#: pinned to odometry. Everything that places the robot on this grid therefore
#: has to use an odom-frame pose, which is why MqttInitPose reads two topics.
COSTMAP_FRAME = "odom"


class MqttLocalCostmap:
    """A `LocalGridSource` over the fleet link. Refuses rather than guesses."""

    def __init__(
        self,
        link: Link,
        *,
        topic: str = COSTMAP_TOPIC,
        updates_topic: str = COSTMAP_UPDATES_TOPIC,
        expected_frame: str = COSTMAP_FRAME,
        stale_s: float = 5.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.link = link
        self.topic = topic
        self.updates_topic = updates_topic
        self.expected_frame = expected_frame
        # Ten publish periods at the real robot's 2 Hz patch rate. Deliberately
        # looser than the 3 s pose gate: Nav2 emits a patch only when something
        # in the window CHANGED, so a robot standing in a still room can go
        # several seconds with nothing to say and still be perfectly healthy.
        self.stale_s = stale_s
        self._now = now
        self._lock = threading.Lock()
        self._grid: CostGrid | None = None
        self._grid_at = 0.0
        self._last_at = 0.0
        self._full_grids = 0
        self._patches = 0
        self._patch_mismatches = 0
        self._live = False
        self._reason = "nothing has arrived on the local costmap topic yet"
        self._watching = False

        watch = getattr(link, "watch", None)
        if callable(watch):
            watch(topic, self._on_grid)
            watch(updates_topic, self._on_patch)
            self._watching = True
        else:
            link.subscribe(topic)
            link.subscribe(updates_topic)

    # ---------------------------------------------------------- watch path --
    def _on_grid(self, msg: Json, arrived: float, retained: bool = False) -> None:
        decoded = decode_occupancy_grid(msg)
        if decoded is None:
            with self._lock:
                self._reason = "the last local costmap message was malformed"
            return
        grid = CostGrid(
            frame=decoded["frame"],
            width=decoded["width"],
            height=decoded["height"],
            resolution=decoded["resolution"],
            origin_x=decoded["origin_x"],
            origin_y=decoded["origin_y"],
            cost=decoded["cost"],
        )
        with self._lock:
            self._grid = grid
            self._grid_at = arrived
            self._last_at = arrived
            self._patches = 0
            self._reason = ""
            # ⚠️ A RETAINED GRID IS NOT EVIDENCE THE ROBOT IS ALIVE. It is the
            # broker handing over what it kept. Liveness needs something that
            # was published while we were listening: a second full grid, or a
            # patch. Until then the age gate below would happily pass a snapshot
            # from last week, because it arrived just now.
            if not retained:
                self._full_grids += 1
                self._live = True

    def _on_patch(self, msg: Json, arrived: float, retained: bool = False) -> None:
        decoded = decode_grid_update(msg)
        if decoded is None:
            return
        with self._lock:
            have_grid = self._grid is not None
        if not have_grid:
            # ⚠️ THE PATCHES ARRIVE FIRST WHEN THE RETAINED GRID WAS MISSED.
            # Dropping them until somebody calls grid() would throw away every
            # change between connecting and the first tool call -- and on the
            # real robot, standing still, those patches are the only news there
            # is. Seed outside the lock: _pull takes it.
            self._pull(seed=True)
        patch = GridUpdate(
            x=decoded["x"],
            y=decoded["y"],
            width=decoded["width"],
            height=decoded["height"],
            cost=tuple(decoded["cost"]),
        )
        with self._lock:
            if self._grid is None:
                return
            patched = self._grid.with_patch(patch)
            if patched is None:
                # The patch describes a window this grid is not. Wait for the
                # next full grid rather than painting part of somewhere else.
                self._patch_mismatches += 1
                return
            self._grid = patched
            self._patches += 1
            self._last_at = arrived
            self._live = True

    # ------------------------------------------------------- LocalGridSource --
    def grid(self) -> CostGrid:
        with self._lock:
            grid = self._grid
            grid_at = self._grid_at
            last_at = self._last_at
            patches = self._patches
            live = self._live
            watching = self._watching
        if grid is None or not watching:
            # Two reasons to read the topic directly.
            #
            # ⚠️ A WATCHER CAN MISS THE RETAINED GRID ENTIRELY. `MqttLink.subscribe`
            # short-circuits a topic it has already subscribed, so if anything
            # else on this link got there first the broker has already delivered
            # the retained message and will not deliver it again for us -- the
            # watcher fires for nothing, and with `always_send_full_costmap:
            # false` on the real robot the next full grid may not come until the
            # robot has walked far enough to move the window. Seeding from
            # `latest()` closes that, and it is seeded as RETAINED so it buys
            # geometry without buying liveness.
            #
            # And a Link with no `watch` at all has nobody reading in the
            # background, so it must pull every time.
            self._pull(seed=grid is None and watching)
            with self._lock:
                grid = self._grid
                grid_at = self._grid_at
                last_at = self._last_at
                patches = self._patches
                live = self._live
        if grid is None:
            raise ValueError(
                f"no local costmap has arrived on {self.topic}. Nav2 may not be "
                "running on the robot, or the fleet agent is not forwarding it. "
                "Without it there is no way to check a local walk is clear."
            )
        if not watching:
            grid, patches, last_at = self._merge_latest_patch(grid, grid_at)
        age = self._now() - last_at
        if age > self.stale_s:
            raise ValueError(
                f"the local costmap is not live: the last message arrived {age:.0f} s "
                f"ago and nothing may be planned on a picture that old. Check Nav2 and "
                "the fleet link on the robot."
            )
        if not live:
            raise ValueError(
                "only a retained local costmap has arrived -- the broker replayed the "
                "last one the robot ever published, and nothing has been published "
                "since we connected. It may describe somewhere the robot no longer is."
            )
        if grid.frame != self.expected_frame:
            raise ValueError(
                f"the local costmap is in frame {grid.frame!r}, not "
                f"{self.expected_frame!r}. Placing the robot on it would put every "
                "obstacle somewhere plausible and wrong."
            )
        return CostGrid(
            frame=grid.frame,
            width=grid.width,
            height=grid.height,
            resolution=grid.resolution,
            origin_x=grid.origin_x,
            origin_y=grid.origin_y,
            cost=grid.cost,
            age_s=age,
            full_grid_age_s=self._now() - grid_at,
            patches_applied=patches,
            live=live,
        )

    def status(self) -> Json:
        """Never raises: a snapshot must render even when the link is dead."""
        try:
            with self._lock:
                grid = self._grid
                state: Json = {
                    "available": grid is not None,
                    "topic": self.topic,
                    "live": self._live,
                    "patches_applied": self._patches,
                    "patch_mismatches": self._patch_mismatches,
                    "full_grids": self._full_grids,
                    "watching": self._watching,
                }
                if self._reason:
                    state["reason"] = self._reason
                if grid is not None:
                    state.update(
                        {
                            "frame": grid.frame,
                            "resolution": grid.resolution,
                            "size_m": [
                                round(grid.size_m[0], 2),
                                round(grid.size_m[1], 2),
                            ],
                            "age_s": round(self._now() - self._last_at, 1),
                            "full_grid_age_s": round(self._now() - self._grid_at, 1),
                        }
                    )
                return state
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------ fallbacks --
    def _pull(self, *, seed: bool = False) -> None:
        """Read the topic directly. `seed` means "assume it was retained"."""
        entry = self.link.latest(self.topic)
        if entry is not None:
            self._on_grid(entry[0], entry[1], retained=seed)

    def _merge_latest_patch(
        self, grid: CostGrid, grid_at: float
    ) -> tuple[CostGrid, int, float]:
        """Without `watch`, at best the newest patch. Better than none, and
        recorded as `watching: false` so the gap is visible rather than assumed
        away."""
        entry = self.link.latest(self.updates_topic)
        if entry is None or entry[1] < grid_at:
            return (grid, 0, self._grid_at)
        decoded = decode_grid_update(entry[0])
        if decoded is None:
            return (grid, 0, self._grid_at)
        patched = grid.with_patch(
            GridUpdate(
                x=decoded["x"],
                y=decoded["y"],
                width=decoded["width"],
                height=decoded["height"],
                cost=tuple(decoded["cost"]),
            )
        )
        if patched is None:
            return (grid, 0, self._grid_at)
        return (patched, 1, entry[1])


class MqttInitPose:
    """Where the robot is, in both the costmap's frame and the goal's frame.

    ⚠️ TWO TOPICS, ONE INSTANT, AND THAT IS THE POINT. The costmap is in `odom`
    and a `/goal_pose` must be in `map`, so the transform between them is
    recovered from a pose read in each -- and it is only correct if the two were
    sampled together. Both go through the same freshness gate for that reason.

    On the simulator they are the same topic: `/odom` carries ground truth in a
    frame that IS the map frame, so the transform is the identity and the
    arithmetic is a no-op rather than a special case.
    """

    def __init__(
        self,
        link: Link,
        *,
        map_topic: str = "/localization_3d",
        odom_topic: str = "/odom",
        stale_s: float = 3.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.link = link
        self.map_topic = map_topic
        self.odom_topic = odom_topic
        self.stale_s = stale_s
        self._now = now
        for topic in {map_topic, odom_topic}:
            link.subscribe(topic)

    def init_pose(self, expected_odom_frame: str = COSTMAP_FRAME) -> InitPose:
        odom, odom_age = self._read(self.odom_topic, "odometry")
        if self.map_topic == self.odom_topic:
            # The simulator. odom IS map here, by construction, so do not invent
            # a second read that could differ by a scheduling jitter.
            map_pose, map_age = odom, odom_age
        else:
            map_pose, map_age = self._read(self.map_topic, "map-frame position")
        if odom["frame"] != expected_odom_frame:
            raise ValueError(
                f"the robot's odometry is in frame {odom['frame']!r}, not "
                f"{expected_odom_frame!r}, so it cannot be placed on the local costmap"
            )
        return InitPose(
            odom=Pose3D(odom["x"], odom["y"], yaw=odom["yaw"], frame=odom["frame"]),
            map=Pose3D(
                map_pose["x"],
                map_pose["y"],
                yaw=map_pose["yaw"],
                frame=map_pose["frame"] if self.map_topic != self.odom_topic else "map",
            ),
            odom_age_s=odom_age,
            map_age_s=map_age,
        )

    def status(self) -> Json:
        state: Json = {"map_topic": self.map_topic, "odom_topic": self.odom_topic}
        for label, topic in (("map", self.map_topic), ("odom", self.odom_topic)):
            entry = self.link.latest(topic)
            state[label] = (
                None if entry is None else {"age_s": round(self._now() - entry[1], 1)}
            )
        return state

    def _read(self, topic: str, what: str) -> tuple[Json, float]:
        entry = self.link.latest(topic)
        if entry is None:
            raise ValueError(
                f"the robot is not reporting its {what} on {topic}; nothing can be "
                "planned relative to a position nobody is publishing"
            )
        msg, arrived = entry
        age = self._now() - arrived
        if age > self.stale_s:
            raise ValueError(
                f"the robot's {what} on {topic} is {age:.0f} s old. It is retained by "
                "the broker, so an old value looks live; a local plan built on one "
                "would aim at where the robot used to be."
            )
        pose = decode_pose(msg)
        if pose is None:
            raise ValueError(f"the {what} message on {topic} could not be read")
        return (pose, age)
