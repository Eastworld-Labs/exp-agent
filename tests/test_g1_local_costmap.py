"""The costmap, depth and pose adapters, with no broker and no clock.

##### THE THEME IS THAT A RETAINED MESSAGE LOOKS EXACTLY LIKE A LIVE ONE. #####
`/local_costmap/costmap` and the depth intrinsics are both retained by the
broker, so a subscriber that has just connected is handed values that arrive at
this instant and may describe a robot that has been switched off for a week.
Everything here is about telling those two apart before a walk is planned on one.
"""
import json
import math
import unittest

from system2_agent.g1.depth import MqttHeadDepth
from system2_agent.g1.local_costmap import MqttInitPose, MqttLocalCostmap
from system2_agent.png16 import encode_png16


class FakeLink:
    """tests/test_g1_nav2_backend.py's FakeLink, plus the optional `watch` hook.

    `watch` is what lets a costmap consumer see EVERY patch instead of only the
    last one before a tool call, so a fake that lacks it is also a fixture: the
    degraded path has to keep working.
    """

    def __init__(self, connected=True, watchable=True):
        self._connected = connected
        self.messages = {}
        self.published = []
        self.time = 100.0
        self.subscribed = set()
        self._watchers = {}
        if not watchable:
            # A Link that implements only the four required methods. The costmap
            # reader probes for `watch` and must degrade, not crash.
            self.watch = None

    def connected(self):
        return self._connected

    def subscribe(self, topic):
        self.subscribed.add(topic)

    def latest(self, topic):
        return self.messages.get(topic)

    def publish_cmd(self, topic, msg, expiry_s=None):
        self.published.append((topic, msg, expiry_s))

    def watch(self, topic, callback):
        self._watchers.setdefault(topic, []).append(callback)
        self.subscribe(topic)

    # -- test helpers -------------------------------------------------------
    def put(self, topic, msg, at=None, retained=False):
        arrived = self.time if at is None else at
        self.messages[topic] = (msg, arrived)
        for watcher in getattr(self, "_watchers", {}).get(topic, ()):
            watcher(msg, arrived, retained)

    def put_grid(self, cost, *, width, height, frame="odom", resolution=0.05,
                 origin=(-4.0, -4.0), retained=False):
        self.put(
            "/local_costmap/costmap",
            {
                "header": {"frame_id": frame},
                "info": {
                    "width": width,
                    "height": height,
                    "resolution": resolution,
                    "origin": {
                        "position": {"x": origin[0], "y": origin[1], "z": 0.0},
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    },
                },
                "data": bytes((value + 256) % 256 for value in cost),
            },
            retained=retained,
        )

    def put_patch(self, x, y, width, height, cost):
        self.put(
            "/local_costmap/costmap_updates",
            {
                "header": {"frame_id": "odom"},
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "data": bytes((value + 256) % 256 for value in cost),
            },
        )

    def put_odom(self, x, y, yaw=0.0, frame="odom"):
        self.put(
            "/odom",
            {
                "header": {"frame_id": frame},
                "pose": {
                    "pose": {
                        "position": {"x": x, "y": y, "z": 0.0},
                        "orientation": {
                            "x": 0.0, "y": 0.0,
                            "z": math.sin(yaw / 2), "w": math.cos(yaw / 2),
                        },
                    }
                },
            },
        )

    def put_map_pose(self, x, y, yaw=0.0):
        self.put(
            "/localization_3d",
            {
                "header": {"frame_id": "map"},
                "pose": {
                    "position": {"x": x, "y": y, "z": 0.0},
                    "orientation": {
                        "x": 0.0, "y": 0.0,
                        "z": math.sin(yaw / 2), "w": math.cos(yaw / 2),
                    },
                },
            },
        )

    def put_depth(self, values, width, height, *, fmt="16UC1; png"):
        self.put(
            "/g1/head/depth/compressed",
            {
                "header": {"frame_id": "d455_color_optical_frame"},
                "format": fmt,
                "data": encode_png16(values, width, height),
            },
        )

    def put_depth_info(self, width, height, *, fx=168.6, cx=None, cy=None):
        self.put(
            "/g1/head/depth_info",
            {
                "data": json.dumps(
                    {
                        "width": width,
                        "height": height,
                        "fx": fx,
                        "fy": fx,
                        "cx": width / 2 if cx is None else cx,
                        "cy": height / 2 if cy is None else cy,
                        "depth_scale": 0.001,
                        "frame_id": "d455_color_optical_frame",
                        "camera": "d455",
                        "source": "g1_vision",
                    }
                )
            },
            retained=True,
        )


def costmap(link, **kwargs):
    kwargs.setdefault("now", lambda: link.time)
    return MqttLocalCostmap(link, **kwargs)


class CostmapTests(unittest.TestCase):
    def test_nothing_on_the_topic_is_a_refusal_that_names_it(self):
        with self.assertRaises(ValueError) as caught:
            costmap(FakeLink()).grid()

        self.assertIn("/local_costmap/costmap", str(caught.exception))

    def test_a_live_grid_is_read_with_its_frame_and_origin(self):
        link = FakeLink()
        reader = costmap(link)
        link.put_grid([0] * 16, width=4, height=4, origin=(-0.1, -0.2), resolution=0.1)

        grid = reader.grid()

        self.assertEqual(grid.frame, "odom")
        self.assertEqual((grid.width, grid.height), (4, 4))
        self.assertAlmostEqual(grid.origin_x, -0.1)
        self.assertAlmostEqual(grid.resolution, 0.1)

    def test_only_a_retained_grid_is_refused_however_fresh_it_looks(self):
        """##### THE TRAP THIS WHOLE FILE EXISTS FOR. ##### The broker replays
        the last grid the robot ever published the moment we subscribe. It
        arrives now, so every age check passes, and it may describe a room the
        robot left days ago."""
        link = FakeLink()
        reader = costmap(link)
        link.put_grid([0] * 16, width=4, height=4, retained=True)

        with self.assertRaises(ValueError) as caught:
            reader.grid()

        self.assertIn("retained", str(caught.exception))
        self.assertFalse(reader.status()["live"])

    def test_a_patch_after_a_retained_grid_proves_the_robot_is_talking(self):
        link = FakeLink()
        reader = costmap(link)
        link.put_grid([0] * 16, width=4, height=4, retained=True)
        link.put_patch(0, 0, 1, 1, [100])

        grid = reader.grid()

        self.assertTrue(grid.live)
        self.assertEqual(grid.patches_applied, 1)

    def test_patches_are_applied_in_order_as_they_land(self):
        """⚠️ WHY `watch` EXISTS. `latest()` keeps one message per topic, and on
        the real robot the full grid is re-sent only when the rolling window
        MOVES -- so a robot standing still (which is when this tool runs) has an
        ageing grid and a stream of patches carrying all the news."""
        link = FakeLink()
        reader = costmap(link)
        link.put_grid([0] * 16, width=4, height=4)
        link.put_patch(0, 0, 1, 1, [100])
        link.put_patch(3, 3, 1, 1, [99])

        grid = reader.grid()

        self.assertEqual(grid.patches_applied, 2)
        self.assertEqual(grid.cost[0], 100)
        self.assertEqual(grid.cost[15], 99)

    def test_a_patch_that_does_not_fit_is_dropped_and_counted(self):
        link = FakeLink()
        reader = costmap(link)
        link.put_grid([0] * 16, width=4, height=4)
        link.put_patch(3, 3, 2, 2, [100, 100, 100, 100])

        grid = reader.grid()

        self.assertEqual(grid.patches_applied, 0)
        self.assertEqual(reader.status()["patch_mismatches"], 1)

    def test_a_new_full_grid_resets_the_patch_count(self):
        link = FakeLink()
        reader = costmap(link)
        link.put_grid([0] * 16, width=4, height=4)
        link.put_patch(0, 0, 1, 1, [100])
        link.put_grid([0] * 16, width=4, height=4)

        grid = reader.grid()

        self.assertEqual(grid.patches_applied, 0)
        self.assertEqual(grid.cost[0], 0)

    def test_an_old_costmap_is_refused_however_well_formed(self):
        link = FakeLink()
        reader = costmap(link)
        link.put_grid([0] * 16, width=4, height=4)
        link.time += 60.0

        with self.assertRaises(ValueError) as caught:
            reader.grid()

        self.assertIn("not live", str(caught.exception))

    def test_a_costmap_in_the_wrong_frame_is_refused(self):
        link = FakeLink()
        reader = costmap(link)
        link.put_grid([0] * 16, width=4, height=4, frame="map")

        with self.assertRaises(ValueError) as caught:
            reader.grid()

        self.assertIn("'map'", str(caught.exception))

    def test_negative_costs_survive_the_unsigned_wire(self):
        """int8[] crosses CBOR as a plain byte string with the sign thrown away,
        so unknown (-1) arrives as 255. Reading it as 255 would make unknown
        space look MORE lethal than lethal."""
        link = FakeLink()
        reader = costmap(link)
        link.put_grid([0, 100, -1, 99], width=2, height=2)

        self.assertEqual(list(reader.grid().cost), [0, 100, -1, 99])

    def test_a_grid_that_arrived_before_the_watcher_is_still_found(self):
        """⚠️ `MqttLink.subscribe` SHORT-CIRCUITS A TOPIC IT ALREADY HAS, so if
        anything else on the link subscribed first, the broker delivered the
        retained grid before this watcher existed and will never deliver it
        again. Without a seed the reader would sit on "nothing has arrived"
        until the robot walked far enough to move the rolling window."""
        link = FakeLink()
        link.put_grid([0] * 16, width=4, height=4)   # before the reader exists
        reader = costmap(link)

        # Found -- but seeded as retained, so it does not count as liveness.
        with self.assertRaises(ValueError) as caught:
            reader.grid()
        self.assertIn("retained", str(caught.exception))

        link.put_patch(0, 0, 1, 1, [100])
        grid = reader.grid()

        self.assertTrue(grid.live)
        self.assertEqual(grid.cost[0], 100)

    def test_patches_arriving_before_the_first_read_are_not_thrown_away(self):
        """The same miss, but nobody calls grid() in between. Standing still on
        the real robot the patches ARE the news -- discarding them until the
        first tool call would mean planning on a picture minutes out of date."""
        link = FakeLink()
        link.put_grid([0] * 16, width=4, height=4)   # before the reader exists
        reader = costmap(link)
        link.put_patch(0, 0, 1, 1, [100])            # before anybody reads

        grid = reader.grid()

        self.assertTrue(grid.live)
        self.assertEqual(grid.patches_applied, 1)
        self.assertEqual(grid.cost[0], 100)

    def test_a_link_without_watch_still_works_and_says_so(self):
        link = FakeLink(watchable=False)
        reader = costmap(link)
        link.put_grid([0] * 16, width=4, height=4)
        link.put_patch(0, 0, 1, 1, [100])

        grid = reader.grid()

        self.assertFalse(reader.status()["watching"])
        self.assertEqual(grid.cost[0], 100)
        self.assertEqual(grid.patches_applied, 1)

    def test_status_never_raises_even_with_a_dead_link(self):
        class Dead:
            def subscribe(self, topic):
                pass

            def latest(self, topic):
                raise OSError("no broker")

        state = MqttLocalCostmap(Dead()).status()

        self.assertFalse(state["available"])


class InitPoseTests(unittest.TestCase):
    def poses(self, link, **kwargs):
        kwargs.setdefault("now", lambda: link.time)
        return MqttInitPose(link, **kwargs)

    def test_both_frames_are_read_and_kept_apart(self):
        link = FakeLink()
        link.put_odom(3.0, -2.0, math.radians(90))
        link.put_map_pose(10.0, 5.0, math.radians(-45))

        init = self.poses(link).init_pose("odom")

        self.assertAlmostEqual(init.odom.x, 3.0)
        self.assertAlmostEqual(init.odom.yaw, math.radians(90), places=6)
        self.assertAlmostEqual(init.map.x, 10.0)
        self.assertAlmostEqual(init.map.yaw, math.radians(-45), places=6)
        self.assertEqual(init.map.frame, "map")

    def test_the_simulator_reads_one_topic_for_both(self):
        """/odom there is ground truth in a frame that IS map, so a second read
        could only differ by scheduling jitter -- and that jitter would show up
        as a phantom odom-to-map drift."""
        link = FakeLink()
        link.put_odom(4.0, 1.0, 1.1)

        init = self.poses(link, map_topic="/odom").init_pose("odom")

        self.assertEqual((init.map.x, init.map.y), (init.odom.x, init.odom.y))
        self.assertEqual(init.map.frame, "map")

    def test_a_stale_pose_is_refused_because_it_is_retained_too(self):
        link = FakeLink()
        link.put_odom(0.0, 0.0)
        link.put_map_pose(0.0, 0.0)
        link.time += 30.0

        with self.assertRaises(ValueError) as caught:
            self.poses(link).init_pose("odom")

        self.assertIn("old", str(caught.exception))

    def test_a_missing_pose_names_the_topic_nobody_is_publishing(self):
        link = FakeLink()
        link.put_odom(0.0, 0.0)

        with self.assertRaises(ValueError) as caught:
            self.poses(link).init_pose("odom")

        self.assertIn("/localization_3d", str(caught.exception))

    def test_odometry_in_an_unexpected_frame_is_refused(self):
        link = FakeLink()
        link.put_odom(0.0, 0.0, frame="base_link")
        link.put_map_pose(0.0, 0.0)

        with self.assertRaises(ValueError) as caught:
            self.poses(link).init_pose("odom")

        self.assertIn("base_link", str(caught.exception))


class DepthTests(unittest.TestCase):
    def depth(self, link, **kwargs):
        kwargs.setdefault("now", lambda: link.time)
        return MqttHeadDepth(link, **kwargs)

    def test_a_frame_and_its_intrinsics_become_a_measurable_image(self):
        link = FakeLink()
        reader = self.depth(link)
        link.put_depth_info(4, 2)
        link.put_depth([1000, 1200, 0, 0, 1500, 1500, 0, 0], 4, 2)

        image = reader.capture()

        self.assertEqual((image.width, image.height), (4, 2))
        self.assertAlmostEqual(image.at(1, 0), 1.2, places=6)
        self.assertEqual(image.at(2, 0), 0.0)  # no return, NOT zero metres
        self.assertAlmostEqual(image.fx, 168.6)

    def test_a_frame_without_intrinsics_is_not_guessed_at(self):
        """A depth image alone is not measurable: the publisher downsampled it,
        so the sensor's own fx is the wrong number and only the publisher knows
        the right one."""
        link = FakeLink()
        reader = self.depth(link)
        link.put_depth([1000] * 8, 4, 2)

        self.assertIsNone(reader.capture())
        self.assertIn("intrinsics", reader.status()["reason"])

    def test_a_size_mismatch_waits_rather_than_scaling_by_luck(self):
        link = FakeLink()
        reader = self.depth(link)
        link.put_depth_info(8, 4)
        link.put_depth([1000] * 8, 4, 2)

        self.assertIsNone(reader.capture())
        self.assertIn("waiting for them to agree", reader.status()["reason"])

    def test_a_stale_depth_frame_is_dropped(self):
        link = FakeLink()
        reader = self.depth(link)
        link.put_depth_info(4, 2)
        link.put_depth([1000] * 8, 4, 2)
        link.time += 30.0

        self.assertIsNone(reader.capture())
        self.assertIn("ago", reader.status()["reason"])

    def test_a_jpeg_on_the_depth_topic_is_refused_not_decoded(self):
        link = FakeLink()
        reader = self.depth(link)
        link.put_depth_info(4, 2)
        link.put_depth([1000] * 8, 4, 2, fmt="jpeg")

        self.assertIsNone(reader.capture())
        self.assertIn("PNG", reader.status()["reason"])

    def test_no_depth_at_all_is_a_status_line_not_an_exception(self):
        """Depth is an upgrade, not a precondition -- a robot without it must
        still be able to run a mission, ranging on the costmap instead."""
        reader = self.depth(FakeLink())

        self.assertIsNone(reader.capture())
        self.assertFalse(reader.status()["available"])


if __name__ == "__main__":
    unittest.main()
