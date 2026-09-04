"""The navigation backend, against a fake link. No broker, no robot, no clock.

What is pinned here is mostly REFUSAL and HONESTY: the cases where a goal must
not be published at all, and the difference between "the planner said it
arrived" and "it looks like it arrived".
"""
import math
import threading
import unittest

from system2_agent.g1.nav2_backend import Nav2MqttBackend
from system2_agent.g1.wire import yaw_from_quat
from system2_agent.modules.semantic_map import Pose3D


class FakeLink:
    """A Link with a clock the test drives."""

    def __init__(self, connected=True):
        self._connected = connected
        self.messages: dict[str, tuple[dict, float]] = {}
        self.published: list[tuple[str, dict, int | None]] = []
        self.time = 100.0
        self.subscribed: set[str] = set()
        self.publish_error: Exception | None = None

    # -- Link ---------------------------------------------------------------
    def connected(self):
        return self._connected

    def subscribe(self, topic):
        self.subscribed.add(topic)

    def latest(self, topic):
        return self.messages.get(topic)

    def publish_cmd(self, topic, msg, expiry_s=None):
        if self.publish_error:
            raise self.publish_error
        self.published.append((topic, msg, expiry_s))

    # -- test helpers -------------------------------------------------------
    def put(self, topic, msg, at=None):
        self.messages[topic] = (msg, self.time if at is None else at)

    def put_pose(self, x, y, yaw=0.0):
        self.put("/localization_3d", {
            "header": {"frame_id": "map"},
            "pose": {"position": {"x": x, "y": y, "z": 0.0},
                     "orientation": {"x": 0, "y": 0, "z": math.sin(yaw / 2),
                                     "w": math.cos(yaw / 2)}},
        })

    def put_status(self, state, x, y, terminal=True, **extra):
        import json
        self.put("/goal_status", {"data": json.dumps(
            {"state": state, "terminal": terminal, "goal": {"x": x, "y": y}, **extra})})


def backend(link, **kw):
    kw.setdefault("poll_s", 0.0)
    return Nav2MqttBackend(
        link, now=lambda: link.time, sleep=lambda s: setattr(link, "time", link.time + 0.5),
        **kw)


GOAL = Pose3D(x=1.77, y=-15.78, yaw=math.radians(15))


class RefusalTests(unittest.TestCase):
    """##### NOTHING IS PUBLISHED IN ANY OF THESE. #####

    Publishing a goal at a robot that cannot act on it is worse than not
    publishing: nothing moves, nothing says why, and the mission burns its
    budget waiting for a walk that was never going to start.
    """

    def test_refuses_with_no_link(self):
        link = FakeLink(connected=False)
        with self.assertRaises(ValueError) as e:
            backend(link).navigate(GOAL)
        self.assertIn("no link", str(e.exception))
        self.assertEqual(link.published, [])

    def test_refuses_while_the_estop_is_engaged(self):
        link = FakeLink()
        link.put_pose(0, 0)
        link.put("/estop_state", {"data": True})
        with self.assertRaises(ValueError) as e:
            backend(link).navigate(GOAL)
        self.assertIn("emergency stop", str(e.exception).lower())
        self.assertEqual(link.published, [])

    def test_refuses_when_the_robot_reports_no_position(self):
        link = FakeLink()
        with self.assertRaises(ValueError) as e:
            backend(link).navigate(GOAL)
        self.assertIn("not reporting a position", str(e.exception))
        self.assertEqual(link.published, [])

    def test_refuses_a_stale_retained_pose(self):
        """/localization_3d is RETAINED, so connecting to a robot that has been
        off for an hour hands you its last known pose instantly, and it looks
        live. Age is measured from arrival here, not from the message stamp."""
        link = FakeLink()
        link.put_pose(0, 0)
        link.time += 60.0
        with self.assertRaises(ValueError):
            backend(link).navigate(GOAL)
        self.assertEqual(link.published, [])

    def test_refuses_while_sonic_is_present_but_not_armed(self):
        link = FakeLink()
        link.put_pose(0, 0)
        link.put("/sonic/enabled", {"data": False})
        with self.assertRaises(ValueError) as e:
            backend(link).navigate(GOAL)
        self.assertIn("NOT ARMED", str(e.exception))
        self.assertEqual(link.published, [])

    def test_an_absent_sonic_topic_is_not_a_refusal(self):
        """Unitree's built-in gait publishes no such topic, and that is the
        ordinary case -- treating silence as 'disarmed' would make the other
        backend unusable."""
        link = FakeLink()
        link.put_pose(1.77, -15.78)
        result = backend(link).navigate(GOAL)
        self.assertEqual(result["state"], "arrived")
        self.assertEqual(backend(link).status()["motion_backend"], "unknown")

    def test_a_stale_retained_sonic_flag_is_not_a_refusal(self):
        """##### THE TRAP AFTER A SONIC SESSION. ##### /sonic/enabled is
        RETAINED and heartbeated at 2 Hz. When SONIC is taken down its last
        value stays on the broker with nobody behind it, and a robot now on
        Unitree's gait would be refused every goal as "not armed". Six missed
        beats and the reading is nobody-is-saying, not disarmed."""
        link = FakeLink()
        link.put("/sonic/enabled", {"data": False})
        link.time += 60.0
        link.put_pose(1.77, -15.78)
        result = backend(link).navigate(GOAL)
        self.assertEqual(result["state"], "arrived")
        self.assertEqual(backend(link).status()["motion_backend"], "unknown")

    def test_a_stale_estop_reading_is_nobody_saying_not_stopped(self):
        """Same retention, same trap: an old `true` from a motion backend that
        has since gone must not read as an enforced stop -- and must not read
        as 'not stopped' either. It is an absence, and it is named as one."""
        link = FakeLink()
        link.put("/estop_state", {"data": True})
        link.time += 60.0
        link.put_pose(0, 0)
        status = backend(link).status()
        self.assertIsNone(status["estop_latched"])
        self.assertFalse(status["motion_backend_reporting"])
        link.put("/estop_state", {"data": False})
        status = backend(link).status()
        self.assertIs(status["estop_latched"], False)
        self.assertTrue(status["motion_backend_reporting"])

    def test_the_pose_can_come_from_odometry_on_another_topic(self):
        """The simulator has no localizer. Its ground-truth Odometry on /odom
        is the map-frame pose, and the backend reads it when told to."""
        link = FakeLink()
        link.put("/odom", {
            "header": {"frame_id": "odom"},
            "pose": {"pose": {"position": {"x": 1.77, "y": -15.78, "z": 0.0},
                              "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}},
                     "covariance": [0.0] * 36},
        })
        sim = backend(link, pose_topic="/odom")
        self.assertIn("/odom", link.subscribed)
        result = sim.navigate(GOAL)
        self.assertEqual(result["state"], "arrived")
        self.assertEqual(sim.status()["pose_topic"], "/odom")
        # And the default backend, reading /localization_3d, sees no pose there.
        with self.assertRaises(ValueError) as e:
            backend(link).navigate(GOAL)
        self.assertIn("/localization_3d", str(e.exception))


class GoalTests(unittest.TestCase):
    def test_publishes_the_goal_with_the_right_pose_and_expiry(self):
        link = FakeLink()
        link.put_pose(1.77, -15.78)
        backend(link).navigate(GOAL)
        topic, msg, expiry = link.published[0]
        self.assertEqual(topic, "/goal_pose")
        self.assertEqual(msg["pose"]["position"]["x"], 1.77)
        self.assertAlmostEqual(
            yaw_from_quat(msg["pose"]["orientation"]), math.radians(15), places=6)
        self.assertIsNone(expiry)   # the link applies the table's own TTL

    def test_the_planner_verdict_wins_and_is_labelled_as_such(self):
        link = FakeLink()
        link.put_pose(0.0, 0.0)          # nowhere near the goal
        result_holder = {}

        def run():
            result_holder["r"] = backend(link, arrive_timeout_s=30).navigate(GOAL)

        link.put_status("succeeded", 1.77, -15.78)
        link.messages["/goal_status"] = (link.messages["/goal_status"][0], link.time + 0.1)
        run()
        result = result_holder["r"]
        self.assertEqual(result["state"], "arrived")
        self.assertEqual(result["verdict_source"], "planner")
        self.assertIn("planner's own verdict", result["arrival_is"])

    def test_an_abort_is_reported_as_an_abort_not_a_timeout(self):
        """The whole reason /goal_status exists: without it this case is
        indistinguishable from a slow walk until the timeout expires."""
        link = FakeLink()
        link.put_pose(0.0, 0.0)
        link.put_status("aborted", 1.77, -15.78, reason="recoveries exhausted")
        link.messages["/goal_status"] = (link.messages["/goal_status"][0], link.time + 0.1)
        result = backend(link, arrive_timeout_s=30).navigate(GOAL)
        self.assertEqual(result["state"], "aborted")
        self.assertEqual(result["verdict_source"], "planner")

    def test_a_stale_retained_verdict_is_not_mistaken_for_ours(self):
        """##### THE TRAP THIS BINDING EXISTS FOR. ##### /goal_status is
        latched, so the previous goal's `succeeded` is sitting there when we
        publish. Reading `state` off it would report an arrival that already
        happened, at a different place, before we asked."""
        link = FakeLink()
        link.put_pose(0.0, 0.0)
        link.put_status("succeeded", 99.0, 99.0, at=link.time - 3600)
        result = backend(link, arrive_timeout_s=2).navigate(GOAL)
        self.assertEqual(result["state"], "timed_out")
        self.assertNotEqual(result["verdict_source"], "planner")

    def test_a_verdict_for_a_different_goal_is_ignored(self):
        link = FakeLink()
        link.put_pose(0.0, 0.0)
        link.put_status("succeeded", 40.0, 40.0)
        link.messages["/goal_status"] = (link.messages["/goal_status"][0], link.time + 0.1)
        result = backend(link, arrive_timeout_s=2).navigate(GOAL)
        self.assertEqual(result["state"], "timed_out")

    def test_without_a_verdict_arrival_is_derived_and_says_so(self):
        link = FakeLink()
        link.put_pose(1.9, -15.9)     # inside the tolerance
        result = backend(link).navigate(GOAL)
        self.assertEqual(result["state"], "arrived")
        self.assertEqual(result["verdict_source"], "derived")
        self.assertIn("DERIVED", result["arrival_is"])
        self.assertIn("indistinguishable", result["arrival_is"])

    def test_a_timeout_reports_the_closest_it_came(self):
        link = FakeLink()
        link.put_pose(0.0, 0.0)
        result = backend(link, arrive_timeout_s=2).navigate(GOAL)
        self.assertEqual(result["state"], "timed_out")
        self.assertGreater(result["remaining_m"], 1.0)
        self.assertIn("NOT known", result["detail"])

    def test_losing_localization_mid_walk_says_nothing_can_be_said(self):
        link = FakeLink()
        link.put_pose(0.0, 0.0)
        original_latest = link.latest

        def vanishing(topic):
            if topic == "/localization_3d" and link.time > 100.5:
                return None
            return original_latest(topic)

        link.latest = vanishing
        result = backend(link, arrive_timeout_s=30).navigate(GOAL)
        self.assertEqual(result["state"], "lost_localization")

    def test_cancelling_says_it_did_not_stop_the_robot(self):
        """##### THE MOST IMPORTANT SENTENCE THIS BACKEND EMITS. ##### An
        operator who presses Stop and reads 'cancelled' will otherwise believe
        the robot stopped. It did not: Nav2 already has the goal."""
        link = FakeLink()
        link.put_pose(0.0, 0.0)
        cancel = threading.Event()
        cancel.set()
        result = backend(link, cancel=cancel).navigate(GOAL)
        self.assertEqual(result["state"], "cancelled")
        self.assertIn("DID NOT STOP THE ROBOT", result["detail"])

    def test_a_transport_error_becomes_a_failed_result_not_an_exception(self):
        """Tool.run only catches (KeyError, TypeError, ValueError). Anything
        else raised from here would end the mission with a traceback instead of
        a step the model could react to."""
        link = FakeLink()
        link.put_pose(0.0, 0.0)
        link.publish_error = ConnectionError("broker went away")
        result = backend(link).navigate(GOAL)
        self.assertEqual(result["state"], "failed")
        self.assertIn("broker went away", result["error"])


class StatusTests(unittest.TestCase):
    def test_status_is_cheap_and_never_raises(self):
        link = FakeLink()
        link.latest = lambda topic: (_ for _ in ()).throw(RuntimeError("boom"))
        got = backend(link).status()
        self.assertEqual(got["state"], "unknown")
        self.assertIn("boom", got["error"])

    def test_status_reports_the_latch_as_the_robot_reports_it(self):
        link = FakeLink()
        link.put_pose(0, 0)
        self.assertIsNone(backend(link).status()["estop_latched"])
        link.put("/estop_state", {"data": True})
        self.assertIs(backend(link).status()["estop_latched"], True)

    def test_status_says_whether_a_planner_verdict_is_available_at_all(self):
        link = FakeLink()
        link.put_pose(0, 0)
        self.assertFalse(backend(link).status()["goal_status_available"])
        link.put_status("succeeded", 0, 0)
        self.assertTrue(backend(link).status()["goal_status_available"])


if __name__ == "__main__":
    unittest.main()
