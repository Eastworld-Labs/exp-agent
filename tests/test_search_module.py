"""The `find_object` loop: what it does, and every way it refuses to keep going.

The scene is the one from test_search: the robot faces a counter with the target
hidden behind it. Here the question is not the geometry but the LOOP -- whether
it stops when it should, spends what it said it would, and hands off correctly.
"""
from __future__ import annotations

import math
import threading
import unittest

from system2_agent.grounding import Box, Grounding
from system2_agent.local_planner import InitPose
from system2_agent.modules.camera import CameraFrame
from system2_agent.modules.search import SWEEP_QUARTERS, SearchModule
from system2_agent.modules.semantic_map import Pose3D

from tests.test_search import HFOV, ROBOT, room


class FakeGeometry:
    horizontal_fov_deg = 87.0


class FakeGrid:
    def __init__(self, grid):
        self._grid = grid

    def grid(self):
        return self._grid

    def status(self):
        return {"available": True}


class FakeInitPose:
    """The robot's pose, in odom and map. The two frames are identical here."""

    def __init__(self, pose=ROBOT):
        self.pose = pose

    def init_pose(self, expected_odom_frame: str) -> InitPose:
        x, y, yaw = self.pose
        return InitPose(
            odom=Pose3D(x, y, yaw=yaw, frame="odom"),
            map=Pose3D(x, y, yaw=yaw, frame="map"),
        )


class FakeGrounder:
    """Finds the target only once `visible_after` calls have been made."""

    def __init__(self, visible_after: int | None = None):
        self.visible_after = visible_after
        self.calls = 0

    def ground(self, target: str, frame: CameraFrame) -> Grounding:
        self.calls += 1
        if self.visible_after is not None and self.calls >= self.visible_after:
            return Grounding(
                found=True, box=Box(0.4, 0.4, 0.6, 0.6), confidence=0.9,
                label=target, note="", model_calls=1,
            )
        raise ValueError(f"{target!r} is not visible in the current camera frame")


class FakeBackend:
    """Records every goal. Reports the planner's own verdict, as Nav2 would."""

    def __init__(self, init: FakeInitPose):
        self.goals: list[Pose3D] = []
        self.init = init

    def navigate(self, goal: Pose3D):
        self.goals.append(goal)
        # Walking moves the robot: the next candidate round must see a new pose,
        # or a test would pass on a search that never actually went anywhere.
        self.init.pose = (goal.x, goal.y, goal.yaw)
        return {"state": "arrived", "verdict_source": "planner"}

    def status(self):
        x, y, yaw = self.init.pose
        # ⚠️ "odom", LIKE THE SIMULATOR. Its pose arrives on /odom stamped
        # `odom` in a frame that IS the map frame; MqttInitPose normalises that
        # and the backend's status does not. A fake that reported "map" here
        # would let a halt in the wrong frame pass every test.
        return {"state": "idle", "pose": {"x": x, "y": y, "yaw": yaw, "frame": "odom"}}


class FakePlanner:
    """Only the parts of LocalPlanner that SearchModule reaches for."""

    expected_grid_frame = "odom"
    footprint_radius_m = 0.35
    max_range_m = 6.0

    def __init__(self, grounder, init, grid):
        self.grounder = grounder
        self.init_pose_source = init
        self.grid_source = FakeGrid(grid)
        self.geometry = FakeGeometry()
        self.frames = 0

    def _colour_frame(self) -> CameraFrame:
        self.frames += 1
        return CameraFrame("head_colour", "data:image/jpeg;base64,AAAA")


def build(visible_after=None, **kwargs):
    init = FakeInitPose(ROBOT)
    grid = room()
    planner = FakePlanner(FakeGrounder(visible_after), init, grid)
    backend = FakeBackend(init)
    module = SearchModule(planner, backend, **kwargs)
    return module, planner, backend


def run(module, **arguments):
    arguments.setdefault("target", "red bucket")
    arguments.setdefault("reason", "the mission needs it")
    tool = module.tools()[0]
    result = tool.run(arguments)
    return result


class TestFindsIt(unittest.TestCase):
    def test_already_in_frame_moves_nothing(self) -> None:
        """The cheapest possible outcome, and it must cost zero goals."""
        module, _, backend = build(visible_after=1)
        result = run(module)
        self.assertTrue(result.ok)
        self.assertEqual(result.data["outcome"], "found")
        self.assertEqual(backend.goals, [])
        self.assertEqual(result.data["legs_walked"], 0)

    def test_found_during_the_sweep_walks_no_legs(self) -> None:
        """Turning is cheap; the loop must exhaust it before walking anywhere."""
        module, _, backend = build(visible_after=3)
        result = run(module)
        self.assertEqual(result.data["outcome"], "found")
        self.assertEqual(result.data["legs_walked"], 0)
        self.assertGreater(result.data["turns"], 0)
        for goal in backend.goals:
            self.assertAlmostEqual(goal.x, ROBOT[0])
            self.assertAlmostEqual(goal.y, ROBOT[1])

    def test_finding_it_hands_off_to_the_local_planner(self) -> None:
        """`find_object` must not try to own the approach as well."""
        module, _, _ = build(visible_after=1)
        result = run(module)
        self.assertIn("local_planner", result.data["next_step"])
        self.assertIsNotNone(result.data["grounding"])

    def test_it_walks_when_turning_is_not_enough(self) -> None:
        module, _, backend = build(visible_after=SWEEP_QUARTERS + 3)
        result = run(module)
        self.assertEqual(result.data["outcome"], "found")
        self.assertGreater(result.data["legs_walked"], 0)
        walked = [g for g in backend.goals
                  if abs(g.x - ROBOT[0]) > 1e-6 or abs(g.y - ROBOT[1]) > 1e-6]
        self.assertTrue(walked, "a walking leg must move the goal off the anchor")


class TestBudget(unittest.TestCase):
    def test_it_stops_at_the_leg_budget(self) -> None:
        module, _, _ = build(visible_after=None, max_legs=2)
        result = run(module, max_legs=2)
        self.assertEqual(result.data["outcome"], "budget")
        self.assertLessEqual(result.data["legs_walked"], 2)

    def test_asking_for_more_legs_than_configured_is_refused(self) -> None:
        module, _, backend = build(max_legs=3)
        result = run(module, max_legs=99)
        self.assertFalse(result.ok)
        self.assertIn("3", str(result.error))
        self.assertEqual(backend.goals, [], "a refused search must not move the robot")

    def test_asking_for_a_bigger_radius_than_configured_is_refused(self) -> None:
        module, _, backend = build(radius_m=4.0)
        result = run(module, radius_m=50)
        self.assertFalse(result.ok)
        self.assertEqual(backend.goals, [])

    def test_the_mission_wide_leg_budget_refuses_a_later_search(self) -> None:
        """The per-call budget cannot be dodged by calling it again and again."""
        module, _, _ = build(max_legs=2)
        module.legs_walked = 2
        result = run(module)
        self.assertFalse(result.ok)
        self.assertIn("budget", str(result.error))

    def test_the_clock_ends_it(self) -> None:
        ticks = iter([0.0] + [999.0] * 200)
        module, _, backend = build(clock=lambda: next(ticks), max_seconds=10.0)
        result = run(module)
        self.assertEqual(result.data["outcome"], "budget")
        self.assertEqual(backend.goals, [], "an expired clock must not walk a leg")


class TestRepeating(unittest.TestCase):
    """The brake that a real sim run proved was missing.

    Observed 2026-09-05: find_object reported "found, nothing moved",
    local_planner then refused because the approach was blocked, and the model
    -- reading a refusal that says "not visible" -- called find_object again. It
    found it again instantly. Five pairs, two of them zero-motion, no step taken.
    """

    def test_searching_again_from_the_same_spot_is_refused(self) -> None:
        module, _, backend = build(visible_after=1)
        self.assertEqual(run(module).data["outcome"], "found")
        again = run(module)
        self.assertFalse(again.ok)
        self.assertIn("APPROACH, NOT THE FINDING", str(again.error))
        self.assertEqual(backend.goals, [], "the refused repeat must not move anything")

    def test_the_refusal_names_both_reasons_local_planner_can_give(self) -> None:
        """The model has just read one of them; the message has to connect."""
        module, _, _ = build(visible_after=1)
        run(module)
        error = str(run(module).error)
        self.assertIn("no free place to stand", error)
        self.assertIn("not visible", error)

    def test_moving_makes_a_second_search_legitimate(self) -> None:
        """Measured on the POSE, not on the last outcome."""
        module, planner, _ = build(visible_after=1)
        self.assertEqual(run(module).data["outcome"], "found")
        planner.init_pose_source.pose = (ROBOT[0] + 1.5, ROBOT[1], ROBOT[2])
        self.assertTrue(run(module).ok, "after walking a metre it is a new question")

    def test_a_different_target_is_never_blocked(self) -> None:
        module, _, _ = build(visible_after=1)
        run(module, target="red bucket")
        self.assertTrue(run(module, target="blue box").ok)

    def test_a_search_that_did_not_find_it_does_not_arm_the_brake(self) -> None:
        """Only success records a spot: an exhausted search may be retried."""
        module, _, _ = build(visible_after=None)
        self.assertEqual(run(module, radius_m=0.3).data["outcome"], "exhausted")
        self.assertTrue(run(module, radius_m=0.3).ok)


class TestStopping(unittest.TestCase):
    def test_cancelling_halts_the_robot_with_a_goal(self) -> None:
        """⚠️ The one place in this stack where stopping the mission stops the robot.

        Everywhere else a goal Nav2 already accepted keeps running. A tool that
        walks six legs under one approval cannot inherit that, so it publishes
        the robot's own pose to supersede the leg in flight.
        """
        cancel = threading.Event()
        cancel.set()
        module, _, backend = build(cancel=cancel)
        result = run(module)
        self.assertEqual(result.data["outcome"], "cancelled")
        self.assertEqual(len(backend.goals), 1, "exactly one goal: the halt")
        halt = backend.goals[0]
        self.assertAlmostEqual(halt.x, ROBOT[0])
        self.assertAlmostEqual(halt.y, ROBOT[1])
        # ⚠️ AND IT MUST BE STAMPED `map`. The sim's pose arrives on /odom with
        # frame_id "odom" in a frame that IS the map frame; the init-pose source
        # normalises that and `backend.status()` does not. A halt built from the
        # status would publish the one goal that matters in the wrong frame.
        self.assertEqual(halt.frame, "map")

    def test_the_halt_survives_a_pose_source_that_is_down(self) -> None:
        """Not stopping is the worst outcome, so it falls back and forces the frame."""
        class NoPose:
            def init_pose(self, expected_odom_frame):
                raise ValueError("nothing on /odom")

        cancel = threading.Event()
        module, planner, backend = build()
        planner.init_pose_source = NoPose()
        module.cancel = cancel
        cancel.set()
        # The search itself cannot start without a pose; drive the halt directly.
        module._halt()
        self.assertEqual(len(backend.goals), 1)
        self.assertEqual(backend.goals[0].frame, "map")

    def test_the_cancel_message_does_not_overclaim(self) -> None:
        cancel = threading.Event()
        cancel.set()
        module, _, _ = build(cancel=cancel)
        detail = run(module).data["detail"]
        self.assertIn("emergency stop", detail)


class TestExhaustion(unittest.TestCase):
    def test_nowhere_left_to_look_is_a_clear_answer(self) -> None:
        """Not a failure: the robot has looked everywhere it can reach."""
        module, planner, _ = build()
        result = run(module, radius_m=0.3)
        self.assertEqual(result.data["outcome"], "exhausted")
        self.assertIn("not in this part of the room", result.data["detail"])

    def test_a_picker_saying_none_ends_the_search(self) -> None:
        class NoPicker:
            def pick(self, target, hint, frame, candidates):
                return (None, "this is a bathroom, a bucket would not be here")

        module, _, backend = build(picker=NoPicker())
        result = run(module)
        self.assertEqual(result.data["outcome"], "exhausted")
        self.assertIn("bathroom", result.data["detail"])
        walked = [g for g in backend.goals
                  if abs(g.x - ROBOT[0]) > 1e-6 or abs(g.y - ROBOT[1]) > 1e-6]
        self.assertEqual(walked, [], "a picker refusal must not walk anywhere")


class TestPicker(unittest.TestCase):
    def test_a_broken_picker_falls_back_to_geometry(self) -> None:
        """A picker that throws must cost a worse choice, never a stopped search."""
        class Broken:
            def pick(self, *args):
                raise RuntimeError("the model endpoint is down")

        module, _, _ = build(visible_after=SWEEP_QUARTERS + 3, picker=Broken())
        result = run(module)
        self.assertEqual(result.data["outcome"], "found")
        why = [leg.get("why", "") for leg in result.data["legs"] if leg["kind"] == "walk"]
        self.assertTrue(any("could not answer" in w for w in why))

    def test_a_picker_naming_a_place_not_offered_is_ignored(self) -> None:
        class OutOfRange:
            def pick(self, target, hint, frame, candidates):
                return (99, "the one by the door")

        module, _, _ = build(visible_after=SWEEP_QUARTERS + 3, picker=OutOfRange())
        result = run(module)
        self.assertEqual(result.data["outcome"], "found")

    def test_the_hint_reaches_the_picker_verbatim(self) -> None:
        seen = {}

        class Recording:
            def pick(self, target, hint, frame, candidates):
                seen["hint"] = hint
                seen["target"] = target
                return (0, "left end of the counter")

        module, _, _ = build(visible_after=SWEEP_QUARTERS + 3, picker=Recording())
        run(module, hint="probably behind the kitchen island")
        self.assertEqual(seen["hint"], "probably behind the kitchen island")
        self.assertEqual(seen["target"], "red bucket")


class TestReporting(unittest.TestCase):
    def test_every_leg_carries_a_verdict_source(self) -> None:
        """The same honesty every other walk in this stack owes."""
        module, _, _ = build(visible_after=SWEEP_QUARTERS + 3)
        result = run(module)
        for leg in result.data["legs"]:
            self.assertIn("verdict_source", leg)

    def test_it_reports_what_it_looked_at(self) -> None:
        module, _, _ = build(visible_after=1)
        result = run(module)
        looked = result.data["looked_at"]
        self.assertIsNotNone(looked)
        self.assertGreater(looked["observed_m2"], 0.0)

    def test_the_snapshot_never_raises_and_is_cheap(self) -> None:
        module, _, _ = build()
        self.assertIn("legs_left", module.snapshot())
        run(module)
        snapshot = module.snapshot()
        self.assertIn("looked_at", snapshot)
        self.assertIsNotNone(snapshot["last"])

    def test_candidates_offered_are_recorded_for_audit(self) -> None:
        module, _, _ = build(visible_after=SWEEP_QUARTERS + 3)
        result = run(module)
        self.assertTrue(result.data["candidates_offered"])
        first = result.data["candidates_offered"][0]["candidates"][0]
        for key in ("x", "y", "yaw_deg", "reveals_m2", "walk_m", "bearing_deg"):
            self.assertIn(key, first)


class TestSchema(unittest.TestCase):
    def test_an_empty_target_is_refused(self) -> None:
        module, _, backend = build()
        self.assertFalse(run(module, target="   ").ok)
        self.assertEqual(backend.goals, [])

    def test_the_tool_moves_the_robot_and_says_so(self) -> None:
        module, _, _ = build()
        tool = module.tools()[0]
        self.assertEqual(tool.kind, "action")
        self.assertTrue(tool.requires_approval)
        self.assertIn("WALKS THE ROBOT", tool.description)

    def test_reason_is_required_because_an_operator_reads_it(self) -> None:
        module, _, _ = build()
        tool = module.tools()[0]
        self.assertIn("reason", tool.parameters["required"])
        self.assertIn("target", tool.parameters["required"])


if __name__ == "__main__":
    unittest.main()
