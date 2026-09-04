"""`local_planner` end to end: a mission, a picture, and one published goal.

##### THE ASSERTION THAT MATTERS IS WHERE THE /goal_pose LANDED. ##### Every
other check in this repo can pass while the robot walks to the wrong place: the
grounding can be right, the range can be right, the path can be clear, and a
transform applied in the wrong direction still sends it into the next room. So
these tests compute the expected map coordinates by hand and compare the bytes
that would leave the process.

Nothing here touches a broker, a model or a clock.
"""
import json
import math
import unittest

from system2_agent.agent import System2Agent
from system2_agent.g1.depth import MqttHeadDepth
from system2_agent.g1.local_costmap import MqttInitPose, MqttLocalCostmap
from system2_agent.g1.nav2_backend import Nav2MqttBackend
from system2_agent.grounding import VisionGrounder
from system2_agent.local_planner import LocalPlanner
from system2_agent.modules.camera import CameraModule
from system2_agent.modules.local_planner import LocalPlannerModule
from system2_agent.sim.head_camera import D455
from system2_agent.types import AssistantTurn, ToolCall

from tests.test_g1_local_costmap import FakeLink

#: A robot standing at the odom origin, believing itself somewhere else in map.
#: Both frames rotated and offset, so a transform composed backwards cannot
#: coincidentally produce the right answer.
ODOM = (0.0, 0.0, 0.0)
MAP_XY = (10.0, 5.0)
MAP_YAW = math.radians(-45)

#: The target: a flat face 2.0 m in front of the lens, centred.
TARGET_M = 2.0
DEPTH_SIZE = (320, 180)


class ScriptedModel:
    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append((list(messages), list(tools)))
        return self.turns.pop(0)


def call(index, name, **arguments):
    return AssistantTurn(tool_calls=(ToolCall(f"call_{index}", name, arguments),))


def grounded(index, box=(0.35, 0.35, 0.65, 0.65), confidence=0.9):
    return AssistantTurn(
        tool_calls=(
            ToolCall(
                f"g_{index}",
                "report_grounding",
                {
                    "found": True,
                    "box": dict(zip(("x0", "y0", "x1", "y1"), box)),
                    "confidence": confidence,
                    "label": "sink",
                    "note": "stainless basin",
                },
            ),
        )
    )


class Harness:
    """A link carrying everything a real robot would, and the modules over it."""

    def __init__(self, *, metres=TARGET_M, map_topic="/localization_3d", grid_m=8.0):
        self.link = FakeLink()
        self.metres = metres
        self.cells = int(grid_m / 0.05)
        self.origin = (-grid_m / 2, -grid_m / 2)
        self.grounder = ScriptedModel([grounded(1)])
        self.backend = Nav2MqttBackend(
            self.link,
            pose_topic=map_topic,
            poll_s=0.0,
            now=lambda: self.link.time,
            sleep=self._sleep,
        )
        from system2_agent.g1.camera import MqttHeadCamera

        self.camera = MqttHeadCamera(self.link, now=lambda: self.link.time)
        self.planner = LocalPlanner(
            camera=self.camera,
            grounder=VisionGrounder(self.grounder),
            grid_source=MqttLocalCostmap(self.link, now=lambda: self.link.time),
            init_pose=MqttInitPose(
                self.link, map_topic=map_topic, now=lambda: self.link.time
            ),
            geometry=D455,
            depth=MqttHeadDepth(self.link, now=lambda: self.link.time),
            # Pinned, not the shipped 0.60 -- see the same note on
            # test_local_planner.planner(). The goal-arithmetic assertions below
            # are hand-derived against 0.90; the shipped default and its derived
            # floor are pinned by test_local_planner.DefaultsTests.
            standoff_m=0.90,
        )
        self.module = LocalPlannerModule(self.planner, self.backend)
        self.tool = self.module.tools()[0]
        # ⚠️ THE ROBOT STARTS TALKING AFTER THE SUBSCRIBERS EXIST, as it does in
        # a real mission: the service builds its modules and then messages flow.
        # Publishing first would leave the costmap's watcher with nothing to
        # have seen, which is a fixture bug that looks exactly like a dead link.
        self.publish_world()
        self.answer_goals()

    def _sleep(self, seconds):
        self.link.time += 0.5

    # -- the robot's side ---------------------------------------------------
    def publish_world(self, *, lethal=(), robot=ODOM):
        cost = [0] * (self.cells * self.cells)
        for col, row in lethal:
            cost[row * self.cells + col] = 100
        self.link.put_grid(
            cost, width=self.cells, height=self.cells, origin=self.origin
        )
        # A patch, because a retained grid alone is refused as not-live.
        self.link.put_patch(0, 0, 1, 1, [0])
        self.link.put_odom(*robot)
        self.link.put_map_pose(*MAP_XY, MAP_YAW)
        self.publish_depth()
        self.publish_preview()
        self.link.put("/estop_state", {"data": False})

    def publish_preview(self):
        self.link.put(
            "/g1/d435c/preview/compressed",
            {"header": {"frame_id": "d455"}, "format": "jpeg", "data": b"\xff\xd8fake"},
        )

    def refresh_except_costmap(self):
        """Everything the tool reads goes fresh except the grid, so a refusal can
        be attributed to the costmap rather than to whatever is checked first."""
        self.publish_preview()
        self.link.put_odom(*ODOM)
        self.link.put_map_pose(*MAP_XY, MAP_YAW)
        self.publish_depth()
        self.link.put("/estop_state", {"data": False})

    def publish_depth(self, metres=None, *, region=(0.3, 0.3, 0.7, 0.7)):
        metres = self.metres if metres is None else metres
        width, height = DEPTH_SIZE
        values = [0] * (width * height)
        for row in range(int(region[1] * height), int(region[3] * height)):
            for col in range(int(region[0] * width), int(region[2] * width)):
                values[row * width + col] = int(round(metres * 1000))
        fx, fy, cx, cy = D455.intrinsics
        ratio = width / D455.width
        self.link.put_depth_info(width, height, fx=fx * ratio)
        self.link.put_depth(values, width, height)

    def answer_goals(self, state="succeeded"):
        """Have the robot report Nav2's verdict the instant a goal is published.

        Synchronous on purpose. The real exchange is asynchronous, but a test
        that waited on a thread while the backend spun a fake clock would race
        the pose freshness gate and fail as `lost_localization` now and then --
        a flake that says nothing about the code under test.
        """
        publish = self.link.publish_cmd

        def answer(topic, msg, expiry_s=None):
            publish(topic, msg, expiry_s)
            if topic == "/goal_pose":
                position = msg["pose"]["position"]
                self.link.put_status(state, position["x"], position["y"])

        self.link.publish_cmd = answer

    def step(self, metres):
        """The robot walked: everything is republished, the target is nearer."""
        self.metres = metres
        self.publish_world()

    def run(self, **arguments):
        arguments.setdefault("target", "sink")
        arguments.setdefault("reason", "the mission needs the sink")
        return self.tool.run(arguments)


def put_status(self, state, x, y, terminal=True, **extra):
    self.put("/goal_status", {"data": json.dumps(
        {"state": state, "terminal": terminal, "goal": {"x": x, "y": y}, **extra})})


FakeLink.put_status = put_status


class GoalTests(unittest.TestCase):
    def test_the_published_goal_lands_where_the_arithmetic_says(self):
        """##### THE WHOLE FEATURE, IN ONE NUMBER. #####

        The target is 2.0 m in front of the lens, the lens is 5.76 cm in front of
        base_footprint, and the standoff is 0.90 m -- so the robot should stop
        1.16 m ahead of where it stands. It believes it stands at map (10, 5)
        facing -45 degrees, so that is 1.16 m along that heading.
        """
        harness = Harness()

        result = harness.run()

        self.assertTrue(result.ok, result.error)
        published = [entry for entry in harness.link.published if entry[0] == "/goal_pose"]
        self.assertEqual(len(published), 1)
        position = published[0][1]["pose"]["position"]
        expected = TARGET_M + D455.mount_xyz[0] - 0.9
        self.assertAlmostEqual(
            position["x"], MAP_XY[0] + expected * math.cos(MAP_YAW), delta=0.06
        )
        self.assertAlmostEqual(
            position["y"], MAP_XY[1] + expected * math.sin(MAP_YAW), delta=0.06
        )
        # Facing the target: the robot's own heading, since the thing is dead ahead.
        orientation = published[0][1]["pose"]["orientation"]
        yaw = 2 * math.atan2(orientation["z"], orientation["w"])
        self.assertAlmostEqual(yaw, MAP_YAW, delta=math.radians(3))

    def test_the_result_reports_the_walk_and_the_plan_behind_it(self):
        harness = Harness()

        data = harness.run().data

        self.assertEqual(data["state"], "arrived")
        self.assertEqual(data["verdict_source"], "planner")
        plan = data["local_plan"]
        self.assertTrue(plan["reached_standoff"])
        self.assertEqual(plan["range"]["method"], "depth")
        # Nothing lethal on that bearing in this fixture, so there was no
        # second opinion to report. Absent, not a claim of agreement.
        self.assertNotIn("agrees", plan["range"])
        self.assertEqual(plan["trajectory"]["frame"], "local")
        self.assertEqual(plan["trajectory"]["points"][0], [0.0, 0.0])
        self.assertAlmostEqual(plan["target_local"]["x"], 2.06, delta=0.05)
        self.assertEqual(plan["costmap"]["frame"], "odom")

    def test_the_model_is_shown_the_picture_and_asked_for_one_box(self):
        harness = Harness()

        harness.run()

        messages, tools = harness.grounder.requests[0]
        self.assertEqual([tool["function"]["name"] for tool in tools], ["report_grounding"])
        images = [p for p in messages[1]["content"] if p.get("type") == "image_url"]
        self.assertEqual(len(images), 1)
        self.assertTrue(images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))


class LegTests(unittest.TestCase):
    def test_a_target_beyond_one_leg_walks_part_way_and_asks_for_another(self):
        """##### A LOCAL PLAN IS OFTEN NOT THE WHOLE APPROACH. ##### The costmap
        is a 5 m window; the camera sees a sink across the room. Refusing there
        would be useless -- the robot can see it and could obviously walk at it."""
        harness = Harness(metres=4.5, grid_m=5.0)

        data = harness.run().data

        plan = data["local_plan"]
        self.assertFalse(plan["reached_standoff"])
        self.assertGreater(plan["remaining_m"], 1.0)
        self.assertIn("call local_planner again", plan["next_step"])
        self.assertLessEqual(plan["goal"]["leg_m"], 2.3)  # inside the 5 m window

    def test_a_second_leg_is_allowed_once_the_robot_has_actually_closed_in(self):
        harness = Harness(metres=4.5, grid_m=5.0)
        harness.run()

        # The robot walked; the sink is nearer now.
        harness.grounder.turns.append(grounded(2))
        harness.step(2.0)
        harness.link.published.clear()

        second = harness.run()

        self.assertTrue(second.ok, second.error)
        self.assertTrue(second.data["local_plan"]["reached_standoff"])
        self.assertEqual(len(harness.module.legs), 2)

    def test_a_leg_that_got_nowhere_is_refused_rather_than_repeated(self):
        """⚠️ MEASURED AGAINST THE RANGE, NOT THE PLAN. Whether the last leg was
        actually walked is a different question from whether it was planned, and
        the only evidence is that the thing is nearer. A goal Nav2 aborted and a
        robot the collision monitor stopped look identical from the plan side."""
        harness = Harness(metres=4.5, grid_m=5.0)
        harness.run()

        harness.grounder.turns.append(grounded(2))
        harness.step(4.5)  # unchanged: nothing moved

        result = harness.run()

        self.assertFalse(result.ok)
        self.assertIn("did not get the robot closer", result.error)
        self.assertIn("request_human", result.error)

    def test_the_leg_budget_stops_an_endless_approach(self):
        harness = Harness()
        harness.module.max_approaches = 1
        harness.module.legs.append({"target": "other", "range_m": 9.9})

        result = harness.run()

        self.assertFalse(result.ok)
        self.assertIn("budget", result.error)
        self.assertEqual(harness.link.published, [])


class RefusalTests(unittest.TestCase):
    def test_nothing_is_published_when_the_target_is_not_visible(self):
        harness = Harness()
        harness.grounder.turns = [
            AssistantTurn(
                tool_calls=(
                    ToolCall(
                        "g",
                        "report_grounding",
                        {"found": False, "confidence": 0.0, "note": "no sink in view"},
                    ),
                )
            )
        ]

        result = harness.run()

        self.assertFalse(result.ok)
        self.assertIn("no sink in view", result.error)
        self.assertEqual(harness.link.published, [])

    def test_nothing_is_published_when_the_costmap_is_stale(self):
        harness = Harness()
        harness.link.time += 120.0
        harness.refresh_except_costmap()

        result = harness.run()

        self.assertFalse(result.ok)
        self.assertIn("not live", result.error)
        self.assertEqual(harness.link.published, [])

    def test_nothing_is_published_when_a_wall_blocks_the_way(self):
        harness = Harness(metres=3.0)
        col = int((1.0 - harness.origin[0]) / 0.05)
        harness.publish_world(lethal=[(col, row) for row in range(harness.cells)])

        result = harness.run()

        self.assertFalse(result.ok)
        self.assertIn("no collision-free path", result.error)
        self.assertEqual(harness.link.published, [])

    def test_the_backends_own_preflight_refusals_still_apply(self):
        """A latched E-stop stops this tool exactly as it stops navigate_to: the
        refusal lives in the backend, and routing a new tool around it would be
        the kind of bypass this stack is built to make impossible."""
        harness = Harness()
        harness.link.put("/estop_state", {"data": True})

        result = harness.run()

        self.assertFalse(result.ok)
        self.assertIn("stop", result.error.lower())
        self.assertEqual(harness.link.published, [])


class AgentTests(unittest.TestCase):
    def test_the_mission_controller_can_drive_the_whole_thing(self):
        harness = Harness()
        model = ScriptedModel(
            [
                call(1, "local_planner", target="sink", reason="walk up to it"),
                call(2, "finish", summary="Stopped in front of the sink."),
            ]
        )
        events = []
        agent = System2Agent(
            model,
            [harness.module, CameraModule(harness.camera)],
            approval=lambda call, tool: True,
            on_event=events.append,
        )

        outcome = agent.run("go to the sink")

        self.assertEqual(outcome.status, "completed")
        self.assertEqual(len(harness.link.published), 1)
        moves = [e for e in events if e.get("type") == "call" and e["name"] == "local_planner"]
        self.assertTrue(moves[0]["moves_robot"], "an approach must be gated as motion")

    def test_the_tool_is_refused_when_the_operator_declines(self):
        harness = Harness()
        model = ScriptedModel(
            [
                call(1, "local_planner", target="sink", reason="walk up to it"),
                call(2, "finish", summary="Operator declined."),
            ]
        )
        agent = System2Agent(
            model, [harness.module], approval=lambda call, tool: False
        )

        agent.run("go to the sink")

        self.assertEqual(harness.link.published, [])

    def test_the_schema_the_model_sees_names_the_thing_not_a_coordinate(self):
        """The model chooses a NOUN. If a coordinate ever appears in this schema
        the safety story changes completely, so the shape is pinned here."""
        harness = Harness()

        schema = harness.tool.schema()["function"]

        self.assertEqual(schema["name"], "local_planner")
        self.assertEqual(
            sorted(schema["parameters"]["properties"]),
            ["clearance_m", "reason", "standoff_m", "target"],
        )
        # ⚠️ `clearance_m` IS STILL A DISTANCE, NOT A COORDINATE. The safety
        # story is about the model never choosing WHERE; choosing HOW CLOSE to a
        # thing it named is the same kind of number `standoff_m` always was.
        self.assertIn("FRONT OF THE ROBOT'S BODY",
                      schema["parameters"]["properties"]["clearance_m"]["description"])
        self.assertEqual(sorted(schema["parameters"]["required"]), ["reason", "target"])
        self.assertIn("WALKS THE ROBOT", schema["description"])


class SnapshotTests(unittest.TestCase):
    def test_the_snapshot_reports_the_link_without_raising_on_a_dead_one(self):
        harness = Harness()

        snapshot = harness.module.snapshot()

        self.assertEqual(snapshot["legs_walked"], 0)
        self.assertTrue(snapshot["costmap"]["available"])
        self.assertTrue(snapshot["depth"]["available"])
        self.assertEqual(snapshot["view"]["standoff_m"], 0.9)
        # Stated every turn so a model reads the closest legal approach rather
        # than inferring it from a refusal. 0.90 - 0.3079 of body.
        self.assertAlmostEqual(snapshot["view"]["closest_body_clearance_m"], 0.29, places=2)

    def test_the_snapshot_warns_when_the_camera_is_aimed_at_the_floor(self):
        """A 48-degree-down sensor cannot see a worktop two metres away, and a
        model that does not know that reads every refusal as a fault. This lives
        in the snapshot rather than the prompt because it differs per robot."""
        from system2_agent.sim.head_camera import HeadCameraSpec

        harness = Harness()
        harness.planner.geometry = HeadCameraSpec(
            name="pitched", width=640, height=480, horizontal_fov_deg=69.4,
            pitch_down_deg=47.87,
        )

        note = harness.module.snapshot()["view"]["note"]

        self.assertIn("48 degrees DOWN", note)


if __name__ == "__main__":
    unittest.main()
