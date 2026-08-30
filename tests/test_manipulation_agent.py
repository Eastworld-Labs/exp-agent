import unittest

from system2_agent.manipulation_agent import (
    AgenticManipulationBackend,
    NestedManipulationAgent,
    SonicUpperBodyControlApi,
    WbcCartesianManipulationEmbodiment,
)
from system2_agent.modules import CameraFrame, ManipulationModule
from system2_agent.types import AssistantTurn, ToolCall


class ScriptedModel:
    def __init__(self, turns):
        self.turns = iter(turns)

    def complete(self, messages, tools):
        return next(self.turns)


def call(index, name, **arguments):
    return AssistantTurn(tool_calls=(ToolCall(f"call_{index}", name, arguments),))


class FakeEmbodiment:
    def __init__(self):
        self.moves = []
        self.aperture = 1.0

    def observe(self):
        return {"right_wrist_xyz": [0.4, -0.2, 0.8], "aperture": self.aperture}

    def camera_frames(self):
        return [CameraFrame("head", "data:image/jpeg;base64,AA=="),
                CameraFrame("right_wrist", "data:image/jpeg;base64,AA==")]

    def move_end_effector(self, arm, translation_m, rotation_rpy_rad, duration_s):
        self.moves.append((arm, tuple(translation_m), tuple(rotation_rpy_rad), duration_s))
        return {"executed": True}

    def set_gripper(self, arm, aperture):
        self.aperture = aperture
        return {"arm": arm, "aperture": aperture}

    def verify(self, instruction):
        return {"succeeded": self.aperture == 0.0, "instruction": instruction}


class ManipulationAgentTests(unittest.TestCase):
    def test_nested_agent_runs_actions_until_verified_completion(self):
        embodiment = FakeEmbodiment()
        model = ScriptedModel([
            call(1, "move_end_effector", arm="right", translation_m=[0.0, 0.0, -0.04],
                 rotation_rpy_rad=[0.0, 0.0, 0.0], duration_s=1.0, reason="approach"),
            call(2, "set_gripper", arm="right", aperture=0.0, reason="grasp"),
            call(3, "finish_manipulation", summary="object grasped"),
        ])
        backend = AgenticManipulationBackend(NestedManipulationAgent(model, embodiment))

        result = backend.manipulate("pick up the object")

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(embodiment.moves), 1)
        self.assertEqual(embodiment.aperture, 0.0)

    def test_outer_module_exposes_one_blocking_manipulate_tool(self):
        embodiment = FakeEmbodiment()
        backend = AgenticManipulationBackend(
            NestedManipulationAgent(ScriptedModel([]), embodiment)
        )
        names = [tool.name for tool in ManipulationModule(backend).tools()]
        self.assertIn("manipulate", names)
        self.assertNotIn("pick_object", names)

    def test_rejects_unbounded_cartesian_action(self):
        embodiment = FakeEmbodiment()
        model = ScriptedModel([
            call(1, "move_end_effector", arm="right", translation_m=[0.5, 0.0, 0.0],
                 rotation_rpy_rad=[0.0, 0.0, 0.0], duration_s=1.0, reason="unsafe"),
            call(2, "fail_manipulation", reason="bounded action rejected"),
        ])
        outcome = NestedManipulationAgent(model, embodiment).run("reach far away")
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(embodiment.moves, [])

    def test_wbc_cartesian_adapter_turns_deltas_into_pose_targets(self):
        class Api:
            def get_current_wrist_pose(self, arm="right"):
                return [0.4, -0.2, 0.8], [1.0, 0.0, 0.0, 0.0]

            def goto_pose(self, **kwargs):
                self.target = kwargs

            def set_gripper(self, aperture, arm="right"):
                self.grip = (arm, aperture)

        class Cameras:
            def capture(self):
                return [CameraFrame("right_wrist", "data:image/jpeg;base64,AA==")]

        api = Api()
        embodiment = WbcCartesianManipulationEmbodiment(
            api, Cameras(), lambda _: {"succeeded": True}
        )
        embodiment.move_end_effector("right", [0.02, 0.0, -0.03], [0, 0, 0], 1.0)
        embodiment.set_gripper("right", 0.25)
        for actual, expected in zip(api.target["position"], [0.42, -0.2, 0.77]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(api.grip, ("right", 0.25))

    def test_sonic_upper_body_api_streams_ik_then_holds(self):
        class Bridge:
            def __init__(self):
                self.commands = []

            def command_upper_body(self, position, velocity):
                self.commands.append((tuple(position), tuple(velocity)))

        bridge = Bridge()
        api = SonicUpperBodyControlApi(
            bridge,
            lambda: (0.0,) * 29,
            lambda arm: ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            lambda **kwargs: (0.5,) * 29,
            lambda arm, aperture: None,
            control_hz=2.0,
            sleeper=lambda seconds: None,
        )
        api.goto_pose((0.1, 0.0, 0.2), (1.0, 0.0, 0.0, 0.0), "right", 1.0)
        self.assertEqual(len(bridge.commands), 3)
        self.assertEqual(bridge.commands[-1][0], (0.5,) * 29)
        self.assertEqual(bridge.commands[-1][1], (0.0,) * 29)


if __name__ == "__main__":
    unittest.main()
