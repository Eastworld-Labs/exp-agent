import math
import unittest

from system2_agent.agent import System2Agent
from system2_agent.modules import (
    CameraFrame,
    CameraModule,
    DryRunManipulationBackend,
    DryRunNavigationBackend,
    ManipulationModule,
    NavigationModule,
    Pose3D,
    SemanticMapModule,
)
from system2_agent.types import AssistantTurn, ToolCall
from system2_agent.navigation_core import (
    AStarPlanner,
    GridMap,
    PathFollower,
    PlannedNavigationBackend,
    VelocityCommand,
)


class ScriptedModel:
    def __init__(self, turns):
        self.turns = iter(turns)

    def complete(self, messages, tools):
        return next(self.turns)


def call(index, name, **arguments):
    return AssistantTurn(tool_calls=(ToolCall(f"call_{index}", name, arguments),))


class AgentTests(unittest.TestCase):
    def test_agent_can_explicitly_refresh_camera_observations(self):
        class Cameras:
            def __init__(self):
                self.captures = 0

            def capture(self):
                self.captures += 1
                return [CameraFrame("head_rgb", "data:image/jpeg;base64,AA==")]

        class InspectingModel(ScriptedModel):
            def __init__(self):
                super().__init__([
                    call(1, "observe_surroundings"),
                    call(2, "finish", summary="fresh view inspected"),
                ])
                self.requests = []

            def complete(self, messages, tools):
                self.requests.append(messages)
                return super().complete(messages, tools)

        cameras = Cameras()
        model = InspectingModel()
        outcome = System2Agent(model, [CameraModule(cameras)]).run("look around")

        self.assertEqual(outcome.status, "completed")
        self.assertEqual(cameras.captures, 2)
        refreshed = [
            message["content"]
            for message in model.requests[1]
            if isinstance(message.get("content"), list)
        ][-1]
        self.assertTrue(any(item.get("type") == "image_url" for item in refreshed))

    def test_complete_mission(self):
        semantic_map = SemanticMapModule({"kitchen table": Pose3D(4.2, 1.8)})
        navigation = DryRunNavigationBackend()
        manipulation = DryRunManipulationBackend()
        model = ScriptedModel(
            [
                call(1, "navigate_to", location="kitchen table", reason="reach workspace"),
                call(2, "inspect_workspace"),
                call(3, "pick_object", object="red cup", reason="mission target"),
                call(4, "finish", summary="Arrived and verified the red cup is held."),
            ]
        )
        agent = System2Agent(
            model,
            [
                semantic_map,
                NavigationModule(semantic_map, navigation, requires_approval=False),
                ManipulationModule(manipulation, requires_approval=False),
            ],
        )

        outcome = agent.run("go to the kitchen table and pick up the red cup")

        self.assertEqual(outcome.status, "completed")
        self.assertEqual(manipulation.holding, "red cup")
        self.assertEqual(navigation.status()["state"], "succeeded")

    def test_motion_defaults_to_denied(self):
        semantic_map = SemanticMapModule({"kitchen": Pose3D(1.0, 1.0)})
        navigation = DryRunNavigationBackend()
        model = ScriptedModel(
            [
                call(1, "navigate_to", location="kitchen", reason="go there"),
                call(2, "request_human", reason="motion approval was denied"),
            ]
        )
        agent = System2Agent(model, [semantic_map, NavigationModule(semantic_map, navigation)])

        outcome = agent.run("go to the kitchen")

        self.assertEqual(outcome.status, "needs_human")
        self.assertEqual(navigation.status()["state"], "idle")

    def test_agent_drives_closed_loop_navigation(self):
        class FakeBase:
            name = "fake-locomotion"

            def __init__(self):
                self.value = Pose3D(0.0, 0.0)

            def pose(self):
                return self.value

            def command_velocity(self, command: VelocityCommand, dt: float):
                c, s = math.cos(self.value.yaw), math.sin(self.value.yaw)
                self.value = Pose3D(
                    self.value.x + (c * command.vx - s * command.vy) * dt,
                    self.value.y + (s * command.vx + c * command.vy) * dt,
                    yaw=self.value.yaw + command.yaw_rate * dt,
                )

            def stop(self):
                pass

        goal = Pose3D(1.0, 0.5, yaw=0.3)
        semantic_map = SemanticMapModule({"kitchen": goal})
        base = FakeBase()
        navigation = PlannedNavigationBackend(
            AStarPlanner(
                GridMap(20, 20, 0.25, -1.0, -1.0, frozenset()),
                footprint_radius_m=0.0,
            ),
            PathFollower(),
            base,
            timeout_s=2,
        )
        model = ScriptedModel(
            [
                call(1, "navigate_to", location="kitchen", reason="execute mission"),
                call(2, "finish", summary="Navigation verified."),
            ]
        )
        agent = System2Agent(
            model,
            [semantic_map, NavigationModule(semantic_map, navigation, requires_approval=False)],
        )

        outcome = agent.run("walk to the kitchen")

        self.assertEqual(outcome.status, "completed")
        self.assertEqual(navigation.status()["state"], "succeeded")
        self.assertLess(math.hypot(base.value.x - goal.x, base.value.y - goal.y), 0.15)


if __name__ == "__main__":
    unittest.main()
