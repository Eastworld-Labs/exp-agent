import unittest

from system2_agent.agent import System2Agent
from system2_agent.modules import (
    DryRunManipulationBackend,
    DryRunNavigationBackend,
    ManipulationModule,
    NavigationModule,
    Pose3D,
    SemanticMapModule,
)
from system2_agent.types import AssistantTurn, ToolCall


class ScriptedModel:
    def __init__(self, turns):
        self.turns = iter(turns)

    def complete(self, messages, tools):
        return next(self.turns)


def call(index, name, **arguments):
    return AssistantTurn(tool_calls=(ToolCall(f"call_{index}", name, arguments),))


class AgentTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
