import math
import unittest

from system2_agent.agent import MAX_STALLED_TURNS, System2Agent, turn_fault
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
from system2_agent.tools import Tool, object_schema
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
        # What the model was ASKED, kept so a test can assert on the schema it
        # was shown and on what the loop echoed back -- both of which are
        # invisible from the outcome alone.
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append((list(messages), list(tools)))
        return next(self.turns)


def call(index, name, **arguments):
    return AssistantTurn(tool_calls=(ToolCall(f"call_{index}", name, arguments),))


def truncated(index, name, **arguments):
    """A turn the provider cut off at the output-token limit."""
    return AssistantTurn(
        tool_calls=(ToolCall(f"call_{index}", name, arguments),),
        finish_reason="length",
    )


class SpyModule:
    """One tool that records every time it actually ran."""

    name = "spy"

    def __init__(self):
        self.runs = 0

    def tools(self):
        return (
            Tool(
                name="poke",
                description="Poke the world.",
                parameters=object_schema({}),
                handler=self._poke,
                kind="action",
            ),
        )

    def _poke(self, arguments):
        self.runs += 1
        return {"pokes": self.runs}

    def snapshot(self):
        return {"pokes": self.runs}


class AgentTests(unittest.TestCase):
    def test_agent_can_explicitly_refresh_camera_observations(self):
        class Cameras:
            def __init__(self):
                self.captures = 0

            def capture(self):
                self.captures += 1
                return [CameraFrame("head_rgb", "data:image/jpeg;base64,AA==")]

        cameras = Cameras()
        # ScriptedModel records every request itself, so there is nothing to
        # subclass: requests[n] is (messages, tools) for the n-th turn.
        model = ScriptedModel([
            call(1, "observe_surroundings"),
            call(2, "finish", summary="fresh view inspected"),
        ])
        outcome = System2Agent(model, [CameraModule(cameras)]).run("look around")

        self.assertEqual(outcome.status, "completed")
        self.assertEqual(cameras.captures, 2)
        messages, _tools = model.requests[1]
        refreshed = [
            message["content"]
            for message in messages
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


    def test_a_truncated_turn_runs_nothing_and_says_why(self):
        """A reply cut off at the token limit may carry truncated ARGUMENTS.

        The call below looks perfectly well formed and would pass every schema check.
        It must still not run, because `finish_reason` is the only evidence that its
        arguments are whole.
        """
        spy = SpyModule()
        model = ScriptedModel([
            truncated(1, "poke"),
            call(2, "poke"),
            call(3, "finish", summary="poked once, deliberately"),
        ])
        outcome = System2Agent(model, [spy]).run("poke the world once")

        self.assertEqual(outcome.status, "completed")
        self.assertEqual(spy.runs, 1, "the truncated call must not have executed")
        faults = [e for e in outcome.events if e["type"] == "protocol_error"]
        self.assertEqual(len(faults), 1)
        self.assertIn("token limit", faults[0]["error"])

    def test_repeated_stalls_end_the_mission_naming_the_cause(self):
        """Without a stall budget this burns every model call and reports the symptom."""
        spy = SpyModule()
        model = ScriptedModel([truncated(i, "poke") for i in range(MAX_STALLED_TURNS)])
        outcome = System2Agent(model, [spy], max_model_calls=40).run("poke the world")

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(spy.runs, 0)
        self.assertEqual(outcome.model_calls, MAX_STALLED_TURNS)
        self.assertIn("token limit", outcome.summary)

    def test_turn_fault_reports_truncation_ahead_of_the_call_count(self):
        """Truncation is the cause; the call count is a symptom of it."""
        both = AssistantTurn(
            tool_calls=(
                ToolCall("a", "poke", {}),
                ToolCall("b", "poke", {}),
            ),
            finish_reason="length",
        )
        self.assertIn("token limit", turn_fault(both) or "")
        self.assertIsNone(turn_fault(call(1, "poke")))
        self.assertIn("exactly one", turn_fault(AssistantTurn()) or "")


if __name__ == "__main__":
    unittest.main()


class AgentHookTests(unittest.TestCase):
    """The seams the host-side mission service is built on."""

    def _agent(self, turns, **kwargs):
        module = SpyModule()
        return System2Agent(ScriptedModel(turns), [module], **kwargs), module

    def test_a_call_is_announced_before_it_runs(self):
        """##### A NAVIGATION TOOL BLOCKS FOR THE WHOLE WALK. ##### A watcher
        told about the step only on completion shows nothing at all while the
        robot is moving."""
        seen = []
        agent, _ = self._agent(
            [call(1, "poke"), call(2, "finish", summary="done")],
            on_event=seen.append,
        )
        agent.run("mission")
        kinds = [event["type"] for event in seen]
        self.assertLess(kinds.index("call"), kinds.index("tool"))
        self.assertEqual(kinds[0], "turn")

    def test_a_watcher_that_throws_does_not_end_the_mission(self):
        def explode(_event):
            raise RuntimeError("the UI fell over")

        agent, _ = self._agent([call(1, "finish", summary="done")], on_event=explode)
        self.assertEqual(agent.run("mission").status, "completed")

    def test_should_stop_ends_the_run_without_touching_a_tool(self):
        agent, spy = self._agent(
            [call(1, "poke"), call(2, "finish", summary="done")],
            should_stop=lambda: True,
        )
        outcome = agent.run("mission")
        self.assertEqual(outcome.status, "cancelled")
        self.assertEqual(spy.runs, 0)

    def test_reasoning_details_are_echoed_verbatim_on_the_next_request(self):
        """A reasoning model's provider rejects the next request of a
        tool-calling exchange without them."""
        blocks = ({"type": "reasoning.text", "text": "thinking"},)
        turns = [
            AssistantTurn(
                tool_calls=(ToolCall(id="c1", name="poke", arguments={}),),
                reasoning_details=blocks,
            ),
            call(2, "finish", summary="done"),
        ]
        model = ScriptedModel(turns)
        System2Agent(model, [SpyModule()]).run("mission")
        second_request = model.requests[1][0]
        assistant = [m for m in second_request if m.get("role") == "assistant"][0]
        self.assertEqual(assistant["reasoning_details"], list(blocks))

    def test_usage_is_summed_across_turns_and_absent_stays_none(self):
        turns = [
            AssistantTurn(
                tool_calls=(ToolCall(id="c1", name="poke", arguments={}),),
                usage={"total_tokens": 10},
            ),
            AssistantTurn(
                tool_calls=(ToolCall(id="c2", name="finish", arguments={"summary": "d"}),),
                usage={"total_tokens": 5},
            ),
        ]
        outcome = System2Agent(ScriptedModel(turns), [SpyModule()]).run("mission")
        self.assertEqual(outcome.usage, {"total_tokens": 15})
        quiet = System2Agent(
            ScriptedModel([call(1, "finish", summary="d")]), [SpyModule()]).run("m")
        self.assertIsNone(quiet.usage)


class EnumValidationTests(unittest.TestCase):
    def test_a_value_outside_the_enum_is_refused_naming_the_valid_set(self):
        tool = Tool(
            name="go",
            description="",
            parameters=object_schema({"where": {"type": "string", "enum": ["a", "b"]}},
                                     ["where"]),
            handler=lambda args: args,
        )
        refused = tool.run({"where": "c"})
        self.assertFalse(refused.ok)
        self.assertIn("'a'", refused.error)
        self.assertIn("'b'", refused.error)
        self.assertTrue(tool.run({"where": "a"}).ok)

    def test_an_empty_enum_refuses_everything(self):
        """An unlabelled map is nowhere to go, and that is the honest shape."""
        tool = Tool(
            name="go",
            description="",
            parameters=object_schema({"where": {"type": "string", "enum": []}}, ["where"]),
            handler=lambda args: args,
        )
        self.assertFalse(tool.run({"where": "anywhere"}).ok)
