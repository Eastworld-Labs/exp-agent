"""The nested call that turns a word into a box.

##### THE FAILURE THAT MATTERS HERE IS A CONFIDENT BOX OVER NOTHING. ##### It
becomes a metric range, a standoff pose and a robot walking at a blank wall, and
every step after the first one looks completely healthy. So most of this file is
about refusing, and about refusing with a sentence the mission controller can act
on rather than a generic error.
"""
import json
import unittest

from system2_agent.grounding import Box, VisionGrounder
from system2_agent.modules.camera import CameraFrame
from system2_agent.types import AssistantTurn, ToolCall

FRAME = CameraFrame("head_colour", "data:image/jpeg;base64,AA==")


class ScriptedModel:
    """The fake from tests/test_agent.py, with the requests kept for inspection."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append((list(messages), list(tools)))
        return self.turns.pop(0)


def reported(index=1, **arguments):
    return AssistantTurn(tool_calls=(ToolCall(f"call_{index}", "report_grounding", arguments),))


def found(**overrides):
    payload = {
        "found": True,
        "box": {"x0": 0.4, "y0": 0.5, "x1": 0.6, "y1": 0.8},
        "confidence": 0.87,
        "label": "sink",
        "note": "stainless basin under the window",
    }
    payload.update(overrides)
    return payload


class BoxTests(unittest.TestCase):
    def test_a_box_outside_the_image_or_inside_out_is_refused(self):
        with self.assertRaises(ValueError):
            Box(0.6, 0.1, 0.4, 0.9)        # x1 before x0
        with self.assertRaises(ValueError):
            Box(0.1, 0.1, 1.4, 0.9)        # off the right edge
        with self.assertRaises(ValueError):
            Box(0.1, 0.5, 0.9, 0.5)        # zero height

    def test_shrinking_keeps_the_centre_and_a_fraction_of_each_side(self):
        """The border of a box is mostly not the object -- it is silhouette and
        whatever is behind it, at the range of the background."""
        inner = Box(0.2, 0.2, 0.8, 0.6).shrunk(0.5)

        self.assertAlmostEqual(inner.centre[0], 0.5, places=9)
        self.assertAlmostEqual(inner.centre[1], 0.4, places=9)
        self.assertAlmostEqual(inner.x1 - inner.x0, 0.3, places=9)
        self.assertAlmostEqual(inner.y1 - inner.y0, 0.2, places=9)


class GroundingTests(unittest.TestCase):
    def test_one_image_one_call_one_box(self):
        model = ScriptedModel([reported(**found())])

        grounding = VisionGrounder(model).ground("sink", FRAME)

        self.assertTrue(grounding.found)
        self.assertEqual(grounding.box.as_json(), [0.4, 0.5, 0.6, 0.8])
        self.assertEqual(grounding.confidence, 0.87)
        self.assertEqual(grounding.model_calls, 1)

    def test_the_request_carries_the_picture_and_exactly_one_tool(self):
        """Cheap by construction: the colour frame only (the range preview is a
        false-colour picture, not evidence for where a thing is), no world
        snapshot, no mission history."""
        model = ScriptedModel([reported(**found())])

        VisionGrounder(model).ground("sink", FRAME)

        messages, tools = model.requests[0]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["function"]["name"], "report_grounding")
        self.assertEqual(len(messages), 2)
        images = [
            part for part in messages[1]["content"] if part.get("type") == "image_url"
        ]
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["image_url"]["url"], FRAME.url)

    def test_not_visible_is_an_answer_and_it_carries_the_reason(self):
        model = ScriptedModel(
            [reported(found=False, confidence=0.0, note="I can see a counter but no sink")]
        )

        with self.assertRaises(ValueError) as caught:
            VisionGrounder(model).ground("sink", FRAME)

        message = str(caught.exception)
        self.assertIn("not visible", message)
        self.assertIn("I can see a counter but no sink", message)

    def test_an_unsure_answer_is_refused_rather_than_acted_on(self):
        model = ScriptedModel([reported(**found(confidence=0.31))])

        with self.assertRaises(ValueError) as caught:
            VisionGrounder(model).ground("sink", FRAME)

        self.assertIn("0.31", str(caught.exception))

    def test_a_confidence_off_the_scale_discredits_the_box_too(self):
        """A model that answers 5 to a 0-to-1 field misread the convention -- and
        the coordinates are documented on that same scale by the same sentence."""
        model = ScriptedModel([reported(**found(confidence=5.0))])

        with self.assertRaises(ValueError) as caught:
            VisionGrounder(model).ground("sink", FRAME)

        self.assertIn("outside 0 to 1", str(caught.exception))

    def test_a_box_too_small_to_range_is_refused_where_it_can_say_why(self):
        """Downstream this would surface as "no depth return", which sends the
        model looking for a sensor fault instead of walking closer."""
        model = ScriptedModel(
            [reported(**found(box={"x0": 0.500, "y0": 0.500, "x1": 0.505, "y1": 0.505}))]
        )

        with self.assertRaises(ValueError) as caught:
            VisionGrounder(model).ground("sink", FRAME)

        self.assertIn("too small", str(caught.exception))

    def test_found_with_no_box_is_refused(self):
        model = ScriptedModel([reported(found=True, confidence=0.9, note="over there")])

        with self.assertRaises(ValueError) as caught:
            VisionGrounder(model).ground("sink", FRAME)

        self.assertIn("no box", str(caught.exception))

    def test_a_prose_answer_with_the_right_fields_is_accepted(self):
        """⚠️ `tool_choice` IS "auto" IN model.py, DELIBERATELY. A model that
        answered correctly in a fenced block has answered; spending a second call
        to make it repeat itself in a tool call is waste, not rigour. The text is
        validated through the SAME schema, so it is not a laxer path."""
        model = ScriptedModel(
            [AssistantTurn(content="Here you go:\n```json\n" + json.dumps(found()) + "\n```")]
        )

        grounding = VisionGrounder(model).ground("sink", FRAME)

        self.assertTrue(grounding.found)
        self.assertEqual(grounding.model_calls, 1)

    def test_a_truncated_reply_is_retried_once_with_the_fault_fed_back(self):
        """A reply cut off at the token limit can carry a box whose numbers still
        parse -- 0.85 truncated to 0.8 is a valid coordinate and a different
        object. agent.turn_fault exists for exactly this and is shared, not
        re-implemented."""
        model = ScriptedModel(
            [
                AssistantTurn(
                    tool_calls=(ToolCall("c1", "report_grounding", found()),),
                    finish_reason="length",
                ),
                reported(2, **found()),
            ]
        )

        grounding = VisionGrounder(model).ground("sink", FRAME)

        self.assertEqual(grounding.model_calls, 2)
        second = model.requests[1][0]
        self.assertIn("cut off", second[-1]["content"])

    def test_two_useless_replies_end_in_a_refusal_not_a_loop(self):
        model = ScriptedModel(
            [AssistantTurn(content="I am not sure."), AssistantTurn(content="Still not sure.")]
        )

        with self.assertRaises(ValueError) as caught:
            VisionGrounder(model).ground("sink", FRAME)

        self.assertIn("did not report a usable box", str(caught.exception))
        self.assertEqual(len(model.requests), 2)

    def test_a_reply_that_calls_the_wrong_tool_is_fed_back(self):
        model = ScriptedModel(
            [
                AssistantTurn(tool_calls=(ToolCall("c1", "navigate_to", {"location": "x"}),)),
                reported(2, **found()),
            ]
        )

        grounding = VisionGrounder(model).ground("sink", FRAME)

        self.assertEqual(grounding.model_calls, 2)
        self.assertIn("report_grounding", model.requests[1][0][-1]["content"])

    def test_usage_is_summed_across_the_nested_calls(self):
        """The outer loop only sums ITS own turns, so a grounding call's cost
        would otherwise vanish from the bill entirely."""
        model = ScriptedModel(
            [
                AssistantTurn(content="no", usage={"total_tokens": 10}),
                AssistantTurn(
                    tool_calls=(ToolCall("c2", "report_grounding", found()),),
                    usage={"total_tokens": 25},
                ),
            ]
        )

        grounding = VisionGrounder(model).ground("sink", FRAME)

        self.assertEqual(grounding.usage["total_tokens"], 35)


if __name__ == "__main__":
    unittest.main()
