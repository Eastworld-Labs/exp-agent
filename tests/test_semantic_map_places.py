"""Loading the dashboard's semantic map, and refusing the wrong one."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from system2_agent.modules.semantic_map import SemanticMapModule


def write(doc: dict) -> Path:
    path = Path(tempfile.mkdtemp()) / "places.json"
    path.write_text(json.dumps(doc))
    return path


DOC = {
    "map": "g1_map_small_upright",
    "updated": "2026-09-03",
    "places": [
        {"name": "kitchen", "x": 1.77, "y": -15.78, "yaw": 15,
         "tags": ["counter"], "provenance": "measured", "note": "by the sink"},
        {"name": "pantry", "x": 0.91, "y": -17.52, "yaw": -170,
         "tags": [], "provenance": "derived", "note": ""},
    ],
}


class PlacesTests(unittest.TestCase):
    def test_loads_places_and_converts_yaw_to_radians(self):
        """The file is authored in degrees by a person pointing at a map; the
        agent works in radians. One conversion, in one place."""
        module = SemanticMapModule.from_places_json(write(DOC))
        self.assertEqual(module.names(), ["kitchen", "pantry"])
        pose = module.resolve("kitchen")
        self.assertAlmostEqual(pose.x, 1.77)
        self.assertAlmostEqual(pose.yaw, math.radians(15))
        self.assertEqual(pose.frame, "map")

    def test_metadata_reaches_the_model(self):
        module = SemanticMapModule.from_places_json(write(DOC))
        described = module.describe("kitchen")
        self.assertEqual(described["provenance"], "measured")
        self.assertEqual(described["yaw_deg"], 15)
        self.assertEqual(described["note"], "by the sink")

    def test_name_lookup_folds_case_spaces_and_dashes(self):
        doc = {"map": "m", "places": [
            {"name": "kitchen_table", "x": 0, "y": 0, "yaw": 0}]}
        module = SemanticMapModule.from_places_json(write(doc))
        for spelling in ("kitchen_table", "Kitchen Table", "KITCHEN-TABLE", " kitchen table "):
            self.assertIsNotNone(module.resolve(spelling))

    def test_a_row_without_yaw_is_dropped_and_reported(self):
        """A waypoint in a doorway means THROUGH the doorway, and yaw is the
        only thing that says which way. Pointing it north instead would be a
        confident walk into a door frame."""
        doc = {"map": "m", "places": [
            {"name": "ok", "x": 0, "y": 0, "yaw": 0},
            {"name": "no_yaw", "x": 1, "y": 1},
        ]}
        module = SemanticMapModule.from_places_json(write(doc))
        self.assertEqual(module.names(), ["ok"])
        self.assertTrue(any("no_yaw" in e for e in module.errors))
        self.assertIn("unusable_entries", module.snapshot())

    def test_a_duplicate_name_is_refused_not_last_wins(self):
        doc = {"map": "m", "places": [
            {"name": "a", "x": 0, "y": 0, "yaw": 0},
            {"name": "a", "x": 9, "y": 9, "yaw": 0},
        ]}
        module = SemanticMapModule.from_places_json(write(doc))
        self.assertEqual(module.resolve("a").x, 0.0)
        self.assertTrue(any("duplicate" in e for e in module.errors))

    def test_an_unknown_place_is_refused_with_the_real_list(self):
        """##### REFUSE, NEVER RELOCATE. ##### A nearest-match would send the
        robot somewhere nobody named."""
        module = SemanticMapModule.from_places_json(write(DOC))
        with self.assertRaises(ValueError) as caught:
            module.resolve("garage")
        self.assertIn("kitchen", str(caught.exception))
        self.assertIn("pantry", str(caught.exception))

    def test_a_map_mismatch_makes_every_destination_refuse(self):
        """A label from another building's map is a confident walk to the wrong
        place, and nothing downstream could detect it."""
        module = SemanticMapModule.from_places_json(write(DOC), expect_map="some_other_map")
        self.assertTrue(module.fault)
        with self.assertRaises(ValueError):
            module.resolve("kitchen")
        self.assertIn("unusable", module.snapshot())

    def test_a_matching_map_is_fine(self):
        module = SemanticMapModule.from_places_json(
            write(DOC), expect_map="g1_map_small_upright")
        self.assertEqual(module.fault, "")
        self.assertEqual(module.snapshot()["map"], "g1_map_small_upright")

    def test_load_still_reads_the_original_locations_format(self):
        doc = {"locations": {"home": {"x": 0.0, "y": 0.0, "yaw": 0.0}}}
        module = SemanticMapModule.load(write(doc))
        self.assertEqual(module.names(), ["home"])

    def test_load_dispatches_on_content(self):
        module = SemanticMapModule.load(write(DOC))
        self.assertEqual(module.names(), ["kitchen", "pantry"])


if __name__ == "__main__":
    unittest.main()
