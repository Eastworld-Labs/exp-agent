import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from system2_agent.scene_bundle import SceneBundle
from system2_agent.scene_loader import SceneLoader
from system2_agent.modules.semantic_map import Pose3D, SemanticMapModule


class SceneLoaderTests(unittest.TestCase):
    def test_scene_bundle_reads_isaac_stage_and_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in (
                ("stage.usda", "#usda 1.0\n"),
                ("g1.usda", "#usda 1.0\n"),
                ("grid.json", "{}"),
                ("semantics.json", "{}"),
            ):
                (root / name).write_text(content, encoding="utf-8")
            manifest = root / "scene.json"
            manifest.write_text(
                json.dumps(
                    {
                        "external_mjcf": None,
                        "navigation_grid": "grid.json",
                        "semantic_map": "semantics.json",
                        "isaac_sim": {
                            "stage_usd": "stage.usda",
                            "robot_usd": "g1.usda",
                            "robot_prim": "/World/G1",
                            "renderer": "RaytracedLighting",
                            "cameras": [
                                {
                                    "label": "g1_head_rgb",
                                    "prim_path": "/World/G1/head_camera",
                                    "width": 320,
                                    "height": 240,
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            scene = SceneBundle.from_json(manifest)

            self.assertIsNotNone(scene.isaac_sim)
            assert scene.isaac_sim is not None
            self.assertEqual(scene.isaac_sim.stage_usd, (root / "stage.usda").resolve())
            self.assertEqual(scene.isaac_sim.robot_prim, "/World/G1")
            self.assertEqual(scene.isaac_sim.robot_usd, (root / "g1.usda").resolve())
            self.assertEqual(scene.isaac_sim.cameras[0].label, "g1_head_rgb")
            self.assertEqual(scene.isaac_sim.cameras[0].width, 320)

    def test_semantic_map_preserves_descriptive_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "locations.json"
            path.write_text(
                '{"locations":{"table approach":{"x":1,"y":2,'
                '"category":"approach","description":"left side",'
                '"interactive":false}}}',
                encoding="utf-8",
            )
            semantic = SemanticMapModule.from_json(path)

            description = semantic.describe("TABLE APPROACH")

            self.assertEqual(description["pose"]["x"], 1.0)
            self.assertEqual(description["category"], "approach")
            self.assertEqual(description["description"], "left side")

    def test_semantic_object_requires_navigation_approach(self):
        semantic = SemanticMapModule(
            {"cup": Pose3D(1, 2)},
            {"cup": {"navigation_target": False}},
        )

        with self.assertRaisesRegex(ValueError, "approach pose"):
            semantic.resolve_navigation_goal("cup")

    def test_scene_bundle_reads_external_initial_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            navigation = root / "grid.json"
            semantic = root / "locations.json"
            manifest = root / "scene.json"
            navigation.write_text('{"width":1,"height":1,"resolution":1}', encoding="utf-8")
            semantic.write_text('{"locations":{}}', encoding="utf-8")
            manifest.write_text(
                '{"navigation_grid":"grid.json","semantic_map":"locations.json",'
                '"initial_pose":{"x":2,"y":-11,"yaw":0.25}}',
                encoding="utf-8",
            )

            scene = SceneBundle.from_json(manifest)

            self.assertEqual(scene.initial_pose, (2.0, -11.0, 0.25))

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is optional")
    def test_attaches_external_scene_without_editing_robot(self):
        import mujoco

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            robot = root / "robot.xml"
            external = root / "external.xml"
            navigation = root / "grid.json"
            semantic = root / "locations.json"
            robot.write_text(
                '<mujoco><worldbody><body name="pelvis"><freejoint/>'
                '<geom name="robot" type="sphere" size=".1"/></body></worldbody></mujoco>',
                encoding="utf-8",
            )
            original = robot.read_bytes()
            external.write_text(
                '<mujoco><worldbody><geom name="wall" type="box" '
                'pos="2 0 1" size=".1 1 1"/></worldbody></mujoco>',
                encoding="utf-8",
            )
            navigation.write_text('{"width":1,"height":1,"resolution":1}', encoding="utf-8")
            semantic.write_text('{"locations":{}}', encoding="utf-8")
            scene = SceneBundle(external, navigation, semantic)

            with SceneLoader(robot).load(scene) as loaded:
                model = mujoco.MjModel.from_xml_path(str(loaded.model_path))
                names = [model.geom(index).name for index in range(model.ngeom)]
                self.assertIn("external_scene/wall", names)

            self.assertEqual(robot.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
