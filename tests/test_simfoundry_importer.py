import json
import tempfile
import unittest
from pathlib import Path

from system2_agent.modules.semantic_map import SemanticMapModule
from system2_agent.scene_bundle import SceneBundle
from system2_agent.simfoundry_importer import SimFoundryMuJoCoImporter


class SimFoundryImporterTests(unittest.TestCase):
    def test_imports_saved_scene_as_physical_rigid_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            obj = root / "cube.obj"
            obj.write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                "f 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n",
                encoding="ascii",
            )
            scene = root / "scene.json"
            scene.write_text(
                json.dumps(
                    {
                        "objects_info": {
                            "init_info": {
                                "robot0": {"class_name": "Yam", "args": {"name": "robot0"}},
                                "red_cube_0": {
                                    "class_name": "USDObject",
                                    "args": {
                                        "name": "red_cube_0",
                                        "category": "red_cube",
                                        "usd_path": "cube.obj",
                                        "mass": 0.3,
                                    },
                                },
                            }
                        },
                        "state": {
                            "registry": {
                                "object_registry": {
                                    "red_cube_0": {
                                        "root_link": {
                                            "pos": [0.4, 0.2, 0.8],
                                            "ori": [0, 0, 0, 1],
                                        }
                                    }
                                }
                            }
                        },
                        "ground_plane_info": {"position": [0, 0, 0]},
                    }
                ),
                encoding="utf-8",
            )
            imported = SimFoundryMuJoCoImporter(scene, root / "out").import_scene()
            xml = imported.external_mjcf.read_text(encoding="utf-8")
            self.assertIn('<body name="red_cube_0"', xml)
            self.assertIn('<freejoint name="red_cube_0_free"', xml)
            self.assertNotIn("robot0", xml)
            bundle = SceneBundle.from_json(imported.scene_manifest)
            self.assertEqual(bundle.mujoco_xml, imported.external_mjcf)
            semantics = json.loads(imported.semantic_map.read_text())
            self.assertTrue(semantics["locations"]["red_cube_0"]["interactive"])
            self.assertFalse(semantics["locations"]["red_cube_0"]["navigation_target"])
            self.assertTrue(semantics["locations"]["red_cube_0_approach"]["navigation_target"])
            semantic_map = SemanticMapModule.from_json(imported.semantic_map)
            self.assertEqual(semantic_map.resolve("red_cube_0").x, 0.4)
            self.assertIsNotNone(bundle.initial_pose)


if __name__ == "__main__":
    unittest.main()
