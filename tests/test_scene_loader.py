import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from system2_agent.scene_bundle import SceneBundle
from system2_agent.scene_loader import SceneLoader
from system2_agent.modules.semantic_map import Pose3D, SemanticMapModule
from system2_agent.sim.head_camera import D455, HeadCameraSpec


class _FakeCamera:
    pass


class _FakeBody:
    def __init__(self, name):
        self.name = name
        self.cameras = []

    def add_camera(self):
        camera = _FakeCamera()
        self.cameras.append(camera)
        return camera


class _FakeGlobal:
    offwidth = 640
    offheight = 480


class _FakeVisual:
    def __init__(self):
        self.global_ = _FakeGlobal()


class _FakeSpec:
    """Enough of MjSpec for attach_head_camera: body lookup, cameras, offscreen size."""

    def __init__(self, *names):
        self.bodies = [_FakeBody(name) for name in names]
        self.visual = _FakeVisual()

    def body(self, name):
        for body in self.bodies:
            if body.name == name:
                return body
        return None


class HeadCameraAttachmentTests(unittest.TestCase):
    def test_attaches_to_the_first_mount_body_present(self):
        spec = _FakeSpec("world", "pelvis", "torso_link")
        camera = HeadCameraSpec(width=1280, height=720)

        mounted = SceneLoader.attach_head_camera(spec, camera)

        torso = spec.body("torso_link")
        self.assertEqual(torso.cameras, [mounted])
        self.assertEqual(spec.body("pelvis").cameras, [])
        self.assertEqual(mounted.name, "head_d455")
        self.assertEqual(mounted.pos, list(camera.mount_xyz))
        self.assertEqual(mounted.quat, list(camera.mujoco_quat()))
        self.assertAlmostEqual(mounted.fovy, camera.vertical_fov_deg)
        self.assertEqual(mounted.resolution, [1280, 720])
        # The offscreen buffer grew to fit the camera.
        self.assertEqual((spec.visual.global_.offwidth, spec.visual.global_.offheight), (1280, 720))

    def test_falls_back_through_the_mount_list_and_names_missing_bodies(self):
        spec = _FakeSpec("world", "pelvis")

        mounted = SceneLoader.attach_head_camera(spec, D455)
        self.assertEqual(spec.body("pelvis").cameras, [mounted])

        with self.assertRaisesRegex(ValueError, "torso_link.*pelvis.*world"):
            SceneLoader.attach_head_camera(_FakeSpec("world"), D455)

    def test_supports_older_mjspec_find_body(self):
        class Older(_FakeSpec):
            find_body = _FakeSpec.body
            body = None

        spec = Older("torso_link")
        SceneLoader.attach_head_camera(spec, D455)
        self.assertEqual(len(spec.find_body("torso_link").cameras), 1)


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
                                    "mount": {
                                        "prim": "/World/G1",
                                        "xyz": [0.05, 0.0, 0.4],
                                        "pitch_down_deg": 10,
                                        "hfov_deg": 87,
                                    },
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
            head = scene.isaac_sim.cameras[0]
            self.assertEqual(head.mount_prim, "/World/G1")
            self.assertEqual(head.mount_xyz, (0.05, 0.0, 0.4))
            self.assertEqual(head.pitch_down_deg, 10.0)
            self.assertEqual(head.horizontal_fov_deg, 87.0)

    def test_mounted_isaac_camera_must_live_under_its_mount_prim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("stage.usda", "grid.json", "semantics.json"):
                (root / name).write_text("{}" if name.endswith(".json") else "#usda 1.0\n")
            manifest = root / "scene.json"
            manifest.write_text(json.dumps({
                "navigation_grid": "grid.json",
                "semantic_map": "semantics.json",
                "isaac_sim": {
                    "stage_usd": "stage.usda",
                    "robot_prim": "/World/G1",
                    "cameras": [{
                        "label": "head",
                        "prim_path": "/World/HeadCamera",
                        "mount": {"prim": "/World/G1/torso_link"},
                    }],
                },
            }))
            with self.assertRaisesRegex(ValueError, "child of mount.prim"):
                SceneBundle.from_json(manifest)

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
