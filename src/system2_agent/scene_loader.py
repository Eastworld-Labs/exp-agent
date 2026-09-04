from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scene_bundle import SceneBundle
from .sim.head_camera import HeadCameraSpec


@dataclass
class LoadedPhysicsScene:
    """A runtime-composed MuJoCo model and the temporary storage it owns."""

    model_path: Path
    _temporary_directory: tempfile.TemporaryDirectory[str] | None = None

    def close(self) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def __enter__(self) -> "LoadedPhysicsScene":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class SceneLoader:
    """Compose immutable robot MJCF with independently supplied scene assets.

    The robot model is never edited. An external MJCF fragment and/or raw
    collision mesh is attached with MuJoCo's MjSpec API, validated, and written
    into a private temporary directory for the simulator to load. Gaussian
    splats stay in SceneBundle as a render layer and do not masquerade as
    collision geometry. A head depth camera can be attached the same way, so a
    robot MJCF that ships without cameras still gets the D455 the real G1 has.
    """

    def __init__(self, robot_scene: str | Path) -> None:
        self.robot_scene = Path(robot_scene).expanduser().resolve()
        if not self.robot_scene.exists():
            raise FileNotFoundError(f"robot MJCF does not exist: {self.robot_scene}")

    def load(
        self, scene: SceneBundle, *, head_camera: HeadCameraSpec | None = None
    ) -> LoadedPhysicsScene:
        if scene.mujoco_xml is None and scene.collision_mesh is None and head_camera is None:
            return LoadedPhysicsScene(self.robot_scene)

        try:
            import mujoco
        except ImportError as exc:
            raise ImportError("SceneLoader requires MuJoCo") from exc

        composed = mujoco.MjSpec.from_file(str(self.robot_scene))
        self._absolutize_assets(composed)
        if head_camera is not None:
            self.attach_head_camera(composed, head_camera)
        if scene.mujoco_xml is not None:
            external = mujoco.MjSpec.from_file(str(scene.mujoco_xml))
            self._absolutize_assets(external)
            root = composed.worldbody.add_frame(name="external_scene_root")
            composed.attach(external, prefix="external_scene/", frame=root)
        if scene.collision_mesh is not None:
            mesh_xml = self._collision_mesh_xml(scene)
            collision = mujoco.MjSpec.from_string(mesh_xml)
            root = composed.worldbody.add_frame(name="external_collision_root")
            composed.attach(collision, prefix="external_collision/", frame=root)

        temporary = tempfile.TemporaryDirectory(prefix="exp-agent-scene-")
        output = Path(temporary.name) / "composed_scene.xml"
        try:
            composed.compile()
            composed.to_file(str(output))
            # Fail at the scene boundary, before starting SONIC or a renderer.
            mujoco.MjModel.from_xml_path(str(output))
        except Exception:
            temporary.cleanup()
            raise
        return LoadedPhysicsScene(output, temporary)

    @staticmethod
    def attach_head_camera(spec: Any, camera: HeadCameraSpec) -> Any:
        """Add ``camera`` to the first of its mount bodies present in ``spec``.

        The camera sits at the mount offset, faces the body's +X (pitched down
        by the spec), and gets the spec's vertical FOV and resolution; the
        offscreen buffer is widened to fit it. Returns the new MjsCamera.
        """
        body = None
        for name in camera.mount_bodies:
            body = SceneLoader._find_body(spec, name)
            if body is not None:
                break
        if body is None:
            names = sorted(str(b.name) for b in getattr(spec, "bodies", ()))
            raise ValueError(
                f"head camera mount: none of {list(camera.mount_bodies)} is a body in "
                f"the robot MJCF (bodies: {names[:40]})"
            )
        mounted = body.add_camera()
        mounted.name = camera.name
        mounted.pos = list(camera.mount_xyz)
        mounted.quat = list(camera.mujoco_quat())
        mounted.fovy = camera.vertical_fov_deg
        mounted.resolution = [camera.width, camera.height]
        offscreen = spec.visual.global_
        offscreen.offwidth = max(int(offscreen.offwidth), camera.width)
        offscreen.offheight = max(int(offscreen.offheight), camera.height)
        return mounted

    @staticmethod
    def _find_body(spec: Any, name: str) -> Any:
        """MjSpec body lookup across MuJoCo versions (``body`` vs ``find_body``)."""
        for attribute in ("body", "find_body"):
            lookup = getattr(spec, attribute, None)
            if lookup is None:
                continue
            try:
                found = lookup(name)
            except Exception:  # noqa: BLE001 - some versions raise on a miss
                found = None
            if found is not None:
                return found
        return None

    @staticmethod
    def _absolutize_assets(spec: object) -> None:
        """Keep asset references valid after writing the composed temp XML."""
        model_dir = Path(str(spec.modelfiledir))

        def absolute(value: str, subdirectory: str) -> str:
            path = Path(value)
            if path.is_absolute():
                return str(path)
            return str((model_dir / subdirectory / path).resolve())

        mesh_dir = str(spec.meshdir)
        for mesh in spec.meshes:
            if mesh.file:
                mesh.file = absolute(str(mesh.file), mesh_dir)
        texture_dir = str(spec.texturedir)
        for texture in spec.textures:
            if texture.file:
                texture.file = absolute(str(texture.file), texture_dir)
            if texture.cubefiles:
                texture.cubefiles = [
                    absolute(str(path), texture_dir) for path in texture.cubefiles
                ]
        for heightfield in spec.hfields:
            if heightfield.file:
                heightfield.file = absolute(str(heightfield.file), "")
        spec.meshdir = ""
        spec.texturedir = ""

    @staticmethod
    def _collision_mesh_xml(scene: SceneBundle) -> str:
        assert scene.collision_mesh is not None
        scale = " ".join(str(value) for value in scene.collision_mesh_scale)
        position = " ".join(str(value) for value in scene.collision_mesh_position)
        quaternion = " ".join(str(value) for value in scene.collision_mesh_quaternion)
        # Paths in XML attributes cannot contain an unescaped quote. Rejecting
        # one is clearer than producing a malformed runtime model.
        mesh_path = str(scene.collision_mesh)
        if '"' in mesh_path:
            raise ValueError("collision mesh path cannot contain a quote")
        return f'''<mujoco model="external collision mesh">
  <asset><mesh name="scene_mesh" file="{mesh_path}" scale="{scale}"/></asset>
  <worldbody>
    <geom name="scene_collision" type="mesh" mesh="scene_mesh"
          pos="{position}" quat="{quaternion}" rgba="0.65 0.65 0.65 1"/>
  </worldbody>
</mujoco>'''
