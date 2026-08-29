from __future__ import annotations

import base64
import io
from pathlib import Path

from ..modules.camera import CameraFrame
from .g1_mujoco import G1MuJoCoBase


class MuGSCamera:
    """Optional MuGS hybrid camera (MuJoCo foreground + 3DGS background).

    The PLY is visual only. MuJoCo collision geoms and the planning grid remain
    authoritative for contact and safety.
    """

    def __init__(
        self,
        base: G1MuJoCoBase,
        background_ply: str | Path,
        *,
        camera: str,
        width: int = 640,
        height: int = 480,
        robot_geom_names: list[str] | None = None,
        world_T_gs: tuple[tuple[float, ...], ...] | None = None,
    ) -> None:
        try:
            from mugs.sensors import GaussianSensor, GaussianSensorConfig
        except ImportError as exc:
            raise ImportError(
                "MuGS is optional. Clone Renforce-Dynamics/MuGS and install it in the sim environment."
            ) from exc
        ply = Path(background_ply).expanduser().resolve()
        if not ply.exists():
            raise FileNotFoundError(ply)
        self.base = base
        self.camera = camera
        model = base.robot._model
        robot_geom_ids = None
        if model is not None and robot_geom_names is None:
            import mujoco

            pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            robot_geom_ids = []
            for geom_id in range(model.ngeom):
                body_id = int(model.geom_bodyid[geom_id])
                while body_id > 0 and body_id != pelvis:
                    body_id = int(model.body_parentid[body_id])
                if body_id == pelvis:
                    robot_geom_ids.append(geom_id)
        self.sensor = GaussianSensor(
            GaussianSensorConfig(
                width=width,
                height=height,
                background_ply_path=str(ply),
                render_mode="hybrid",
                robot_geom_names=robot_geom_names or [],
                robot_geom_ids=robot_geom_ids,
            )
        )
        if world_T_gs is not None:
            import numpy as np

            transform = np.asarray(world_T_gs, dtype=np.float32).reshape(4, 4)
            original = self.sensor._extract_camera_params

            def aligned(model, data, camera_name):
                params = original(model, data, camera_name)
                params["position"] = transform[:3, :3] @ params["position"] + transform[:3, 3]
                params["rotation_matrix"] = transform[:3, :3] @ params["rotation_matrix"]
                return params

            self.sensor._extract_camera_params = aligned

    def capture(self) -> list[CameraFrame]:
        model, data = self.base.robot._model, self.base.robot._data
        if model is None or data is None:
            raise RuntimeError("MuJoCo is not connected")
        result = self.sensor.render(model, data, self.camera)
        image = result["rgb"] if isinstance(result, dict) else result
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("MuGSCamera needs Pillow") from exc
        output = io.BytesIO()
        Image.fromarray(image).save(output, format="JPEG", quality=88)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return [CameraFrame("g1_head_hybrid_3dgs", f"data:image/jpeg;base64,{encoded}")]
