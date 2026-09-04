"""A forward-facing head depth camera for the simulated G1, shaped like the real one.

The real robot carries an Intel RealSense on its head. g1_auto_navigation's
vision node publishes two ``sensor_msgs/CompressedImage`` previews -- colour,
and a false-colour range image with its legend burned into its own pixels --
plus a health JSON. The fleet agent forwards them over MQTT as CBOR, the
dashboard's head-camera panel renders them, and the mission service's
``MqttHeadCamera`` hands them to the model. This module gives every simulator
the same camera and the same wire:

* :class:`HeadCameraSpec` -- where the camera sits and what it sees. The default
  is a RealSense D455 field of view (87 x 56 degrees at 16:9) at the G1's head
  bracket, pitched level so it faces forward.
* :func:`depth_preview` -- the vision node's colourisation: TURBO over a fixed
  0.3-3.0 m window, red near and blue far, no-return pixels flat grey, and a
  scale bar drawn into the bottom of the picture.
* :class:`HeadCameraBackend` -- a ``CameraBackend`` producing the two frames
  under the labels ``MqttHeadCamera`` uses, colour first.
* :class:`HeadCameraStream` -- publishes both previews and the health JSON on
  the robot's own topics at a steady rate, through a :class:`FramePublisher`
  such as :class:`MqttFramePublisher`. A dashboard opened with
  ``?robot=g1-sim-0001`` and a mission service with ``G1_ROBOT_ID=g1-sim-0001``
  then see the simulator exactly as they would see the robot.

Backends supply a :class:`HeadDepthCamera`: something that renders one colour
image and one metric depth image from the head camera. MuJoCo and Isaac
adapters live next to their simulators.
"""
from __future__ import annotations

import base64
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..modules.camera import CameraFrame

#: Topics the vision node publishes on (g1_vision/grounding_node.py), and the
#: labels the mission service shows the model. ⚠️ COLOUR FIRST -- see g1/camera.py.
COLOUR_TOPIC = "/g1/d435c/preview/compressed"
RANGE_TOPIC = "/g1/d435c/preview_depth/compressed"
STATUS_TOPIC = "/object_grounding_status"

#: MEASURABLE depth, as opposed to the range PREVIEW above. The preview is a
#: Turbo colour map with its legend drawn into it -- evidence for a reader, and
#: nothing a computer can measure. `local_planner` needs metres, so these two
#: carry the millimetres and the lens that produced them.
#:
#: ⚠️ NEUTRAL `head` NAMING, UNLIKE THE `d435c` PREVIEWS ABOVE. Those names are
#: pinned by the dashboard, the fleet manifests and MqttHeadCamera, and on the
#: real robot they will carry a D455's pixels regardless. New topics do not have
#: to inherit that; renaming the old ones is a change across two repos.
DEPTH_TOPIC = "/g1/head/depth/compressed"
DEPTH_INFO_TOPIC = "/g1/head/depth_info"
#: 16-bit single-channel PNG, the one lossless encoding for depth that every
#: image library already reads. ⚠️ NEVER JPEG: a lossy codec on a depth map
#: invents ranges at every edge, exactly where an object's silhouette is.
DEPTH_FORMAT = "16UC1; png"
#: Millimetres per unit, matching `depth_module.depth_units:=0.001` pinned on
#: the robot's RealSense driver. Sent with every info message rather than
#: assumed, because it is a driver setting and not a constant.
DEPTH_SCALE = 0.001
COLOUR_LABEL = "head_colour"
RANGE_LABEL = "head_range"

#: sensor_msgs/CompressedImage.format as the vision node writes it: bare "jpeg".
#: Both previews carry the COLOUR optical frame; depth is aligned to colour.
IMAGE_FORMAT = "jpeg"
FRAME_ID = "d435c_color_optical_frame"

#: The robot id the sim manifest (topics.sim.json) and the dashboard's
#: ``?robot=`` switch already know. The broker ACL lets a ``g1-*`` username
#: publish only under its own id, so this is the MQTT username too.
SIM_ROBOT_ID = "g1-sim-0001"

#: Preview geometry and compression, matching g1_vision's grounding.yaml.
COLOUR_PREVIEW_WIDTH = 480
COLOUR_JPEG_QUALITY = 60
RANGE_PREVIEW_WIDTH = 320
RANGE_JPEG_QUALITY = 55
#: Depth is published smaller and slower than the colour preview: it is ranged,
#: not looked at, and 320 px still spans a 0.3 m object at 3 m across ~15 px.
DEPTH_WIDTH = 320
DEPTH_HZ = 3.0
SCALE_BAR_HEIGHT = 26
NO_RETURN_RGB = (70, 70, 70)


@dataclass(frozen=True)
class HeadCameraSpec:
    """Placement and optics of the simulated head camera.

    ``mount_xyz`` is the optical centre in the mount body's frame. The default
    is the G1's RealSense bracket on ``torso_link`` (robot_class's
    ``G1_DEFAULT_EXTRINSICS``). The real bracket pitches the sensor about 48
    degrees towards the floor; this camera faces forward unless
    ``pitch_down_deg`` says otherwise.
    """

    name: str = "head_d455"
    width: int = 640
    height: int = 360
    #: Horizontal field of view. RealSense D455 depth: 87 degrees.
    horizontal_fov_deg: float = 87.0
    mount_bodies: tuple[str, ...] = ("torso_link", "torso", "pelvis")
    mount_xyz: tuple[float, float, float] = (0.0576235, 0.01753, 0.42987)
    pitch_down_deg: float = 0.0
    #: The false-colour window the vision node uses (depth_preview.py).
    min_range_m: float = 0.3
    max_range_m: float = 3.0
    #: Returns outside the sensor's working range are reported as no return,
    #: as a RealSense does. D455: about 0.4 m to 10 m indoors.
    near_clip_m: float = 0.4
    far_clip_m: float = 10.0
    frame_id: str = FRAME_ID

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("head camera resolution must be positive")
        if not 0 < self.horizontal_fov_deg < 180:
            raise ValueError("horizontal_fov_deg must be in (0, 180)")
        if not -90 <= self.pitch_down_deg <= 90:
            raise ValueError("pitch_down_deg must be within +/-90")
        if not 0 <= self.min_range_m < self.max_range_m:
            raise ValueError("range limits must satisfy 0 <= min < max")
        if not 0 <= self.near_clip_m < self.far_clip_m:
            raise ValueError("clip range must satisfy 0 <= near < far")
        if not self.mount_bodies:
            raise ValueError("at least one mount body is required")

    @property
    def focal_px(self) -> float:
        """Pixel focal length shared by both axes (square pixels)."""
        return (self.width / 2) / math.tan(math.radians(self.horizontal_fov_deg) / 2)

    @property
    def vertical_fov_deg(self) -> float:
        return math.degrees(2 * math.atan((self.height / 2) / self.focal_px))

    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        """(fx, fy, cx, cy) in pixels."""
        return (self.focal_px, self.focal_px, self.width / 2, self.height / 2)

    def mujoco_quat(self) -> tuple[float, float, float, float]:
        """MuJoCo camera quaternion (w, x, y, z) in the mount body frame.

        MuJoCo cameras look down their -Z with +Y up. Right is the mount's -Y;
        the view starts along +X and tips towards -Z as the camera pitches down.
        """
        pitch = math.radians(self.pitch_down_deg)
        cp, sp = math.cos(pitch), math.sin(pitch)
        # Columns: camera x (right), y (up), z (back), in the mount frame.
        return _quaternion_from_columns((0.0, -1.0, 0.0), (sp, 0.0, cp), (-cp, 0.0, sp))

    def usd_focal_length_mm(self, horizontal_aperture_mm: float = 20.955) -> float:
        """USD/Isaac focal length giving this horizontal FOV for the aperture."""
        return horizontal_aperture_mm / (2 * math.tan(math.radians(self.horizontal_fov_deg) / 2))

    def usd_vertical_aperture_mm(self, horizontal_aperture_mm: float = 20.955) -> float:
        return horizontal_aperture_mm * self.height / self.width

    def optical_rotation_in_mount(self) -> tuple[tuple[float, float, float], ...]:
        """Row-major rotation taking optical axes (+X right, +Y down, +Z forward)
        into the mount body frame (+X forward, +Y left, +Z up)."""
        pitch = math.radians(self.pitch_down_deg)
        cp, sp = math.cos(pitch), math.sin(pitch)
        right = (0.0, -1.0, 0.0)
        down = (-sp, 0.0, -cp)
        forward = (cp, 0.0, -sp)
        return tuple((right[axis], down[axis], forward[axis]) for axis in range(3))

    def as_json(self) -> dict:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "hfov_deg": round(self.horizontal_fov_deg, 2),
            "vfov_deg": round(self.vertical_fov_deg, 2),
            "mount_xyz": list(self.mount_xyz),
            "pitch_down_deg": self.pitch_down_deg,
            "range_m": [self.min_range_m, self.max_range_m],
            "clip_m": [self.near_clip_m, self.far_clip_m],
        }


def _quaternion_from_columns(x: tuple[float, ...], y: tuple[float, ...], z: tuple[float, ...]) -> tuple[float, float, float, float]:
    """(w, x, y, z) for the rotation whose columns are the given unit axes."""
    m = ((x[0], y[0], z[0]), (x[1], y[1], z[1]), (x[2], y[2], z[2]))
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)


#: The default camera: a RealSense D455 on the head, facing forward.
D455 = HeadCameraSpec()


@dataclass(frozen=True)
class HeadCameraFrame:
    """One capture: colour (HxWx3 uint8), metric depth (HxW float, metres, NaN or
    non-positive where there is no return), and the capture time (seconds)."""

    rgb: Any
    depth: Any
    stamp_s: float


class HeadDepthCamera(Protocol):
    """A simulator's head camera. ``capture`` renders both images now."""

    spec: HeadCameraSpec

    def capture(self) -> HeadCameraFrame: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------- imaging ----


def _turbo_lut() -> Any:
    """Google's Turbo colormap (fifth-order polynomial fit), 256 RGB entries."""
    import numpy as np

    x = np.linspace(0.0, 1.0, 256)
    r = 0.13572138 + 4.61539260 * x - 42.66032258 * x**2 + 132.13108234 * x**3 - 152.94239396 * x**4 + 59.28637943 * x**5
    g = 0.09140261 + 2.19418839 * x + 4.84296658 * x**2 - 14.18503333 * x**3 + 4.27729857 * x**4 + 2.82956604 * x**5
    b = 0.10667330 + 12.64194608 * x - 60.58204836 * x**2 + 110.36276771 * x**3 - 89.90310912 * x**4 + 27.34824973 * x**5
    return (np.clip(np.stack((r, g, b), axis=-1), 0.0, 1.0) * 255).round().astype(np.uint8)


def colourise_depth(depth: Any, spec: HeadCameraSpec = D455) -> Any:
    """False-colour range image at the depth image's own resolution.

    Metres are normalised over ``[spec.min_range_m, spec.max_range_m]`` and
    the ramp is inverted so RED IS NEAR and blue is far, as on the robot.
    Beyond the window saturates blue; pixels with no return (NaN, non-positive)
    are painted flat grey and never pass through the colormap.
    """
    import numpy as np

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or not depth.size:
        raise ValueError("depth must be a non-empty HxW array")
    valid = np.isfinite(depth) & (depth > 0)
    span = spec.max_range_m - spec.min_range_m
    normalised = np.clip((np.nan_to_num(depth) - spec.min_range_m) / span, 0.0, 1.0)
    code = ((1.0 - normalised) * 255).round().astype(np.intp)
    image = _turbo_lut()[code]
    image[~valid] = NO_RETURN_RGB
    return image


def resize_nearest(image: Any, width: int) -> Any:
    """Nearest-neighbour resize to ``width`` keeping the aspect ratio (numpy only)."""
    import numpy as np

    height, source_width = image.shape[:2]
    if source_width == width:
        return image
    new_height = max(1, int(round(height * width / source_width)))
    rows = (np.arange(new_height) * source_width / width).astype(np.intp)
    cols = (np.arange(width) * source_width / width).astype(np.intp)
    rows = np.clip(rows, 0, height - 1)
    cols = np.clip(cols, 0, source_width - 1)
    return image[rows][:, cols]


def with_scale_bar(image: Any, spec: HeadCameraSpec = D455) -> Any:
    """Append the range legend the robot burns into its depth preview.

    A ``SCALE_BAR_HEIGHT`` strip below the picture: a grey swatch for "no
    return", then the near-to-far ramp, labelled ``--``, ``0.3m`` and
    ``3.0m+``. The scale is in the pixels, so wherever the frame is shown --
    dashboard, model prompt, a saved log -- it explains itself.
    """
    import numpy as np

    height, width = image.shape[:2]
    bar = np.zeros((SCALE_BAR_HEIGHT, width, 3), dtype=np.uint8)
    swatch = max(8, width // 12)
    bar[:, :swatch] = NO_RETURN_RGB
    ramp_width = max(1, width - swatch)
    codes = ((1.0 - np.linspace(0.0, 1.0, ramp_width)) * 255).round().astype(np.intp)
    bar[:, swatch:] = _turbo_lut()[codes]
    bar[0, :] = 0
    out = np.concatenate((image, bar), axis=0)
    labels = (
        (2, "--"),
        (swatch + 3, f"{spec.min_range_m:g}m"),
        (width - 1, f"{spec.max_range_m:g}m+"),
    )
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return out
    canvas = Image.fromarray(out)
    draw = ImageDraw.Draw(canvas)
    top = height + 6
    for x, text in labels:
        anchor = "ra" if x >= width - 1 else "la"
        draw.text((x, top), text, fill=(255, 255, 255), anchor=anchor)
        # A thin dark outline keeps the label readable over any ramp colour.
    return np.asarray(canvas)


def depth_preview(depth: Any, spec: HeadCameraSpec = D455, *, width: int = RANGE_PREVIEW_WIDTH) -> Any:
    """The robot's depth preview: colourise, shrink to ``width``, add the legend."""
    return with_scale_bar(resize_nearest(colourise_depth(depth, spec), width), spec)


def mask_depth(depth: Any, spec: HeadCameraSpec = D455) -> Any:
    """Metric depth with everything outside the sensor's working range set to NaN.

    Renderers report a distance for every pixel, out to their far plane; a
    depth camera does not. Anything nearer than ``near_clip_m`` or farther
    than ``far_clip_m`` becomes "no return", so coverage and the grey pixels in
    the preview mean what they mean on the robot.
    """
    import numpy as np

    depth = np.asarray(depth, dtype=np.float32)
    keep = np.isfinite(depth) & (depth >= spec.near_clip_m) & (depth <= spec.far_clip_m)
    return np.where(keep, depth, np.nan).astype(np.float32)


def depth_coverage(depth: Any) -> float:
    """Fraction of pixels with a return -- the health JSON's ``coverage``."""
    import numpy as np

    depth = np.asarray(depth, dtype=np.float32)
    if not depth.size:
        return 0.0
    return float(np.count_nonzero(np.isfinite(depth) & (depth > 0)) / depth.size)


def encode_jpeg(rgb: Any, *, quality: int = COLOUR_JPEG_QUALITY, width: int | None = None) -> bytes:
    """Encode an HxWx3 uint8 image as JPEG bytes, optionally shrunk to ``width``."""
    import io

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("head camera JPEG encoding needs Pillow: pip install -e '.[sim]'") from exc
    image = Image.fromarray(rgb).convert("RGB")
    if width is not None and image.width != width:
        image = image.resize((width, max(1, round(image.height * width / image.width))), Image.BOX)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality)
    return output.getvalue()


def compressed_image(data: bytes, *, stamp_s: float, frame_id: str = FRAME_ID, fmt: str = IMAGE_FORMAT) -> dict:
    """A ``sensor_msgs/CompressedImage`` as the fleet link carries it.

    Field names verbatim, ``data`` as a plain byte string: that is what the
    fleet agent's CBOR codec produces and what the dashboard decodes.
    """
    seconds = int(math.floor(stamp_s))
    return {
        "header": {
            "stamp": {"sec": seconds, "nanosec": int(round((stamp_s - seconds) * 1e9))},
            "frame_id": frame_id,
        },
        "format": fmt,
        "data": bytes(data),
    }


def downsample_depth(depth: Any, width: int = DEPTH_WIDTH) -> Any:
    """Shrink metric depth by an INTEGER STRIDE, sampling, never averaging.

    ⚠️ AVERAGING DEPTH IS NOT A RESIZE, IT IS AN INVENTION. Interpolating across
    an object's edge produces pixels at ranges nothing occupies -- half way
    between the mug and the wall behind it -- and those are precisely the pixels
    a bounding box's border is full of. Nearest-neighbour keeps every surviving
    value a range something actually returned.
    """
    import numpy as np

    depth = np.asarray(depth)
    if width <= 0 or depth.shape[1] <= width:
        return depth
    stride = max(1, int(depth.shape[1] // width))
    return depth[::stride, ::stride]


def encode_depth_png(depth: Any, *, width: int = DEPTH_WIDTH, scale: float = DEPTH_SCALE) -> bytes:
    """Metric depth (metres, NaN for no return) -> a 16-bit PNG of millimetres.

    ⚠️ ZERO MEANS NO RETURN, NOT ZERO METRES, and that is the convention the
    RealSense driver already publishes 16UC1 with. NaN, non-positive and
    anything past the 65.535 m a uint16 of millimetres can hold all collapse to
    it: a saturated maximum would read as a real wall at the far clip.
    """
    import numpy as np

    from ..png16 import encode_png16

    array = np.asarray(downsample_depth(depth, width), dtype=np.float32)
    units = array / float(scale)
    valid = np.isfinite(units) & (units > 0) & (units <= 65535)
    # ROUND, do not truncate: 1.234 m / 0.001 is 1233.9999 in float32, and
    # truncating turns every measurement into a systematic millimetre short.
    pixels = np.where(valid, np.rint(units), 0).astype(np.uint16)
    return encode_png16(pixels.ravel().tolist(), int(pixels.shape[1]), int(pixels.shape[0]))


def depth_info(
    spec: HeadCameraSpec,
    width: int,
    height: int,
    *,
    stamp_s: float,
    source: str = "sim",
    camera: str = "d455",
    scale: float = DEPTH_SCALE,
) -> dict:
    """The intrinsics of the depth image AS PUBLISHED, not as the sensor has them.

    ⚠️ SCALED BY THE RESIZE. `encode_depth_png` downsamples, so fx, fy, cx and cy
    all shrink with it. Sending the full-size intrinsics with a shrunk image is
    the classic silent error: every deprojected point lands at the right range
    and the wrong bearing, growing with distance from the image centre.
    """
    fx, fy, cx, cy = spec.intrinsics
    x_ratio = width / spec.width
    y_ratio = height / spec.height
    return {
        "width": int(width),
        "height": int(height),
        "fx": round(fx * x_ratio, 4),
        "fy": round(fy * y_ratio, 4),
        "cx": round(cx * x_ratio, 4),
        "cy": round(cy * y_ratio, 4),
        "depth_scale": float(scale),
        "frame_id": spec.frame_id,
        "camera": camera,
        "source": source,
        "stamp_ms": int(round(stamp_s * 1000)),
    }


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


# -------------------------------------------------------------- the model ----


class HeadCameraBackend:
    """Colour and range frames for the model, labelled like the real robot's.

    The mission service shows ``head_colour`` then ``head_range``, at the
    robot's preview sizes; a model that learned the robot's pictures should
    see the simulator's the same way.
    """

    def __init__(
        self,
        camera: HeadDepthCamera,
        *,
        encoder: Callable[..., bytes] = encode_jpeg,
        include_range: bool = True,
    ) -> None:
        self.camera = camera
        self.encoder = encoder
        self.include_range = include_range

    def capture(self) -> list[CameraFrame]:
        frame = self.camera.capture()
        colour = self.encoder(frame.rgb, quality=COLOUR_JPEG_QUALITY, width=COLOUR_PREVIEW_WIDTH)
        frames = [CameraFrame(COLOUR_LABEL, _data_url(colour))]
        if self.include_range:
            preview = depth_preview(frame.depth, self.camera.spec)
            frames.append(CameraFrame(RANGE_LABEL, _data_url(self.encoder(preview, quality=RANGE_JPEG_QUALITY))))
        return frames

    def close(self) -> None:
        self.camera.close()


# ------------------------------------------------------------- the stream ----


class FramePublisher(Protocol):
    """Where messages go. ``publish`` takes the robot-relative ROS topic."""

    def publish(self, ros_topic: str, msg: dict, *, retain: bool = False, qos: int = 0) -> None: ...

    def close(self) -> None: ...


class MqttFramePublisher:
    """paho-mqtt + CBOR onto ``g1/<robot_id>/ros<topic>``, as the fleet agent does.

    ##### CONNECTS AS THE ROBOT. ##### The broker's ACL only lets a ``g1-*``
    username publish telemetry, and only under its own id, so the username IS
    the robot id. That is the same identity the fleet agent uses; do not point
    this at a broker while the real robot of the same id is on it. Keep the id
    ``g1-sim-0001`` unless you know why not.

    This publisher is deliberately separate from ``g1.link.MqttLink``: that link
    is the mission service's operator identity with a fixed allow-list of
    commands, and it must stay unable to publish telemetry.
    """

    def __init__(
        self,
        *,
        broker: str = "127.0.0.1",
        port: int = 1883,
        robot_id: str = SIM_ROBOT_ID,
        client_id: str = "",
    ) -> None:
        if not robot_id.startswith("g1-"):
            raise ValueError("robot_id must start with 'g1-': the broker ACL keys telemetry on it")
        try:
            import cbor2
            import paho.mqtt.client as mqtt
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("head camera streaming needs paho-mqtt and cbor2: pip install -e '.[g1]'") from exc
        self._cbor = cbor2
        self.robot_id = robot_id
        self.broker = f"{broker}:{port}"
        self.connected = False
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id or f"{robot_id}-headcam-{int(time.time() * 1000) % 100000}",
            protocol=mqtt.MQTTv5,
        )
        self._client.username_pw_set(robot_id)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.connect_async(broker, port, keepalive=30)
        self._client.loop_start()

    def _on_connect(self, *_args, **_kwargs) -> None:
        self.connected = True

    def _on_disconnect(self, *_args, **_kwargs) -> None:
        self.connected = False

    def topic(self, ros_topic: str) -> str:
        return f"g1/{self.robot_id}/ros{ros_topic}"

    def publish(self, ros_topic: str, msg: dict, *, retain: bool = False, qos: int = 0) -> None:
        if not self.connected:
            raise ConnectionError(f"no link to the broker at {self.broker}")
        self._client.publish(self.topic(ros_topic), self._cbor.dumps(msg), qos=qos, retain=retain)

    def close(self) -> None:
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:  # noqa: BLE001
            pass


class HeadCameraStream:
    """Publish the head previews and health at a steady rate, from a thread or a tick.

    ``tick`` renders and publishes only when a frame is due, so a simulator
    that must render on its own thread (Isaac) calls it from its step loop,
    while one that can render anywhere (MuJoCo, under the base's lock) runs
    :meth:`start` for a background thread. Frames are never retained and a
    frame is never sent twice: silence must mean the camera stopped.
    """

    def __init__(
        self,
        camera: HeadDepthCamera,
        publisher: FramePublisher,
        *,
        hz: float = 6.0,
        status_hz: float = 2.0,
        depth_hz: float = DEPTH_HZ,
        depth_width: int = DEPTH_WIDTH,
        source: str = "sim",
        encoder: Callable[..., bytes] = encode_jpeg,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if hz <= 0 or status_hz <= 0:
            raise ValueError("stream rates must be positive")
        if depth_hz < 0:
            raise ValueError("depth rate cannot be negative")
        self.camera = camera
        self.publisher = publisher
        self.interval_s = 1.0 / hz
        self.status_interval_s = 1.0 / status_hz
        # Metric depth is published SLOWER than the previews and on its own
        # clock: it is 16-bit and lossless, so it costs several times a preview
        # per frame, and nothing reads it between tool calls. 0 turns it off.
        self.depth_interval_s = (1.0 / depth_hz) if depth_hz else 0.0
        self.depth_hz = depth_hz
        self.depth_width = depth_width
        self.max_hz = hz
        self.source = source
        self.encoder = encoder
        self._clock = clock
        self._monotonic = monotonic
        self._started = monotonic()
        self._next_frame = 0.0
        self._next_status = 0.0
        self._next_depth = 0.0
        self._recent: list[float] = []
        self.frames = 0
        self.errors = 0
        self.last_error = ""
        self.last_colour_bytes = 0
        self.last_range_bytes = 0
        self.last_lag_ms = 0.0
        self.last_coverage = 0.0
        self.depth_frames = 0
        self.last_depth_bytes = 0
        self.last_depth_size = (0, 0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ frames --
    def publish_once(self) -> None:
        began = self._monotonic()
        frame = self.camera.capture()
        stamp = frame.stamp_s if frame.stamp_s else self._clock()
        spec = self.camera.spec
        colour = self.encoder(frame.rgb, quality=COLOUR_JPEG_QUALITY, width=COLOUR_PREVIEW_WIDTH)
        range_jpeg = self.encoder(depth_preview(frame.depth, spec), quality=RANGE_JPEG_QUALITY)
        self.publisher.publish(COLOUR_TOPIC, compressed_image(colour, stamp_s=stamp, frame_id=spec.frame_id))
        self.publisher.publish(RANGE_TOPIC, compressed_image(range_jpeg, stamp_s=stamp, frame_id=spec.frame_id))
        now = self._monotonic()
        # ⚠️ THE SAME `frame`, NOT A SECOND CAPTURE. Rendering again for the
        # metric copy would put the previews and the measurement a frame apart,
        # and the box a model drew on one would be ranged against the other.
        if self.depth_interval_s and now >= self._next_depth:
            self._next_depth = now + self.depth_interval_s
            self.publish_depth(frame.depth, stamp)
        self.frames += 1
        self.last_colour_bytes = len(colour)
        self.last_range_bytes = len(range_jpeg)
        self.last_lag_ms = (now - began) * 1000.0
        self.last_coverage = depth_coverage(frame.depth)
        self._recent.append(now)
        self._recent = [t for t in self._recent if now - t <= 2.0]

    def publish_depth(self, depth: Any, stamp_s: float) -> None:
        """Metric millimetres plus the intrinsics that read them."""
        spec = self.camera.spec
        png = encode_depth_png(depth, width=self.depth_width)
        self.publisher.publish(
            DEPTH_TOPIC,
            compressed_image(png, stamp_s=stamp_s, frame_id=spec.frame_id, fmt=DEPTH_FORMAT),
        )
        import numpy as np

        height, width = np.asarray(downsample_depth(depth, self.depth_width)).shape[:2]
        # Retained, like the robot's own latched CameraInfo: a consumer that
        # connects between frames must be able to read the lens immediately, or
        # it drops depth frames it could have used.
        self.publisher.publish(
            DEPTH_INFO_TOPIC,
            {"data": json.dumps(depth_info(spec, width, height, stamp_s=stamp_s, source=self.source))},
            retain=True,
            qos=1,
        )
        self.depth_frames += 1
        self.last_depth_bytes = len(png)
        self.last_depth_size = (int(width), int(height))

    def measured_hz(self) -> float:
        if len(self._recent) < 2:
            return float(len(self._recent)) / 2.0
        return (len(self._recent) - 1) / max(1e-6, self._recent[-1] - self._recent[0])

    # ------------------------------------------------------------ health --
    def status(self) -> dict:
        """The vision node's health JSON, with the keys the dashboard reads."""
        spec = self.camera.spec
        hz = round(self.measured_hz(), 2)
        colour = {
            "enabled": True,
            "topic": COLOUR_TOPIC,
            "published": self.frames,
            "hz": hz,
            "lag_ms": round(self.last_lag_ms, 1),
            "last_bytes": self.last_colour_bytes,
            "max_hz": self.max_hz,
            "width": COLOUR_PREVIEW_WIDTH,
            "jpeg_quality": COLOUR_JPEG_QUALITY,
            "rate_capped": False,
            "encode_errors": self.errors,
        }
        depth = {
            **colour,
            "topic": RANGE_TOPIC,
            "last_bytes": self.last_range_bytes,
            "width": RANGE_PREVIEW_WIDTH,
            "jpeg_quality": RANGE_JPEG_QUALITY,
            "range_m": [spec.min_range_m, spec.max_range_m],
            "coverage": round(self.last_coverage, 3),
        }
        metric = {
            "enabled": bool(self.depth_interval_s),
            "topic": DEPTH_TOPIC,
            "info_topic": DEPTH_INFO_TOPIC,
            "published": self.depth_frames,
            "max_hz": self.depth_hz,
            "last_bytes": self.last_depth_bytes,
            "size": list(self.last_depth_size),
            "format": DEPTH_FORMAT,
            "depth_scale": DEPTH_SCALE,
        }
        return {
            "stamp_ms": int(self._clock() * 1000),
            "uptime_s": round(self._monotonic() - self._started, 1),
            "source": self.source,
            "model": None,
            "frame_id": spec.frame_id,
            "grounder": {"alive": False, "device": None, "error": "no grounder in simulation", "loading": False},
            "preview": colour,
            "depth_preview": depth,
            "metric_depth": metric,
            "last_target": None,
            "camera": spec.as_json(),
        }

    def publish_status(self) -> None:
        # std_msgs/String, retained like the manifest says: health is a state.
        self.publisher.publish(STATUS_TOPIC, {"data": json.dumps(self.status())}, retain=True, qos=1)

    # -------------------------------------------------------------- loop --
    def tick(self) -> bool:
        """Publish whatever is due. Returns whether a frame went out.

        A failing render or publish is counted and reported, not raised: the
        stream is a window onto the mission, and the mission must not end
        because the window blinked.
        """
        now = self._monotonic()
        sent = False
        if now >= self._next_frame:
            self._next_frame = now + self.interval_s
            try:
                self.publish_once()
                sent = True
            except Exception as exc:  # noqa: BLE001 - see docstring
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
        if now >= self._next_status:
            self._next_status = now + self.status_interval_s
            try:
                self.publish_status()
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"
        return sent

    def run(self) -> None:
        """Publish until :meth:`close`; the body of the background thread.

        The camera is closed here, on this thread, because a renderer's GL
        context belongs to the thread that made it.
        """
        try:
            while not self._stop.is_set():
                self.tick()
                remaining = min(self._next_frame, self._next_status) - self._monotonic()
                if remaining > 0:
                    self._stop.wait(min(remaining, self.interval_s))
        finally:
            self.camera.close()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self.run, name="head-camera-stream", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        try:
            if self._thread is not None:
                self._thread.join(timeout=5.0)
                self._thread = None
            else:
                # Tick-driven: the caller's thread did the rendering, so it
                # is the right one to release the renderer.
                self.camera.close()
        finally:
            self.publisher.close()

    def summary(self) -> dict:
        return {
            "frames": self.frames,
            "depth_frames": self.depth_frames,
            "errors": self.errors,
            "last_error": self.last_error,
            "hz": round(self.measured_hz(), 2),
        }


__all__ = [
    "COLOUR_LABEL",
    "COLOUR_TOPIC",
    "D455",
    "DEPTH_FORMAT",
    "DEPTH_INFO_TOPIC",
    "DEPTH_SCALE",
    "DEPTH_TOPIC",
    "DEPTH_WIDTH",
    "depth_info",
    "downsample_depth",
    "encode_depth_png",
    "FRAME_ID",
    "FramePublisher",
    "HeadCameraBackend",
    "HeadCameraFrame",
    "HeadCameraSpec",
    "HeadCameraStream",
    "HeadDepthCamera",
    "IMAGE_FORMAT",
    "MqttFramePublisher",
    "RANGE_LABEL",
    "RANGE_TOPIC",
    "SIM_ROBOT_ID",
    "STATUS_TOPIC",
    "colourise_depth",
    "compressed_image",
    "depth_coverage",
    "depth_preview",
    "encode_jpeg",
    "mask_depth",
    "resize_nearest",
    "with_scale_bar",
]
