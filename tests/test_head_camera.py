import importlib.util
import json
import math
import unittest

from system2_agent.sim.head_camera import (
    COLOUR_LABEL,
    COLOUR_TOPIC,
    D455,
    DEPTH_FORMAT,
    DEPTH_INFO_TOPIC,
    DEPTH_TOPIC,
    FRAME_ID,
    RANGE_LABEL,
    RANGE_TOPIC,
    SCALE_BAR_HEIGHT,
    SIM_ROBOT_ID,
    STATUS_TOPIC,
    HeadCameraBackend,
    HeadCameraFrame,
    HeadCameraSpec,
    HeadCameraStream,
    MqttFramePublisher,
    compressed_image,
)
from system2_agent.png16 import decode_png16
from system2_agent.sim.isaac import rotation_from_quaternion

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def _rotate(quaternion, vector):
    rotation = rotation_from_quaternion(quaternion)
    return tuple(sum(rotation[row][col] * vector[col] for col in range(3)) for row in range(3))


class FakePublisher:
    def __init__(self) -> None:
        self.sent = []
        self.closed = False

    def publish(self, ros_topic, msg, *, retain=False, qos=0):
        self.sent.append((ros_topic, msg, retain, qos))

    def close(self):
        self.closed = True


def fake_encoder(image, **_kwargs):
    # A JPEG SOI marker followed by the image shape: enough for the plumbing.
    return b"\xff\xd8" + repr(tuple(image.shape)).encode("ascii")


class HeadCameraSpecTests(unittest.TestCase):
    def test_d455_field_of_view_is_87_by_about_56_degrees(self):
        self.assertEqual(D455.horizontal_fov_deg, 87.0)
        self.assertAlmostEqual(D455.vertical_fov_deg, 56.2, delta=0.3)
        fx, fy, cx, cy = D455.intrinsics
        self.assertAlmostEqual(fx, fy)
        self.assertAlmostEqual(math.degrees(2 * math.atan(cx / fx)), 87.0, places=6)
        self.assertEqual((cx, cy), (320.0, 180.0))

    def test_forward_facing_camera_looks_along_mount_x(self):
        quaternion = D455.mujoco_quat()
        # MuJoCo/USD cameras look down -Z with +Y up.
        forward = _rotate(quaternion, (0.0, 0.0, -1.0))
        up = _rotate(quaternion, (0.0, 1.0, 0.0))
        right = _rotate(quaternion, (1.0, 0.0, 0.0))
        for actual, expected in ((forward, (1, 0, 0)), (up, (0, 0, 1)), (right, (0, -1, 0))):
            for a, e in zip(actual, expected):
                self.assertAlmostEqual(a, e, places=9)

    def test_pitch_tips_the_view_towards_the_floor(self):
        tilted = HeadCameraSpec(pitch_down_deg=30.0)
        forward = _rotate(tilted.mujoco_quat(), (0.0, 0.0, -1.0))
        self.assertAlmostEqual(forward[0], math.cos(math.radians(30)), places=9)
        self.assertAlmostEqual(forward[2], -math.sin(math.radians(30)), places=9)
        optical = tilted.optical_rotation_in_mount()
        optical_forward = tuple(optical[row][2] for row in range(3))
        for a, e in zip(optical_forward, forward):
            self.assertAlmostEqual(a, e, places=9)

    def test_usd_focal_length_reproduces_the_fov(self):
        aperture = 20.955
        focal = D455.usd_focal_length_mm(aperture)
        self.assertAlmostEqual(math.degrees(2 * math.atan(aperture / (2 * focal))), 87.0, places=9)
        self.assertAlmostEqual(D455.usd_vertical_aperture_mm(aperture) / aperture, 360 / 640)

    def test_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            HeadCameraSpec(horizontal_fov_deg=180.0)
        with self.assertRaises(ValueError):
            HeadCameraSpec(pitch_down_deg=120.0)
        with self.assertRaises(ValueError):
            HeadCameraSpec(min_range_m=3.0, max_range_m=1.0)
        with self.assertRaises(ValueError):
            HeadCameraSpec(near_clip_m=11.0, far_clip_m=10.0)


@unittest.skipUnless(HAS_NUMPY, "head camera imaging needs NumPy")
class DepthPreviewTests(unittest.TestCase):
    def test_colourisation_is_red_near_blue_far_grey_nothing(self):
        import numpy as np

        from system2_agent.sim.head_camera import NO_RETURN_RGB, colourise_depth

        depth = np.array([[0.3, 2.6, 3.0, 9.0, float("nan"), 0.0]], dtype=np.float32)
        image = colourise_depth(depth, D455)

        near, far, limit, beyond, nothing, zero = image[0]
        self.assertGreater(int(near[0]), int(near[2]) + 100)  # red up close
        self.assertGreater(int(far[2]), int(far[0]) + 100)  # blue at range
        self.assertEqual(beyond.tolist(), limit.tolist())  # saturates: "3.0m+"
        self.assertNotEqual(limit.tolist(), far.tolist())  # the ramp keeps going to the limit
        self.assertEqual(nothing.tolist(), list(NO_RETURN_RGB))
        self.assertEqual(zero.tolist(), list(NO_RETURN_RGB))

    def test_preview_is_320_wide_with_the_legend_appended(self):
        import numpy as np

        from system2_agent.sim.head_camera import depth_preview

        depth = np.full((360, 640), 1.0, dtype=np.float32)
        preview = depth_preview(depth, D455)

        self.assertEqual(preview.shape, (180 + SCALE_BAR_HEIGHT, 320, 3))
        self.assertEqual(preview.dtype, np.uint8)
        # The legend's grey swatch sits bottom-left; the picture above it is coloured.
        self.assertEqual(preview[-1, 0].tolist(), [70, 70, 70])
        self.assertNotEqual(preview[0, 0].tolist(), [70, 70, 70])

    def test_mask_depth_applies_the_sensor_working_range(self):
        import numpy as np

        from system2_agent.sim.head_camera import depth_coverage, mask_depth

        depth = np.array([[0.2, 0.5, 9.0, 12.0, float("inf")]], dtype=np.float32)
        masked = mask_depth(depth, D455)

        self.assertTrue(np.isnan(masked[0, 0]))  # nearer than 0.4 m
        self.assertEqual(masked[0, 1], np.float32(0.5))
        self.assertEqual(masked[0, 2], np.float32(9.0))
        self.assertTrue(np.isnan(masked[0, 3]))  # beyond 10 m
        self.assertTrue(np.isnan(masked[0, 4]))
        self.assertAlmostEqual(depth_coverage(masked), 0.4)


class WireTests(unittest.TestCase):
    def test_compressed_image_matches_the_fleet_codec_shape(self):
        msg = compressed_image(b"\xff\xd8\xff\xd9", stamp_s=1725400000.25)

        self.assertEqual(msg["format"], "jpeg")
        self.assertEqual(msg["header"]["frame_id"], FRAME_ID)
        self.assertEqual(msg["header"]["stamp"], {"sec": 1725400000, "nanosec": 250000000})
        self.assertIsInstance(msg["data"], bytes)

    @unittest.skipUnless(importlib.util.find_spec("cbor2"), "needs cbor2")
    def test_compressed_image_round_trips_through_cbor(self):
        import cbor2

        msg = compressed_image(b"\xff\xd8\xff\xd9", stamp_s=12.5)
        decoded = cbor2.loads(cbor2.dumps(msg))

        self.assertEqual(decoded, msg)
        self.assertIsInstance(decoded["data"], bytes)

    def test_publisher_topics_follow_the_fleet_layout_and_acl(self):
        publisher = object.__new__(MqttFramePublisher)
        publisher.robot_id = SIM_ROBOT_ID

        self.assertEqual(
            publisher.topic(COLOUR_TOPIC), "g1/g1-sim-0001/ros/g1/d435c/preview/compressed"
        )
        # The broker only lets a g1-* username publish telemetry; refuse anything else early.
        with self.assertRaisesRegex(ValueError, "g1-"):
            MqttFramePublisher(robot_id="operator-mission")


@unittest.skipUnless(HAS_NUMPY, "head camera imaging needs NumPy")
class StreamTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        depth = np.full((360, 640), np.nan, dtype=np.float32)
        depth[100:260, 200:440] = 1.2
        self.depth = depth
        self.rgb = np.zeros((360, 640, 3), dtype=np.uint8)
        tests = self

        class Camera:
            spec = D455
            captures = 0
            closed = False

            def capture(self):
                self.captures += 1
                return HeadCameraFrame(tests.rgb, tests.depth, 1725400000.0)

            def close(self):
                self.closed = True

        self.camera = Camera()
        self.now = [100.0]

    def _stream(self, publisher, **kwargs):
        return HeadCameraStream(
            self.camera,
            publisher,
            encoder=fake_encoder,
            monotonic=lambda: self.now[0],
            clock=lambda: 1725400000.0,
            **kwargs,
        )

    def test_backend_labels_frames_like_the_mission_service(self):
        frames = HeadCameraBackend(self.camera, encoder=fake_encoder).capture()

        self.assertEqual([frame.label for frame in frames], [COLOUR_LABEL, RANGE_LABEL])
        self.assertTrue(all(frame.url.startswith("data:image/jpeg;base64,") for frame in frames))

    def test_tick_publishes_both_previews_unretained_and_health_retained(self):
        publisher = FakePublisher()
        stream = self._stream(publisher)

        self.assertTrue(stream.tick())

        topics = [(topic, retain, qos) for topic, _, retain, qos in publisher.sent]
        # Colour, then the range PREVIEW, then the METRIC depth and the lens that
        # reads it, then health. The two previews are pictures for the model and
        # are never retained -- silence must mean the camera stopped. The depth
        # intrinsics are retained because they are state: a consumer connecting
        # between frames has to be able to read the lens immediately or it drops
        # depth it could have used.
        self.assertEqual(
            topics,
            [
                (COLOUR_TOPIC, False, 0),
                (RANGE_TOPIC, False, 0),
                (DEPTH_TOPIC, False, 0),
                (DEPTH_INFO_TOPIC, True, 1),
                (STATUS_TOPIC, True, 1),
            ],
        )
        colour, depth = publisher.sent[0][1], publisher.sent[1][1]
        self.assertEqual(colour["format"], "jpeg")
        self.assertEqual(colour["header"]["frame_id"], FRAME_ID)
        self.assertEqual(depth["header"]["frame_id"], FRAME_ID)
        self.assertTrue(depth["data"].startswith(b"\xff\xd8"))

    def test_metric_depth_carries_millimetres_and_its_own_intrinsics(self):
        """##### THE ONE TOPIC local_planner MEASURES WITH. #####

        The range preview above is a colour map; this is the number. It has to
        survive the wire as millimetres, and its intrinsics have to describe the
        SIZE THAT WAS SENT rather than the sensor's native one -- a full-size fx
        with a downsampled image is the classic silent error, every point landing
        at the right range and the wrong bearing.
        """
        publisher = FakePublisher()
        stream = self._stream(publisher)
        stream.tick()

        frame = dict(publisher.sent[2][1])
        self.assertEqual(frame["format"], DEPTH_FORMAT)
        self.assertEqual(frame["header"]["frame_id"], FRAME_ID)
        values, width, height = decode_png16(frame["data"])
        self.assertEqual((width, height), (320, 180))
        # 1.2 m in the block, no return everywhere else -- NOT zero metres.
        self.assertEqual(sorted(set(values)), [0, 1200])
        self.assertEqual(values[(height // 2) * width + width // 2], 1200)

        info = json.loads(publisher.sent[3][1]["data"])
        self.assertEqual((info["width"], info["height"]), (320, 180))
        self.assertEqual(info["depth_scale"], 0.001)
        self.assertEqual(info["frame_id"], FRAME_ID)
        fx, fy, cx, cy = D455.intrinsics
        self.assertAlmostEqual(info["fx"], fx / 2, places=3)
        self.assertAlmostEqual(info["cx"], cx / 2, places=3)
        self.assertAlmostEqual(info["cy"], cy / 2, places=3)

    def test_metric_depth_can_be_turned_off_without_touching_the_previews(self):
        publisher = FakePublisher()
        stream = self._stream(publisher, depth_hz=0)

        stream.tick()

        topics = [topic for topic, *_ in publisher.sent]
        self.assertNotIn(DEPTH_TOPIC, topics)
        self.assertNotIn(DEPTH_INFO_TOPIC, topics)
        self.assertEqual(topics, [COLOUR_TOPIC, RANGE_TOPIC, STATUS_TOPIC])
        self.assertFalse(stream.status()["metric_depth"]["enabled"])

    def test_health_json_carries_the_blocks_the_dashboard_reads(self):
        publisher = FakePublisher()
        stream = self._stream(publisher)
        stream.tick()

        status = json.loads(publisher.sent[-1][1]["data"])
        self.assertEqual(status["frame_id"], FRAME_ID)
        self.assertFalse(status["grounder"]["alive"])
        preview, depth = status["preview"], status["depth_preview"]
        self.assertEqual(preview["topic"], COLOUR_TOPIC)
        self.assertEqual(depth["topic"], RANGE_TOPIC)
        self.assertEqual(depth["range_m"], [0.3, 3.0])
        self.assertAlmostEqual(depth["coverage"], 160 * 240 / (360 * 640), places=3)
        for key in ("enabled", "published", "hz", "lag_ms", "last_bytes", "max_hz", "width", "jpeg_quality", "rate_capped", "encode_errors"):
            self.assertIn(key, preview)
            self.assertIn(key, depth)
        self.assertEqual(preview["width"], 480)
        self.assertEqual(depth["width"], 320)
        metric = status["metric_depth"]
        self.assertTrue(metric["enabled"])
        self.assertEqual(metric["topic"], DEPTH_TOPIC)
        self.assertEqual(metric["info_topic"], DEPTH_INFO_TOPIC)
        self.assertEqual(metric["format"], DEPTH_FORMAT)
        self.assertEqual(metric["size"], [320, 180])

    def test_frames_are_rate_limited_and_never_resent(self):
        publisher = FakePublisher()
        stream = self._stream(publisher, hz=5.0, status_hz=1.0)

        self.assertTrue(stream.tick())
        self.assertFalse(stream.tick())  # same instant: nothing is due
        self.now[0] = 100.1
        self.assertFalse(stream.tick())  # 0.1 s later: still not due at 5 Hz
        self.now[0] = 100.3
        self.assertTrue(stream.tick())

        self.assertEqual(stream.frames, 2)
        self.assertEqual(self.camera.captures, 2)
        previews = [
            topic for topic, *_ in publisher.sent if topic in (COLOUR_TOPIC, RANGE_TOPIC)
        ]
        self.assertEqual(len(previews), 4)

    def test_metric_depth_runs_slower_than_the_previews_on_its_own_clock(self):
        """Depth is lossless and several times the size of a preview, and nothing
        reads it between tool calls. It rides the same CAPTURE -- ranging a frame
        the model never saw would be measuring a different moment -- but not the
        same rate."""
        publisher = FakePublisher()
        # Quarter-second steps: exactly representable, so the two schedules are
        # compared rather than the floating-point accumulation in either.
        stream = self._stream(publisher, hz=4.0, status_hz=1.0, depth_hz=2.0)

        for step in range(4):
            self.now[0] = 100.0 + step * 0.25
            stream.tick()

        self.assertEqual(stream.frames, 4)
        self.assertEqual(stream.depth_frames, 2)
        self.assertEqual(self.camera.captures, 4)

    def test_a_failing_capture_is_counted_not_raised(self):
        publisher = FakePublisher()

        def broken():
            raise RuntimeError("renderer lost its context")

        self.camera.capture = broken
        stream = self._stream(publisher)

        self.assertFalse(stream.tick())
        self.assertEqual(stream.errors, 1)
        self.assertIn("renderer lost its context", stream.last_error)
        # Health still goes out, and says so.
        status = json.loads(publisher.sent[-1][1]["data"])
        self.assertEqual(status["preview"]["encode_errors"], 1)

    def test_close_releases_camera_and_publisher(self):
        publisher = FakePublisher()
        stream = self._stream(publisher)

        stream.close()

        self.assertTrue(self.camera.closed)
        self.assertTrue(publisher.closed)

    def test_threaded_stream_closes_camera_on_its_own_thread(self):
        import threading

        publisher = FakePublisher()
        closed_on = []
        original_close = self.camera.close

        def close():
            closed_on.append(threading.current_thread().name)
            original_close()

        self.camera.close = close
        stream = self._stream(publisher)
        stream.start()
        stream.close()

        self.assertEqual(closed_on, ["head-camera-stream"])
        self.assertTrue(publisher.closed)
        self.assertGreaterEqual(stream.frames, 1)


class CliHelperTests(unittest.TestCase):
    def test_broker_parsing(self):
        from system2_agent.sim_cli import parse_broker

        self.assertEqual(parse_broker("192.168.1.20"), ("192.168.1.20", 1883))
        self.assertEqual(parse_broker("legion:8883"), ("legion", 8883))
        with self.assertRaises(ValueError):
            parse_broker(":1883")
        with self.assertRaises(ValueError):
            parse_broker("host:mqtt")

    def test_head_camera_flags(self):
        import argparse

        from system2_agent.sim_cli import add_head_camera_arguments, head_camera_from_args

        parser = argparse.ArgumentParser()
        add_head_camera_arguments(parser)

        default = head_camera_from_args(parser.parse_args([]))
        self.assertEqual(default, D455)
        tilted = head_camera_from_args(parser.parse_args(["--head-pitch-deg", "20"]))
        self.assertEqual(tilted.pitch_down_deg, 20.0)
        self.assertIsNone(head_camera_from_args(parser.parse_args(["--no-head-camera"])))
        self.assertEqual(parser.parse_args([]).robot_id, SIM_ROBOT_ID)
        with self.assertRaises(SystemExit):
            head_camera_from_args(parser.parse_args(["--no-head-camera", "--stream-mqtt", "h"]))


if __name__ == "__main__":
    unittest.main()
