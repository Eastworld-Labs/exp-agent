"""The arithmetic behind `local_planner`, with nothing installed.

##### EVERY SIGN IN HERE IS ONE THAT DOES NOT LOOK WRONG WHEN IT IS WRONG. #####
A flipped bearing puts the goal on the far side of the object; a swapped
rotation puts it in the next room; reading depth as native-endian turns 1.2 m
into 53 m. All three present as "the planner did something odd", which is the
hardest kind of fault to find on a robot and the cheapest kind to pin here.
"""
import importlib.util
import math
import unittest

from system2_agent.grounding import Box
from system2_agent.local_planner import (
    CostGrid,
    CostmapRaycastRanger,
    DepthImage,
    DepthRanger,
    GridUpdate,
    InitPose,
    LocalPlanner,
    NoDepthReturn,
    approach_goal,
    back_off_until_free,
    local_to_map,
    local_to_odom,
    odom_to_local,
    pixel_azimuth,
    standoff_pose,
)
from system2_agent.modules.camera import CameraFrame
from system2_agent.modules.semantic_map import Pose3D
from system2_agent.png16 import decode_png16, encode_png16
from system2_agent.sim.head_camera import D455, HeadCameraSpec

LEVEL = D455
#: The real robot's OTHER camera: the D435i that feeds the costmap, pitched
#: 47.87 degrees down from a measured 1.254 m (g1_auto_navigation
#: docs/DEPTH_OBSTACLES.md). Not what the model looks through, but the geometry
#: has to be right for it or a pitched sensor can never be used.
PITCHED = HeadCameraSpec(
    name="d435i", width=640, height=480, horizontal_fov_deg=69.4, pitch_down_deg=47.87
)


def grid(
    *,
    width=160,
    height=160,
    resolution=0.05,
    origin=(-4.0, -4.0),
    lethal=(),
    unknown=(),
    frame="odom",
    **kwargs,
):
    """A costmap centred on the origin, with named cells made lethal."""
    cost = [0] * (width * height)
    for col, row in lethal:
        cost[row * width + col] = 100
    for col, row in unknown:
        cost[row * width + col] = -1
    return CostGrid(
        frame=frame,
        width=width,
        height=height,
        resolution=resolution,
        origin_x=origin[0],
        origin_y=origin[1],
        cost=cost,
        **kwargs,
    )


def wall(x_m, *, resolution=0.05, origin=(-4.0, -4.0), height=160, thickness=2):
    """Cells forming a wall across the whole grid at a world x."""
    col = int(math.floor((x_m - origin[0]) / resolution))
    return [(col + offset, row) for offset in range(thickness) for row in range(height)]


class GeometryTests(unittest.TestCase):
    def test_a_pixel_right_of_centre_bears_clockwise(self):
        """##### THE SIGN THE WHOLE TOOL TURNS ON. #####

        Optical +X is to the right; the robot's +Y is to the LEFT. Get this
        backwards and every approach mirrors about the robot's nose -- which
        still produces a confident goal, a clean path and a walk to the wrong
        side of the room.
        """
        centre = pixel_azimuth(LEVEL, LEVEL.intrinsics, LEVEL.width / 2, LEVEL.height / 2)
        left = pixel_azimuth(LEVEL, LEVEL.intrinsics, LEVEL.width * 0.25, LEVEL.height / 2)
        right = pixel_azimuth(LEVEL, LEVEL.intrinsics, LEVEL.width * 0.75, LEVEL.height / 2)

        self.assertAlmostEqual(centre, 0.0, places=9)
        self.assertGreater(left, 0.15)
        self.assertLess(right, -0.15)
        self.assertAlmostEqual(left, -right, places=9)

    def test_bearings_do_not_depend_on_the_image_size(self):
        """A box is reported as fractions, and three different resolutions are in
        play (a 480 px preview, a 320 px depth image, the sensor's own grid). The
        same fraction has to mean the same bearing in all of them."""
        full = pixel_azimuth(LEVEL, LEVEL.intrinsics, 0.75 * LEVEL.width, LEVEL.height / 2)
        fx, fy, cx, cy = LEVEL.intrinsics
        half = pixel_azimuth(LEVEL, (fx / 2, fy / 2, cx / 2, cy / 2), 0.75 * 320, 90)

        self.assertAlmostEqual(full, half, places=9)

    def test_a_pitched_camera_sees_the_floor_where_the_robot_docs_say(self):
        """The one independent check available on this arithmetic.

        g1_auto_navigation derived, from a measured 47.87 degree pitch at 1.254 m,
        that the centre of the D435i's frame lands on the floor about 1.13 m
        ahead. That number was computed by another person in another repo from
        the same two measurements, so agreeing with it exercises the optical to
        base rotation end to end rather than against itself.
        """
        fx, fy, cx, cy = PITCHED.intrinsics
        from system2_agent.local_planner import pixel_ray_base

        ray = pixel_ray_base(PITCHED, PITCHED.intrinsics, cx, cy)
        self.assertLess(ray[2], 0.0)  # the centre ray points DOWN
        scale = 1.254 / -ray[2]
        self.assertAlmostEqual(math.hypot(ray[0] * scale, ray[1] * scale), 1.135, places=2)

    def test_bearing_shifts_with_the_row_on_a_pitched_camera(self):
        """⚠️ WHY A FAN OF RAYS MUST SHARE ONE ROW. Once the sensor is tipped, an
        image row is not a constant-bearing line, so sampling each column at a
        different height would smear the fan across bearings it never saw."""
        top = pixel_azimuth(PITCHED, PITCHED.intrinsics, 600, 40)
        bottom = pixel_azimuth(PITCHED, PITCHED.intrinsics, 600, 440)

        self.assertGreater(abs(top - bottom), math.radians(5))


class FrameTests(unittest.TestCase):
    #: A robot facing +y in odom, believed to be somewhere else entirely in map.
    #: Both rotations non-trivial and different, so a transform that happens to
    #: work for the identity cannot pass.
    ODOM = Pose3D(3.0, -2.0, yaw=math.radians(90), frame="odom")
    MAP = Pose3D(10.0, 5.0, yaw=math.radians(-45), frame="map")

    def test_local_and_odom_round_trip_through_a_yawed_pose(self):
        odom = local_to_odom(2.0, 0.5, self.ODOM)

        self.assertAlmostEqual(odom[0], 2.5, places=9)
        self.assertAlmostEqual(odom[1], 0.0, places=9)
        back = odom_to_local(*odom, self.ODOM)
        self.assertAlmostEqual(back[0], 2.0, places=9)
        self.assertAlmostEqual(back[1], 0.5, places=9)

    def test_a_local_pose_becomes_a_map_goal(self):
        goal = local_to_map(2.0, 0.5, 0.3, self.MAP)

        self.assertAlmostEqual(goal.x, 11.767767, places=5)
        self.assertAlmostEqual(goal.y, 3.939340, places=5)
        self.assertAlmostEqual(goal.yaw, math.radians(-45) + 0.3, places=9)
        self.assertEqual(goal.frame, "map")

    def test_on_the_simulator_odom_is_map_and_the_transform_vanishes(self):
        """The sim publishes ground truth on /odom in a frame that IS map, so
        both reads are the same pose and the composition must be the identity --
        not merely close to it."""
        same = Pose3D(4.0, 1.0, yaw=1.1, frame="odom")
        local = odom_to_local(6.0, -0.5, same)
        goal = local_to_map(local[0], local[1], 0.0, same)

        self.assertAlmostEqual(goal.x, 6.0, places=9)
        self.assertAlmostEqual(goal.y, -0.5, places=9)


class CostGridTests(unittest.TestCase):
    def test_a_cell_is_read_row_major_from_the_origin_corner(self):
        cells = grid(width=4, height=3, origin=(0.0, 0.0), resolution=1.0, lethal=[(3, 2)])

        self.assertEqual(cells.cost_at(3.5, 2.5), 100)
        self.assertEqual(cells.cost_at(0.5, 0.5), 0)
        self.assertIsNone(cells.cost_at(-0.5, 0.5))
        self.assertIsNone(cells.cost_at(4.5, 0.5))

    def test_a_ray_stops_at_the_first_lethal_cell(self):
        hit = grid(lethal=wall(2.0)).raycast(0.0, 0.0, 0.0, max_range_m=3.5)

        self.assertEqual(hit.end, "lethal")
        self.assertAlmostEqual(hit.range_m, 2.0, delta=0.05)

    def test_a_ray_passes_through_unknown_and_counts_it(self):
        """The local costmap does not track unknown space on this stack, so a run
        of it means the ray left what the sensors have described. Worth saying,
        not worth stopping on -- stopping would report a wall made of ignorance."""
        unknown = [(col, 80) for col in range(90, 100)]
        hit = grid(lethal=wall(3.0), unknown=unknown).raycast(0.0, 0.0, 0.0, max_range_m=3.5)

        self.assertEqual(hit.end, "lethal")
        self.assertGreater(hit.unknown_crossed, 0)

    def test_a_ray_that_leaves_the_window_says_edge_not_a_range(self):
        hit = grid().raycast(0.0, 0.0, 0.0, max_range_m=99.0)

        self.assertEqual(hit.end, "edge")
        self.assertIsNone(hit.range_m)

    def test_a_ray_that_finds_nothing_in_range_says_so(self):
        hit = grid().raycast(0.0, 0.0, 0.0, max_range_m=1.0)

        self.assertEqual(hit.end, "max_range")
        self.assertIsNone(hit.range_m)

    def test_planning_treats_unknown_as_occupied_and_inflation_as_free(self):
        """99 and 100 forbid a pose; the 1..98 gradient does NOT. Nav2's own
        planners route through inflation, and treating it as a wall would refuse
        every doorway on the robot."""
        cells = grid(width=4, height=1, origin=(0.0, 0.0), resolution=1.0)
        cells = CostGrid(**{**cells.__dict__, "cost": [0, 50, 99, -1]})

        occupied = cells.to_gridmap().occupied

        self.assertNotIn((0, 0), occupied)
        self.assertNotIn((1, 0), occupied)
        self.assertIn((2, 0), occupied)
        self.assertIn((3, 0), occupied)

    def test_a_patch_overwrites_its_rectangle_and_nothing_else(self):
        before = grid(width=4, height=4, origin=(0.0, 0.0), resolution=1.0)
        after = before.with_patch(GridUpdate(x=1, y=2, width=2, height=1, cost=(100, 99)))

        self.assertEqual(after.cost_at(1.5, 2.5), 100)
        self.assertEqual(after.cost_at(2.5, 2.5), 99)
        self.assertEqual(after.cost_at(0.5, 2.5), 0)
        self.assertEqual(after.patches_applied, 1)

    def test_a_patch_that_does_not_fit_is_refused_whole(self):
        """It means this grid and that patch describe different windows -- the
        rolling costmap moved. Half-applying one makes a plausible map of
        nowhere, which is worse than waiting for the next full grid."""
        before = grid(width=4, height=4, origin=(0.0, 0.0), resolution=1.0)

        self.assertIsNone(before.with_patch(GridUpdate(3, 3, 2, 2, (0, 0, 0, 0))))
        self.assertIsNone(before.with_patch(GridUpdate(0, 0, 2, 2, (0, 0))))


class DepthTests(unittest.TestCase):
    def depth(self, *, metres=2.0, region=(0.3, 0.3, 0.7, 0.7), width=320, height=180,
              background=0.0):
        fx, fy, cx, cy = LEVEL.intrinsics
        ratio = width / LEVEL.width
        values = [int(round(background * 1000))] * (width * height)
        col0, col1 = int(region[0] * width), int(region[2] * width)
        row0, row1 = int(region[1] * height), int(region[3] * height)
        for row in range(row0, row1):
            for col in range(col0, col1):
                values[row * width + col] = int(round(metres * 1000))
        return DepthImage(
            width=width,
            height=height,
            fx=fx * ratio,
            fy=fy * ratio,
            cx=cx * ratio,
            cy=cy * ratio,
            depth_mm=values,
        )

    def test_a_flat_surface_ranges_at_its_distance_plus_the_mount_offset(self):
        estimate = DepthRanger().range(Box(0.35, 0.35, 0.65, 0.65), LEVEL, self.depth())

        # The camera sits 5.76 cm forward of base_footprint, so the range from
        # the ROBOT is that much more than the range from the lens. Small, and
        # free to get right; the standoff arithmetic is all robot-relative.
        self.assertAlmostEqual(estimate.range_m, 2.0 + LEVEL.mount_xyz[0], places=2)
        self.assertAlmostEqual(math.degrees(estimate.azimuth_rad), 0.0, delta=1.0)
        self.assertEqual(estimate.method, "depth")
        self.assertGreater(estimate.valid_frac, 0.9)

    def test_the_near_mode_wins_over_the_background_behind_it(self):
        """##### WHY THE 30th PERCENTILE AND NOT THE MEDIAN. ##### A box around a
        real object is bimodal: the object, and whatever shows past its
        silhouette. A median lands between the two the moment the box is more
        than half background, which is a range at which nothing exists."""
        image = self.depth(metres=1.5, region=(0.40, 0.35, 0.56, 0.65), background=3.0)

        estimate = DepthRanger().range(Box(0.35, 0.3, 0.65, 0.7), LEVEL, image)

        self.assertAlmostEqual(estimate.range_m, 1.5 + LEVEL.mount_xyz[0], places=1)

    def test_an_object_left_of_centre_bears_left(self):
        image = self.depth(region=(0.10, 0.4, 0.30, 0.6))

        estimate = DepthRanger().range(Box(0.10, 0.4, 0.30, 0.6), LEVEL, image)

        self.assertGreater(estimate.azimuth_rad, math.radians(10))
        self.assertGreater(estimate.target_base[1], 0.3)

    def test_too_few_returns_is_a_fallback_not_a_measurement(self):
        """Glass, a dark surface, past the working range: the honest answer is
        "this image cannot measure that", which sends the caller to the costmap
        rather than producing a number from four speckles."""
        image = self.depth(region=(0.49, 0.49, 0.51, 0.51))

        with self.assertRaises(NoDepthReturn):
            DepthRanger().range(Box(0.2, 0.2, 0.8, 0.8), LEVEL, image)

    def test_no_return_is_zero_and_zero_is_not_a_range(self):
        image = self.depth(metres=2.0, region=(0.3, 0.3, 0.7, 0.7), background=0.0)

        self.assertEqual(image.at(0, 0), 0.0)
        estimate = DepthRanger().range(Box(0.3, 0.3, 0.7, 0.7), LEVEL, image)
        self.assertGreater(estimate.range_m, 1.9)


class PngTests(unittest.TestCase):
    def test_sixteen_bit_samples_survive_the_wire(self):
        values = [0, 1, 1234, 40000, 65535, 300]

        self.assertEqual(decode_png16(encode_png16(values, 3, 2)), (values, 3, 2))

    def test_a_truncated_or_foreign_png_decodes_to_nothing(self):
        good = encode_png16([1, 2, 3, 4], 2, 2)

        self.assertIsNone(decode_png16(good[:20]))
        self.assertIsNone(decode_png16(b"not a png at all"))

    @unittest.skipUnless(
        importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"),
        "needs OpenCV, which the robot has and this host may not",
    )
    def test_a_png_from_the_robots_own_encoder_decodes_byte_for_byte(self):
        """##### THE DECODER'S REAL COUNTERPARTY IS libpng, NOT ITSELF. #####

        g1_vision encodes depth with `cv2.imencode(".png", uint16)`, whose libpng
        picks a filter per row adaptively and may split the stream across several
        IDAT chunks. A decoder tested only against our own encoder would pass
        while failing on every frame the robot actually sends, so this drives the
        real one -- over noisy data, because a flat image gets the trivial filter
        and proves nothing.
        """
        import cv2
        import numpy as np

        rng = np.random.default_rng(7)
        rows, cols = np.mgrid[0:240, 0:320]
        image = (2500 + rows * 3 + rng.normal(0, 8, (240, 320))).astype(np.uint16)
        image[rng.random((240, 320)) < 0.15] = 0  # no-return speckle
        ok, buffer = cv2.imencode(".png", image)
        self.assertTrue(ok)

        decoded = decode_png16(buffer.tobytes())

        self.assertIsNotNone(decoded, "cv2's PNG must be readable without Pillow")
        values, width, height = decoded
        self.assertEqual((width, height), (320, 240))
        self.assertEqual(values, image.ravel().tolist())

    def test_every_row_filter_is_understood(self):
        """⚠️ THE ENCODER ON THE ROBOT PICKS THE FILTER, NOT US. OpenCV's libpng
        chooses adaptively per row, so a decoder that only handled filter 0 would
        work against our own encoder and fail against the robot's."""
        import struct
        import zlib

        from system2_agent.png16 import _chunk, PNG_MAGIC

        width, height = 4, 5
        rows = bytearray()
        expected = []
        for row in range(height):
            samples = [1000 + row * 10 + col for col in range(width)]
            expected.extend(samples)
            raw = struct.pack(f">{width}H", *samples)
            rows.append(row % 5)  # cycle through all five filter types
            previous = (
                struct.pack(f">{width}H", *[1000 + (row - 1) * 10 + c for c in range(width)])
                if row
                else bytes(width * 2)
            )
            rows.extend(_filter(row % 5, raw, previous))
        png = b"".join(
            (
                PNG_MAGIC,
                _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)),
                _chunk(b"IDAT", zlib.compress(bytes(rows))),
                _chunk(b"IEND", b""),
            )
        )

        self.assertEqual(decode_png16(png), (expected, width, height))


def _filter(kind, raw, previous, bpp=2):
    """Apply one PNG row filter, so the decoder can be tested against all five."""
    out = bytearray(len(raw))
    for index, value in enumerate(raw):
        left = raw[index - bpp] if index >= bpp else 0
        up = previous[index]
        upper_left = previous[index - bpp] if index >= bpp else 0
        if kind == 0:
            out[index] = value
        elif kind == 1:
            out[index] = (value - left) & 0xFF
        elif kind == 2:
            out[index] = (value - up) & 0xFF
        elif kind == 3:
            out[index] = (value - (left + up) // 2) & 0xFF
        else:
            p = left + up - upper_left
            pa, pb, pc = abs(p - left), abs(p - up), abs(p - upper_left)
            best = left if pa <= pb and pa <= pc else (up if pb <= pc else upper_left)
            out[index] = (value - best) & 0xFF
    return bytes(out)


class ApproachTests(unittest.TestCase):
    def test_the_standoff_sits_between_the_robot_and_the_thing(self):
        """⚠️ APPROACHED FROM THE ROBOT'S SIDE, NOT AN ABSOLUTE BEARING. Any other
        side means walking PAST the object through space nothing said was clear."""
        x, y, yaw, distance = standoff_pose((0.0, 0.0), (3.0, 0.0), 0.9)

        self.assertAlmostEqual(x, 2.1, places=9)
        self.assertAlmostEqual(y, 0.0, places=9)
        self.assertAlmostEqual(yaw, 0.0, places=9)
        self.assertAlmostEqual(distance, 3.0, places=9)

    def test_a_standoff_longer_than_the_gap_does_not_walk_backwards(self):
        x, y, _, distance = standoff_pose((0.0, 0.0), (0.5, 0.0), 0.9)

        self.assertAlmostEqual(distance, 0.5, places=9)
        self.assertAlmostEqual(math.hypot(x, y), 0.0, places=9)

    def test_a_far_target_becomes_a_leg_rather_than_a_refusal(self):
        """##### THE COSTMAP IS A WINDOW; THE CAMERA SEES PAST IT. ##### A sink
        across a room is visible, correctly ranged, and further than this planner
        may commit to. Refusing there would be useless, so the goal is clamped
        and the caller is told to come back for the rest."""
        x, y, yaw, distance, final = approach_goal(
            (0.0, 0.0), (5.0, 0.0), standoff_m=0.9, max_leg_m=2.0
        )

        self.assertFalse(final)
        self.assertAlmostEqual(x, 2.0, places=9)
        self.assertAlmostEqual(distance, 5.0, places=9)
        self.assertAlmostEqual(yaw, 0.0, places=9)

    def test_a_reachable_target_is_one_final_leg(self):
        x, _, _, _, final = approach_goal((0.0, 0.0), (2.5, 0.0), standoff_m=0.9, max_leg_m=4.0)

        self.assertTrue(final)
        self.assertAlmostEqual(x, 1.6, places=9)

    def test_a_goal_inside_an_obstacle_is_pulled_back_towards_the_robot(self):
        """The thing being approached IS an obstacle, so a nominal standoff
        landing inside its inflation is the ordinary case."""
        cells = grid(lethal=wall(1.8, thickness=8)).to_gridmap().inflated(0.35)

        x, y, given_up = back_off_until_free(cells, (0.0, 0.0), (1.9, 0.0))

        self.assertGreater(given_up, 0.3)
        self.assertLess(x, 1.5)
        self.assertAlmostEqual(y, 0.0, places=9)
        self.assertNotIn(cells.world_to_cell(x, y), cells.occupied)


class RaycastRangerTests(unittest.TestCase):
    def test_the_fan_finds_a_wall_ahead(self):
        estimate = CostmapRaycastRanger().range(
            Box(0.45, 0.45, 0.55, 0.55), LEVEL, grid(lethal=wall(2.0)), Pose3D(0.0, 0.0)
        )

        self.assertEqual(estimate.method, "costmap_raycast")
        self.assertAlmostEqual(estimate.range_m, 2.0, delta=0.1)
        self.assertEqual(estimate.hits, 9)

    def test_nothing_solid_on_that_bearing_is_a_refusal_with_a_next_step(self):
        with self.assertRaises(ValueError) as caught:
            CostmapRaycastRanger().range(
                Box(0.45, 0.45, 0.55, 0.55), LEVEL, grid(), Pose3D(0.0, 0.0)
            )

        message = str(caught.exception)
        self.assertIn("nothing solid", message)
        self.assertIn("navigate_to", message)


# --------------------------------------------------------------------- fakes --
class FakeCamera:
    def __init__(self, labels=("head_colour",)):
        self.labels = labels

    def capture(self):
        return [CameraFrame(label, "data:image/jpeg;base64,AA==") for label in self.labels]


class FakeGrounder:
    def __init__(self, box=Box(0.35, 0.35, 0.65, 0.65), confidence=0.9):
        from system2_agent.grounding import Grounding

        self.grounding = Grounding(True, box, confidence, "sink", "", 1)
        self.calls = []

    def ground(self, target, frame):
        self.calls.append((target, frame.label))
        return self.grounding


class FakeGrid:
    def __init__(self, value):
        self.value = value

    def grid(self):
        return self.value

    def status(self):
        return {"available": True}


class FakePose:
    def __init__(self, odom=Pose3D(0.0, 0.0, frame="odom"), map_pose=Pose3D(0.0, 0.0)):
        self.value = InitPose(odom=odom, map=map_pose)

    def init_pose(self, expected):
        return self.value


class FakeDepth:
    def __init__(self, image):
        self.image = image

    def capture(self):
        return self.image

    def status(self):
        return {"available": self.image is not None}


def depth_of(metres, *, region=(0.3, 0.3, 0.7, 0.7), width=320, height=180):
    fx, fy, cx, cy = LEVEL.intrinsics
    ratio = width / LEVEL.width
    values = [0] * (width * height)
    for row in range(int(region[1] * height), int(region[3] * height)):
        for col in range(int(region[0] * width), int(region[2] * width)):
            values[row * width + col] = int(round(metres * 1000))
    return DepthImage(width, height, fx * ratio, fy * ratio, cx * ratio, cy * ratio, values)


def planner(**kwargs):
    settings = {
        "camera": FakeCamera(),
        "grounder": FakeGrounder(),
        "grid_source": FakeGrid(grid()),
        "init_pose": FakePose(),
        "geometry": LEVEL,
        "depth": FakeDepth(depth_of(2.0)),
        # ⚠️ PINNED, AND NOT THE PRODUCTION DEFAULT (0.60). The geometry tests
        # below assert distances worked out by hand against 0.90 -- "0.9 m short
        # of a target 2.06 m ahead" -- and those assertions are about the
        # standoff ARITHMETIC, not about which standoff ships. Letting the
        # default leak in here would mean re-deriving four hand-checked numbers
        # every time the default is retuned, which is how a geometry test
        # quietly becomes a test of a constant. The shipped default and its
        # derived floor are pinned by DefaultsTests instead.
        "standoff_m": 0.90,
    }
    settings.update(kwargs)
    return LocalPlanner(**settings)


class PlannerTests(unittest.TestCase):
    def test_a_visible_thing_becomes_a_goal_short_of_it_facing_it(self):
        plan = planner().plan("sink")

        self.assertTrue(plan.final)
        self.assertEqual(plan.range.method, "depth")
        self.assertAlmostEqual(plan.target_local[0], 2.06, delta=0.05)
        # 0.9 m short of a target 2.06 m ahead.
        self.assertAlmostEqual(plan.standoff_local.x, 1.16, delta=0.06)
        self.assertAlmostEqual(math.degrees(plan.standoff_local.yaw), 0.0, delta=2.0)
        self.assertAlmostEqual(plan.remaining_m, 0.0, places=6)

    def test_the_goal_is_expressed_in_map_even_though_planning_was_in_odom(self):
        """The costmap is in odom and a /goal_pose must be in map. Both poses are
        read at one instant so the composition is a rigid transform, and a robot
        whose two frames disagree by 7 m and 90 degrees is the case that catches
        a transform applied in the wrong direction."""
        plan = planner(
            init_pose=FakePose(
                odom=Pose3D(0.0, 0.0, yaw=math.radians(90), frame="odom"),
                map_pose=Pose3D(10.0, 5.0, yaw=math.radians(-45)),
            )
        ).plan("sink")

        # 1.16 m ahead of a robot facing map-frame -45 degrees.
        self.assertAlmostEqual(plan.standoff_map.x, 10.0 + 1.16 * math.cos(math.radians(-45)), delta=0.06)
        self.assertAlmostEqual(plan.standoff_map.y, 5.0 + 1.16 * math.sin(math.radians(-45)), delta=0.06)
        self.assertEqual(plan.standoff_map.frame, "map")

    def test_a_far_target_walks_one_leg_and_asks_to_be_called_again(self):
        far = planner(depth=FakeDepth(depth_of(5.0)), max_leg_m=2.0).plan("sink")

        self.assertFalse(far.final)
        self.assertAlmostEqual(far.leg_m, 2.0, delta=0.05)
        self.assertAlmostEqual(far.remaining_m, 5.06 - 0.9 - 2.0, delta=0.1)
        self.assertIn("call local_planner again", far.next_step)
        self.assertIn("NOT THERE YET", far.next_step)

    def test_a_leg_is_never_longer_than_the_costmap_window(self):
        """A goal on the rim of a rolling window is a goal the window may have
        left behind by the time Nav2 plans to it."""
        small = grid(width=100, height=100)  # 5 x 5 m, like the simulator's
        plan = planner(
            depth=FakeDepth(depth_of(5.0)), grid_source=FakeGrid(small), max_leg_m=4.0
        ).plan("sink")

        self.assertLessEqual(plan.leg_m, small.half_window_m - 0.3)
        self.assertFalse(plan.final)

    def test_standing_already_close_enough_turns_instead_of_walking(self):
        plan = planner(depth=FakeDepth(depth_of(0.7))).plan("sink")

        self.assertTrue(plan.turn_only)
        self.assertTrue(plan.final)
        self.assertAlmostEqual(plan.leg_m, 0.0, places=9)
        self.assertIn("turns to face it", " ".join(plan.notes))

    def test_a_wall_between_robot_and_goal_is_refused_before_anything_moves(self):
        """The goal itself is clear; there is simply no way to it. A* is the
        completeness check, and it runs before a single byte is published."""
        blocked = grid(lethal=wall(1.0, thickness=1))

        with self.assertRaises(ValueError) as caught:
            planner(depth=FakeDepth(depth_of(3.0)), grid_source=FakeGrid(blocked)).plan("sink")

        message = str(caught.exception)
        self.assertIn("no collision-free path", message)
        self.assertIn("navigate_to", message)

    def test_an_obstacle_in_the_way_shortens_the_leg_rather_than_lying(self):
        """A thick wall short of the target swallows the standoff pose, so the
        goal is pulled back to the near side. That is a real, safe leg -- and it
        must NOT be reported as having reached the standoff, or the model would
        stop one obstacle short of the thing and call it arrival."""
        plan = planner(grid_source=FakeGrid(grid(lethal=wall(1.4, thickness=6)))).plan("sink")

        self.assertFalse(plan.final)
        self.assertGreater(plan.remaining_m, 0.0)
        self.assertGreater(plan.given_up_m, 0.0)
        self.assertLess(plan.standoff_local.x, 1.1)
        self.assertFalse(plan.range.agrees)
        self.assertIn("costmap puts the first solid thing", " ".join(plan.notes))

    def test_without_depth_it_falls_back_to_the_costmap_and_says_so(self):
        plan = planner(depth=None, grid_source=FakeGrid(grid(lethal=wall(2.0)))).plan("sink")

        self.assertEqual(plan.range.method, "costmap_raycast")
        self.assertIn("no metric depth", " ".join(plan.notes))
        self.assertEqual(plan.intrinsics_source, "spec_fov")

    def test_depth_over_nothing_falls_back_rather_than_failing(self):
        empty = DepthImage(320, 180, 168.6, 168.6, 160.0, 90.0, [0] * (320 * 180))

        plan = planner(
            depth=FakeDepth(empty), grid_source=FakeGrid(grid(lethal=wall(2.0)))
        ).plan("sink")

        self.assertEqual(plan.range.method, "costmap_raycast")
        self.assertIn("no usable depth", " ".join(plan.notes))

    def test_the_costmap_cross_checks_the_camera(self):
        plan = planner(grid_source=FakeGrid(grid(lethal=wall(2.05)))).plan("sink")

        self.assertEqual(plan.range.method, "depth")
        self.assertTrue(plan.range.agrees)
        self.assertAlmostEqual(plan.range.costmap_check_m, 2.06, delta=0.15)

    def test_a_costmap_in_the_wrong_frame_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            planner(grid_source=FakeGrid(grid(frame="map"))).plan("sink")

        self.assertIn("frame", str(caught.exception))

    def test_no_colour_frame_means_nothing_to_ground(self):
        with self.assertRaises(ValueError) as caught:
            planner(camera=FakeCamera(labels=("head_range",))).plan("sink")

        self.assertIn("head_colour", str(caught.exception))

    def test_a_standoff_inside_the_collision_stop_zone_is_refused(self):
        """0.40 m stop zone plus a 0.10 m goal tolerance plus 0.10 m of margin: a
        goal any closer can be satisfied inside the circle that halts the robot,
        which looks like a walk that stopped for no reason."""
        with self.assertRaises(ValueError) as caught:
            planner().plan("sink", standoff_m=0.3)

        self.assertIn("0.6", str(caught.exception))

    def test_the_refusal_explains_itself_in_clearance_not_just_stop_zones(self):
        """##### THE FAILURE THIS WHOLE PARAMETER EXISTS FOR. #####

        A mission asked to get 0.3 m from a thing, read "minimum 0.65 m", and
        reported the task impossible. It was not: 0.3 m of CLEARANCE is a 0.61 m
        standoff and perfectly legal. So the refusal has to say that the number
        is centre-to-object, what clearance it actually buys, and that
        `clearance_m` is the way to ask -- otherwise a reader retries with 0.31
        and gets refused again, or concludes the robot cannot do it.
        """
        with self.assertRaises(ValueError) as caught:
            planner().plan("sink", standoff_m=0.3)
        message = str(caught.exception)

        self.assertIn("clearance_m", message)
        self.assertIn("CENTRE", message)
        # 0.3 - 0.3079 of body: the object would be INSIDE the robot, and the
        # message must show that rather than describing a tight margin.
        self.assertIn("-0.01", message)

    def test_asking_in_clearance_gets_closer_than_the_standoff_floor_looks(self):
        """0.2 m of clearance is a 0.51 m standoff -- BELOW the 0.60 m floor, so
        it is still refused, but the refusal is now about a real 0.2 m gap rather
        than a number the caller never meant."""
        p = planner()

        self.assertAlmostEqual(p.standoff_for_clearance(0.3), 0.6079, places=4)
        self.assertAlmostEqual(p.clearance_for_standoff(0.60), 0.2921, places=4)
        # The one that has to stay clear of the 0.40 m stop zone.
        self.assertAlmostEqual(p.worst_case_clearance_for_standoff(0.60), 0.1921, places=4)

        # 0.3 m of clearance is legal and lands where the arithmetic says.
        plan = p.plan("sink", clearance_m=0.3)
        self.assertAlmostEqual(plan.standoff_m, 0.6079, places=4)
        self.assertAlmostEqual(plan.body_clearance[0], 0.3, places=4)

    def test_clearance_and_standoff_together_is_refused_rather_than_ranked(self):
        """They measure the same approach from origins 0.31 m apart. Silently
        preferring one would honour a request nobody made."""
        with self.assertRaises(ValueError) as caught:
            planner().plan("sink", standoff_m=0.9, clearance_m=0.3)

        self.assertIn("not both", str(caught.exception))


class DefaultsTests(unittest.TestCase):
    """What actually ships, pinned away from the fixture's hand-checked 0.90.

    These are the numbers a robot walks on, and every one of them is coupled to
    a number in another repo (Nav2's `xy_goal_tolerance`, collision_monitor's
    `StopZone`). A change here that is not matched there is a silent collision
    or a silent refusal, so they get asserted rather than inherited.
    """

    def _bare(self, **kwargs):
        return LocalPlanner(
            camera=FakeCamera(), grounder=FakeGrounder(), grid_source=FakeGrid(grid()),
            init_pose=FakePose(), geometry=LEVEL, depth=FakeDepth(depth_of(2.0)),
            **kwargs,
        )

    def test_the_floor_is_derived_from_the_stop_zone_and_the_arrival_box(self):
        p = self._bare()

        self.assertAlmostEqual(p.standoff_m, 0.60)
        # 0.40 stop zone + 0.10 arrival box + 0.10 margin. NOT a literal.
        self.assertAlmostEqual(p.standoff_min_m, 0.60)
        self.assertAlmostEqual(p.min_clearance_m, 0.2921, places=4)

    def test_a_wider_arrival_box_raises_the_floor_instead_of_being_ignored(self):
        """##### THE COUPLING, ASSERTED. #####

        Under SONIC locomotion Nav2 keeps a 0.25 m box (it commits to 0.8 m/s off
        a 0.15 deadband and would fly through a 0.10 m one). If this planner went
        on believing the box was 0.10 it would promise a clearance the robot does
        not have. So the box is configuration, and the floor moves with it.
        """
        sonic = self._bare(arrival_box_m=0.25)

        self.assertAlmostEqual(sonic.standoff_min_m, 0.75)
        # And the guarantee that matters is IDENTICAL on both backends: the
        # worst-case body clearance at the floor. Only the nominal stand-off
        # differs, which is the honest cost of a backend that cannot creep.
        self.assertAlmostEqual(
            sonic.worst_case_clearance_for_standoff(sonic.standoff_min_m),
            self._bare().worst_case_clearance_for_standoff(0.60),
            places=6,
        )

    def test_an_explicit_floor_still_overrides_the_derivation(self):
        """A robot with a different body is a constructor argument, not a fork."""
        self.assertAlmostEqual(self._bare(standoff_min_m=0.9).standoff_min_m, 0.9)

    def test_a_transport_failure_becomes_a_refusal_not_a_traceback(self):
        """`Tool.run` only catches KeyError, TypeError and ValueError, so anything
        else raised in here would end the mission with a stack trace instead of a
        step the model could react to."""
        class Broken:
            def grid(self):
                raise OSError("broker went away")

            def status(self):
                return {}

        with self.assertRaises(ValueError) as caught:
            planner(grid_source=Broken()).plan("sink")

        self.assertIn("before anything moved", str(caught.exception))
        self.assertIn("OSError", str(caught.exception))

    def test_the_reported_trajectory_starts_under_the_robot(self):
        plan = planner().plan("sink")

        first = plan.trajectory_local[0]
        self.assertAlmostEqual(first[0], 0.0, places=6)
        self.assertAlmostEqual(first[1], 0.0, places=6)
        self.assertGreater(plan.length_m, 1.0)
        self.assertGreater(plan.min_clearance_m, 0.0)
        self.assertEqual(plan.as_json()["trajectory"]["frame"], "local")


if __name__ == "__main__":
    unittest.main()
