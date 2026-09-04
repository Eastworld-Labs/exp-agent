"""The bytes that reach the robot, and the ones that come back."""
import math
import unittest

from system2_agent.g1.wire import (
    PUBLISHABLE,
    SUBSCRIBED,
    decode_bool,
    decode_goal_status,
    decode_pose_stamped,
    downlink,
    image_mime,
    pose_stamped,
    quat_from_yaw,
    uplink,
    yaw_from_quat,
)


class TopicTests(unittest.TestCase):
    def test_topic_layout_matches_the_fleet_manifest(self):
        self.assertEqual(uplink("g1-0001", "/odom"), "g1/g1-0001/ros/odom")
        self.assertEqual(downlink("g1-0001", "/goal_pose"), "g1/g1-0001/cmd/goal_pose")

    def test_the_service_may_publish_goals_and_nothing_else(self):
        """##### THE ACCESS CONTROL, ASSERTED. #####

        /cmd_vel bypasses the costmaps, the collision monitor's stop zones and
        the backend's stale-scan watchdog. /estop is the operator's latch.
        Neither may ever be reachable from a model's tool call, and keeping
        them out of this table is how that is guaranteed rather than promised.
        """
        self.assertEqual(set(PUBLISHABLE), {"/goal_pose"})
        self.assertNotIn("/cmd_vel", PUBLISHABLE)
        self.assertNotIn("/estop", PUBLISHABLE)

    def test_a_goal_expires_rather_than_arriving_late(self):
        self.assertEqual(PUBLISHABLE["/goal_pose"], 3)

    def test_it_subscribes_to_what_it_needs_to_refuse_and_to_verify(self):
        self.assertIn("/localization_3d", SUBSCRIBED)
        self.assertIn("/estop_state", SUBSCRIBED)
        self.assertIn("/goal_status", SUBSCRIBED)


class ClientIdTests(unittest.TestCase):
    def test_each_robot_gets_its_own_mqtt_client_id(self):
        """Two links with one client id take turns being kicked off the broker.
        The service opens one per target in the same millisecond."""
        from system2_agent.g1.link import default_client_id
        real, sim = default_client_id("g1-0001"), default_client_id("g1-sim-0001")
        self.assertNotEqual(real, sim)
        self.assertIn("g1-0001", real)
        self.assertIn("g1-sim-0001", sim)
        self.assertEqual(default_client_id("g1-0001"), real)


class PoseTests(unittest.TestCase):
    def test_yaw_ninety_degrees(self):
        q = quat_from_yaw(math.pi / 2)
        self.assertAlmostEqual(q["z"], 0.7071, places=4)
        self.assertAlmostEqual(q["w"], 0.7071, places=4)
        self.assertEqual((q["x"], q["y"]), (0.0, 0.0))

    def test_quaternion_round_trip(self):
        for yaw in (0.0, 0.5, -1.2, math.pi - 0.01, -math.pi + 0.01):
            self.assertAlmostEqual(yaw_from_quat(quat_from_yaw(yaw)), yaw, places=9)

    def test_pose_stamped_uses_the_ros_field_name_and_a_zero_stamp(self):
        """`nanosec`, not `nsec`. And zero, meaning "use the latest transform":
        this pose is authored now, and a host clock stamp would be wrong by
        whatever the host/robot offset is."""
        msg = pose_stamped(1.77, -15.78, 0.0)
        self.assertEqual(msg["header"]["frame_id"], "map")
        self.assertEqual(msg["header"]["stamp"], {"sec": 0, "nanosec": 0})
        self.assertEqual(msg["pose"]["position"]["x"], 1.77)
        self.assertEqual(msg["pose"]["position"]["z"], 0.0)

    def test_decode_pose_stamped_round_trips(self):
        got = decode_pose_stamped(pose_stamped(2.0, 3.0, 1.0))
        self.assertAlmostEqual(got["x"], 2.0)
        self.assertAlmostEqual(got["yaw"], 1.0, places=6)

    def test_decode_pose_reads_odometry_too(self):
        """nav_msgs/Odometry nests the pose one level deeper. One decoder for
        both shapes is what lets a target be just a topic name."""
        from system2_agent.g1.wire import decode_pose
        odom = {"header": {"frame_id": "odom"},
                "pose": {"pose": {"position": {"x": 1.0, "y": 2.0, "z": 0.0},
                                  "orientation": quat_from_yaw(math.pi / 2)},
                         "covariance": [0.0] * 36}}
        got = decode_pose(odom)
        self.assertEqual((got["x"], got["y"]), (1.0, 2.0))
        self.assertAlmostEqual(got["yaw"], math.pi / 2, places=6)
        self.assertEqual(got["frame"], "odom")

    def test_decode_pose_stamped_returns_none_for_junk(self):
        for junk in (None, {}, {"pose": {}}, {"pose": {"position": {"x": "a", "y": 1}}}):
            self.assertIsNone(decode_pose_stamped(junk))


class DecodeTests(unittest.TestCase):
    def test_absent_bool_is_none_not_false(self):
        """"Nobody is saying whether the E-stop is engaged" is a different fact
        from "it is not", and collapsing them is how a dashboard shows a robot
        as safe because nothing answered."""
        self.assertIsNone(decode_bool(None))
        self.assertIsNone(decode_bool({}))
        self.assertIs(decode_bool({"data": False}), False)
        self.assertIs(decode_bool({"data": True}), True)

    def test_goal_status_is_parsed_out_of_the_string_message(self):
        got = decode_goal_status({"data": '{"state":"succeeded","goal":{"x":1.0}}'})
        self.assertEqual(got["state"], "succeeded")

    def test_goal_status_tolerates_junk_rather_than_raising(self):
        for junk in (None, {}, {"data": "not json"}, {"data": "[1,2]"}):
            self.assertIsNone(decode_goal_status(junk))

    def test_image_mime_reads_the_format_field_rather_than_assuming(self):
        self.assertEqual(image_mime("bgr8; jpeg compressed bgr8"), "image/jpeg")
        self.assertEqual(image_mime("rgb8; png compressed rgb8"), "image/png")
        self.assertIsNone(image_mime("16UC1"))
        self.assertIsNone(image_mime(None))


if __name__ == "__main__":
    unittest.main()
