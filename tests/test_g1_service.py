"""The mission service end to end: "go to kitchen" over HTTP, no robot.

A scripted model stands in for the LLM and a fake link for the broker, so this
exercises the real loop, the real gate, the real navigation backend and the real
HTTP surface -- everything except the two things that cost money and move a
robot. It is the closest thing to a mission that is safe to run unattended.
"""
import json
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

from system2_agent.g1.config import ServiceConfig
from system2_agent.g1.service import MissionRunner, serve
from system2_agent.types import AssistantTurn, ToolCall

from test_g1_nav2_backend import FakeLink


class LiveFakeLink(FakeLink):
    """FakeLink stamped with the REAL clock.

    The backend tests drive a fake clock; the service builds its backend the
    ordinary way, with `time.monotonic`. A link whose arrivals were stamped at
    a fixed t=100 would make every message look hours stale to it, and every
    goal would be refused for a reason that has nothing to do with the test.
    """

    @property
    def time(self):
        return time.monotonic()

    @time.setter
    def time(self, _value):
        pass

MAP = "g1_map_small_upright"
SIM_MAP = "sim_house"
SIM_PLACES = {
    "map": SIM_MAP,
    "places": [
        {"name": "kitchen_2", "x": 4.0, "y": 6.0, "yaw": 0,
         "provenance": "derived", "tags": [], "note": ""},
    ],
}
PLACES = {
    "map": MAP,
    "places": [
        {"name": "kitchen", "x": 1.77, "y": -15.78, "yaw": 15,
         "provenance": "measured", "tags": [], "note": ""},
        {"name": "pantry", "x": 0.91, "y": -17.52, "yaw": -170,
         "provenance": "derived", "tags": [], "note": ""},
    ],
}


def call(index, name, **arguments):
    return AssistantTurn(
        content="", tool_calls=(ToolCall(id=f"c{index}", name=name, arguments=arguments),))


class ScriptedModel:
    """Returns pre-written turns. Records what it was asked, so a test can
    assert on the schema the model was shown -- which is where the destination
    enum lives."""

    def __init__(self, turns):
        self.turns = iter(turns)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append((list(messages), list(tools)))
        return next(self.turns)


class Harness:
    """A service on a free port, with a scripted model and a fake link."""

    def __init__(self, turns, *, places=PLACES, camera=False):
        self.dir = TemporaryDirectory()
        maps = Path(self.dir.name) / "maps"
        maps.mkdir()
        (maps / ".map_active.json").write_text(json.dumps({"name": MAP}))
        if places is not None:
            (maps / f"{MAP}.places.json").write_text(json.dumps(places))
        (maps / f"{SIM_MAP}.places.json").write_text(json.dumps(SIM_PLACES))
        self.config = ServiceConfig(
            nav_dir=Path(self.dir.name), camera=camera, gate_timeout_s=5.0,
            max_model_calls=10, wall_clock_s=30.0, api_key="test-key",
            sim_map=SIM_MAP,
            # Port 0: the OS picks a free one, so tests never collide with each
            # other or with a service somebody left running on 8765.
            port=0)
        # One fake link PER ROBOT ID, exactly as the service holds them, so a
        # test can assert that a goal for the simulator never leaves under the
        # real robot's id.
        self.links: dict = {}
        self.link = self.links.setdefault(self.config.robot_id, LiveFakeLink())
        self.link.put_pose(0.0, 0.0)
        self.model = ScriptedModel(turns)

        runner = MissionRunner(
            self.config, lambda robot_id: self.links.setdefault(robot_id, LiveFakeLink()))
        runner._model = lambda: self.model
        self.runner = runner
        self.server = serve(self.config, runner)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def get(self, path):
        with urllib.request.urlopen(f"{self.url}{path}", timeout=5) as r:
            return json.loads(r.read())

    def post(self, path, payload):
        request = urllib.request.Request(
            f"{self.url}{path}", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=5) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    def wait_for(self, predicate, timeout=8.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            state = self.runner.state()
            if predicate(state):
                return state
            time.sleep(0.02)
        raise AssertionError(f"timed out; last state={self.runner.state()}")

    def frames(self):
        return self.runner.log.since(0)

    def close(self):
        self.runner.stop()
        self.server.shutdown()
        self.server.server_close()
        self.dir.cleanup()


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.harness = None

    def tearDown(self):
        if self.harness:
            self.harness.close()

    # ---------------------------------------------------------- the mission --
    def test_go_to_kitchen(self):
        """##### THE WHOLE FEATURE, IN ONE TEST. #####

        A sentence goes in; a map-frame goal goes out on the topic the robot's
        fleet agent republishes; the planner's verdict comes back; the model
        finishes. Everything between the goal and the motors is on the robot.
        """
        self.harness = h = Harness([
            call(1, "list_locations"),
            call(2, "navigate_to", location="kitchen", reason="the mission names it"),
            call(3, "finish", summary="arrived at the kitchen; the planner confirmed it"),
        ])

        # The robot's answer, injected once the goal is on the wire.
        def robot():
            for _ in range(400):
                if h.link.published:
                    h.link.put_pose(1.77, -15.78)
                    h.link.put_status("succeeded", 1.77, -15.78)
                    h.link.messages["/goal_status"] = (
                        h.link.messages["/goal_status"][0], h.link.time + 1)
                    return
                time.sleep(0.01)

        threading.Thread(target=robot, daemon=True).start()
        status, started = h.post("/mission/start", {"mission": "go to kitchen", "gate": "auto"})
        self.assertEqual(status, 200)
        self.assertTrue(started["ok"])

        state = h.wait_for(lambda s: s["status"] == "completed")
        self.assertEqual(state["status"], "completed")

        topic, msg, _ = h.link.published[0]
        self.assertEqual(topic, "/goal_pose")
        self.assertEqual(msg["pose"]["position"]["x"], 1.77)
        self.assertEqual(msg["pose"]["position"]["y"], -15.78)

        kinds = [f["t"] for f in h.frames()]
        self.assertEqual(kinds[0], "mission")
        self.assertEqual(kinds[-1], "end")
        self.assertIn("call", kinds)
        self.assertIn("result", kinds)

        navigation = [f for f in h.frames() if f["t"] == "result" and f["name"] == "navigate_to"]
        self.assertEqual(len(navigation), 1)
        self.assertTrue(navigation[0]["ok"])
        self.assertIn("planner", navigation[0]["summary"])

    def test_the_destination_enum_is_built_from_the_loaded_map(self):
        """A hallucinated place is structurally impossible, not caught after
        the fact -- and the list the model sees is the one on this map."""
        self.harness = h = Harness([call(1, "finish", summary="nothing to do")])
        h.post("/mission/start", {"mission": "hello", "gate": "auto"})
        h.wait_for(lambda s: s["status"] == "completed")
        _, tools = h.model.requests[0]
        navigate = next(t for t in tools if t["function"]["name"] == "navigate_to")
        enum = navigate["function"]["parameters"]["properties"]["location"]["enum"]
        self.assertEqual(enum, ["kitchen", "pantry"])

    def test_an_unknown_place_is_refused_and_nothing_is_published(self):
        self.harness = h = Harness([
            call(1, "navigate_to", location="garage", reason="try it"),
            call(2, "request_human", reason="the garage is not on this map"),
        ])
        h.post("/mission/start", {"mission": "go to the garage", "gate": "auto"})
        state = h.wait_for(lambda s: s["status"] == "needs_human")
        self.assertEqual(state["status"], "needs_human")
        self.assertEqual(h.link.published, [])
        # ⚠️ THE ENUM CAUGHT IT BEFORE THE MAP DID, and the refusal names the
        # destinations that DO exist -- so the model's next move is informed
        # rather than another guess.
        refusal = next(f for f in h.frames() if f["t"] == "result" and not f["ok"])
        self.assertIn("kitchen", refusal["summary"])
        self.assertIn("pantry", refusal["summary"])

    # ------------------------------------------------------------- the gate --
    def test_the_gate_stops_before_the_robot_moves_and_the_answer_releases_it(self):
        self.harness = h = Harness([
            call(1, "navigate_to", location="kitchen", reason="go"),
            call(2, "finish", summary="done"),
        ])

        def robot():
            for _ in range(400):
                if h.link.published:
                    h.link.put_pose(1.77, -15.78)
                    return
                time.sleep(0.01)

        threading.Thread(target=robot, daemon=True).start()
        _, started = h.post("/mission/start", {"mission": "go to kitchen", "gate": "confirm"})
        state = h.wait_for(lambda s: s["pending"] is not None)
        self.assertEqual(state["pending"]["name"], "navigate_to")
        # ##### NOTHING HAS BEEN PUBLISHED WHILE IT WAITS. #####
        self.assertEqual(h.link.published, [])

        status, _ = h.post("/mission/approve",
                           {"id": started["id"], "call_id": state["pending"]["id"], "ok": True})
        self.assertEqual(status, 200)
        h.wait_for(lambda s: s["status"] == "completed")
        self.assertEqual(len(h.link.published), 1)

    def test_declining_is_a_normal_answer_the_model_reads(self):
        self.harness = h = Harness([
            call(1, "navigate_to", location="kitchen", reason="go"),
            call(2, "request_human", reason="the operator declined"),
        ])
        _, started = h.post("/mission/start", {"mission": "go to kitchen", "gate": "confirm"})
        state = h.wait_for(lambda s: s["pending"] is not None)
        h.post("/mission/approve",
               {"id": started["id"], "call_id": state["pending"]["id"], "ok": False})
        h.wait_for(lambda s: s["status"] == "needs_human")
        self.assertEqual(h.link.published, [])
        declined = next(f for f in h.frames() if f["t"] == "gate")
        self.assertEqual(declined["verdict"], "declined")

    def test_approving_a_step_that_is_not_pending_is_refused(self):
        self.harness = h = Harness([call(1, "finish", summary="ok")])
        status, body = h.post("/mission/approve", {"id": "x", "call_id": "nope", "ok": True})
        self.assertEqual(status, 409)
        self.assertFalse(body["ok"])

    def test_stop_settles_a_pending_gate_rather_than_hanging_the_run(self):
        """##### A GATE THAT NEVER SETTLES HANGS A MISSION with the robot's
        next step pending and nothing on screen saying why. #####"""
        self.harness = h = Harness([
            call(1, "navigate_to", location="kitchen", reason="go"),
            call(2, "finish", summary="unreachable"),
        ])
        h.post("/mission/start", {"mission": "go to kitchen", "gate": "confirm"})
        h.wait_for(lambda s: s["pending"] is not None)
        status, body = h.post("/mission/stop", {})
        self.assertEqual(status, 200)
        self.assertIn("DOES NOT STOP THE ROBOT", body["note"])
        state = h.wait_for(lambda s: s["status"] not in ("running",))
        self.assertIn(state["status"], ("cancelled", "needs_human", "completed", "failed"))
        self.assertEqual(h.link.published, [])

    # ------------------------------------------------------------ dry run ---
    def test_dry_run_publishes_nothing_at_all(self):
        """The way to exercise a prompt change hundreds of times for free. The
        real publisher is not merely unused -- it is never constructed."""
        self.harness = h = Harness([
            call(1, "navigate_to", location="kitchen", reason="go"),
            call(2, "finish", summary="dry"),
        ])
        h.post("/mission/start", {"mission": "go to kitchen", "gate": "dry-run"})
        h.wait_for(lambda s: s["status"] == "completed")
        self.assertEqual(h.link.published, [])
        result = next(f for f in h.frames() if f["t"] == "result" and f["name"] == "navigate_to")
        self.assertTrue(result["ok"])

    # ------------------------------------------------------------- the HTTP --
    def test_one_mission_at_a_time(self):
        """Two missions is two planners sharing one set of legs."""
        self.harness = h = Harness([
            call(1, "navigate_to", location="kitchen", reason="go"),
            call(2, "finish", summary="done"),
        ])
        h.post("/mission/start", {"mission": "first", "gate": "confirm"})
        h.wait_for(lambda s: s["pending"] is not None)
        status, body = h.post("/mission/start", {"mission": "second", "gate": "confirm"})
        self.assertEqual(status, 409)
        self.assertIn("already running", body["error"])

    def test_config_reports_the_key_presence_never_the_key(self):
        self.harness = h = Harness([call(1, "finish", summary="ok")])
        config = h.get("/mission/config")
        self.assertTrue(config["keyPresent"])
        self.assertNotIn("test-key", json.dumps(config))
        self.assertEqual(config["map"], MAP)
        self.assertEqual(config["places"], ["kitchen", "pantry"])
        self.assertEqual(config["engine"], "host")

    def test_an_unlabelled_map_is_an_empty_list_not_a_crash(self):
        self.harness = h = Harness([call(1, "finish", summary="ok")], places=None)
        config = h.get("/mission/config")
        self.assertEqual(config["places"], [])
        self.assertTrue(config["placesErrors"])

    def test_a_bad_gate_is_refused(self):
        self.harness = h = Harness([call(1, "finish", summary="ok")])
        status, _ = h.post("/mission/start", {"mission": "x", "gate": "whatever"})
        self.assertEqual(status, 400)

    def test_an_empty_mission_is_refused(self):
        self.harness = h = Harness([call(1, "finish", summary="ok")])
        status, _ = h.post("/mission/start", {"mission": "   ", "gate": "auto"})
        self.assertEqual(status, 400)

    def test_events_replay_from_a_sequence_number(self):
        self.harness = h = Harness([call(1, "finish", summary="ok")])
        h.post("/mission/start", {"mission": "x", "gate": "auto"})
        h.wait_for(lambda s: s["status"] == "completed")
        every = h.runner.log.since(0)
        self.assertGreater(len(every), 2)
        self.assertEqual([f["seq"] for f in every], list(range(1, len(every) + 1)))
        self.assertEqual(h.runner.log.since(2), every[2:])

    def test_unknown_endpoints_say_so(self):
        self.harness = h = Harness([call(1, "finish", summary="ok")])
        with self.assertRaises(urllib.error.HTTPError) as caught:
            h.get("/mission/nonsense")
        self.assertEqual(caught.exception.code, 404)

    # ----------------------------------------------------------- the links --
    def test_the_link_is_open_before_the_first_mission(self):
        """##### THE DEADLOCK THIS FIXES. ##### The dashboard refuses to offer
        Start while the service reports no link to the broker. A link that only
        existed once a mission had started could therefore never be reported
        as connected before one, and nothing could ever start the first."""
        self.harness = h = Harness([call(1, "finish", summary="ok")])
        self.assertFalse(h.get("/mission/config")["mqtt"]["connected"])
        h.runner.connect()
        self.assertTrue(h.get("/mission/config")["mqtt"]["connected"])
        # And both targets got one, under their own ids.
        self.assertEqual(set(h.links), {h.config.robot_id, h.config.sim_robot_id})

    # ------------------------------------------------------------ targets --
    def test_the_config_is_per_robot(self):
        self.harness = h = Harness([call(1, "finish", summary="ok")])
        real = h.get("/mission/config")
        sim = h.get("/mission/config?robot=g1-sim-0001")
        self.assertEqual(real["robot"], h.config.robot_id)
        self.assertEqual(real["map"], MAP)
        self.assertEqual(real["poseTopic"], "/localization_3d")
        self.assertEqual(sim["robot"], "g1-sim-0001")
        self.assertEqual(sim["target"], "sim")
        self.assertEqual(sim["map"], SIM_MAP)
        self.assertEqual(sim["places"], ["kitchen_2"])
        self.assertEqual(sim["poseTopic"], "/odom")
        by_name = h.get("/mission/config?robot=sim")
        self.assertEqual(by_name["robot"], "g1-sim-0001")

    def test_an_unknown_robot_is_named_not_served(self):
        self.harness = h = Harness([call(1, "finish", summary="ok")])
        answer = h.get("/mission/config?robot=g1-9999")
        self.assertFalse(answer["ok"])
        self.assertIn("g1-9999", answer["error"])
        self.assertIn("g1-sim-0001", answer["error"])
        status, body = h.post("/mission/start", {"mission": "x", "gate": "auto",
                                                  "robot": "g1-9999"})
        self.assertEqual(status, 400)
        self.assertEqual(h.link.published, [])

    def test_a_sim_mission_goes_out_under_the_sim_id_and_reads_odometry(self):
        """The simulator has no localizer: its pose is ground-truth Odometry on
        /odom, in a frame that is the map frame. The goal must leave on the
        SIM's link and never on the real robot's."""
        self.harness = h = Harness([
            call(1, "navigate_to", location="kitchen_2", reason="the mission names it"),
            call(2, "finish", summary="arrived"),
        ])
        sim = h.links.setdefault("g1-sim-0001", LiveFakeLink())
        sim.put("/odom", {
            "header": {"frame_id": "odom"},
            "pose": {"pose": {"position": {"x": 0.5, "y": 0.5, "z": 0.0},
                              "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}},
                     "covariance": [0.0] * 36},
        })

        def robot():
            for _ in range(400):
                if sim.published:
                    sim.put("/odom", {"pose": {"pose": {
                        "position": {"x": 4.0, "y": 6.0, "z": 0.0},
                        "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}}}})
                    return
                time.sleep(0.01)

        threading.Thread(target=robot, daemon=True).start()
        status, started = h.post("/mission/start", {
            "mission": "go to kitchen_2", "gate": "auto", "robot": "g1-sim-0001"})
        self.assertEqual(status, 200)
        self.assertEqual(started["robot"], "g1-sim-0001")
        state = h.wait_for(lambda s: s["status"] == "completed")
        self.assertEqual(state["robot"], "g1-sim-0001")
        self.assertEqual(h.link.published, [], "nothing may leave under the real id")
        self.assertEqual(len(sim.published), 1)
        self.assertEqual(sim.published[0][1]["pose"]["position"]["x"], 4.0)
        opening = h.frames()[0]
        self.assertEqual(opening["t"], "mission")
        self.assertEqual(opening["robot"], "g1-sim-0001")

    def test_a_running_mission_refuses_the_other_robot_and_says_where_it_is(self):
        self.harness = h = Harness([
            call(1, "navigate_to", location="kitchen", reason="go"),
            call(2, "finish", summary="done"),
        ])
        h.post("/mission/start", {"mission": "go", "gate": "confirm"})
        h.wait_for(lambda s: s["pending"] is not None)
        status, body = h.post("/mission/start", {"mission": "other", "gate": "auto",
                                                  "robot": "sim"})
        self.assertEqual(status, 409)
        self.assertIn(h.config.robot_id, body["error"])

    # --------------------------------------------------------- the transcript --
    def test_the_sequence_numbers_never_restart_between_missions(self):
        """##### A SECOND MISSION MUST REACH A TAB THAT WATCHED THE FIRST. #####
        The client replays from the last seq it saw. If the log restarted at 1
        per mission, a tab holding cursor 57 would never see mission B's 12
        frames and would sit on "running" for ever."""
        self.harness = h = Harness([
            call(1, "finish", summary="first"),
            call(2, "finish", summary="second"),
        ])
        h.post("/mission/start", {"mission": "one", "gate": "auto"})
        h.wait_for(lambda s: s["status"] == "completed" and s["summary"] == "first")
        after_first = h.runner.state()["seq"]
        self.assertGreater(after_first, 0)

        h.post("/mission/start", {"mission": "two", "gate": "auto"})
        h.wait_for(lambda s: s["status"] == "completed" and s["summary"] == "second")
        later = h.runner.log.since(after_first)
        self.assertEqual([f["t"] for f in later][0], "mission")
        self.assertEqual([f["t"] for f in later][-1], "end")
        self.assertTrue(all(f["seq"] > after_first for f in later))
        self.assertEqual(later[0]["text"], "two")
        # And the full record is still one continuous count.
        every = h.runner.log.since(0)
        self.assertEqual([f["seq"] for f in every], list(range(1, len(every) + 1)))


class LocalPlannerWiringTests(unittest.TestCase):
    """That the approach tool reaches the model at all, and only when it should.

    The tool itself is tested in test_local_planner_module.py against a fake
    link. What can only be checked here is the WIRING: a module list assembled
    from config, and a dry run that must not offer a tool whose whole job is to
    look at real pictures and plan on a real costmap.
    """

    def _schema_names(self, harness):
        harness.post("/mission/start", {"mission": "go to the sink", "gate": "auto"})
        harness.wait_for(lambda state: state["status"] != "running")
        messages, tools = harness.model.requests[0]
        return [tool["function"]["name"] for tool in tools]

    def test_the_tool_is_offered_when_the_camera_is(self):
        harness = Harness([call(1, "finish", summary="nothing to do")], camera=True)
        self.addCleanup(harness.close)
        harness.runner._grounding_model = lambda: harness.model

        names = self._schema_names(harness)

        self.assertIn("local_planner", names)
        self.assertIn("navigate_to", names)

    def test_without_a_camera_there_is_nothing_to_ground_and_no_tool(self):
        harness = Harness([call(1, "finish", summary="nothing to do")], camera=False)
        self.addCleanup(harness.close)

        self.assertNotIn("local_planner", self._schema_names(harness))

    def test_a_dry_run_does_not_offer_it(self):
        """##### A DRY RUN HAS NO PICTURES AND NO COSTMAP. ##### Offering the
        tool there would give the model something that can only refuse, and
        every refusal would read as a broken robot rather than a chosen mode."""
        harness = Harness([call(1, "finish", summary="nothing to do")], camera=True)
        self.addCleanup(harness.close)
        harness.post("/mission/start", {"mission": "go to the sink", "gate": "dry-run"})
        harness.wait_for(lambda state: state["status"] != "running")

        _, tools = harness.model.requests[0]

        self.assertNotIn(
            "local_planner", [tool["function"]["name"] for tool in tools]
        )

    def test_the_dashboard_is_told_whether_the_robot_can_approach_things(self):
        harness = Harness([call(1, "finish", summary="x")], camera=True)
        self.addCleanup(harness.close)

        config = harness.get("/mission/config")

        self.assertTrue(config["localPlanner"])


if __name__ == "__main__":
    unittest.main()
