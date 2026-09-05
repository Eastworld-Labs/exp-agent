"""The mission service: one HTTP surface over one System-2 loop.

##### THE MODEL BEHIND THIS CAN WALK A 35 kg BIPED. #####
Everything here that looks like plumbing is shaped by that. The gate blocks the
agent thread rather than queueing an approval; stop settles the gate rather than
leaving it pending; the runner refuses to start a second mission rather than
interleaving two; and none of it can publish a velocity, because the link it
holds will not carry one (see wire.PUBLISHABLE).

ONE MISSION AT A TIME, deliberately -- across BOTH targets. Two missions sharing
one robot is two planners sharing one set of legs, and there is no useful way
to arbitrate that in software -- the second request is refused with a 409 and
the operator decides. The refusal names the robot the running mission is on.

TWO TARGETS, ONE SERVICE. The real robot and the SONIC simulator are two robot
ids on the same broker (config.TargetConfig). Every request that is about a
robot names one (`robot=<id>` / `{"robot": ...}`); an unnamed request means the
real robot, because that is the only default that cannot surprise somebody
standing next to a physical machine.

##### THE LINKS ARE OPENED AT STARTUP, NOT AT THE FIRST MISSION. ##### The
dashboard refuses to offer Start while /mission/config says the service has no
link to the broker -- correctly, since a goal could not reach the robot. A link
that was only created when a mission began therefore deadlocked the dashboard:
nothing could ever start the first mission. `connect()` runs before the server
takes its first request.

##### THE TRANSCRIPT NUMBERING IS MONOTONIC FOR THE LIFE OF THE PROCESS. #####
A client replays from the last sequence number it saw. If each mission restarted
the count at 1, a dashboard tab that had watched mission A (say 57 frames) would
ask for everything after 57 and mission B, with its 12 frames, would never reach
it -- the tab sits on "running" for ever. One log, one counter, and a client's
cursor is always meaningful.

Endpoints (all JSON; the dashboard reaches them through the Vite dev server's
proxy, so the browser stays same-origin and the model key stays here):

    GET  /mission/config?robot=  what this service is configured with, for that target
    POST /mission/start          {mission, gate, robot} -> {id}
    GET  /mission/state          a snapshot, for a client that just connected
    GET  /mission/events         server-sent events, replayed from ?since=<seq>
    POST /mission/approve        {id, call_id, ok}
    POST /mission/stop           {id}
    POST /mission/gate           {gate}
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from ..agent import System2Agent
from ..model import OpenAICompatibleModel
from ..modules.camera import CameraModule
from ..modules.navigation import DryRunNavigationBackend, NavigationModule
from ..modules.semantic_map import SemanticMapModule
from ..types import Json, ToolCall
from .config import ServiceConfig, TargetConfig
from .events import end_frame, gate_frame, to_frames
from .prompt import G1_SYSTEM_PROMPT

GATES = ("confirm", "auto", "dry-run")

#: How many frames the transcript keeps. Old frames are dropped from the
#: front; sequence numbers keep counting, so a client that asks for a frame
#: that has been dropped simply gets everything that is still held.
LOG_KEEP = 5000


def _now_ms() -> int:
    return int(time.time() * 1000)


class Gate:
    """The operator's yes/no, in front of every tool that moves the robot.

    ⚠️ IT BLOCKS THE AGENT THREAD, and that is the point rather than a
    limitation: the alternative is queueing an approval and letting the loop
    carry on, which means the model reasons about a step nobody has agreed to.
    ⚠️ AND IT MUST ALWAYS SETTLE. A gate that never resolves hangs a mission
    with the robot's next step pending and nothing on screen saying why -- so
    stop settles it, a timeout settles it, and shutdown settles it.
    """

    def __init__(self, mode: str = "confirm", timeout_s: float = 300.0) -> None:
        self.mode = mode
        self.timeout_s = timeout_s
        self._cond = threading.Condition()
        self._pending: Json | None = None
        self._answer: bool | None = None

    def pending(self) -> Json | None:
        with self._cond:
            return dict(self._pending) if self._pending else None

    def request(self, call: ToolCall, _tool: Any) -> bool:
        if self.mode == "auto":
            return True
        if self.mode == "dry-run":
            # Nothing can reach the robot in this mode anyway (the backend
            # publishes nothing), so approving keeps the loop exercising the
            # same branches a real run takes.
            return True
        deadline = time.monotonic() + self.timeout_s
        with self._cond:
            self._pending = {
                "id": call.id, "name": call.name, "arguments": dict(call.arguments)
            }
            self._answer = None
            self._cond.notify_all()
            while self._answer is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._answer = False
                    break
                self._cond.wait(timeout=min(remaining, 1.0))
            answer = bool(self._answer)
            self._pending = None
            self._cond.notify_all()
        return answer

    def decide(self, call_id: str, ok: bool) -> bool:
        with self._cond:
            if not self._pending or self._pending["id"] != call_id:
                return False
            self._answer = bool(ok)
            self._cond.notify_all()
            return True

    def settle(self, ok: bool = False) -> None:
        with self._cond:
            if self._pending is not None:
                self._answer = bool(ok)
                self._cond.notify_all()


class EventLog:
    """An append-only transcript with sequence numbers that NEVER restart, so a
    client that reconnects -- or that watched the previous mission -- can
    replay from where it left off instead of losing the run."""

    def __init__(self, keep: int = LOG_KEEP) -> None:
        self._cond = threading.Condition()
        self._frames: deque[Json] = deque()
        self._keep = keep
        self._seq = 0

    def append(self, frame: Json) -> None:
        with self._cond:
            self._seq += 1
            self._frames.append({**frame, "seq": self._seq})
            while len(self._frames) > self._keep:
                self._frames.popleft()
            self._cond.notify_all()

    def since(self, seq: int) -> list[Json]:
        with self._cond:
            return [frame for frame in self._frames if frame["seq"] > seq]

    def wait(self, seq: int, timeout: float) -> list[Json]:
        with self._cond:
            if self._seq <= seq:
                self._cond.wait(timeout=timeout)
            return [frame for frame in self._frames if frame["seq"] > seq]

    @property
    def seq(self) -> int:
        with self._cond:
            return self._seq

    def __len__(self) -> int:
        return self.seq


class MissionRunner:
    """Owns the one mission that may be running, and everything it needs --
    including one link per target, so a goal for the simulator can never be
    published under the real robot's id or the other way round."""

    def __init__(
        self, config: ServiceConfig, link_factory: Callable[[str], Any]
    ) -> None:
        self.config = config
        self._link_factory = link_factory
        self.links: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.log = EventLog()
        self.gate = Gate(timeout_s=config.gate_timeout_s)
        self.cancel = threading.Event()
        self.id = ""
        self.status = "idle"
        self.summary = ""
        #: Robot id of the running (or last) mission. Every consumer that
        #: shows a mission shows which robot it is on.
        self.robot = ""
        self.started_at: float | None = None
        self.model_calls = 0
        self.usage: Json | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------- targets --
    def target(self, robot_id: str = "") -> TargetConfig:
        """Raises ValueError for a robot this service does not know."""
        return self.config.target_for(robot_id)

    # ------------------------------------------------------------- the link --
    def link(self, robot_id: str = "") -> Any:
        target = self.target(robot_id)
        with self._lock:
            link = self.links.get(target.robot_id)
            if link is None:
                link = self._link_factory(target.robot_id)
                self.links[target.robot_id] = link
            return link

    def connect(self) -> None:
        """Open a link to every target NOW. See the module header for why
        this is not left to the first mission."""
        for target in self.config.targets.values():
            self.link(target.robot_id)

    def linked(self, robot_id: str = "") -> bool:
        try:
            target = self.target(robot_id)
        except ValueError:
            return False
        link = self.links.get(target.robot_id)
        return bool(link and link.connected())

    # ------------------------------------------------------- the semantic map --
    def semantic_map(self, robot_id: str = "") -> SemanticMapModule:
        target = self.target(robot_id)
        map_name = self.config.map_for(target)
        path = self.config.places_path(map_name)
        if not path.exists():
            # An unlabelled map is an ordinary state, not an error: the answer
            # is an empty destination list the model can read and refuse from,
            # and an operator labels places in the dashboard's Places sheet.
            return SemanticMapModule(
                {},
                map_name=map_name,
                errors=[f"no semantic map at {path} -- nothing on this map is labelled yet"],
            )
        return SemanticMapModule.from_places_json(path, expect_map=map_name)

    # ----------------------------------------------------------- the running --
    def start(self, mission: str, gate_mode: str, robot_id: str = "") -> str:
        target = self.target(robot_id)          # ValueError for an unknown robot
        with self._lock:
            if self.status == "running":
                raise RuntimeError(
                    f"a mission is already running on {self.robot or 'a robot'}; "
                    "stop it first"
                )
            self.gate.mode = gate_mode
            self.cancel.clear()
            self.id = f"m{int(time.time() * 1000)}"
            self.status = "running"
            self.summary = ""
            self.robot = target.robot_id
            self.started_at = time.time()
            self.model_calls = 0
            self.usage = None
            self._thread = threading.Thread(
                target=self._run, args=(mission, target), name="mission", daemon=True
            )
            self._thread.start()
            return self.id

    def _modules(self, target: TargetConfig) -> list[Any]:
        semantic = self.semantic_map(target.robot_id)
        if self.gate.mode == "dry-run":
            backend: Any = DryRunNavigationBackend()
        else:
            from .nav2_backend import Nav2MqttBackend

            backend = Nav2MqttBackend(
                self.link(target.robot_id),
                cancel=self.cancel,
                pose_topic=target.pose_topic,
            )
        modules: list[Any] = [semantic, NavigationModule(semantic, backend)]
        if target.camera and self.gate.mode != "dry-run":
            from .camera import MqttHeadCamera

            # ⚠️ ONE CAMERA OBJECT, SHARED. The model looks through it and
            # local_planner grounds through it, and they must be looking at the
            # same thing: two backends would hold two `latest` reads and a box
            # drawn on one frame could be ranged against another.
            camera = MqttHeadCamera(self.link(target.robot_id))
            modules.append(CameraModule(camera, max_looks=self.config.max_looks))
            if target.local_planner:
                from ..modules.local_planner import LocalPlannerModule

                planner = self._local_planner(target, camera)
                modules.append(LocalPlannerModule(planner, backend))
                if self.config.search:
                    modules.append(self._search(planner, backend))
        return modules

    def _search(self, planner: Any, backend: Any) -> Any:
        """`find_object`: the loop that goes and looks when nothing is in frame.

        Built on the SAME planner object as `local_planner`, deliberately. They
        share the camera, the grounder, the costmap and the pose source, so the
        picture the search grounds on is the picture the approach then measures
        against -- two planners would be two `latest()` reads and a box drawn on
        one frame ranged against another.
        """
        from ..grounding import StandpointPicker
        from ..modules.search import SearchModule

        return SearchModule(
            planner,
            backend,
            # The same cheap vision model the grounder uses: this is one more
            # one-image question, not a second class of model.
            picker=StandpointPicker(self._grounding_model()),
            cancel=self.cancel,
            max_legs=self.config.search_max_legs,
            radius_m=self.config.search_radius_m,
            max_seconds=self.config.search_max_seconds,
        )

    def _local_planner(self, target: TargetConfig, camera: Any) -> Any:
        """The `LocalPlanner` both `local_planner` and `find_object` run on.

        Returns the PLANNER, not the module: two tools are built over it now
        (the approach, and the search that goes and looks first), and they must
        share one -- see `_search`.

        Not built in dry-run: it grounds on real pictures and plans on a real
        costmap, and a dry run has neither. The camera is omitted there too, so
        this would have nothing to look at.
        """
        from ..grounding import VisionGrounder
        from ..local_planner import LocalPlanner
        from .depth import MqttHeadDepth
        from .local_costmap import MqttInitPose, MqttLocalCostmap

        link = self.link(target.robot_id)
        config = self.config
        planner = LocalPlanner(
            camera=camera,
            grounder=VisionGrounder(
                self._grounding_model(),
                min_confidence=config.grounding_min_confidence,
            ),
            grid_source=MqttLocalCostmap(link, stale_s=config.local_costmap_stale_s),
            init_pose=MqttInitPose(link, map_topic=target.pose_topic),
            geometry=target.head_camera_spec(),
            depth=MqttHeadDepth(link),
            footprint_radius_m=config.local_footprint_m,
            standoff_m=config.local_standoff_m,
            arrival_box_m=config.local_arrival_box_m,
            max_leg_m=config.local_max_leg_m,
            max_range_m=config.local_range_max_m,
        )
        return planner

    def _grounding_model(self) -> OpenAICompatibleModel:
        """The model that finds a box in a photograph.

        ⚠️ A SEPARATE CLIENT AT LOW EFFORT, NOT THE MISSION MODEL'S SETTINGS.
        The mission model runs at high reasoning effort because it is deciding
        what a robot should do; "where is the sink in this picture" is not that
        job, and paying mission-grade thinking for it makes every approach leg
        slow and expensive. `MISSION_GROUNDING_MODEL` can point it at a cheaper
        vision model entirely.
        """
        extra: Json = {"usage": {"include": True}}
        if self.config.grounding_effort:
            extra["reasoning"] = {"effort": self.config.grounding_effort, "exclude": True}
        return OpenAICompatibleModel(
            model=self.config.grounding_model or self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout_s=self.config.timeout_s,
            max_tokens=self.config.max_tokens,
            extra_body=extra,
        )

    def _model(self) -> OpenAICompatibleModel:
        extra: Json = {"usage": {"include": True}}
        if self.config.effort:
            extra["reasoning"] = {"effort": self.config.effort, "exclude": False}
        return OpenAICompatibleModel(
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout_s=self.config.timeout_s,
            max_tokens=self.config.max_tokens,
            extra_body=extra,
        )

    def _run(self, mission: str, target: TargetConfig) -> None:
        self.log.append({
            "t": "mission", "text": mission, "gate": self.gate.mode,
            "robot": target.robot_id, "target": target.name, "id": self.id,
            "at": _now_ms(),
        })
        deadline = time.monotonic() + self.config.wall_clock_s

        def on_event(event: Json) -> None:
            if event.get("type") == "turn":
                self.model_calls = int(event.get("model_call") or self.model_calls)
                self.usage = event.get("usage") or self.usage
            for frame in to_frames(event, _now_ms()):
                self.log.append(frame)

        def approval(call: ToolCall, tool: Any) -> bool:
            approved = self.gate.request(call, tool)
            self.log.append(gate_frame(call, approved, self.gate.mode, _now_ms()))
            return approved

        def should_stop() -> bool:
            return self.cancel.is_set() or time.monotonic() >= deadline

        try:
            agent = System2Agent(
                self._model(),
                self._modules(target),
                approval=approval,
                max_model_calls=self.config.max_model_calls,
                system_prompt=G1_SYSTEM_PROMPT,
                on_event=on_event,
                should_stop=should_stop,
            )
            outcome = agent.run(mission)
        except Exception as exc:  # noqa: BLE001
            # ⚠️ NEVER LET A MISSION THREAD DIE SILENTLY. The operator is
            # watching a transcript; a thread that vanished leaves it frozen
            # mid-step with the robot possibly still walking.
            self.log.append({"t": "note", "tone": "error",
                             "text": f"{type(exc).__name__}: {exc}", "at": _now_ms()})
            self.status, self.summary = "failed", f"{type(exc).__name__}: {exc}"
            self.log.append({"t": "end", "status": "failed", "summary": self.summary,
                             "modelCalls": self.model_calls, "usage": self.usage,
                             "robot": target.robot_id, "at": _now_ms()})
            return
        finally:
            self.gate.settle(False)

        self.status = outcome.status
        self.summary = outcome.summary
        self.model_calls = outcome.model_calls
        self.usage = outcome.usage or self.usage
        self.log.append({**end_frame(outcome, _now_ms()), "robot": target.robot_id})

    def stop(self) -> None:
        self.cancel.set()
        # Settle the gate too, or a mission parked at an approval never notices
        # it was stopped.
        self.gate.settle(False)

    def state(self) -> Json:
        return {
            "id": self.id,
            "status": self.status,
            "summary": self.summary,
            "robot": self.robot,
            "gate": self.gate.mode,
            "pending": self.gate.pending(),
            "modelCalls": self.model_calls,
            "usage": self.usage,
            "startedAt": self.started_at,
            "elapsedS": None if self.started_at is None else round(
                time.time() - self.started_at, 1),
            "seq": self.log.seq,
        }

    def config_for(self, robot_id: str = "") -> Json:
        """What /mission/config says about one target. Never raises for an
        unknown robot: the dashboard shows the reason, and the list of robots
        this service does know, instead of a blank control."""
        config = self.config
        base: Json = {
            "ok": True,
            "engine": "host",
            "available": True,
            **config.redacted(),
        }
        try:
            target = self.target(robot_id)
        except ValueError as exc:
            return {**base, "ok": False, "error": str(exc), "robot": robot_id}
        places = self.semantic_map(target.robot_id)
        return {
            **base,
            **target.redacted(),
            "mqtt": {
                "connected": self.linked(target.robot_id),
                "broker": f"{config.broker}:{config.broker_port}",
            },
            "map": config.map_for(target),
            "places": places.names(),
            "placesErrors": list(places.errors),
            "running": {"id": self.id, "robot": self.robot} if self.status == "running" else None,
        }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def make_handler(runner: MissionRunner) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "exp-agent-g1"

        def log_message(self, *_args) -> None:  # quieter than the default
            pass

        # ------------------------------------------------------------ util --
        def _send(self, payload: Json, code: int = 200) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> Json:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                parsed = json.loads(self.rfile.read(length))
            except ValueError:
                return {}
            return parsed if isinstance(parsed, dict) else {}

        def _path(self) -> str:
            return (self.path or "").split("?")[0].rstrip("/") or "/"

        def _query(self, key: str, default: str = "") -> str:
            _, _, query = (self.path or "").partition("?")
            for part in query.split("&"):
                name, _, value = part.partition("=")
                if name == key:
                    return value
            return default

        # ------------------------------------------------------------- GET --
        def do_GET(self) -> None:  # noqa: N802
            path = self._path()
            if path == "/mission/config":
                self._send(runner.config_for(self._query("robot")))
            elif path == "/mission/state":
                self._send(runner.state())
            elif path == "/mission/events":
                self._events()
            else:
                self._send({"ok": False, "error": f"unknown endpoint: {path}"}, 404)

        def _events(self) -> None:
            try:
                since = int(self._query("since", "0"))
            except ValueError:
                since = 0
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            # Vite's proxy and any reverse proxy in front of it must not buffer
            # this, or the transcript arrives in one lump when the run ends.
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            state = runner.state()
            self._frame({"t": "hello", "id": state["id"], "status": state["status"],
                         "robot": state["robot"], "seq": state["seq"],
                         "gate": state["gate"], "pending": state["pending"]})
            cursor = since
            last_pending = state["pending"]
            try:
                while True:
                    for frame in runner.log.wait(cursor, timeout=5.0):
                        self._frame(frame)
                        cursor = frame["seq"]
                    pending = runner.gate.pending()
                    if pending != last_pending:
                        last_pending = pending
                        self._frame({"t": "pending", "pending": pending,
                                     "status": runner.status})
                    # A comment line keeps the connection (and any proxy) alive
                    # through a long walk, when nothing else is being sent.
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def _frame(self, frame: Json) -> None:
            self.wfile.write(
                f"data: {json.dumps(frame, default=str)}\n\n".encode("utf-8"))
            self.wfile.flush()

        # ------------------------------------------------------------ POST --
        def do_POST(self) -> None:  # noqa: N802
            path = self._path()
            body = self._body()
            if path == "/mission/start":
                mission = str(body.get("mission") or "").strip()
                gate = str(body.get("gate") or runner.gate.mode)
                robot = str(body.get("robot") or "")
                if not mission:
                    return self._send({"ok": False, "error": "mission text is required"}, 400)
                if gate not in GATES:
                    return self._send(
                        {"ok": False, "error": f"gate must be one of {list(GATES)}"}, 400)
                try:
                    mission_id = runner.start(mission, gate, robot)
                except ValueError as exc:
                    # An unknown robot. Nothing started; say which ones exist.
                    return self._send({"ok": False, "error": str(exc)}, 400)
                except RuntimeError as exc:
                    # ⚠️ 409, NOT A QUEUE. Two missions is two planners sharing
                    # one set of legs; the operator decides which one runs.
                    return self._send(
                        {"ok": False, "error": str(exc), "robot": runner.robot}, 409)
                self._send({"ok": True, "id": mission_id, "robot": runner.robot})
            elif path == "/mission/approve":
                ok = bool(body.get("ok"))
                call_id = str(body.get("call_id") or "")
                if not runner.gate.decide(call_id, ok):
                    return self._send(
                        {"ok": False, "error": "no step is waiting for that approval"}, 409)
                self._send({"ok": True})
            elif path == "/mission/stop":
                runner.stop()
                self._send({"ok": True, "note": "the MISSION was stopped. ##### THIS DOES "
                                                "NOT STOP THE ROBOT ##### -- a goal the "
                                                "planner already accepted keeps running."})
            elif path == "/mission/gate":
                gate = str(body.get("gate") or "")
                if gate not in GATES:
                    return self._send(
                        {"ok": False, "error": f"gate must be one of {list(GATES)}"}, 400)
                runner.gate.mode = gate
                self._send({"ok": True, "gate": gate})
            else:
                self._send({"ok": False, "error": f"unknown endpoint: {path}"}, 404)

    return Handler


def build_runner(config: ServiceConfig) -> MissionRunner:
    def link_factory(robot_id: str):
        from .link import MqttLink

        link = MqttLink(
            broker=config.broker,
            port=config.broker_port,
            robot_id=robot_id,
            username=config.mqtt_username,
            password=config.mqtt_password,
        )
        # The real robot's read-only set. A target with another pose topic
        # (the sim) adds its own when its backend is built; subscribing
        # /localization_3d on the sim costs nothing (nobody publishes it).
        link.subscribe_defaults()
        return link

    return MissionRunner(config, link_factory)


def serve(config: ServiceConfig, runner: MissionRunner | None = None) -> ThreadingHTTPServer:
    runner = runner or build_runner(config)
    server = ThreadingHTTPServer((config.bind, config.port), make_handler(runner))
    server.daemon_threads = True
    return server
