"""Where the mission service gets its settings.

Same names the dashboard's own dev-server middleware already reads, so one
`webui/.env.local` configures both and nobody has to keep two files in step.

##### ONE SERVICE, TWO TARGETS. #####
The dashboard can look at the real robot (g1-016, `g1-0001` on the broker) or
at the SONIC simulator on the workstation (`g1-sim-0001`), and the operator
switches between them with a reload. The service serves both from one process
so the switch costs nothing on this side: each target is a robot id on the
same broker, the semantic map that belongs to it, and the topic its pose
arrives on. A mission names its target; nothing is inferred from which tab
happened to be open.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def read_env_file(path: str | Path) -> dict[str, str]:
    """Parse a KEY=VALUE file. Absent is not an error -- it is the normal state
    of a checkout nobody has configured yet, and the service says so on
    /mission/config instead of refusing to start."""
    values: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _flag(value: str, default: bool) -> bool:
    if value == "":
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


@dataclass(frozen=True)
class TargetConfig:
    """One thing a mission can be sent to: a robot id on the broker and what
    is true about it."""

    #: Short handle an operator types: "real" or "sim".
    name: str
    #: The MQTT identity -- `g1/<robot_id>/ros/...` and `g1/<robot_id>/cmd/...`.
    robot_id: str
    #: What the dashboard calls it.
    label: str
    #: Which `maps/<map>.places.json` holds its destinations. Empty means "the
    #: map the stack is localizing against", read from maps/.map_active.json.
    map_name: str = ""
    #: Where the robot's map-frame pose arrives. The real robot has a localizer
    #: (`/localization_3d`, PoseStamped). The sim publishes ground-truth
    #: odometry in a frame that IS the map frame (`/odom`, Odometry) -- see
    #: g1_nav/launch/nav2_sonic_sim.launch.py for why that is not cheating.
    pose_topic: str = "/localization_3d"
    #: Whether head-camera frames are offered to the model.
    camera: bool = True
    #: Whether `local_planner` is offered: walking up to a SEEN object rather
    #: than a labelled place. Needs the camera, and is silently off without it.
    local_planner: bool = True
    #: The head camera's placement and optics, for turning a pixel into a
    #: bearing. ⚠️ THE FALLBACK ONLY -- when the robot publishes
    #: `/g1/head/depth_info` those intrinsics win, because they describe the
    #: image that actually arrived rather than a datasheet.
    #: The D455 sits LEVEL above the pitched D435i, so `pitch_down_deg` is 0
    #: here; the D435i's measured 47.87 degrees belongs to the camera that feeds
    #: the costmap, not the one the model looks through.
    camera_width: int = 640
    camera_height: int = 360
    camera_hfov_deg: float = 87.0
    camera_pitch_down_deg: float = 0.0
    #: Optical centre in base_footprint: the G1 bracket's x/y, and the D435i's
    #: MEASURED 1.254 m plus the 0.05 m the D455 sits above it.
    #: ⚠️ THE HEIGHT IS ASSUMED UNTIL SOMEBODY MEASURES IT. Only z is unused by
    #: the planar arithmetic today, so an error there is currently harmless --
    #: it stops being harmless the moment anything reasons about height.
    camera_mount_xyz: tuple[float, float, float] = (0.0576, 0.0175, 1.304)

    def head_camera_spec(self):
        """This target's camera as a `sim.head_camera.HeadCameraSpec`.

        Imported here rather than at module scope: `sim.head_camera` is a large
        module and the config is read by the CLI before anything decides whether
        a camera is even in play.
        """
        from ..sim.head_camera import HeadCameraSpec

        return HeadCameraSpec(
            name=f"{self.name}_head",
            width=self.camera_width,
            height=self.camera_height,
            horizontal_fov_deg=self.camera_hfov_deg,
            pitch_down_deg=self.camera_pitch_down_deg,
            mount_xyz=self.camera_mount_xyz,
        )

    def redacted(self) -> dict:
        return {
            "target": self.name,
            "robot": self.robot_id,
            "label": self.label,
            "poseTopic": self.pose_topic,
            "camera": self.camera,
            "localPlanner": self.camera and self.local_planner,
        }


@dataclass
class ServiceConfig:
    # ---- where the robot is ------------------------------------------------
    nav_dir: Path = Path(".")
    broker: str = "127.0.0.1"
    broker_port: int = 1883
    robot_id: str = "g1-0001"
    mqtt_username: str = "operator-mission"
    mqtt_password: str = ""
    # ---- the simulator, as a second target ---------------------------------
    # Empty disables it. The map is named rather than read from
    # maps/.map_active.json because the sim is never "the map the stack is
    # localizing against" -- it is always the surveyed ProcTHOR house.
    sim_robot_id: str = "g1-sim-0001"
    sim_map: str = "procthor_val_0"
    sim_pose_topic: str = "/odom"
    # ---- the model ---------------------------------------------------------
    # ⚠️ PINNED, NEVER "latest". A model id that moves underneath a physically
    # consequential prompt changes behaviour with no diff anywhere in the repo.
    model: str = "anthropic/claude-opus-4.1"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    effort: str = "high"
    max_tokens: int | None = None
    timeout_s: float = 300.0
    # ---- the loop ----------------------------------------------------------
    max_model_calls: int = 20
    wall_clock_s: float = 900.0
    gate_timeout_s: float = 300.0
    max_looks: int = 12
    # ON by default: the head camera is what lets the model verify a walk by
    # looking rather than by trusting a pose. The camera backend hands over
    # nothing when the topics are silent, so a stack without the camera
    # container simply gives the model no pictures.
    camera: bool = True
    # ---- walking up to something the robot can see -------------------------
    local_planner: bool = True
    #: How far short of a target to stop, CENTRE-TO-OBJECT. 0.60 m leaves
    #: 0.29 m of clear air in front of the body (which reaches 0.31 m forward)
    #: and 0.19 m at worst-case arrival. WAS 0.90, which was the right number
    #: for the 0.25 m arrival box Nav2 used to have; see local_arrival_box_m.
    local_standoff_m: float = 0.60
    #: The longest single leg. Not a limit on how far a target may BE: further
    #: just means the model calls the tool again from where the last leg ended.
    local_max_leg_m: float = 4.0
    #: The robot's circumscribed radius, matching Nav2's `robot_radius`.
    local_footprint_m: float = 0.35
    # ⚠️ MUST MATCH NAV2'S `xy_goal_tolerance`, WHICH LIVES IN ANOTHER REPO:
    # g1_auto_navigation src/g1_bridge/config/nav2_g1_robot.yaml. The goal
    # checker latches anywhere inside this radius, so it is subtracted from the
    # clearance the robot is promised, and the local planner's standoff FLOOR is
    # derived from it. Raising it here without raising it there makes the robot
    # think it has less room than it does (safe, and refuses close approaches);
    # lowering it here without lowering it there lets a goal latch inside the
    # 0.40 m stop zone, which halts the walk with nothing in any log.
    local_arrival_box_m: float = 0.10
    local_costmap_stale_s: float = 5.0
    #: Past this the depth camera's own ranging is not trusted.
    local_range_max_m: float = 6.0
    #: Whether `find_object` is offered: the loop that TURNS AND WALKS to go
    #: look for something not in frame. Needs the local planner (it runs on the
    #: same camera, grounder and costmap) and is silently off without it.
    #:
    #: ⚠️ ONE APPROVAL BUYS EVERY LEG OF ONE SEARCH. That is the deliberate
    #: trade -- see modules/search.py -- and the three budgets below are what
    #: the operator is actually approving, so they belong in the config an
    #: operator can see rather than in the model's arguments alone.
    search: bool = True
    search_max_legs: int = 6
    search_radius_m: float = 4.0
    #: Wall clock, and the binding limit in practice: a G1 walks about 0.5 m/s
    #: and Nav2's own per-goal timeout is 120 s.
    search_max_seconds: float = 180.0
    #: The grounding model. Empty means "the mission model", which is the sane
    #: default; a cheaper vision model is a legitimate override because finding
    #: a box in a photograph is not the job the mission model is chosen for.
    grounding_model: str = ""
    grounding_effort: str = "low"
    grounding_min_confidence: float = 0.5
    # ---- the server --------------------------------------------------------
    # Loopback by default. The dashboard reaches it through the Vite dev
    # server's proxy, which is the surface that is already exposed on the
    # tailnet -- this process does not add a second one.
    bind: str = "127.0.0.1"
    port: int = 8765
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        *,
        nav_dir: str | Path = ".",
        env_file: str | Path | None = None,
        overrides: dict | None = None,
    ) -> "ServiceConfig":
        env = dict(read_env_file(env_file) if env_file else {})
        env.update(os.environ)   # a real environment variable wins over the file

        def get(key: str, default: str = "") -> str:
            value = env.get(key)
            return default if value is None else str(value)

        broker = get("G1_MQTT_HOST", "127.0.0.1") or "127.0.0.1"
        base_url = (get("MISSION_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        sim_robot = get("G1_SIM_ROBOT_ID", "g1-sim-0001").strip()
        if sim_robot.lower() in ("none", "off", "0", "false"):
            sim_robot = ""
        config = cls(
            nav_dir=Path(nav_dir).resolve(),
            broker=broker,
            broker_port=int(get("G1_MQTT_PORT", "1883") or 1883),
            robot_id=get("G1_ROBOT_ID", "g1-0001") or "g1-0001",
            mqtt_username=get("G1_MQTT_USERNAME", "operator-mission") or "operator-mission",
            mqtt_password=get("G1_MQTT_PASSWORD", ""),
            sim_robot_id=sim_robot,
            sim_map=get("G1_SIM_MAP", "procthor_val_0") or "procthor_val_0",
            sim_pose_topic=get("G1_SIM_POSE_TOPIC", "/odom") or "/odom",
            model=get("MISSION_MODEL", "anthropic/claude-opus-4.1") or "anthropic/claude-opus-4.1",
            base_url=base_url,
            # OPENROUTER_API_KEY first, matching webui/.env.example; MISSION_API_KEY
            # is the name for any other OpenAI-compatible server.
            api_key=get("OPENROUTER_API_KEY") or get("MISSION_API_KEY"),
            effort=get("MISSION_EFFORT", "high"),
            max_tokens=int(get("MISSION_MAX_TOKENS")) if get("MISSION_MAX_TOKENS") else None,
            timeout_s=float(get("MISSION_TIMEOUT_S", "300") or 300),
            max_model_calls=int(get("MISSION_MAX_MODEL_CALLS", "20") or 20),
            wall_clock_s=float(get("MISSION_WALL_CLOCK_S", "900") or 900),
            gate_timeout_s=float(get("MISSION_GATE_TIMEOUT_S", "300") or 300),
            max_looks=int(get("MISSION_MAX_LOOKS", "12") or 12),
            camera=_flag(get("MISSION_CAMERA"), True),
            local_planner=_flag(get("MISSION_LOCAL_PLANNER"), True),
            local_standoff_m=float(get("MISSION_LOCAL_STANDOFF_M", "0.6") or 0.6),
            local_arrival_box_m=float(get("MISSION_LOCAL_ARRIVAL_BOX_M", "0.10") or 0.10),
            local_max_leg_m=float(get("MISSION_LOCAL_MAX_LEG_M", "4") or 4),
            local_footprint_m=float(get("MISSION_LOCAL_FOOTPRINT_M", "0.35") or 0.35),
            local_costmap_stale_s=float(get("MISSION_LOCAL_COSTMAP_STALE_S", "5") or 5),
            local_range_max_m=float(get("MISSION_LOCAL_RANGE_MAX_M", "6") or 6),
            search=_flag(get("MISSION_SEARCH"), True),
            search_max_legs=int(get("MISSION_SEARCH_MAX_LEGS", "6") or 6),
            search_radius_m=float(get("MISSION_SEARCH_RADIUS_M", "4") or 4),
            search_max_seconds=float(get("MISSION_SEARCH_MAX_SECONDS", "180") or 180),
            grounding_model=get("MISSION_GROUNDING_MODEL", ""),
            grounding_effort=get("MISSION_GROUNDING_EFFORT", "low"),
            grounding_min_confidence=float(
                get("MISSION_GROUNDING_MIN_CONFIDENCE", "0.5") or 0.5
            ),
            bind=get("MISSION_BIND", "127.0.0.1") or "127.0.0.1",
            port=int(get("MISSION_PORT", "8765") or 8765),
            env=env,
        )
        for key, value in (overrides or {}).items():
            if value is not None:
                setattr(config, key, value)
        return config

    # ---- targets -------------------------------------------------------------
    @property
    def targets(self) -> dict[str, TargetConfig]:
        """Every robot this service will send a mission to, by short name.
        `real` always; `sim` when a sim robot id is configured."""
        targets = {
            "real": TargetConfig(
                name="real",
                robot_id=self.robot_id,
                label="real robot",
                map_name="",
                pose_topic="/localization_3d",
                camera=self.camera,
                local_planner=self.local_planner,
            ),
        }
        if self.sim_robot_id:
            targets["sim"] = TargetConfig(
                name="sim",
                robot_id=self.sim_robot_id,
                label="SONIC simulator",
                map_name=self.sim_map,
                pose_topic=self.sim_pose_topic,
                camera=self.camera,
                local_planner=self.local_planner,
                # The simulated head camera is the D455 spec, level, at the
                # bracket -- sim/head_camera.D455. Its height above the FLOOR is
                # not the mount_xyz z (which is in torso frame), and nothing
                # planar uses it, so it is left at the real robot's figure.
                camera_width=640,
                camera_height=360,
                camera_hfov_deg=87.0,
                camera_pitch_down_deg=0.0,
            )
        return targets

    @property
    def default_target(self) -> TargetConfig:
        return self.targets["real"]

    def target_for(self, robot_id: str = "") -> TargetConfig:
        """The target for a robot id, or for a target's short name. Empty
        means the real robot -- the only default that cannot surprise anyone
        watching a physical machine."""
        wanted = (robot_id or "").strip()
        if not wanted:
            return self.default_target
        for target in self.targets.values():
            if wanted in (target.robot_id, target.name):
                return target
        known = ", ".join(f"{t.name} ({t.robot_id})" for t in self.targets.values())
        raise ValueError(f"unknown robot {wanted!r}; this service knows: {known}")

    # ---- the semantic map --------------------------------------------------
    @property
    def maps_dir(self) -> Path:
        return self.nav_dir / "maps"

    def active_map(self) -> str:
        """Which map the stack is localizing against, from maps/.map_active.json.

        ⚠️ THE SAME FILE `./g1 map use` WRITES. Reading it rather than taking a
        map name as an argument is what stops the service offering labels from
        one building while the robot is localized in another -- and the places
        document is checked against this name, not trusted to match.
        """
        import json

        try:
            raw = json.loads((self.maps_dir / ".map_active.json").read_text())
            return str(raw.get("name") or "")
        except (OSError, ValueError):
            return ""

    def map_for(self, target: TargetConfig | None = None) -> str:
        target = target or self.default_target
        return target.map_name or self.active_map()

    def places_path(self, map_name: str = "") -> Path:
        name = map_name or self.active_map()
        return self.maps_dir / f"{name}.places.json"

    def redacted(self) -> dict:
        """What /mission/config may say. ⚠️ THE KEY'S PRESENCE, NEVER THE KEY."""
        return {
            "model": self.model,
            "effort": self.effort,
            "keyPresent": bool(self.api_key),
            "baseUrl": self.base_url,
            "robotId": self.robot_id,
            "broker": f"{self.broker}:{self.broker_port}",
            "maxModelCalls": self.max_model_calls,
            "camera": self.camera,
            "targets": [target.redacted() for target in self.targets.values()],
        }
