# Exp Agent

A deliberately small mission-level agent for robots. It follows the architecture
used by systems such as Flexion Reflect:

```text
human mission
     |
frontier VLM/LLM mission controller       this package
     |
semantic tools: map, navigate, pick, place
     |
deterministic skill and safety layer
     |
navigation / VLA / motion planner / WBC
     |
robot hardware
```

The model decides **what capability to invoke**. It never produces joint torques,
foot contacts, or raw base velocities. Real-time behavior remains in conventional
or learned robot controllers.

The default CLI is dry-run only. The simulation CLI selects MuJoCo or Isaac Sim,
and the SONIC bridge is opt-in; nothing targets physical hardware by default.

`robot_class` is the shared embodiment/controller layer. This repository does
not duplicate its G1, MuJoCo, SONIC variant, checkpoint, or reference-protocol
implementations. It adds the mission agent, semantic modules, global planning,
and the navigation-specific target-velocity planner adapter.

## Driving a real G1: the host mission service

`system2_agent/g1/` runs this agent as a small service beside the
[g1_auto_navigation](../g1_auto_navigation) stack's MQTT broker, so an operator
can type "go to kitchen" in the dashboard and the robot walks there.

```text
webui chatbox / ./g1 mission ──HTTP+SSE──> exp-agent-g1 serve   (this machine)
                                             System2Agent
                                             semantic_map + navigate_to
                                                 │ CBOR/MQTT: /goal_pose
                                                 ▼
                              the robot's fleet agent -> goal_relay -> Nav2
                              -> /cmd_vel -> collision monitor -> the legs
```

```bash
pip install -e '.[g1]'
exp-agent-g1 serve --nav-dir ../g1_auto_navigation \
                   --env-file ../g1_auto_navigation/webui/.env.local
exp-agent-g1 run "go to kitchen"                 # the real robot; approves each step
exp-agent-g1 run --robot sim "go to kitchen_2"   # the SONIC simulator
```

Normally nothing above is typed: the dashboard's dev server starts the service
itself when it is not answering (`MISSION_AUTOSTART=0` turns that off), and the
dashboard's **Send on a mission** control is its client.

**One service, two targets.** The real robot (`g1-0001`) and the SONIC simulator
on the workstation (`g1-sim-0001`) are two robot ids on the same broker. The
service opens one link per target at startup, plans the sim over its own map
(`maps/procthor_val_0.places.json`) and reads the sim's pose off ground-truth
`/odom` because it has no localizer. Every request names its robot; unnamed means
the real one. One mission at a time across both, refused with a 409 that says
which robot the running one is on.

Three properties are worth stating because they are load-bearing rather than
incidental:

- **The agent chooses a destination, never a velocity.** The service's link
  refuses to publish any topic outside a one-entry table, so `/cmd_vel` and
  `/estop` are unreachable from a tool call by construction rather than by
  prompt. Everything between the goal and the motors — costmaps, obstacle
  avoidance, recovery behaviours, the collision monitor — stays on the robot.
- **Destinations come from the map the robot is actually localized against.**
  `maps/<map>.places.json` is read against `maps/.map_active.json`, and a
  document describing a different map refuses every destination instead of
  walking confidently to the wrong building. `navigate_to`'s `location` is an
  `enum` built from that file, so a hallucinated place cannot be emitted.
- **Arrival is labelled.** Nav2's own verdict arrives on `/goal_status` when the
  robot publishes it; otherwise arrival is inferred from the pose converging on
  the goal — which cannot tell an abort from a slow walk. Every result carries
  `verdict_source` saying which one it was.
- **Retained heartbeats are aged, not trusted.** `/estop_state` and
  `/sonic/enabled` are retained on the broker and heartbeated at 2 Hz. A value
  older than six beats reads as "nobody is saying", so a `/sonic/enabled: false`
  left behind by last week's SONIC session cannot refuse every goal on a robot
  now walking on Unitree's gait.

`--gate dry-run` publishes nothing at all and needs no robot; it is how a prompt
change gets exercised for free.

## Design goals

- Small enough to read in an afternoon.
- OpenAI-compatible model interface: OpenAI, Gemini, DeepSeek, OpenRouter, vLLM,
  Ollama, or another compatible endpoint.
- One physical action per reasoning turn.
- Fresh world state after every physical action.
- Typed module tools with deterministic validation.
- Hardware motion denied by default unless an approval callback allows it.
- Semantic maps, navigation stacks, VLAs, and robot SDKs remain replaceable modules.

The important files are:

```text
src/system2_agent/agent.py                 reasoning loop
src/system2_agent/model.py                 model/provider adapter
src/system2_agent/tools.py                 module and tool contracts
src/system2_agent/modules/semantic_map.py  names -> map poses
src/system2_agent/modules/navigation.py    navigation backend boundary
src/system2_agent/modules/manipulation.py  manipulation backend boundary
src/system2_agent/modules/camera.py        fresh multimodal observations
```

## Quick start

Python 3.10 or newer is required. The package itself has no runtime dependencies.

```bash
cd exp-agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For local simulation development with the sibling checkout:

```bash
pip install -e '../robot_class[sim]'
pip install -e '.[sim]'
```

Set one API key and run a completely simulated mission:

```bash
export OPENAI_API_KEY=...
exp-agent \
  --model openai/YOUR_TOOL_CAPABLE_MODEL \
  "go to the kitchen table, pick up the red cup, and verify that you hold it"
```

Other providers use the same agent:

```bash
export GEMINI_API_KEY=...
exp-agent --model google/YOUR_MODEL "go to the charging station"

export DEEPSEEK_API_KEY=...
exp-agent --model deepseek/YOUR_MODEL "inspect the kitchen table"

export OPENROUTER_API_KEY=...
exp-agent --model openrouter/google/YOUR_MODEL "inspect the kitchen table"
```

For vLLM, Ollama, or another server:

```bash
export LOCAL_MODEL_KEY=anything
exp-agent \
  --model your-model-id \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key-env LOCAL_MODEL_KEY \
  "go home"
```

The selected model must support reliable structured tool calling and, once camera
observations are connected, vision. Do not silently fall back to parsing Python or
JSON from prose on real hardware.

### Adding camera images

Add a `CameraModule` whose backend returns `CameraFrame` objects containing HTTPS
or base64 data URLs:

```python
from system2_agent.modules import CameraFrame, CameraModule


class HeadCamera:
    def capture(self):
        jpeg_data_url = camera.latest_jpeg_data_url()
        return [CameraFrame(label="head_rgb", url=jpeg_data_url)]


modules.append(CameraModule(HeadCamera()))
```

The initial model request and every post-action request will then contain fresh
frames. Keep safety and success detection independent of the VLM: camera reasoning
can be wrong, stale, or temporarily unavailable.

`CameraModule` also exposes `observe_surroundings()`. Calling it does not move the
robot; it injects fresh head/wrist images into the next System-2 reasoning turn.
This is for semantic inspection and mission decisions, not for producing
control-rate velocity commands.

## The System-2 loop

`System2Agent` depends only on the exported `ChatModel` protocol—not on a
provider SDK or simulator. Any tool-capable text/vision model can be used by
implementing `complete(messages, tools) -> AssistantTurn`. The included
`OpenAICompatibleModel` covers OpenAI, Gemini, DeepSeek, local compatible
servers, and Claude through OpenRouter; a native-provider adapter can implement
the same protocol without changing the agent, tools, navigation, manipulation,
SONIC, or either simulator backend.

`System2Agent.run()` keeps a linear conversation:

1. Send the mission, available tools, and current module snapshots.
2. Require exactly one structured tool call.
3. Validate its arguments and ask the safety approval layer when required.
4. Execute the selected module capability.
5. After a physical action, attach a fresh world snapshot.
6. Let the model verify, recover, or choose the next skill.
7. End only through `finish` or `request_human`.

This is the same abstract loop used by coding agents. The action language is a
small robot skill API rather than a shell. A sandboxed code-as-policy module can be
added later when loops and program synthesis are genuinely useful.

## Adding a module

A module has a name, a collection of tools, and an optional state snapshot:

```python
from system2_agent.tools import Tool, object_schema


class DoorModule:
    name = "doors"

    def tools(self):
        return (
            Tool(
                name="open_door",
                description="Open the named door with the learned door skill.",
                parameters=object_schema(
                    {
                        "door": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    ["door", "reason"],
                ),
                handler=lambda args: self.skill_server.open(args["door"]),
                kind="action",
                requires_approval=True,
            ),
        )

    def snapshot(self):
        return {"known_doors": self.tracker.known_doors()}
```

Add the instance to the list passed to `System2Agent`. Tool names must be globally
unique. Keep a module's model-facing interface semantic and put timeouts, retries,
collision checking, and control-rate loops inside its backend.

`examples/custom_module.py` contains an even smaller read-only example.

### Module rules

- Observation tools may inspect state but must not move hardware.
- Action tools should be idempotent where practical and report explicit states such
  as `accepted`, `running`, `succeeded`, `failed`, or `cancelled`.
- Long skills should return a job ID and expose a separate status/cancel tool.
- Every real action should have precondition checks, a timeout, and an independent
  safety stop.
- Tool results should report measured outcomes, not merely that commands were sent.
- Never place a safety invariant only in the prompt.

## Semantic maps

The included semantic map is intentionally just a dictionary:

```json
{
  "locations": {
    "kitchen table": {"x": 4.2, "y": 1.8, "yaw": 1.57}
  }
}
```

In a real deployment, split mapping into two layers:

1. **Metric map/localization:** SLAM, an offline building scan, VIO, LiDAR
   localization, occupancy maps, TSDF/ESDF, and the current robot transform.
2. **Semantic index:** language names, object/room identities, approach poses,
   affordances, and links into the metric map.

`SemanticMapModule.resolve()` should return a stable map-frame goal. It should not
implement path planning. A future semantic backend can use a scene graph, database,
3-D map service, or learned open-vocabulary mapper without changing the agent.

## Navigation for a G1 humanoid

For a mostly flat office or home, the practical first stack is ROS 2 Nav2:

```text
navigate_to("kitchen")
        |
SemanticMapModule: "kitchen" -> PoseStamped(map)
        |
Nav2 NavigateToPose
        |
Planner Server: global collision-free path
        |
Controller Server: local trajectory + vx, vy, yaw_rate
        |
velocity smoother + collision monitor + robot safety supervisor
        |
G1 locomotion policy / WBC velocity-reference interface
```

Recommended starting choices are a Smac 2-D global planner and the Nav2 MPPI local
controller. Treat the G1 as an omnidirectional footprint only if its locomotion
controller genuinely accepts lateral velocity. Otherwise constrain the Nav2 model.
Use a conservative humanoid footprint and much lower velocity/acceleration limits
than a wheeled base.

The adapter should call Nav2's `NavigateToPose` action and monitor feedback. Do not
have the LLM publish `/cmd_vel`. Nav2's Controller Server owns local obstacle
avoidance and publishes `TwistStamped`; a small bridge translates that into the
exact velocity-reference message consumed by your locomotion policy or WBC.

The existing sibling `robot_class` package already exposes G1 `base_velocity`
commands shaped like:

```python
robot.send_action({"base.vx": vx, "base.vy": vy, "base.vyaw": yaw_rate})
```

That is the natural output boundary for a Nav2-to-G1 bridge. The bridge must enforce
rate limits, stale-command timeouts, zero-on-disconnect, localization validity, and
an independent emergency stop.

For richer 3-D perception on NVIDIA hardware, Isaac ROS cuVSLAM can provide visual
odometry/localization and nvblox can build a TSDF/ESDF plus a planning costmap from
depth or LiDAR. Nav2 can still plan on a projected traversability/cost map. Stairs,
large height changes, stepping stones, and rough terrain require a separate
terrain/footstep or learned locomotion planner; expose those as bounded skills such
as `climb_stairs(staircase_id)`, rather than pretending a 2-D `/cmd_vel` planner is
sufficient.

Useful references:

- [Nav2 navigation servers](https://docs.nav2.org/rolling/getting_started/navigation_concepts/navigation_servers/)
- [Nav2 planner/controller selection](https://docs.nav2.org/rolling/configuration_and_development/first_time_robot_setup_guide/navigation_plugins/setup_navigation_plugins/)
- [NVIDIA nvblox](https://nvidia-isaac-ros.github.io/concepts/scene_reconstruction/nvblox/index.html)
- [Flexion Reflect v1](https://flexion.ai/news/flexion-reflect-v1.0)

Flexion publicly states that its semantic map is generated from a building scan and
can return global paths, but it does not publish the exact SLAM or planner stack.

## Manipulation after navigation

Make navigation stop at a named **approach pose**, not merely the table's center.
Then transition ownership from locomotion to a manipulation backend:

```text
navigate_to("kitchen table approach")
stabilize_for_manipulation()
inspect_workspace()
pick_object("red cup")
verify_grasp()
```

Three sensible manipulation backends are:

1. A trained VLA such as GR00T/openpi, returning whole action chunks.
2. A classical perception/grasp/motion stack: segmentation, 6-D pose or grasp
   generation, whole-body IK/planning, collision checking, and WBC tracking.
3. A RoboCurve-style frontier VLM that repeatedly observes cameras and proprioception
   and calls bounded `move_to`/`move_joints` tools.

For a G1, manipulation is possible while standing if the object is reachable, the
base is accurately staged, the gripper and cameras are calibrated, and the motion
backend preserves balance and self-collision constraints. Mobile manipulation is
harder: use a whole-body manipulation policy or planner rather than independently
commanding the waist, arms, and walking controller.

Inspect Robots currently provides a Unitree G1 arms/standing embodiment and can run
its tool-calling frontier-model agent or a GR00T policy against compatible action
spaces. Its agent issues one approved motion chunk, receives fresh images and
proprioception, and replans. It is an evaluation/runtime harness—not a complete
global navigation or whole-body autonomy stack.

- [Inspect Robots](https://github.com/robocurve/inspect-robots)
- [Inspect Robots agent plugin](https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-agent)

To integrate it here, implement `ManipulationBackend.pick()` as a client to an
Inspect Robots/VLA skill service. Keep the outer System-2 agent responsible for
mission sequencing and the inner manipulation policy responsible for short-horizon
physical control.

## Connecting real hardware safely

The dry-run modules disable approvals only in the demo CLI. Real modules should
leave `requires_approval=True` and pass an approval function:

```python
def approve(call, tool):
    return safety_supervisor.check(
        capability=call.name,
        arguments=call.arguments,
        robot_state=robot_state.latest(),
    )

agent = System2Agent(model, modules, approval=approve)
```

The approval function is only a gate. The backend and lower controller must still
enforce hard limits. First validate new modules in a mock environment, then
simulation, then attended hardware at reduced speed.

## Tests

The tests use a deterministic scripted model and do not call the network or need
third-party packages:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Selectable MuJoCo and Isaac Sim environments

The simulation entry point and Python factory accept either backend while keeping
the System-2 modules, semantic map, global planner, local-observer contract, and
tool calls unchanged:

```bash
# Lightweight/default backend
exp-agent-sim --backend mujoco --scene examples/sim_scene.json \
  --goal "kitchen table"

# RTX/PhysX backend; run from Isaac Sim's Python 3.12 environment
exp-agent-sim --backend isaac --scene /data/office/scene_bundle.json \
  --goal "west room"
```

The same choice is available when embedding the agent:

```python
from system2_agent.scene_bundle import SceneBundle
from system2_agent.sim import create_simulation_environment

scene = SceneBundle.from_json("/data/office/scene_bundle.json")
environment = create_simulation_environment(
    "isaac", scene, workspace="/home/robot/workspace", with_vision=True
)
```

An Isaac-capable scene bundle adds a backend-specific block alongside the common
navigation grid and semantic map:

```json
{
  "navigation_grid": "navigation_grid.json",
  "semantic_map": "semantic_locations.json",
  "initial_pose": {"x": 0, "y": 0, "yaw": 0},
  "isaac_sim": {
    "stage_usd": "interactive_office.usdz",
    "robot_usd": "/data/robots/g1.usd",
    "robot_prim": "/World/G1",
    "renderer": "RaytracedLighting",
    "cameras": [
      {
        "label": "g1_head_rgb",
        "prim_path": "/World/G1/head_camera",
        "width": 640,
        "height": 480
      },
      {
        "label": "g1_left_wrist_rgb",
        "prim_path": "/World/G1/left_wrist_camera"
      }
    ]
  }
}
```

The environment USD/USDZ stage owns RTX materials, lights, native Gaussian-splat
prims, and PhysX rigid/articulated objects. `robot_usd` is optional: when the G1
prim is not already in the environment stage, it is referenced at `robot_prim` at
runtime without modifying either source asset. The splat remains appearance;
collision meshes, rigid bodies, mass, friction, and joints remain physics. This
prevents a visible table from silently being treated as a contact surface.

By default, `IsaacSimBase` moves the G1 root kinematically for fast System-2 and
navigation evaluation and identifies itself as `isaac-sim-kinematic-velocity`.
It pauses PhysX during that render-only mode so an uncontrolled articulation does
not fall, and it must not be reported as learned locomotion or object-interaction
performance. Pass an `IsaacVelocityActuator` (SONIC, a perceptive locomotion
controller, or another policy bridge) to enable PhysX and execute the same bounded
velocity intents through real simulated dynamics. Interactive-object manipulation
likewise needs the existing nested manipulation module connected to an Isaac
arm/WBC control API; merely loading a rigid object does not command a grasp.

Configured Isaac cameras provide occasional RGB frames to the System-2/VLM loop.
A camera entry may carry a `mount` block -- `{"prim": "/World/G1/torso_link",
"xyz": [0.0576, 0.0175, 0.4299], "pitch_down_deg": 0, "hfov_deg": 87}` -- in
which case the runtime authors the camera prim under that link with a D455's
optics when the stage does not already have it; the first camera is the head
camera, and with the default head-camera spec the model sees its colour and
range previews (`HeadCameraBackend`) rather than every camera's RGB.
The first camera also supplies a separate depth-based local obstacle observer at
navigation control rate. It back-projects every depth pixel with the camera's
intrinsics and world pose, discards returns within 0.15 m of the floor or more
than 2 m above it, and reports the nearest remaining return inside the robot's
forward body corridor. Because the floor is rejected by height rather than by a
fixed band of image rows, a head camera pitched towards the ground does not
report the floor as an obstacle. The stage's floor is assumed to lie at z=0;
pass `IsaacDepthObserver(runtime, ground_z_m=...)` for scenes authored
otherwise. Anything lower than 0.15 m counts as floor, so kerbs and steps are a
job for terrain perception. The observer can be replaced by nvblox/ESDF or a
learned perceptive controller without changing `navigate_to`.

For recorded indoor routes, the third-person camera is selected from close
rear-quarter candidates whose sight lines are free in the navigation grid. It
falls back to an overhead view rather than rendering through a wall. Gaussian
LOD generation is deterministic and record-stratified: it deliberately avoids
global raw-opacity ranking because real PLY exports may contain saturated or
infinite opacity logits that otherwise consume the LOD budget and remove most
of the scene.

SimFoundry's native application runtime is OmniGibson on Omniverse/PhysX. Its USD
objects and Gaussian USDZ background can therefore be assembled/exported as the
Isaac stage above. Alternatively, its saved-scene JSON can continue through the
included MuJoCo importer. The semantic map and navigation grid are deliberately
external to both physics formats.

Isaac Sim is an optional, large NVIDIA runtime. Install `.[isaac]` in a supported
Linux Python 3.12 environment or run the package with an existing Isaac Sim
`python.sh`. Merely installing the base `exp-agent` package does not start or load
Isaac. Its first launch may require the operator to review and accept NVIDIA's
Omniverse license; the agent does not accept licenses automatically.

## Included G1 simulation stack

There are three deliberately distinct paths:

```text
Fast integration test
System-2 -> semantic goal -> smooth trajectory planner -> body velocity -> robot_class MuJoCo G1

Photorealistic environment evaluation
System-2 -> shared map/tools -> Isaac USD/USDZ + RTX cameras + local depth
          -> kinematic root or optional perceptive/SONIC actuator -> PhysX

Policy-faithful sim2sim (Linux + NVIDIA GPU)
System-2 -> semantic goal -> smooth trajectory planner -> SONIC planner ZMQ command
          -> kinematic planner at 10 Hz -> SONIC v1.1 at 50 Hz
          -> low-level joint commands -> official GEAR-SONIC MuJoCo G1
```

The fast path is runnable in the current workspace and is useful for agent, map,
planner, camera, and failure-recovery development. It moves the floating base
kinematically; it must not be used to claim locomotion-policy performance.

Even this path needs the G1 MJCF from the official stack: `create_simulation_environment`
loads `../GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml`.
That file exists only on the **`gear-sonic`** branch -- `main` ships
`decoupled_wbc/control/robot_model/...` instead and the MuJoCo backend will not
start against it. The meshes are Git LFS, so `git lfs` must be installed before
the clone or MuJoCo fails to parse the pointer files:

```bash
sudo apt-get install -y git-lfs && git lfs install
git clone --depth 1 --branch gear-sonic \
  https://github.com/NVlabs/GR00T-WholeBodyControl.git ../GR00T-WholeBodyControl
```

```bash
cd exp-agent
PYTHONPATH=src:../robot_class ../robot_class/.venv/bin/python \
  -m system2_agent.sim_cli --goal "kitchen table"
```

`scene_43dof.xml` defines no MuJoCo cameras of its own. The scene loader
therefore attaches the G1's head depth camera (below) to the robot at load
time, so `--with-vision` shows the model that camera by default and
`--splat`/MuGS can render from it as `--camera head_d455`. Pass `--camera` to
use some other named camera, or `--no-head-camera` to leave the model as
shipped. Headless rendering needs `MUJOCO_GL=egl`.

### The head depth camera

Every simulator here carries the same forward-facing head depth camera the
real G1 has -- a RealSense D455-shaped one: 87 x 56 degree field of view at
640 x 360, at the RealSense bracket on `torso_link`, level rather than the
hardware bracket's 48-degree downward pitch (`--head-pitch-deg` tilts it).
`system2_agent.sim.head_camera` owns it:

- `HeadCameraSpec` / `D455` describe placement and optics; `SceneLoader`
  attaches the camera to a MuJoCo robot, and an Isaac scene bundle camera with
  a `mount` block is authored under a robot link by the runtime (see below).
- `MuJoCoHeadCamera` and `IsaacHeadCamera` render colour plus metric depth,
  with returns outside the sensor's 0.4-10 m working range reported as no
  return, as a depth camera would.
- `HeadCameraBackend` gives the model the robot's own two previews under the
  labels the mission service uses -- `head_colour`, then `head_range`, the
  false-colour range image with its legend burned in -- so a mission prompt
  that works on hardware sees the same evidence in simulation.
- `HeadCameraStream` publishes those previews and the vision node's health JSON
  on the robot's topics (`/g1/d435c/preview/compressed`,
  `/g1/d435c/preview_depth/compressed`, `/object_grounding_status`) as CBOR
  over MQTT, byte-for-byte the shape the fleet agent produces.

```bash
# Stream the sim's head camera to the dashboard's broker as the sim robot:
exp-agent-sim --backend mujoco --goal "kitchen table" --stream-mqtt 127.0.0.1:1883
# then open the dashboard as  http://<host>:5173/?robot=g1-sim-0001
```

The stream connects as MQTT username `g1-sim-0001` (`--robot-id`), because the
broker's ACL only lets a `g1-*` username publish telemetry under its own id.
##### That is the same identity a fleet agent for that robot id would use;
never point the stream at a broker while the real robot of the same id is on
it. A mission service that should see the simulated camera needs
`G1_ROBOT_ID=g1-sim-0001` and `MISSION_CAMERA=1`; nothing else changes. Frames
go out at `--stream-hz` (6 by default), are never retained and never re-sent,
so a frozen picture on the dashboard means the simulator stopped, exactly as
it means the camera stopped on the robot.

To let a frontier VLM choose the semantic navigation tool and inspect fresh
MuJoCo frames:

```bash
PYTHONPATH=src:../robot_class ../robot_class/.venv/bin/python \
  -m system2_agent.sim_cli \
  --mission "go to the kitchen table and check that you arrived" \
  --model openai/YOUR_TOOL_AND_VISION_MODEL --with-vision
```

### Actual SONIC v1.1 sim2sim

The official stack has been cloned at `../GR00T-WholeBodyControl`. The latest
`../robot_class` owns SONIC v1/v1.1 variant selection, asset layout, deploy setup,
and the canonical `command`/reference protocol. This package only adds the
navigation-specific `planner` publisher. Simulation ground truth supplies
localization; on the robot, replace it with SLAM/VIO.

On a supported Ubuntu NVIDIA system:

```bash
# One setup path for cloning, v1.1 assets, deploy build, checks, and protocol smoke tests.
cd ../robot_class
python scripts/setup_sonic_deploy.py \
  --all --variant sonic_v1_1 \
  --wbc-dir ../GR00T-WholeBodyControl

# NVIDIA's simulator environment is still installed from its upstream checkout.
cd ../GR00T-WholeBodyControl
bash install_scripts/install_mujoco_sim.sh

cd ../exp-agent
PYTHONPATH=src:../robot_class:../GR00T-WholeBodyControl \
  ../GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m system2_agent.official_sonic_sim_cli --check

# Deterministic semantic navigation through the full policy:
PYTHONPATH=src:../robot_class:../GR00T-WholeBodyControl \
  ../GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m system2_agent.official_sonic_sim_cli \
  --sonic-variant sonic_v1_1 --goal "charging station"

# Preplan one collision-checked route through several semantic destinations:
PYTHONPATH=src:../robot_class:../GR00T-WholeBodyControl \
  ../GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m system2_agent.official_sonic_sim_cli \
  --route "charging station,kitchen table,home" \
  --record artifacts/sonic_semantic_route.mp4

# System-2 + live head camera + full SONIC sim2sim:
PYTHONPATH=src:../robot_class:../GR00T-WholeBodyControl \
  ../GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m system2_agent.official_sonic_sim_cli \
  --mission "go to the charging station and verify the scene" \
  --model openai/YOUR_TOOL_AND_VISION_MODEL --with-vision

# ... and stream that head camera to the dashboard while it runs:
PYTHONPATH=src:../robot_class:../GR00T-WholeBodyControl \
  ../GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m system2_agent.official_sonic_sim_cli \
  --goal "charging station" --stream-mqtt BROKER_HOST:1883
```

With the head camera attached (the default), `--with-vision` shows the model
the D455 head camera's colour and range previews instead of GEAR-SONIC's
`ego_view` ZMQ stream; `--no-head-camera` restores the upstream camera.

SONIC v1.1 is a 50 Hz whole-body motion tracker. NVIDIA's target-velocity
kinematic planner supplies its reference animation. The navigation backend uses
footprint-inflated A* as its complete global search, resamples and smooths that
seed with a collision-checked elastic-band optimizer, and tracks the resulting
trajectory with regulated look-ahead, clearance-aware speed control, and a
forward collision rollout. It emits bounded movement/facing/speed commands to
SONIC. The agent only calls `navigate_to(location, reason)`; none of the map,
trajectory, controller, or joint-level details are exposed to the model.

The `navigate_to` result includes the planning pipeline, trajectory point count,
length, and minimum map clearance. The static splat-derived grid is the current
collision source. In deployment, an SLAM/VIO pose and a live ESDF can replace
the simulation pose/grid without changing the tool contract.

### Global goals versus local perception

The navigation boundary deliberately has two different information paths:

```text
System-2 agent -> semantic map -> map-frame destination -> global trajectory
depth / LiDAR / vision costmap -> local observer -> regulated follower -> SONIC
```

`SemanticMapModule` owns language-grounded names, descriptions, affordances, and
approach poses. The agent can call `describe_location()` before `navigate_to()`.
The planner owns the building-scale collision-free route. During execution,
`LocalNavigationObserver` can provide localized obstacles, sensor health, and an
emergency-stop signal on every control tick. The regulated follower fuses those
observations into its forward collision guard. Unhealthy perception fails closed;
a temporary obstacle brakes the robot and navigation resumes when it clears.

The recorded InteriorGS demo uses a static generated grid because that dataset
does not provide a live depth stream. Real deployment must attach a depth, LiDAR,
or ESDF adapter as `local_observer`; the interface is present and tested, but RGB
images alone are not treated as metric collision geometry. Nav2 MPPI, nvblox,
X-Mobility, or another local controller can replace the regulated follower behind
the same `navigate_to(location, reason)` tool.

### Walking up to something the robot can see: `local_planner`

`navigate_to` can only reach places somebody labelled, so "go to the sink" has
nowhere to go: a sink is not a place. `local_planner(target, reason)` is the
second stage, and it never opens the semantic map.

```text
"go to the sink"
  navigate_to("pantry")            a labelled place, Nav2, as before
  observe_surroundings()           is the sink actually in frame?
  local_planner("sink")            grounds it, measures it, walks up to it
```

Inside one call: a fresh `head_colour` frame goes to a small vision model that
returns one bounding box or refuses; the box is measured against the robot's
metric depth frame (30th percentile inside the middle 60% of the box, so the
object's face wins over the background past its silhouette); the point is placed
in the frame of the robot's pose **at that instant**; a standoff pose is put
0.6 m short of it on the line from the robot, pulled back until it is somewhere
the robot could stand; A\* on the Nav2 local costmap proves a path exists; and
the pose is handed to the same `NavigationBackend.navigate()` that `navigate_to`
uses, so every pre-flight refusal (E-stop latched, not localized, SONIC
disarmed) applies unchanged.

**The model names a noun.** It never sees a cell, a coordinate or a velocity,
and the tool's schema has no place to put one. Everything metric happens inside
the call, where it is checked against the costmap before a byte is published.

**How close it will go, and why that number is not the one you asked for.** Every
standoff here is measured from `base_footprint`'s centre, and the G1's measured
footprint reaches **0.31 m** in front of that origin — so a 0.60 m standoff is
0.29 m of clear air, and a "0.30 m standoff" would put the object *inside the
robot*. Missions should therefore ask for `clearance_m`, the gap a person would
measure, and let the planner convert:

| ask | goal sent | gap you see | at worst-case arrival |
| --- | --- | --- | --- |
| `clearance_m: 0.29` (the floor) | 0.60 m | 0.29 m | 0.19 m |
| `clearance_m: 0.5` | 0.81 m | 0.50 m | 0.40 m |
| `standoff_m: 0.3` | *refused* | −0.01 m | the robot standing on it |

The floor is **derived, not chosen**: `StopZone` (0.40 m) + Nav2's arrival box
(0.10 m) + 0.10 m of margin = 0.60 m. `StopZone` halts the robot with nothing in
any log and is the only guard that reads `/scan` directly, so a goal the checker
could latch inside it is a walk that stops for no visible reason. Because the
floor is derived, the arrival box is configuration
(`MISSION_LOCAL_ARRIVAL_BOX_M`) and must mirror Nav2's `xy_goal_tolerance` — 0.10
under Unitree's gait, 0.25 under SONIC, which cannot creep into a tighter box.
Under SONIC the floor rises to 0.75 m on its own rather than promising a
clearance the backend cannot deliver; the worst-case body clearance is 0.19 m
either way.

⚠️ **0.29 m is where the robot can STAND, not how well it can see.** This G1
stack has no arm control (the prompt says so to the model), so the standoff is
the whole approach: there is no reaching step to close the last stretch. What
0.29 m buys is a head camera whose floor coverage starts at ~0.48 m still having
the object in frame on arrival, which is what makes the approach verifiable.

**One call is often one leg.** The local costmap is a rolling window a few
metres across while the camera sees across a room, so a distant target is
clamped to the longest leg that fits inside both the window and the walk budget,
and the result says `reached_standoff: false` with the metres remaining. The
model looks at the fresh frame and calls again. Each leg re-grounds on a new
picture, which is also how an approach survives the target being half-occluded
at the start. Two brakes stop that loop from running away: a budget of legs per
mission, and a check that the measured range actually shrank — a robot that is
no closer than last time is not approaching anything, and the tool says so
rather than walking the same leg again.

Ranging prefers metric depth and falls back to the costmap. The two topics the
robot publishes for it are new:

| topic | payload |
| --- | --- |
| `/g1/head/depth/compressed` | `sensor_msgs/CompressedImage`, `format: "16UC1; png"` — 16-bit PNG of **millimetres**, 0 meaning no return, downsampled to 320 px wide, in the colour camera's pixel grid |
| `/g1/head/depth_info` | `std_msgs/String`, retained JSON — `width, height, fx, fy, cx, cy, depth_scale, frame_id, camera, source` |

⚠️ **The intrinsics travel with the pixels and both are required.** The publisher
downsamples before sending, so the sensor's own `fx` is the wrong number and only
the publisher knows the right one. A frame that arrives without a model is
dropped rather than ranged against an assumption. Decoding needs no new
dependency: `system2_agent.png16` reads 16-bit greyscale PNG with `zlib` alone.

⚠️ **Aligned depth, in the colour grid.** "The box is at (u, v), so the range is
`depth[v][u]`" is only true because the driver resamples depth into the colour
camera's grid. Raw depth is a different imager a baseline away, and indexing it
with a colour-frame box is wrong by a parallax that grows as the object gets
closer — which is exactly the regime this tool works in.

Without depth on the link the tool ranges by casting the pixel's bearing into the
local costmap until a lethal cell, and says `method: "costmap_raycast"` in the
result. That cannot tell the target from anything else on the same bearing, so a
chair in between yields a shorter, safer standoff — which is also how "get as
close as you can" behaves when something is in the way: the goal is placed
against the **blocker**, not the target, and then pulled back until the body
fits. `pulled_back_m` and `achieved_standoff_m` in the result say how much the
costmap took, and `body_clearance_m` says what the robot actually ended up with.
A target the obstacle layers never marked (glass, a thin rail) is a refusal
rather than a guess. When depth
does answer, the ray-cast still runs as a cross-check and disagreement beyond
0.5 m is reported.

The simulator publishes both topics through `HeadCameraStream`, from the same
capture as the previews, so a mission that works in the SONIC sim sees the same
wire on hardware. `exp-agent-g1 local-check` prints the costmap, the depth
status and an ASCII picture of the grid with the robot on it, and can dry-run a
goal placement for a given bearing and range — read-only, no model call, nothing
published. Configuration lives in `.env.example` under `MISSION_LOCAL_*` and
`MISSION_GROUNDING_*`.

The official deploy build requires Ubuntu, CUDA and the exact supported
TensorRT version. It cannot be built or policy-tested on macOS. The lightweight
MuJoCo integration above does run on macOS.

On macOS, launch camera/viewer runs from an interactive desktop session with
MuJoCo's `mjpython`; a headless shell has no CoreGraphics context. Navigation
without `--with-vision` does not need a graphics context.

### Going to look for it: `find_object`

`local_planner` needs the thing in the current frame. When it is not there, the
robot has to go and look — and the whole question is *where*.

The scene that motivates it: the robot arrives at a labelled place and faces a
counter, and the object is on the floor behind it. On the local costmap that
strip of floor reads **free**, because nothing has ever sensed it and Nav2's
local costmap does not track unknown space. A candidate generator reading only
"is this cell free" therefore proposes standing *inside* the region nothing has
looked at — useless, and the one place a biped should not walk.

So `find_object(target, reason, hint)` keeps its own memory: a visibility map
built by ray-casting the camera cone through the costmap, so a ray **stops** at
the counter. The strip behind it stays unobserved, and the cells that are
observed-and-free beside it — the frontier, which in that scene lines the
counter's two ends — are where the robot goes to see more.

```text
"find the sofa"
  navigate_to("living_room")     a labelled place, Nav2, as before
  find_object("sofa")            turns, then walks to where it can see more
  local_planner("sofa")          now that it is in frame, close the last metres
```

One call runs the whole loop on the host: ground on the current frame, turn four
quarters grounding each one, then repeatedly pick a standpoint, walk one Nav2
goal, and ground again. **One approval buys every leg**, and the approval names
the budget — at most *N* legs inside *R* metres for at most *T* seconds
(`MISSION_SEARCH_MAX_LEGS`, `_RADIUS_M`, `_MAX_SECONDS`).

That is the deliberate trade. A tool per leg would cost a reasoning turn, an
operator approval and a retained image *per leg*; the realistic response to eight
approvals per search is switching the gate to auto, which is strictly less safe
than one approval a person actually reads.

**It still never picks a coordinate.** Candidate standpoints come from geometry
and are verified against the same inflated costmap `local_planner` uses. A cheap
vision model may only choose *among* them, and `hint` is words, not a position —
"probably behind the kitchen island" breaks a tie between places the map already
says are worth looking.

`outcome` is the field that matters: `found` (in frame now, hand off to
`local_planner`), `exhausted` (nowhere left within the radius reveals anything
new, so the object is not in this part of the room), `budget`, or `cancelled`.

⚠️ **Cancelling this one stops the robot.** Everywhere else in this stack,
stopping a mission does not stop a goal Nav2 already accepted. A tool that walks
six legs under one approval cannot inherit that, so cancelling publishes the
robot's **current pose** as a new goal, which supersedes the leg in flight. That
is still inside the one-entry publish table; the emergency stop remains the only
guarantee.

⚠️ **A second search from the same spot is refused.** Measured on the pose, not
on the last result. Without that brake a mission loops: `find_object` reports
"found, nothing moved", `local_planner` refuses because the approach is blocked,
and the model — reading a refusal that says *not visible* — calls `find_object`
again. Observed in the SONIC sim before the brake existed: five
find_object/local_planner pairs, two of them zero-motion, no step taken. The
refusal now names the real problem, which is the approach rather than the
finding.

### Two head cameras

The G1 ships with a D435i pitched **47.87°** at the floor — that is what feeds
`/scan_depth` and the trip-hazard layer — and a D455 mounted level above it. The
two see different worlds and neither is a superset:

| | level D455 | pitched D435i |
| --- | --- | --- |
| field | 87° × 56° | 69.4° × 42.5° |
| floor visible from | 2.44 m | 0.48 m to 2.51 m |
| a cup on a 0.9 m counter | 0.76–1.41 m | out of frame past 0.51 m |
| a cup on a 0.45 m table | never in frame | 0.60–1.41 m |

On the robot the pitched camera's **colour has never been published** — the
vision node carries one hardcoded pair of preview topic names. The **simulator
publishes both**, on a second topic set (`/g1/floor/preview/compressed`,
`/g1/floor/preview_depth/compressed`, `/g1/floor/depth/compressed`,
`/g1/floor/depth_info`, `/g1/floor/status`), and the dashboard mounts its camera
panel once per camera. Making this true on hardware is a robot-side change
nobody has made, so `topics.json` deliberately carries no floor rows and the
real-robot profile resolves them to `""`.

⚠️ **The depth topics are separate and that is load-bearing.** Aligned depth is
aligned to its *own* colour frame. Ranging a box grounded in the level camera's
picture against the pitched camera's depth is wrong by the 5 cm offset and the
48° between them — and wrong in a way that grows as the object gets nearer,
which is exactly the regime an approach works in.

## COMPASS / X-Mobility versus Nav2

The NVIDIA post linked in the design discussion shows COMPASS adapting a
pretrained X-Mobility policy to a new robot/environment with residual RL. It is
not a minimal replacement for every part of Nav2. The open X-Mobility runtime
takes one RGB image, odometry speed, and a 20-point route, then emits forward
and yaw velocity. In its mapless example that route is merely a straight line
to the goal; with a real building map, another component must still produce a
safe global route.

X-Mobility is attractive later as a learned **local controller** because it can
react directly to vision and transfer between embodiments. For an understandable
first stack, the included global search + trajectory optimizer + regulated
controller is smaller and deterministic.
Nav2 becomes worthwhile when lifecycle management, recovery behaviors, mature
costmaps, multiple planner/controller plugins, ROS tooling, and production
monitoring matter.

Relevant upstream implementations:

- [NVIDIA COMPASS post](https://x.com/NVIDIARobotics/status/2093381258243895657)
- [NVlabs X-Mobility](https://github.com/NVlabs/X-MOBILITY)
- [NVIDIA GEAR-SONIC / SONIC v1.1](https://github.com/NVlabs/GR00T-WholeBodyControl)

To add X-Mobility or COMPASS, implement the `MobileBase` boundary or replace
`PathFollower`; keep `AStarPlanner` (or Nav2's global planner) as the route
source. COMPASS training itself is not yet present in the public X-Mobility
repository as of this implementation, so it is intentionally not vendored.

## Gaussian splats in MuJoCo

The workspace also contains `../MuGS`. It does not turn Gaussian ellipsoids into
contact geometry. It renders a 3DGS PLY as the background, renders robot and
interactive objects in MuJoCo, and composites the images using segmentation.
`MuGSCamera` wraps its `GaussianSensor` API for VLM observations.

A scene therefore contains three aligned representations:

1. MJCF plus meshes/primitives for collision and dynamics.
2. An inflated 2-D occupancy/traversability grid for this planner.
3. An optional 3DGS PLY for photorealistic RGB.

`SceneLoader` keeps these assets independent from the robot. The scene manifest
uses `external_mjcf` for a scene-only MJCF fragment, or `collision_mesh` for a
raw static collision mesh. At startup the loader attaches those assets to the
unchanged G1 MJCF with MuJoCo `MjSpec`, validates the result, and gives the
simulator a temporary composed model. Nothing is copied into or edited inside
the SONIC/G1 repository.

```json
{
  "external_mjcf": "../assets/scenes/apartment/scene.xml",
  "collision_mesh": null,
  "gaussian_splat": "/data/apartment/point_cloud.ply",
  "navigation_grid": "apartment_grid.json",
  "semantic_map": "apartment_locations.json"
}
```

For a raw mesh, `collision_mesh` may instead be an object containing `path`,
`scale`, `position`, and a MuJoCo `wxyz` `quaternion`. A splat alone supplies
RGB but no contact surface; pair it with aligned MJCF/mesh collision geometry
and a grid. This makes swapping a ProcTHOR/MolmoSpaces MJCF, a reconstructed
mesh, or a splat-plus-mesh bundle a manifest change rather than a robot-model
change.

For the DISCOVERSE lab3 example, the occupancy layer is derived from the splat
itself. Gaussian centers between 0.18 m and 1.4 m above the aligned floor are
accumulated into 10 cm cells; dense cells become obstacles and are inflated by
the G1 footprint before A*. Rebuild the checked-in derived map with:

```bash
../GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/build_splat_navigation_grid.py \
  --scene examples/sim_scene.json \
  --output examples/lab3_splat_navigation_grid.json
```

This provides planning collision constraints from the scan, but Gaussian
ellipsoids are still visual—not MuJoCo contact shapes. Add aligned MJCF
geometry when physical obstacle contact is required.

`examples/sim_scene.json` is the manifest and includes a `world_T_gs`-style 4x4
alignment field. Replace its `gaussian_splat` value with a PLY path and pass a
named MJCF head camera:

```bash
pip install -e ../MuGS
PYTHONPATH=src:../robot_class ../robot_class/.venv/bin/python \
  -m system2_agent.sim_cli --mission "inspect the kitchen" \
  --model openai/YOUR_TOOL_AND_VISION_MODEL \
  --splat /data/kitchen/point_cloud.ply --camera head_camera
```

MuGS currently requires PyTorch/gsplat and an NVIDIA CUDA GPU for the 3DGS
renderer. A splat reconstructed from video must be metrically aligned, and a
collision mesh or authored MuJoCo geometry must still be supplied separately.

- [MuGS](https://github.com/Renforce-Dynamics/MuGS)

## Nested visual manipulation agent

`ManipulationModule` can expose one blocking `manipulate(instruction, reason)`
tool backed by `AgenticManipulationBackend`. The outer System-2 mission agent
does not micromanage a grasp. A separate `NestedManipulationAgent` repeatedly:

1. receives fresh head, left-wrist, and right-wrist frames plus proprioception;
2. selects one bounded Cartesian hand delta or Dex1 aperture command;
3. executes it through a `ManipulationEmbodiment` implemented by the robot/WBC;
4. observes again and continues until the embodiment independently verifies
   completion or the nested agent returns a safe failure.

This is an Inspect-Robots-style policy/embodiment boundary. A frontier VLM can
be the policy, but its 1-3 Hz tool loop is not the balance controller. SONIC
continues running at its native control rate while the nested agent supplies
sparse, collision-gated end-effector intent.
`WbcCartesianManipulationEmbodiment` connects this loop to robot-class's
CAP-X-compatible `get_current_wrist_pose`, `goto_pose`, `set_gripper`, and
head/wrist camera APIs.

`SonicUpperBodyControlApi` executes accepted IK solutions through the upstream
planner message's native 17-DOF `upper_body_position` and
`upper_body_velocity` fields. It maps robot-class's 29-joint MuJoCo ordering to
SONIC's interleaved IsaacLab ordering, interpolates each accepted target, and
ends with a zero-velocity hold. Dex1 aperture remains a separately bounded
robot-class command. This supports standing manipulation and simultaneous
upper-body references during walking; it does not bypass IK, collision checks,
joint limits, or SONIC's high-rate stabilization.

## Importing SimFoundry scenes into MuJoCo

`SimFoundryMuJoCoImporter` reads the published SimFoundry saved-scene JSON
without modifying its USD assets. It converts USD/PLY geometry into a generated
OBJ cache, preserves object pose, scale, mass and friction, creates free MuJoCo
bodies for interactable props, applies the authored support plane, emits a
semantic object map, and turns occupied navigation cells into conservative
invisible collision proxies. A separately aligned 3DGS PLY remains the
photorealistic visual background.

```bash
python -m venv .venv_scene_import
pip install -e '.[scene-import]'

PYTHONPATH=src .venv_scene_import/bin/python \
  -m system2_agent.simfoundry_importer \
  assets/simfoundry/assets/scenes/YAM/stack_dishware/stack_dishware_scene_state_latest.json \
  assets/generated/lab3_simfoundry_dishware \
  --gaussian-splat ../MuGS/assets/scenes/discoverse_unpacked/lab3/point_cloud.ply \
  --gaussian-alignment-json examples/sim_scene.json \
  --navigation-grid examples/lab3_splat_navigation_grid.json \
  --without-background-mesh \
  --scene-offset -1.15 -0.25 0.77
```

The generated manifest is
`assets/generated/lab3_simfoundry_dishware/scene_bundle.json`. It combines the
Lab3 splat, physical navigation collisions, a finite tabletop support surface,
and SimFoundry's plate, bowl and mug as separate rigid bodies. Lab3 and the YAM
workcell are different captures, so this is a compositor/physics diagnostic,
not a coherent scene and not a valid visual benchmark. SimFoundry's
currently published examples use mesh backgrounds; its automatic splat
background release is still pending. The importer intentionally accepts an
metric 3DGS layer only when its registration belongs to the same capture.

## Coherent bounded indoor scene

The prepared InteriorGS scene uses its matching Gaussian and Habitat navmesh in
one coordinate frame. Build a deterministic 300k-Gaussian LOD and conservative
MuJoCo collision layer without running CUDA:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 60s \
  .venv_scene_import/bin/python scripts/import_habitat_gs_indoor.py \
  assets/scenes/habitat_gs/train/interior_0048_839893 \
  assets/generated/interior_0048_coherent \
  --max-gaussians 300000 \
  --initial-pose X Y YAW
```

The importer centers the immutable scene transform on a high-clearance cell in
the largest connected region and never edits the source scene. Geometric clearance
does not prove that a Gaussian reconstruction is visually complete. Probe the
candidate spawn and destinations, then pass the verified pose through
`--initial-pose`; the manifest records whether it was operator-verified. Runtime
checks reject a spawn outside the navigable component or without footprint
clearance, and the simulator persists it through MuJoCo resets.

InteriorGS is a navigation asset: visible furniture is baked into the splat and
is not independently movable. Manipulation requires rigid props reconstructed
and registered from the same room (for example a SimFoundry capture) or a
coherent rigid-object scene such as ReplicaCAD. Treating pixels from a splat as
free bodies would produce the floating-object error this architecture forbids.
