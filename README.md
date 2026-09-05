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
The first camera also supplies a separate depth-based local obstacle observer at
navigation control rate. That observer can be replaced by nvblox/ESDF or a learned
perceptive controller without changing `navigate_to`.

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
kinematically; it must not be used to claim locomotion-policy performance:

```bash
cd exp-agent
PYTHONPATH=src:../robot_class ../robot_class/.venv/bin/python \
  -m system2_agent.sim_cli --goal "kitchen table"
```

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
the canonical `command`/reference protocol, and the native `planner` publisher.
This package converts navigation and manipulation intent into namespaced
robot_class actions; it does not construct SONIC wire packets or own the ZMQ
transport. Simulation ground truth supplies localization; on the robot, replace
it with SLAM/VIO.

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
```

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

The official deploy build requires Ubuntu, CUDA and the exact supported
TensorRT version. It cannot be built or policy-tested on macOS. The lightweight
MuJoCo integration above does run on macOS.

On macOS, launch camera/viewer runs from an interactive desktop session with
MuJoCo's `mjpython`; a headless shell has no CoreGraphics context. Navigation
without `--with-vision` does not need a graphics context.

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

## G1 Dex-1 manipulation experiments

The isolated [loco-manipulation experiments](experiments/locomanipulation/README.md)
provide a three-camera tabletop pick-and-place scene and a two-hand floor-basket
pickup scene, with native SONIC 1.1 planner/three-point commands and private
physics scoring. Scene previews work locally; learned-controller task execution
requires the Linux SONIC runtime and has not yet been validated on the Dex-1 model.
