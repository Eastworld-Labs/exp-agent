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

The default CLI is dry-run only. Separate simulation CLIs connect to MuJoCo, and
the SONIC bridge is opt-in; nothing targets physical hardware by default.

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

Python 3.11 or newer is required. The package itself has no runtime dependencies.

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

## The System-2 loop

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

## Included G1 simulation stack

There are two deliberately distinct paths:

```text
Fast integration test
System-2 -> A* -> path follower -> body velocity -> robot_class MuJoCo G1

Policy-faithful sim2sim (Linux + NVIDIA GPU)
System-2 -> A* -> path follower -> SONIC planner ZMQ command
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
  --sonic-variant sonic_v1_1 --goal "kitchen table"

# System-2 + live head camera + full SONIC sim2sim:
PYTHONPATH=src:../robot_class:../GR00T-WholeBodyControl \
  ../GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m system2_agent.official_sonic_sim_cli \
  --mission "go to the kitchen table and verify the scene" \
  --model openai/YOUR_TOOL_AND_VISION_MODEL --with-vision
```

SONIC v1.1 is a 50 Hz whole-body motion tracker. NVIDIA's target-velocity
kinematic planner supplies its reference animation. The global A* planner in
this package supplies the route; the local follower turns that route into
bounded movement/facing/speed commands. The agent only calls `navigate_to`.

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
first stack, the included A* + path follower is much smaller and deterministic.
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
