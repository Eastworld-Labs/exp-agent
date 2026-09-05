# G1 Dex-1 loco-manipulation experiments

Two isolated MuJoCo experiments using **native SONIC 1.1 planner + three-point
teleoperation**, not arm IK or joint-space playback:

| Task | Goal | Private physical success condition |
| --- | --- | --- |
| `tabletop` | Pick red block; place on green pad | Object lifted above 0.80 m, then released on the pad, inside its boundary and stationary for 2 s |
| `floor_basket` | Squat, grasp both handles, stand and lift | Pelvis lowered below 0.64 m then above 0.70 m; basket centre above 0.50 m, level within 20°, stationary with both hands contacting for 2 s |

The basket is approximately 0.54 kg. Both tasks use actual MuJoCo collision,
friction and gravity. There are no grasp welds, object teleportation, root fixing
or gravity compensation during task execution. A startup-only pelvis support is
removed before a two-second unsupported standing preflight. Falls invalidate the
result. `finish` from an agent is recorded separately from physical success.

## Current validation status

Implemented: dynamic scene construction, 29 body motor + 4 finger actuator
mapping, three-camera rendering, restricted observation boundary, private task
evaluator, smooth native VR/planner packets, loopback DDS physics integration,
and the existing model-portable agent loop.

Verified locally: both models compile; all three camera views render; unit tests
cover transforms, packet shapes, bounds, preflight gating and observation isolation.
**Not yet verified:** unsupported SONIC tracking on this Dex-1 body, squat/reach
coordination, contact accuracy, successful pickups, or VLM task success rates.
The development host is macOS; official SONIC execution requires a Linux NVIDIA
GPU host and the encoder/decoder/planner weights, which are absent locally.
This is an experimental harness, not a demonstrated pick-and-place controller.

## Robot and camera provenance

Reuse the local `g1_sim_pipeline/models/dex1_urdf` asset directory. It must contain
`g1_dex1_converted.xml` and `meshes/`. Do not use its replay program as a dynamics
controller: the converted XML has a fixed root and no motor actuators.

The source robot is Unitree's `g1_29dof_mode_15_with_dex1_1` (29 body joints,
two coupled sliding fingers per hand, 5010 wrists), described in the
[official G1 model catalogue](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/README.md).
Our composer restores the free pelvis/inertia and adds torque/position actuators
without modifying the source files. The SONIC training embodiment differs; mass,
inertia and actuator transfer must be checked before interpreting failures.
The finger aperture is a **simulation-normalized jaw target**, not the real
Dex-1 DDS motor value (0..5.5). It is not sent through SONIC's seven-joint Dex3
hand fields.

The local MountCamera dataset has wrist and head streams, but it does not provide
a verified camera calibration here. Camera mounts/FOV and the ~0.13 m wrist-to-
pinch offset are **experiment approximations**, not measured hardware extrinsics.
We replace the replay camera angles with forward/downward robot-mounted views:

- `head_camera`: torso + (0.08, 0, 0.42) m; 35° down; 90° vertical FOV.
- `cam_left_wrist`, `cam_right_wrist`: wrist-yaw + (0.045, 0, 0.055) m;
  25° down; 90° vertical FOV.

Exactly these three cameras exist in the generated model. The agent sees only
their RGB images and robot proprioception. No external/overview camera, depth,
segmentation, object pose, contact identity, semantic map or success oracle is
exposed. Private evaluator truth goes only to `result.json` and
`private_telemetry.jsonl`, not model messages. Do not add either file to a prompt.

## Local scene previews

From `exp-agent`, install into an isolated environment:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '../robot_class' -e '.[locomanipulation]' pytest

.venv/bin/python -m system2_agent.experiments.locomanipulation \
  --task tabletop --preview --output artifacts/tabletop-preview
.venv/bin/python -m system2_agent.experiments.locomanipulation \
  --task floor_basket --preview --output artifacts/basket-preview
```

Use `--dex1-assets /path/to/dex1_urdf` on another machine. Large external meshes
are deliberately not copied into this repo. Copy the complete local asset
directory, not just its XML. The default finds the sibling `g1_sim_pipeline`.
Output directories must be new/empty so previous evidence is not overwritten.
On macOS the renderer needs graphics access; on headless Linux use `MUJOCO_GL=egl`.
Preview writes three JPEGs and an explicit `preview_only` result with no success
claim. `--check` compiles without rendering and reports missing SONIC files.

## Linux SONIC run

First prepare the existing official SONIC 1.1 installation, including
`policy/sonic_v1_1/model_{encoder,decoder}.onnx`, its observation YAML and the shared
`planner/target_vel/V2/planner_sonic.onnx`. Use the existing `robot_class` SONIC
setup instructions; this harness does not download weights or build TensorRT.
Run all three processes **on the same isolated simulation host**. Do not run a
second navigation publisher/simulator, or a real robot on DDS domain 0.

Terminal 1, from `exp-agent` (use the upstream simulation environment with DDS):

```bash
../GR00T-WholeBodyControl/.venv_sim/bin/python scripts/sonic_dds_sidecar.py
```

Terminal 2, from `exp-agent` (official deployment, explicitly SONIC 1.1):

```bash
python ../robot_class/scripts/setup_sonic_deploy.py \
  --wbc-dir ../GR00T-WholeBodyControl --launch-deploy \
  --variant sonic_v1_1 --robot-interface sim \
  --deploy-input-type zmq_manager \
  --deploy-zmq-host 127.0.0.1 \
  --deploy-planner planner/target_vel/V2/planner_sonic.onnx
```

Accept the deployer's startup prompt. It may wait for simulator state until
terminal 3 starts. The harness sends start commands while supported, removes
support, verifies standing, then enables native three-point control. There is
currently no model-version handshake over DDS: launching the correct bundle
above is required; the harness label alone does not prove the loaded version.

Terminal 3, from `exp-agent`:

```bash
MUJOCO_GL=egl .venv/bin/python -m system2_agent.experiments.locomanipulation \
  --task tabletop --connect-sonic \
  --actions experiments/locomanipulation/hold.json \
  --record-video \
  --output artifacts/tabletop-hold-001
```

`--record-video` writes `camera_evidence.mp4` continuously. It shows the same
three public robot cameras available to the agent, an evidence-only tracking
view, and the active `move_to` step with its exact parameters. The observer view
is rendered with a free MuJoCo camera: it is not a model camera, agent input or
tool observation. Private evaluator state is never overlaid. Use `--record-fps`
(1..15, default 5) to bound render load.

`hold.json` is only a standing/three-point/gripper transport diagnostic, **not a
scripted pickup or an agent benchmark**. First confirm it remains standing and
inspect measured wrist errors and private telemetry. Next test small wrist moves
and incremental squats before trying contact. No fallback WBC is substituted if
SONIC fails. The sidecar stops relaying stale DDS commands, and the harness aborts
on stale UDP motor commands, invalid motor values, falls or episode timeout.

Then run the vision agent, using an existing configured exp-agent model route:

```bash
MUJOCO_GL=egl .venv/bin/python -m system2_agent.experiments.locomanipulation \
  --task tabletop --connect-sonic --model "$SYSTEM2_MODEL" \
  --max-model-calls 30 --output artifacts/tabletop-agent-001
MUJOCO_GL=egl .venv/bin/python -m system2_agent.experiments.locomanipulation \
  --task floor_basket --connect-sonic --model "$SYSTEM2_MODEL" \
  --max-model-calls 40 --episode-seconds 480 --output artifacts/basket-agent-001
```

Set the corresponding provider API key (see the main README); optional
`--base-url` and `--api-key-env` follow the existing model client. The physics
continues while the model thinks. Each completed action stops commanded walking
and keeps streaming the hand/body hold target; the total wall-clock episode is
bounded. A model-requested finish is **not** an evaluation pass.

The CLI automatically loads the nearest `.env` without overriding variables
already exported by the process supervisor. Set `SYSTEM2_MODEL` plus the matching
provider key there to omit `--model`; `.env` is gitignored. Known provider routes
use `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, or
`OPENROUTER_API_KEY`. `SYSTEM2_API_KEY` is only the fallback for a custom
OpenAI-compatible `SYSTEM2_BASE_URL`, unless selected explicitly with
`--api-key-env`.

## Action representation and source contract

The agent's single `move_to` action contains optional `left_wrist`/`right_wrist`
poses, optional normalized gripper apertures, explicit `body` intent, and
`duration_s`. Wrist poses are **absolute in the full pelvis frame sampled at
action start**, with `position_m` and `quaternion_wxyz`; they target the wrist-yaw
body, not the gripper pinch centre. Omitted wrists hold their measured world poses.
Targets are transformed to world once and re-expressed relative to the current
root on every stream update. Quintic translation and shortest-path quaternion
interpolation run below the slow VLM loop.

Body modes are initially restricted to idle, slow-walk and squat. Squats require
an explicit height in [0.40, 0.80] m and changes <=0.15 m per call. Wrist moves
are <=0.25 m per call and <=0.20 m/s peak; walking <=0.15 m/s and <=0.25 m per call.
These are experiment bounds, **not collision avoidance or feasibility guarantees**.
The planner does not see objects or automatically select a squat for a low target.
There is no contact/force-aware load planner yet. The next baseline after standing
is scripted tracking; do not attribute all subsequent failure to the VLM.

The adapter follows the checked upstream native path:

- `robot_class.robot.ZmqSonicPlannerBridge`: owns validation, canonical planner
  packet serialization, ZMQ lifecycle and SONIC start/stop commands. This
  experiment submits namespaced robot actions and does not own the controller
  transport.

- `gear_sonic/scripts/pico_manager_thread_server.py`, `PLANNER_VR_3PT`:
  planner fields and `vr_position`/`vr_orientation` in one message.
- `gear_sonic_deploy/.../include/input_interface/zmq_manager.hpp`:
  three-point presence selects encoder mode 1; wrist targets are not neural
  kinematic-planner inputs.
- `gear_sonic_deploy/.../src/g1_deploy_onnx_ref.cpp`,
  `GatherVR3PointPosition/Orientation`: virtual wrist offsets
  left=(0.18,-0.025,0), right=(0.18,+0.025,0), head=torso+(0,0,0.35),
  inverse full-root transformation, quaternion wxyz.
- `policy/release/observation_config_sonic_v1_1.yaml`: lower-body references
  plus three-point targets and the deploy-owned heading-anchor observation.

The default head landmark follows the root. Optional `body.head_orientation_wxyz`
requests a bounded waist lean (<=35°), using upstream's root-to-torso/neck chain;
it is not an extra camera or a neck actuator. Dex-1 fingers use their own MuJoCo actuators; their four joint indices
must never be packed into the 29-body-joint DDS state.

Run `pytest -q`. Tests requiring local robot meshes explicitly skip when those
external assets are unavailable. Do not treat passing unit tests or preview
images as evidence that learned whole-body lifting works.
