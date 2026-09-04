"""`exp-agent-g1` -- run the mission service, or drive it from a terminal.

    exp-agent-g1 serve --nav-dir ~/g1_auto_navigation --env-file .../webui/.env.local
    exp-agent-g1 run "go to kitchen"                 # the real robot; approves each step
    exp-agent-g1 run --robot sim "go to kitchen_2"   # the SONIC simulator
    exp-agent-g1 watch
    exp-agent-g1 stop
    exp-agent-g1 config [--robot sim]

##### `run` CAN WALK THE ROBOT. ##### The default gate stops at every step that
would, and asks here. `--gate auto` does not ask; `--gate dry-run` publishes
nothing at all and is the way to exercise a prompt change for free.

`--robot` takes a target's short name (`real`, `sim`) or its broker id
(`g1-0001`, `g1-sim-0001`). Unnamed means the real robot.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .config import ServiceConfig
from .service import GATES, build_runner, serve


def _request(url: str, path: str, payload: dict | None = None, timeout: float = 10.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read() or b"{}")
        except ValueError:
            return {"ok": False, "error": f"HTTP {exc.code}"}
    except urllib.error.URLError as exc:
        return {"ok": False, "error": f"cannot reach the mission service at {url}: {exc.reason}"}


def _stream(url: str, since: int = 0, on_pending=None, mission_id: str = "") -> int:
    """Follow the transcript. Returns a process exit status.

    `since` defaults to the CURRENT sequence number rather than zero: the log
    is one continuous record across missions, and replaying every earlier
    mission into a terminal that asked to watch this one is noise."""
    if since <= 0:
        state = _request(url, "/mission/state")
        since = max(0, int(state.get("seq") or 0) - (0 if mission_id else 60))
    request = urllib.request.Request(f"{url.rstrip('/')}/mission/events?since={since}")
    try:
        response = urllib.request.urlopen(request, timeout=None)
    except urllib.error.URLError as exc:
        print(f"cannot reach the mission service at {url}: {exc.reason}", file=sys.stderr)
        return 1
    status = 1
    with response:
        for raw in response:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if not line.startswith("data: "):
                continue
            frame = json.loads(line[len("data: "):])
            kind = frame.get("t")
            if kind == "mission":
                print(f"mission: {frame.get('text')}   [robot: {frame.get('robot')}  "
                      f"gate: {frame.get('gate')}]")
            elif kind == "reasoning":
                print(f"\n  …{str(frame.get('text'))[:400]}")
            elif kind == "say":
                print(f"\n{frame.get('text')}")
            elif kind == "call":
                mark = "  ##### MOVES THE ROBOT" if frame.get("movesRobot") else ""
                print(f"\n> {frame.get('name')}({_args(frame.get('args'))}){mark}")
            elif kind == "gate":
                print(f"  gate: {frame.get('verdict')} ({frame.get('mode')})")
            elif kind == "result":
                mark = "ok" if frame.get("ok") else "REFUSED"
                print(f"  {mark}: {frame.get('summary')}")
            elif kind == "note":
                print(f"  ! {frame.get('text')}")
            elif kind == "pending" and on_pending is not None:
                on_pending(frame.get("pending"))
            elif kind == "end":
                print(f"\n== {frame.get('status')}: {frame.get('summary')}")
                print(f"   {frame.get('modelCalls')} model calls   usage: {frame.get('usage')}")
                status = 0 if frame.get("status") == "completed" else 2
                break
    return status


def _args(args) -> str:
    if not isinstance(args, dict):
        return ""
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="exp-agent-g1", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve_cmd = sub.add_parser("serve", help="run the mission service")
    serve_cmd.add_argument("--nav-dir", default=".", help="the g1_auto_navigation checkout")
    serve_cmd.add_argument("--env-file", default=None, help="e.g. webui/.env.local")
    serve_cmd.add_argument("--bind", default=None)
    serve_cmd.add_argument("--port", type=int, default=None)
    serve_cmd.add_argument("--camera", action="store_true", default=None,
                           help="offer head-camera frames to the model (the default; "
                                "MISSION_CAMERA=0 turns it off)")
    serve_cmd.add_argument("--no-camera", action="store_true", default=False)
    serve_cmd.add_argument("--sim-robot-id", default=None,
                           help="broker id of the simulator target (default g1-sim-0001; "
                                "'none' disables the target)")
    serve_cmd.add_argument("--sim-map", default=None,
                           help="which maps/<map>.places.json the simulator plans over")

    check_cmd = sub.add_parser(
        "local-check",
        help="what local_planner can see right now (READ-ONLY: publishes nothing)",
    )
    check_cmd.add_argument("--nav-dir", default=".")
    check_cmd.add_argument("--env-file", default=None)
    check_cmd.add_argument("--robot", default="", help="real | sim | a broker id")
    check_cmd.add_argument("--bearing-deg", type=float, default=None,
                           help="with --range-m, plan to a target on this bearing "
                                "(+left) WITHOUT grounding and WITHOUT publishing")
    check_cmd.add_argument("--range-m", type=float, default=None)
    check_cmd.add_argument("--wait-s", type=float, default=5.0,
                           help="how long to let messages arrive before reporting")

    for name, help_text in (
        ("run", "start a mission and follow it"),
        ("watch", "follow whatever is running"),
        ("stop", "end the running mission"),
        ("config", "what the service is configured with"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--url", default="http://127.0.0.1:8765")
        if name in ("run", "config"):
            cmd.add_argument("--robot", default="",
                             help="real | sim | a broker id; unnamed = the real robot")
        if name == "run":
            cmd.add_argument("mission", nargs="+")
            cmd.add_argument("--gate", choices=GATES, default="confirm")

    args = parser.parse_args(argv)

    if args.command == "serve":
        overrides = {
            k: v for k, v in
            (("bind", args.bind), ("port", args.port), ("camera", args.camera))
            if v is not None
        }
        if args.no_camera:
            overrides["camera"] = False
        if args.sim_robot_id is not None:
            overrides["sim_robot_id"] = (
                "" if args.sim_robot_id.lower() in ("none", "off", "") else args.sim_robot_id)
        if args.sim_map is not None:
            overrides["sim_map"] = args.sim_map
        config = ServiceConfig.from_env(
            nav_dir=args.nav_dir, env_file=args.env_file, overrides=overrides)
        runner = build_runner(config)
        server = serve(config, runner)
        # ##### LINKS FIRST. ##### The dashboard will not offer Start until this
        # service reports a link to the broker; a link opened lazily at the first
        # mission is a Start nobody can ever press. See service.py's header.
        runner.connect()
        # flush: this banner is the operator's only confirmation of what the
        # service loaded, and Python block-buffers stdout when it is a pipe --
        # which is exactly how `./g1 mission --serve` runs it.
        print(f"[mission] http://{config.bind}:{config.port}", flush=True)
        print(f"[mission] model  : {config.model}   key: "
              f"{'present' if config.api_key else '##### MISSING #####'}")
        print(f"[mission] broker : {config.broker}:{config.broker_port}   "
              f"camera: {'on' if config.camera else 'off'}")
        for target in config.targets.values():
            places = runner.semantic_map(target.robot_id)
            print(f"[mission] {target.name:<5}: {target.robot_id}   pose {target.pose_topic}   "
                  f"map {config.map_for(target) or '(none selected)'}   "
                  f"places: {', '.join(places.names()) or '(none labelled)'}")
            for problem in places.errors:
                print(f"[mission]   ! {problem}")
        print("[mission] ##### A mission can publish /goal_pose, which WALKS THE ROBOT.",
              flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[mission] stopping")
            runner.stop()
        return 0

    if args.command == "local-check":
        return _local_check(args)

    if args.command == "config":
        query = f"?robot={args.robot}" if args.robot else ""
        print(json.dumps(_request(args.url, f"/mission/config{query}"), indent=2))
        return 0

    if args.command == "stop":
        print(json.dumps(_request(args.url, "/mission/stop", {}), indent=2))
        return 0

    if args.command == "watch":
        return _stream(args.url)

    # run
    mission = " ".join(args.mission)
    # Read the cursor BEFORE starting, so the stream begins at this mission's
    # first frame and not sixty frames into the previous one.
    before = int(_request(args.url, "/mission/state").get("seq") or 0)
    started = _request(args.url, "/mission/start",
                       {"mission": mission, "gate": args.gate, "robot": args.robot})
    if not started.get("ok"):
        print(started.get("error") or started, file=sys.stderr)
        return 1

    def on_pending(pending) -> None:
        if not pending:
            return
        print(f"\n##### APPROVE? {pending.get('name')}({_args(pending.get('arguments'))})")
        try:
            answer = input("      [y/N] ")
        except (EOFError, KeyboardInterrupt):
            answer = ""
        _request(args.url, "/mission/approve",
                 {"id": started.get("id"), "call_id": pending.get("id"),
                  "ok": answer.strip().lower() in ("y", "yes")})

    return _stream(args.url, since=max(before, 1) if before else 0,
                   on_pending=on_pending, mission_id=str(started.get("id") or ""))


def _local_check(args) -> int:
    """Show the operator what the approach tool is working from.

    ##### IT PUBLISHES NOTHING AND CALLS NO MODEL. ##### The cheapest way to
    catch a sign error on a real grid is to look at the grid: a robot cell in
    the wrong place, a costmap that never goes live, a depth frame whose
    intrinsics do not match its size. All of those otherwise surface as "the
    robot walked somewhere odd", days later, with a person standing next to it.
    """
    import time

    from ..local_planner import approach_goal, back_off_until_free, local_to_odom
    from .depth import MqttHeadDepth
    from .local_costmap import MqttInitPose, MqttLocalCostmap

    config = ServiceConfig.from_env(nav_dir=args.nav_dir, env_file=args.env_file)
    target = config.target_for(args.robot)
    runner = build_runner(config)
    runner.connect()
    link = runner.link(target.robot_id)
    costmap = MqttLocalCostmap(link, stale_s=config.local_costmap_stale_s)
    poses = MqttInitPose(link, map_topic=target.pose_topic)
    depth = MqttHeadDepth(link)

    print(f"[check] {target.name}: {target.robot_id} on {config.broker}:{config.broker_port}")
    deadline = time.monotonic() + args.wait_s
    while time.monotonic() < deadline and not costmap.status().get("live"):
        time.sleep(0.25)

    print(f"[check] costmap  : {json.dumps(costmap.status())}")
    print(f"[check] depth    : {json.dumps(depth.status())}")
    print(f"[check] poses    : {json.dumps(poses.status())}")

    try:
        grid = costmap.grid()
        init = poses.init_pose(grid.frame)
    except ValueError as exc:
        print(f"[check] ##### NOT USABLE: {exc}")
        return 1

    print(f"[check] robot    : odom {init.odom.x:.2f},{init.odom.y:.2f} "
          f"@{math.degrees(init.odom.yaw):.0f}deg   "
          f"map {init.map.x:.2f},{init.map.y:.2f} @{math.degrees(init.map.yaw):.0f}deg")
    print(_render(grid, init.odom))

    if args.bearing_deg is None or args.range_m is None:
        print("[check] pass --bearing-deg and --range-m to dry-run a goal placement")
        return 0

    bearing = math.radians(args.bearing_deg)
    target_odom = local_to_odom(
        math.cos(bearing) * args.range_m, math.sin(bearing) * args.range_m, init.odom
    )
    goal_x, goal_y, yaw, distance, final = approach_goal(
        (init.odom.x, init.odom.y), target_odom,
        standoff_m=config.local_standoff_m, max_leg_m=config.local_max_leg_m,
    )
    free = grid.to_gridmap().inflated(config.local_footprint_m)
    goal_x, goal_y, given_up = back_off_until_free(
        free, (init.odom.x, init.odom.y), (goal_x, goal_y))
    print(f"[check] target   : odom {target_odom[0]:.2f},{target_odom[1]:.2f} "
          f"({distance:.2f} m away)")
    print(f"[check] goal     : odom {goal_x:.2f},{goal_y:.2f} @{math.degrees(yaw):.0f}deg   "
          f"final={final}   pulled back {given_up:.2f} m")
    # The number an operator can check by looking, rather than the
    # centre-to-object standoff, which is ~0.31 m more. Same arithmetic as
    # LocalPlan.body_clearance; printed here because this command exists to be
    # believed before a goal is ever sent.
    nose = 0.3079
    achieved = config.local_standoff_m + given_up
    print(f"[check] clearance: {achieved - nose:.2f} m of body clearance nominal, "
          f"{achieved - nose - config.local_arrival_box_m:.2f} m if Nav2 latches at the "
          f"near edge of its {config.local_arrival_box_m:.2f} m box"
          + ("   (to the TARGET -- the costmap moved this goal, so something nearer "
             "stopped it)" if given_up > 1e-9 else ""))
    print("[check] ##### nothing was published.")
    return 0


def _render(grid, robot, *, span: int = 25) -> str:
    """The costmap as characters, robot at the centre. `R` is where we think it is.

    Deliberately coarse: this is for spotting a robot placed in a wall or a map
    rotated 90 degrees, not for reading cell values.
    """
    col, row = grid.cell(robot.x, robot.y)
    step = max(1, grid.width // (span * 2))
    lines = []
    for r in range(row + span * step, row - span * step - 1, -step):
        line = []
        for c in range(col - span * step, col + span * step + 1, step):
            if not grid.in_bounds((c, r)):
                line.append(" ")
                continue
            if abs(c - col) < step and abs(r - row) < step:
                line.append("R")
                continue
            cost = grid.cost[r * grid.width + c]
            line.append("#" if cost >= 100 else ("+" if cost >= 99 else
                        ("?" if cost < 0 else ("." if cost else " "))))
        lines.append("        " + "".join(line))
    scale = span * step * grid.resolution
    lines.append(
        f"        (R robot, # lethal, + inscribed, ? unknown; half-width {scale:.1f} m; "
        f"odom +x RIGHT, +y UP -- the robot faces {math.degrees(robot.yaw):.0f} deg)")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
