#!/usr/bin/env python3
"""Run one GPU command under conservative machine-health limits."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _gpu_sample() -> tuple[float, float, float]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=5,
    )
    temperature, used, total = output.splitlines()[0].split(",")
    return float(temperature), float(used), float(total)


def _system_memory_percent() -> float:
    values: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            values[key] = int(value.split()[0])
    return 100.0 * (1.0 - values["MemAvailable"] / values["MemTotal"])


def _cpu_temperature() -> float | None:
    readings = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            value = float(path.read_text(encoding="utf-8").strip()) / 1000.0
        except (OSError, ValueError):
            continue
        if 0.0 < value < 150.0:
            readings.append(value)
    return max(readings) if readings else None


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    # Give Python rollouts a chance to flush videos, telemetry, and child
    # simulator processes. SIGTERM bypasses normal ``finally`` cleanup unless
    # every target installs a handler explicitly.
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seconds", type=float, default=90.0)
    parser.add_argument("--max-gpu-temperature", type=float, default=78.0)
    parser.add_argument("--max-gpu-memory-percent", type=float, default=70.0)
    parser.add_argument("--max-system-memory-percent", type=float, default=85.0)
    parser.add_argument("--max-cpu-temperature", type=float, default=85.0)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    try:
        temperature, used, total = _gpu_sample()
    except Exception as exc:
        print(f"[watchdog] refusing to start: GPU health query failed: {exc}", file=sys.stderr)
        return 2
    memory_percent = 100.0 * used / total
    system_memory = _system_memory_percent()
    cpu_temperature = _cpu_temperature()
    print(
        "[watchdog] initial "
        f"gpu_temp={temperature:.0f}C gpu_memory={memory_percent:.1f}% "
        f"system_memory={system_memory:.1f}% "
        f"cpu_temp={cpu_temperature if cpu_temperature is not None else 'unavailable'}",
        flush=True,
    )
    if (
        temperature >= args.max_gpu_temperature
        or memory_percent >= args.max_gpu_memory_percent
        or system_memory >= args.max_system_memory_percent
        or (cpu_temperature is not None and cpu_temperature >= args.max_cpu_temperature)
    ):
        print("[watchdog] refusing to start because an initial limit is exceeded", file=sys.stderr)
        return 3

    process = subprocess.Popen(command, start_new_session=True)
    started = time.monotonic()
    consecutive_limit_samples = 0
    reason = ""
    peaks = [temperature, memory_percent, system_memory, cpu_temperature or 0.0]
    try:
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= args.max_seconds:
                reason = f"runtime reached {args.max_seconds:.0f}s"
                break
            try:
                temperature, used, total = _gpu_sample()
                memory_percent = 100.0 * used / total
                system_memory = _system_memory_percent()
                cpu_temperature = _cpu_temperature()
                peaks = [
                    max(peaks[0], temperature),
                    max(peaks[1], memory_percent),
                    max(peaks[2], system_memory),
                    max(peaks[3], cpu_temperature or 0.0),
                ]
            except Exception as exc:
                reason = f"health query failed while running: {exc}"
                break
            exceeded = []
            if temperature >= args.max_gpu_temperature:
                exceeded.append(f"GPU temperature {temperature:.0f}C")
            if memory_percent >= args.max_gpu_memory_percent:
                exceeded.append(f"GPU memory {memory_percent:.1f}%")
            if system_memory >= args.max_system_memory_percent:
                exceeded.append(f"system memory {system_memory:.1f}%")
            if cpu_temperature is not None and cpu_temperature >= args.max_cpu_temperature:
                exceeded.append(f"CPU temperature {cpu_temperature:.0f}C")
            consecutive_limit_samples = consecutive_limit_samples + 1 if exceeded else 0
            # Two consecutive samples avoid reacting to a one-second telemetry spike.
            if consecutive_limit_samples >= 2:
                reason = ", ".join(exceeded)
                break
            time.sleep(args.sample_seconds)
    except KeyboardInterrupt:
        reason = "interrupted"
    finally:
        if process.poll() is None:
            print(f"[watchdog] stopping rollout: {reason}", file=sys.stderr, flush=True)
            _stop_process_group(process)
        print(
            "[watchdog] peaks "
            f"gpu_temp={peaks[0]:.0f}C gpu_memory={peaks[1]:.1f}% "
            f"system_memory={peaks[2]:.1f}% cpu_temp={peaks[3]:.0f}C",
            flush=True,
        )
    return process.returncode if process.returncode is not None else 4


if __name__ == "__main__":
    raise SystemExit(main())
