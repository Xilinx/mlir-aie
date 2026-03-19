#!/usr/bin/env python3

# (c) Copyright 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import deque
import os
import shutil
import subprocess
import sys
import time

# Retry configuration
TRANSIENT_FAILURE_TEXT = "No such device with index"
MAX_ATTEMPTS = 3
TAIL_LINES = 200


# Logging and diagnostics helpers
# TODO: update or disable for a future Windows NPU runner.
def log(message: str) -> None:
    print(f"[run_on_npu.py] {message}", file=sys.stderr, flush=True)


def log_output(label: str, text: str) -> None:
    log(label)
    for line in text.splitlines() or ["<empty>"]:
        log(line)


def find_xrt_smi(xrt_dir: str) -> str | None:
    xrt_smi = os.environ.get("XRT_SMI")
    if xrt_smi and os.path.isfile(xrt_smi):
        return xrt_smi

    xrt_smi = shutil.which("xrt-smi")
    if xrt_smi is not None:
        return xrt_smi

    candidate = os.path.join(os.environ.get("XILINX_XRT") or xrt_dir, "bin", "xrt-smi")
    if os.path.isfile(candidate):
        return candidate

    return None


def emit_failure_diagnostics(xrt_dir: str, attempt: int) -> None:
    log(f"Device-enumeration failure on attempt {attempt}/{MAX_ATTEMPTS}.")

    device_node = "/dev/accel/accel0"
    if os.path.exists(device_node):
        log(f"{device_node} exists")
        try:
            log(f"stat: {os.stat(device_node)}")
        except OSError as error:
            log(f"stat({device_node}) failed: {error}")
    else:
        log(f"{device_node} does not exist")

    xrt_smi = find_xrt_smi(xrt_dir)
    if xrt_smi is None:
        log("xrt-smi not found")
        return

    log(f"Running {xrt_smi} examine")
    try:
        result = subprocess.run(
            [xrt_smi, "examine"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=5,
        )
    except Exception as error:  # pragma: no cover - best-effort diagnostics
        log(f"xrt-smi examine failed: {error}")
        return

    log_output("xrt-smi stdout:", result.stdout)
    log_output("xrt-smi stderr:", result.stderr)


# Command launch helpers
def wrapped_command(xrt_dir: str, command: list[str]) -> list[str]:
    if os.name == "nt":
        return command

    setup_script = os.path.join(xrt_dir, "setup.sh")
    if not os.path.isfile(setup_script):
        return command

    return [
        "bash",
        "-c",
        'XRT_DIR="$1"; source "$XRT_DIR/setup.sh" >/dev/null 2>&1; shift; exec "$@"',
        "run_on_npu",
        xrt_dir,
        *command,
    ]


def run_command(command: list[str]) -> tuple[int, str]:
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    ) as process:
        if process.stdout is not None:
            # Stream child output directly to CI.
            # Keep a tail for failure detection after the command exits.
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                tail.append(line)
        return process.wait(), "".join(tail)


# Main entry point.
def main() -> int:
    if len(sys.argv) < 3:
        print("usage: run_on_npu.py <npu-kind> <command> [args...]", file=sys.stderr)
        return 2

    # Mimic the existing %run_on_npu*% bash functionality, broadening OS support.
    # "NPU kind" is currently unused, but keep the argument for interface parity.
    command = sys.argv[2:]
    xrt_dir = (
        os.environ.get("XRT_DIR")
        or os.environ.get("XRT_ROOT")
        or os.environ.get("XILINX_XRT")
        or "/opt/xilinx/xrt"
    )
    launched_command = wrapped_command(xrt_dir, command)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        returncode, recent_output = run_command(launched_command)
        if returncode == 0:
            return 0
        if TRANSIENT_FAILURE_TEXT not in recent_output:
            return returncode
        emit_failure_diagnostics(xrt_dir, attempt)
        if attempt == MAX_ATTEMPTS:
            return returncode
        delay = attempt * 3
        log(f"Retrying in {delay:g}s.")
        time.sleep(delay)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
