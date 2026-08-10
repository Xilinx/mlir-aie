#!/usr/bin/env python3

# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import deque
import contextlib
import errno
import os
import shutil
import subprocess
import sys
import tempfile
import time

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None

# Retry configuration
#
# GET_INFO/-22 is the driver failing to resume: on runtime-PM resume it restarts
# the hardware contexts, aie2_config_cu() looks up a CU BO that is already gone,
# and the -EINVAL surfaces on whatever ioctl triggered the resume. dmesg shows
#   aie2_config_cu: Lookup GEM object failed
#   aie2_hwctx_restart: Config cu failed, ret -22
#   amdxdna_pm_resume_get: Resume failed: -22
# XRT raises it as an uncaught xrt_core::system_error, so the test aborts. Like
# the other signatures this cannot mask a wrong result: it fails before the
# design runs, so there is no output to compare, and a deterministic failure
# still fails every attempt.
TRANSIENT_FAILURE_TEXT = (
    "No such device",
    "DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed (err=-22)",
)
MAX_ATTEMPTS = 3
TAIL_LINES = 200


# Logging and diagnostics helpers
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

    xrt_root = os.environ.get("XILINX_XRT") or xrt_dir
    candidates = [
        os.path.join(xrt_root, "xrt-smi"),
        os.path.join(xrt_root, "bin", "xrt-smi"),
    ]
    if os.name == "nt":
        candidates.extend(
            [
                os.path.join(xrt_root, "xrt-smi.exe"),
                os.path.join(xrt_root, "bin", "xrt-smi.exe"),
                os.path.join(
                    os.environ.get("SystemRoot", r"C:\Windows"),
                    "System32",
                    "AMD",
                    "xrt-smi.exe",
                ),
            ]
        )

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    return None


def emit_failure_diagnostics(xrt_dir: str, attempt: int) -> None:
    log(f"Device-enumeration failure on attempt {attempt}/{MAX_ATTEMPTS}.")

    if sys.platform.startswith("linux"):
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
# Bound how many device runs are in flight at once.
#   * flock is released by the kernel when the fd closes or the process dies, so
#     a crashed or timed-out test cannot wedge the queue.
#   * the lock files live at stable paths and are never unlinked: flock is per
#     INODE, so deleting and recreating one would let two holders through.
#   * it is machine-wide, which is stronger than a lit parallelism group; that
#     only bounds one lit invocation, and gives nothing if a second job on the
#     same host also touches the device.
# AIE_NPU_MAX_CONCURRENT_DISPATCH overrides the bound; 1 serializes dispatch.
# AIE_NPU_NO_DEVICE_LOCK=1 opts out entirely, for tests that exercise concurrency.
# AIE_NPU_EXCLUSIVE_DEVICE=1 takes the whole device rather than a slot.
#
# npu1 gets 1: the driver caps it at 6 hardware contexts against npu4's 16 and
# enables frame-boundary preemption (AIE2_PREEMPT) only on npu4, so it cannot
# time-share a whole-array design. At 8 the aie2-4col leg lost 6 tests -- five
# aborted in CREATE_HWCTX/-ENOENT, and writebd_tokens returned all zeros instead
# of failing. Anything between 1 and 8 is unmeasured there.
DEFAULT_MAX_CONCURRENT_DISPATCH = {"npu1": 1, "npu2": 8}
FALLBACK_MAX_CONCURRENT_DISPATCH = 1


def take_slot(lock_dir: str, npu_kind: str, slots: int) -> int:
    # Blocking on one chosen file instead would queue behind that holder while
    # another slot sat idle.
    while True:
        for i in range(slots):
            path = os.path.join(lock_dir, f"mlir-aie-npu-{npu_kind}.{i}.lock")
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except OSError as e:
                os.close(fd)
                # Only contention means "try the next slot". Anything else --
                # a lock_dir whose filesystem has no flock, say -- would spin
                # here forever, which is the wedged job this lock exists to avoid.
                if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                    raise
        time.sleep(0.01)


@contextlib.contextmanager
def device_lock(npu_kind: str):
    if os.environ.get("AIE_NPU_NO_DEVICE_LOCK"):
        yield
        return
    # No flock here. test/npu-xrt/lit.local.cfg keys on the same import and keeps
    # the lit-level npu-xrt parallelism group on these platforms, so dispatch stays
    # serialized -- just at test granularity rather than at dispatch granularity.
    if fcntl is None:
        yield
        return
    try:
        slots = int(os.environ.get("AIE_NPU_MAX_CONCURRENT_DISPATCH", ""))
    except ValueError:
        slots = DEFAULT_MAX_CONCURRENT_DISPATCH.get(npu_kind)
        if slots is None:
            log(f"no measured dispatch bound for {npu_kind}, serializing")
            slots = FALLBACK_MAX_CONCURRENT_DISPATCH
    slots = max(1, slots)
    lock_dir = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    # An exclusive run needs no other hardware context to EXIST: opening one
    # clears a core tile's data memory under a run that is idle between
    # dispatches, no dispatch required. A slot count only bounds dispatch.
    exclusive = bool(os.environ.get("AIE_NPU_EXCLUSIVE_DEVICE"))
    gate_path = os.path.join(lock_dir, f"mlir-aie-npu-{npu_kind}.gate.lock")
    gate = os.open(gate_path, os.O_CREAT | os.O_RDWR, 0o666)
    fd = None
    t0 = time.perf_counter()
    try:
        # Gate then slot is the only pair held at once, so the order cannot
        # deadlock. flock is not FIFO-fair: slot holders can delay an exclusive.
        fcntl.flock(gate, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        if not exclusive:
            fd = take_slot(lock_dir, npu_kind, slots)
        waited = time.perf_counter() - t0
        if waited > 1.0:
            wanted = "the device" if exclusive else f"a free slot ({slots} total)"
            log(f"waited {waited:.1f}s for {wanted} on {npu_kind}")
        yield
    finally:
        if fd is not None:
            os.close(fd)
        os.close(gate)


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

    npu_kind = sys.argv[1]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        # Held across one attempt only: the retry backoff below must not keep
        # the device reserved while it sleeps.
        with device_lock(npu_kind):
            returncode, recent_output = run_command(launched_command)
        if returncode == 0:
            return 0
        if not any(text in recent_output for text in TRANSIENT_FAILURE_TEXT):
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
