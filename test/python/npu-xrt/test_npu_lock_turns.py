# test_npu_lock_turns.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test that two processes sharing ``NPU_LOCK`` take turns on the NPU.

The first runs a copy and keeps its context until told to clean up. The second,
with its own cache, compiles the copy while the first holds the lock, and only
then waits for it. It runs once the first has cleaned up, while the first is
still alive.
"""

import argparse
import fcntl
import logging
import os
import queue
import subprocess
import sys
import threading
import time

import aie.utils as aie_utils
from test_npu_idle import run_copy


class Turn:
    """A process taking a turn on the NPU, whose output lines are read with a deadline."""

    def __init__(self, lock_path, cache_home, seed: int, hold: bool):
        self.process = subprocess.Popen(
            [sys.executable, __file__, "--seed", str(seed)]
            + (["--hold"] if hold else []),
            env={**os.environ, "NPU_LOCK": lock_path, "NPU_CACHE_HOME": cache_home},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.lines = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.process.stdout:
            self.lines.put(line.rstrip("\n"))

    def expect(self, prefix: str, timeout: float = 300) -> float:
        """Return the timestamp of the next line that starts with `prefix`."""
        deadline = time.monotonic() + timeout
        while True:
            when, _, line = self.lines.get(
                timeout=deadline - time.monotonic()
            ).partition(" ")
            if line.startswith(prefix):
                return float(when)

    def proceed(self):
        self.process.stdin.write("\n")
        self.process.stdin.flush()


def locked_elsewhere(path) -> bool:
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def test_a_second_process_compiles_then_waits_for_the_first_to_clean_up(tmp_path):
    lock = str(tmp_path / "npu.lock")
    first = Turn(lock, str(tmp_path / "first"), 0, hold=True)
    second = None
    try:
        first.expect("ran")
        assert locked_elsewhere(lock)
        second = Turn(lock, str(tmp_path / "second"), 1, hold=False)
        compiled = second.expect("Waiting for the NPU lock")
        first.proceed()
        released = first.expect("Released the NPU lock")
        took = second.expect("Took the NPU lock")
        second.expect("ran")
        first.proceed()
        exited = first.expect("exiting")
        assert compiled < released <= took < exited
        assert second.process.wait(60) == 0
        assert first.process.wait(60) == 0
        assert not locked_elsewhere(lock)
    finally:
        for turn in (first, second):
            if turn is not None and turn.process.poll() is None:
                turn.process.kill()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--hold", action="store_true", help="wait to be told to clean up, then to exit"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout, format="%(created)f %(message)s"
    )
    run_copy(args.seed)
    print(time.time(), "ran", flush=True)
    if args.hold:
        sys.stdin.readline()
        aie_utils.cleanup_npu_runtime()
        sys.stdin.readline()
        print(time.time(), "exiting", flush=True)
