# test_npu_lock.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# REQUIRES: peano
# RUN: %pytest %s
"""The NPU lock between real processes and open files, without an NPU.

``flock`` locks belong to an open file, so a second open of the lock file sees
it as another process would.
"""

import argparse
import fcntl
import logging
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import NPU2Col1
from aie.utils import get_current_device, set_current_device
from aie.utils.compile.jit.compilabledesign import CompilableDesign
from aie.utils.hostruntime.npulock import NPULock, npu_lock

HOLD = 2.0


class Holder:
    """Stands in for a runtime that has a hardware context."""


def locked_elsewhere(path) -> bool:
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def copy():
    ty = np.ndarray[(32,), np.dtype[np.int32]]
    of_in = ObjectFifo(ty)
    of_out = of_in.cons().forward()

    def sequence(a, b, a_in, b_out):
        a_in.fill(a)
        b_out.drain(b, wait=True)

    rt = Runtime(sequence, [ty, ty, of_in.prod(), of_out.cons()])
    return Program(get_current_device(probe_runtime=False), rt).resolve_program()


def test_an_unset_lock_takes_nothing(tmp_path):
    lock = NPULock(None)
    holder = Holder()
    lock.acquire(holder)
    assert not lock.held
    lock.release(holder)
    assert not any(tmp_path.iterdir())


def test_the_lock_is_held_until_its_last_holder_releases(tmp_path):
    path = tmp_path / "npu.lock"
    lock = NPULock(str(path))
    first, second = Holder(), Holder()
    lock.acquire(first)
    lock.acquire(second)
    assert lock.held and locked_elsewhere(path)
    lock.release(first)
    assert lock.held and locked_elsewhere(path)
    lock.release(second)
    assert not lock.held and not locked_elsewhere(path)


def test_a_second_process_waits_for_the_lock_and_says_so(tmp_path):
    path = tmp_path / "npu.lock"
    lock = NPULock(str(path))
    holder = Holder()
    lock.acquire(holder)
    waiter = subprocess.Popen(
        [sys.executable, __file__, "--wait", str(path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert waiter.stdout.readline().startswith("Waiting for the NPU lock")
        time.sleep(HOLD)
        released = time.time()
        lock.release(holder)
        took = waiter.stdout.readline().split()
        assert took[:4] == ["Took", "the", "NPU", "lock"], took
        assert float(took[-2]) >= HOLD
        assert float(waiter.stdout.readline()) >= released
    finally:
        lock.release(holder)
        assert waiter.wait(60) == 0


@pytest.mark.skipif(shutil.which("xclbinutil") is None, reason="xclbinutil")
def test_compiling_does_not_take_the_lock(tmp_path):
    """A compile finishes while another process holds the lock."""
    path = tmp_path / "npu.lock"
    lock = NPULock(str(path))
    holder = Holder()
    lock.acquire(holder)
    try:
        result = subprocess.run(
            [sys.executable, __file__, "--compile"],
            env={
                **os.environ,
                "NPU_LOCK": str(path),
                "NPU_CACHE_HOME": str(tmp_path / "cache"),
            },
            capture_output=True,
            text=True,
            timeout=300,
        )
    finally:
        lock.release(holder)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.split() == ["compiled", "unlocked"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--wait", metavar="LOCK_FILE")
    mode.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if args.wait:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
        NPULock(args.wait).acquire(Holder())
        print(time.time(), flush=True)
    else:
        set_current_device(NPU2Col1())
        CompilableDesign(copy).compile()
        print("compiled", "locked" if npu_lock.held else "unlocked")
