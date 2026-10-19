# test_npu_idle.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test that a measurement waits out another process on the NPU.

A second process runs a copy on the NPU, keeps its context for ``HOLD``
seconds and exits. ``xrt-smi`` reports its pid while it holds the context, so
``other_npu_users`` names it, ``wait_idle`` returns once it is gone, and
``run_interleaved(idle=True)`` times nothing until then.
"""

import argparse
import subprocess
import sys
import time

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import AIETileType
from aie.iron import Flow, In, Out, Program, Runtime
from aie.iron.device import Tile
from aie.utils.benchmark import other_npu_users, run_interleaved, wait_idle

N = 1024
HOLD = 4.0


@iron.jit
def copy(a: In, c: Out):
    ty = np.ndarray[(N,), np.dtype[np.int32]]
    feed = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    drain = Tile(col=1, row=0, tile_type=AIETileType.ShimNOCTile)
    link = Flow(feed, drain, src_channel=1, dst_channel=0)

    def seq(A, C):
        link.fill(A)
        link.drain(C, wait=True)

    rt = Runtime(seq, [ty, ty])
    rt.add_flow(link)
    return Program(iron.get_current_device(), rt).resolve_program()


def run_copy(seed: int) -> None:
    host = np.random.default_rng(seed).integers(-(2**31), 2**31 - 1, N, dtype=np.int32)
    a = iron.tensor(host, dtype=np.int32, device="npu")
    c = iron.zeros(N, dtype=np.int32, device="npu")
    copy(a, c)
    c.to("cpu")
    np.testing.assert_array_equal(c.numpy(), host)


def holder() -> subprocess.Popen:
    """Start a process that holds a context on the NPU for ``HOLD`` seconds."""
    process = subprocess.Popen(
        [sys.executable, __file__, "--hold", str(HOLD)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout.readline().strip() == "holding"
    return process


def test_another_process_on_the_npu_is_named_and_waited_out():
    run_copy(0)
    process = holder()
    try:
        assert process.pid in other_npu_users()
        said = []
        start = time.monotonic()
        wait_idle(said.append, holding=True)
        assert process.poll() is not None
        assert said == [f"the NPU is in use by pid [{process.pid}]: waiting"]
        assert time.monotonic() - start > 1.0
    finally:
        process.wait()
    assert other_npu_users(holding=True) == set()


def test_run_interleaved_times_nothing_while_another_process_holds_the_npu():
    run_copy(1)
    process = holder()
    try:
        logged = []
        (stats,) = run_interleaved(
            [lambda: run_copy(2)], rounds=2, calls=2, idle=True, log=logged.append
        )
        assert process.poll() is not None
        assert stats.n == 2
        assert logged[0] == f"the NPU is in use by pid [{process.pid}]: waiting"
    finally:
        process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hold", type=float, required=True)
    args = parser.parse_args()
    run_copy(3)
    print("holding", flush=True)
    time.sleep(args.hold)
