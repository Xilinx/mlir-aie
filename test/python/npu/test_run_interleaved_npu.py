# test_run_interleaved_npu.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: env AIE_NPU_EXCLUSIVE_DEVICE=1 %run_on_npu1_xrt% %pytest %s
# RUN: env AIE_NPU_EXCLUSIVE_DEVICE=1 %run_on_npu2_xrt% %pytest %s
# REQUIRES: xrt_python_bindings
"""run_interleaved on the NPU: real copies, timed with the NPU otherwise idle.

The device is taken whole (AIE_NPU_EXCLUSIVE_DEVICE), since a round waits for
every other process to leave the NPU.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from functools import partial

import numpy as np
import pytest

import aie.iron as iron
import aie.utils as utils
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime
from aie.utils.benchmark import CompletionCores, run_interleaved

N = 1 << 20


@iron.jit
def copy(a: In, b: Out, *, n: CompileTime[int]):
    ty = np.ndarray[(n,), np.dtype[np.int32]]
    of_in = ObjectFifo(np.ndarray[(1024,), np.dtype[np.int32]])
    of_out = of_in.cons().forward()

    def sequence(a, b, a_in, b_out):
        a_in.fill(a)
        b_out.drain(b, wait=True)

    rt = Runtime(sequence, [ty, ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt).resolve_program()


def copies(*lengths):
    """One copy callable per length, with its input and output."""
    runs = []
    for n in lengths:
        a = iron.arange(n, dtype=np.int32, device="npu")
        b = iron.zeros(n, dtype=np.int32, device="npu")
        runs.append((partial(copy, a, b, n=n), a, b))
    return runs


def test_a_copy_four_times_as_long_takes_longer_but_at_most_four_times():
    (short, a1, b1), (long, a4, b4) = copies(N, 4 * N)
    t1, t4 = run_interleaved(
        [short, long], rounds=4, calls=10, runtime=utils.DefaultNPURuntime
    )
    np.testing.assert_array_equal(b1.numpy(), a1.numpy())
    np.testing.assert_array_equal(b4.numpy(), a4.numpy())
    assert t1.n == t4.n == 4
    # t = fixed + bytes / bandwidth: the bytes are 4x, the fixed part is not.
    assert 1 < t4.median_us / t1.median_us <= 4, (t1.median_us, t4.median_us)


def test_the_timing_thread_is_held_on_the_completion_cores():
    ((run, a, b),) = copies(N)
    seen = []

    def recorded():
        seen.append(os.sched_getaffinity(0))
        return run()

    before = os.sched_getaffinity(0)
    run_interleaved([recorded], rounds=2, calls=3, runtime=utils.DefaultNPURuntime)
    np.testing.assert_array_equal(b.numpy(), a.numpy())
    # Every CPU an NPU completion interrupt has ever been taken on.
    irq_cpus = CompletionCores({}).cpus
    timed = seen[2:]
    assert timed and all(cpus <= irq_cpus for cpus in timed), (timed, irq_cpus)
    assert os.sched_getaffinity(0) == before


SPIN = """
import os
import sys

os.sched_setaffinity(0, {int(sys.argv[1])})
print("spinning", flush=True)
while True:
    pass
"""


def test_a_figure_taken_beside_another_process_on_the_completion_cores_is_refused():
    # Another session's build on the cores the completions interrupt holds
    # the woken timing thread off them until the core's next tick.
    ((run, a, b),) = copies(N)
    quiet = run_interleaved([run], rounds=4, calls=5, runtime=utils.DefaultNPURuntime)[
        0
    ].median_us
    before = CompletionCores.interrupts()
    run()
    cpus = CompletionCores(before).cpus
    assert cpus
    spinners = [
        subprocess.Popen(
            [sys.executable, "-c", SPIN, str(cpu)], stdout=subprocess.PIPE, text=True
        )
        for cpu in sorted(cpus)
    ]
    try:
        assert all(s.stdout.readline().strip() == "spinning" for s in spinners)
        busy = run_interleaved(
            [run], rounds=4, calls=5, runtime=utils.DefaultNPURuntime
        )[0].median_us
    except RuntimeError as e:
        assert "for a CPU" in str(e), e
    else:
        # The scheduler lets the woken thread through in some stretches; a
        # figure from those stands only if it is the quiet one.
        assert busy == pytest.approx(quiet, rel=0.25), (busy, quiet)
    finally:
        for s in spinners:
            s.kill()
            s.wait()
    np.testing.assert_array_equal(b.numpy(), a.numpy())


def hold(seconds):
    """Hold an NPU context for `seconds`, having said so on stdout."""
    ((run, a, b),) = copies(N)
    run()
    np.testing.assert_array_equal(b.numpy(), a.numpy())
    print("held", flush=True)
    time.sleep(seconds)


def test_a_round_waits_for_another_process_to_leave_the_npu(caplog):
    holder = subprocess.Popen(
        [sys.executable, __file__, "--hold", "8"], stdout=subprocess.PIPE, text=True
    )
    assert holder.stdout.readline().strip() == "held"
    assert holder.pid in utils.DefaultNPURuntime.other_processes()
    ((run, a, b),) = copies(N)
    with caplog.at_level(logging.WARNING, logger="aie.utils.benchmark"):
        run_interleaved([run], rounds=2, calls=3, runtime=utils.DefaultNPURuntime)
    assert holder.wait(timeout=60) == 0
    assert f"in use by pid [{holder.pid}]" in caplog.text, caplog.text
    np.testing.assert_array_equal(b.numpy(), a.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hold", type=float, required=True, metavar="SECONDS")
    hold(parser.parse_args().hold)
