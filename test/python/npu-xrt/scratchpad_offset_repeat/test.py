# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# One shim DMA task with a runtime offset, repeat count and length together:
# iteration i of n passes starts at off + i * stride, for a stepped (stride ROW)
# and a pure-repeat (stride 0) pattern. A zero length moves nothing, whatever
# the repeat count.
#
# REQUIRES: ryzen_ai_npu2, peano, xrt_python_bindings, xrt_python_ctrl_scratchpad_bo
#
# RUN: %python %S/aie_design.py > aie.mlir
# RUN: %aiecc -v --get-full-elf --dynamic-objFifos --get-scratchpad-parameters aie.mlir
# RUN: %run_on_npu2% %pytest %s

import numpy as np
import pytest
import pyxrt

import aie.iron as iron
from aie.utils.hostruntime.xrtruntime.device import acquire_device
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
    ParameterScratchpad,
)

N = 1024
TILE = 16
ROW = 128


@pytest.fixture(scope="module")
def setup():
    device = acquire_device()
    context = pyxrt.hw_context(device, pyxrt.elf("aie.elf"))
    kernel = pyxrt.ext.kernel(context, "test:sequence")
    src = iron.tensor(np.arange(N, dtype=np.int32), device="cpu")
    dst_a = iron.tensor((N,), dtype=np.int32, device="cpu")
    dst_b = iron.tensor((N,), dtype=np.int32, device="cpu")
    src.to("npu")
    run = pyxrt.run(kernel)
    for i, t in enumerate((src, dst_a, dst_b)):
        run.set_arg(i, t.buffer_object())
    return run, ParameterScratchpad(run, "params.txt"), src, dst_a, dst_b


def dispatch(setup, off, n, reps_a, reps_b):
    run, params, _, dst_a, dst_b = setup
    for t in (dst_a, dst_b):
        t.data.fill(-1)
        t.to("npu")
    params.write("off", np.int32(off))
    params.write("n", np.int32(n))
    params.write("reps_a", np.int32(reps_a))
    params.write("reps_b", np.int32(reps_b))
    params.write("tiles_a", np.int32((reps_a + 1) * n))
    params.write("tiles_b", np.int32((reps_b + 1) * n))
    params.sync()
    run.start()
    run.wait2()
    for t in (dst_a, dst_b):
        t.to("cpu")
    return dst_a.numpy().copy(), dst_b.numpy().copy()


def expected(off, n, reps, stride):
    out = np.full(N, -1, dtype=np.int32)
    moved = np.concatenate(
        [
            np.arange(off + i * stride, off + i * stride + n * TILE)
            for i in range(reps + 1)
        ]
    )
    out[: moved.size] = moved
    return out


@pytest.mark.parametrize(
    "off,n,reps_a,reps_b",
    [(0, 1, 0, 0), (16, 2, 3, 2), (48, 4, 7, 7), (32, 3, 5, 1), (0, 1, 7, 0)],
)
def test_offset_repeat_and_length_on_one_task(setup, off, n, reps_a, reps_b):
    a, b = dispatch(setup, off, n, reps_a, reps_b)
    np.testing.assert_array_equal(a, expected(off, n, reps_a, ROW))
    np.testing.assert_array_equal(b, expected(off, n, reps_b, 0))


# Last: a zero-length transfer that never completed would hang this run.
@pytest.mark.parametrize("reps", [0, 3])
def test_zero_length_moves_nothing(setup, reps):
    a, b = dispatch(setup, 16, 0, reps, reps)
    np.testing.assert_array_equal(a, np.full(N, -1, dtype=np.int32))
    np.testing.assert_array_equal(b, np.full(N, -1, dtype=np.int32))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
