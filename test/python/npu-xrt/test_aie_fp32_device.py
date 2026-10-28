# test_aie_fp32_device.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""aie.utils.aie_fp32.aie_add against the AIE2P core's own scalar fp32 adds.

Random pairs whose operands and sums are normal floats, spanning every
alignment of the smaller operand onto the larger one's grid, must match the
model bit for bit under both the core's reset rounding mode and conv_even,
and IEEE addition must not.
"""

import numpy as np
import pytest

import aie.iron as iron
from aie.iron import (
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.utils.aie_fp32 import aie_add

TILE = 64
PAIRS = 4096

SOURCE = """
#include <aie_api/aie.hpp>
extern "C" void add_pairs(const float *in, float *out) {
#if CONV_EVEN
  aie::set_rounding(aie::rounding_mode::conv_even);
#endif
#pragma clang loop vectorize(disable) unroll(disable)
  for (int i = 0; i < TILE; i++)
    out[i] = in[2 * i] + in[2 * i + 1];
}
"""


@iron.jit
def add_pairs(pairs: In, sums: Out, *, func: CompileTime[object]):
    in_tile = np.ndarray[(2 * TILE,), np.dtype[np.float32]]
    out_tile = np.ndarray[(TILE,), np.dtype[np.float32]]
    of_in = ObjectFifo(in_tile, name="in")
    of_out = ObjectFifo(out_tile, name="out")

    def core(of_in, of_out, fn):
        for _ in range_(PAIRS // TILE):
            a = of_in.acquire(1)
            b = of_out.acquire(1)
            fn(a, b)
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core, fn_args=[of_in.cons(), of_out.prod(), func])

    def sequence(p, s, feed, drain):
        feed.fill(p)
        drain.drain(s, wait=True)

    rt = Runtime(
        sequence,
        [
            np.ndarray[(2 * PAIRS,), np.dtype[np.float32]],
            np.ndarray[(PAIRS,), np.dtype[np.float32]],
            of_in.prod(),
            of_out.cons(),
        ],
    )
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def normal_pairs(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = 2 * PAIRS
    big = rng.uniform(1, 2, n) * 2.0 ** rng.integers(-20, 20, n)
    big *= rng.choice([-1, 1], n)
    small = rng.uniform(1, 2, n) * big * 2.0 ** -rng.integers(0, 32, n)
    small *= rng.choice([-1, 1], n)
    u, v = big.astype(np.float32), small.astype(np.float32)
    tiny = np.finfo(np.float32).tiny
    sums = u.astype(np.float64) + v
    keep = (np.abs(v) >= tiny) & ((np.abs(sums) >= tiny) | (sums == 0))
    u, v = u[keep][: PAIRS - 1], v[keep][: PAIRS - 1]
    return (
        np.concatenate([[1.0], u]).astype(np.float32),
        np.concatenate([[-(2.0**-24 + 2.0**-30)], v]).astype(np.float32),
    )


@pytest.mark.parametrize("conv_even", [False, True], ids=["reset_mode", "conv_even"])
def test_aie_add_is_the_cores_scalar_fp32_add(conv_even):
    u, v = normal_pairs(np.random.default_rng(0))
    packed = np.empty(2 * PAIRS, np.float32)
    packed[0::2], packed[1::2] = u, v
    tile = np.ndarray[(2 * TILE,), np.dtype[np.float32]]
    func = ExternalFunction(
        "add_pairs",
        source_string=SOURCE,
        arg_types=[tile, np.ndarray[(TILE,), np.dtype[np.float32]]],
        compile_flags=[f"-DTILE={TILE}", f"-DCONV_EVEN={int(conv_even)}"],
    )
    sums = iron.zeros((PAIRS,), dtype=np.float32)
    add_pairs(iron.tensor(packed, dtype=np.float32), sums, func=func)
    got = sums.numpy()
    np.testing.assert_array_equal(got.view(np.int32), aie_add(u, v).view(np.int32))
    assert got[0] == np.float32(1.0 - 2.0**-23)
    assert np.count_nonzero(got != u + v) > 0
