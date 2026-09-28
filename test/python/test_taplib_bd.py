# test_taplib_bd.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""An access pattern as a value, and whether one buffer descriptor holds it.

``BdLimits.fits`` restates the compiler's descriptor rules over a pattern, so
it is checked against the compiler itself: each pattern is built as a shim
task and lowered, and the lowering must fail exactly where ``fits`` says no.
"""

import itertools

import numpy as np
import pytest
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    T,
    device,
    shim_dma_allocation,
    tile,
)
from aie.dialects.aiex import (
    dma_await_task,
    dma_start_task,
    runtime_sequence,
    shim_dma_single_bd_task,
)
from aie.extras.context import mlir_mod_ctx
from aie.helpers.taplib import BdLimits, TensorAccessPattern
from aie.iron.device import from_name
from aie.passmanager import PassManager

# What aiecc runs on a runtime sequence's descriptors before lowering them,
# less aie-decompose-large-dma-bd: a pattern that fits needs no splitting.
LOWERING = (
    "builtin.module(aie.device(aie-substitute-shim-dma-allocations,canonicalize,"
    "aie-normalize-dma-bd-dims,aie-assign-runtime-sequence-bd-ids,"
    "aie-dma-tasks-to-npu))"
)


def lowers(tap: TensorAccessPattern) -> bool:
    """Whether the compiler lowers ``tap`` as one bf16 shim task, unsplit."""
    elements = int(np.prod(tap.tensor_dims))
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2)
        def device_body():
            shim_dma_allocation("a", tile(0, 0), DMAChannelDir.MM2S, 0)

            @runtime_sequence(T.memref(elements, T.bf16()))
            def sequence(buffer):
                task = shim_dma_single_bd_task("a", buffer, tap=tap, issue_token=True)
                dma_start_task(task)
                dma_await_task(task)

        try:
            PassManager.parse(LOWERING).run(ctx.module.operation)
        except Exception:
            return False
        return True


@pytest.fixture(scope="module")
def shim():
    return from_name("npu2").bd_limits(0, 0)


def test_a_device_gives_each_tiles_limits():
    npu2 = from_name("npu2")
    assert npu2.bd_limits(0, 0) == BdLimits(
        wrap=1023, step=1 << 20, iterations=64, granule_bytes=4, linear=True
    )
    # A mem tile has no buffer length field that exempts a contiguous run.
    assert not npu2.bd_limits(0, 1).linear


# (sizes, strides, elements); each is lowered, and fits must agree.
PATTERNS = [
    ([4096], [1], 1 << 20),  # linear
    ([1 << 20], [1], 1 << 20),  # linear, past the wrap
    ([64, 4096], [8192, 1], 1 << 20),  # d0 past the wrap
    ([64, 4096], [4096, 1], 1 << 20),  # contiguous, so one linear run
    ([2048, 2], [8, 1], 1 << 16),  # d1 past the wrap
    ([4, 8, 64], [0, 64, 1], 1 << 12),  # a leading re-read
    ([2, 4, 8, 64], [0, 0, 64, 1], 1 << 12),  # a re-read outside the iteration
    ([65, 8, 64], [512, 64, 1], 1 << 16),  # d2 has no wrap
    ([65, 1, 8, 64], [512, 0, 64, 1], 1 << 16),  # iterations past 64
    ([64, 1, 8, 64], [512, 0, 64, 1], 1 << 16),
    ([8, 1024, 64], [65536, 64, 1], 1 << 20),
    ([8, 64, 2048], [1 << 17, 4096, 1], 1 << 21),
    ([8, 64, 2046], [1 << 17, 4096, 1], 1 << 21),
    ([2, 3], [3, 1], 6),  # contiguous in 12 bytes
    ([2, 3], [4, 1], 8),  # d0 of 6 bytes
    ([16, 8], [1, 16], 128),  # a stride of 2 bytes
    ([4, 2], [1 << 21, 1], 1 << 23),  # a stride past the step field
    ([4, 2], [1 << 20, 1], 1 << 23),
]


@pytest.mark.parametrize("sizes, strides, elements", PATTERNS)
def test_fits_agrees_with_the_compiler(shim, sizes, strides, elements):
    tap = TensorAccessPattern([elements], 0, sizes, strides)
    assert shim.fits(tap, np.dtype("bfloat16")) == lowers(tap), tap


def test_a_pattern_of_tuples_lowers():
    assert lowers(TensorAccessPattern((64,), 0, (8, 8), (8, 1)))


def test_an_unaligned_offset_does_not_fit(shim):
    tap = TensorAccessPattern([1024], 1, [64], [1])
    assert not shim.fits(tap, np.int16)
    assert shim.fits(TensorAccessPattern([1024], 2, [64], [1]), np.int16)


def test_slots_keep_a_re_read_in_the_iteration_dimension():
    assert BdLimits.slots([4, 8, 64], [0, 64, 1]) == (
        [4, 1, 8, 64],
        [0, 0, 64, 1],
    )
    assert BdLimits.slots([8, 64], [128, 1]) == ([1, 1, 8, 64], [0, 0, 128, 1])
    assert BdLimits.slots([8, 64], None) == ([1, 1, 8, 64], None)


@pytest.mark.parametrize("run, granule", [(4096, 2), (2046, 2), (1 << 19, 2), (7, 1)])
def test_factor_splits_a_run_into_legal_d1_d0(shim, run, granule):
    d1, d0 = shim.factor(run, granule)
    assert d1 * d0 == run
    assert d0 % granule == 0 and d0 <= shim.wrap * granule and d1 <= shim.wrap


def test_factor_is_none_when_no_split_fits(shim):
    assert shim.factor(2 * 1031 * 1031) is None  # two primes past the wrap


def test_a_pattern_is_a_value():
    a = TensorAccessPattern((8, 64), 64, (4, 64), (64, 1))
    b = TensorAccessPattern([8, 64], np.int64(64), [4, 64], [64, 1])
    assert a == b and hash(a) == hash(b)
    assert len({a, b, TensorAccessPattern([8, 64], 0, [4, 64], [64, 1])}) == 2
    assert eval(repr(a), {"TensorAccessPattern": TensorAccessPattern}) == a


@pytest.mark.parametrize(
    "dims, offset, sizes, strides",
    [
        ([8, 64], 0, [8, 64], [64, 1]),
        ([8, 64], 64, [64, 4], [1, 64]),
        ([4, 4], 12, [3, 4], [4, 1]),  # past the end, wrapping as the generator does
        ([2, 3, 4], 1, [2, 2, 2], [12, 4, 2]),
    ],
)
def test_access_indices_are_the_generators(dims, offset, sizes, strides):
    tap = TensorAccessPattern(dims, offset, sizes, strides)
    assert tap.access_indices().tolist() == list(tap.access_generator())


@pytest.mark.parametrize(
    "key, contiguous",
    [
        (np.s_[2:5], True),
        (np.s_[2:5, :], True),
        (np.s_[3], True),
        (np.s_[:, 1], False),
        (np.s_[:, 0:2], False),
        (np.s_[::2], False),
        (np.s_[0:1, 0:4], True),
    ],
)
def test_contiguous_is_one_dense_run(key, contiguous):
    tap = TensorAccessPattern.from_slice((8, 16), key)
    assert tap.contiguous == contiguous
    indices = tap.access_indices()
    dense = bool(
        np.array_equal(indices, np.arange(indices[0], indices[0] + len(indices)))
    )
    assert dense == contiguous


def test_every_small_pattern_agrees_with_the_compiler(shim):
    # Every pattern of up to four dimensions over sizes and strides at the
    # edges of the fields, so the rules are covered together, not one at a
    # time.
    iterations = [(1, 0), (2, 0), (2, 2048), (65, 2)]
    sizes = [1, 2, 1024]
    strides = [0, 1, 2, 2048]
    checked = 0
    for (it, s3), (n2, n1, n0), (s2, s1) in itertools.product(
        iterations,
        itertools.product(sizes, repeat=3),
        itertools.product(strides, repeat=2),
    ):
        span = (it - 1) * s3 + (n2 - 1) * s2 + (n1 - 1) * s1 + n0
        tap = TensorAccessPattern([span], 0, [it, n2, n1, n0], [s3, s2, s1, 1])
        assert shim.fits(tap, np.dtype("bfloat16")) == lowers(tap), tap
        checked += 1
    assert checked == 4 * 27 * 16
