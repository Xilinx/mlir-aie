# test_dma_bd_fits.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Device.dma_bd_fits: one buffer descriptor's limits, as the DMA lowerings verify them.

Every pattern is also lowered as a runtime-sequence task on the same tile
(aie-dma-tasks-to-npu), and the compiler must refuse exactly the patterns
`dma_bd_fits` says do not fit.
"""

import numpy as np
import pytest
from ml_dtypes import bfloat16

from aie.dialects._aie_enum_gen import AIETileType
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Context, Location, MLIRError, Module
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.passmanager import PassManager

TILES = [AIETileType.ShimNOCTile, AIETileType.MemTile, AIETileType.CoreTile]
SPAN = 1 << 24


def lowers(dev, col, row, sizes, strides):
    """Whether aie-dma-tasks-to-npu lowers the int32 pattern as one task's BD.

    aie-normalize-dma-bd-dims runs first, as it does in aiecc's pipelines.

    The buffer is as long as the pattern reaches, and the BD's length leaves
    out the iteration dimension, as the BD verifier requires.
    """
    shim = dev.get_tile_type(col, row) is AIETileType.ShimNOCTile
    reach = 1 + sum((n - 1) * s for n, s in zip(sizes, strides))
    memref = f"memref<{reach}xi32>"
    buffer = "" if shim else f"%buf = aie.buffer(%t) {{address = 0 : i32}} : {memref}"
    length = int(np.prod(sizes[1:]))
    text = f"""
        aie.device({dev.name}) {{
          %t = aie.tile({col}, {row})
          {buffer}
          aie.runtime_sequence({"%buf: " + memref if shim else ""}) {{
            %task = aiex.dma_configure_task(%t, MM2S, 0) {{
              aie.dma_bd(%buf : {memref} offset = 0 len = {length}
                sizes = {list(sizes)} strides = {list(strides)}) {{bd_id = 0 : i32}}
              aie.end
            }}
            aiex.dma_start_task(%task)
          }}
        }}"""
    with Context(), Location.unknown():
        try:
            module = Module.parse(text)
            PassManager.parse(
                "builtin.module(aie.device(aie-normalize-dma-bd-dims,"
                "aie-dma-tasks-to-npu))"
            ).run(module.operation)
        except MLIRError:
            return False
    return True


def tile_of(dev, tile_type):
    return next((0, r) for r in range(dev.rows) if dev.get_tile_type(0, r) is tile_type)


def limits(dev, col, row):
    """The cases at each limit, and one past it, as (name, sizes, strides, fits)."""
    wrap = (1 << dev.get_dma_bd_wrap_bits(col, row)) - 1
    step = 1 << dev.get_dma_bd_step_bits(col, row)
    iterations = 1 << dev.get_dma_bd_iter_bits(col, row)
    repeats = dev.max_repeat_count + 1
    cases = []
    for past in (0, 1):
        fits = not past
        d0, d1 = wrap + past, wrap + past
        cases += [
            (f"d0 {d0}", [1, 1, 2, d0], [0, 0, d0 + 1, 1], fits),
            (f"d1 {d1}", [1, 1, d1, 2], [0, 0, 4, 1], fits),
            (f"d1 step {step + past}", [1, 1, 2, 4], [0, 0, step + past, 1], fits),
            (f"d2 step {step + past}", [1, 2, 2, 4], [0, step + past, 8, 1], fits),
            (
                f"iterations {iterations + past}",
                [iterations + past, 1, 1, 4],
                [8, 0, 0, 1],
                fits,
            ),
            (
                f"repeats {repeats + past}",
                [repeats + past, 1, 1, 4],
                [0, 0, 0, 1],
                fits,
            ),
        ]
    return cases


@pytest.mark.parametrize("device", [NPU1Col1, NPU2Col1])
@pytest.mark.parametrize("tile_type", TILES, ids=lambda t: t.name)
def test_each_limit_fits_and_one_past_it_does_not(device, tile_type):
    dev = device()
    col, row = tile_of(dev, tile_type)
    for name, sizes, strides, fits in limits(dev, col, row):
        tap = TensorAccessPattern((SPAN,), 0, sizes, strides)
        assert dev.dma_bd_fits(col, row, tap, np.int32) is fits, name
        assert lowers(dev, col, row, sizes, strides) is fits, name


@pytest.mark.parametrize("device", [NPU1Col1, NPU2Col1])
@pytest.mark.parametrize("tile_type", TILES, ids=lambda t: t.name)
def test_a_linear_transfer_fits_past_the_wrap(device, tile_type):
    dev = device()
    col, row = tile_of(dev, tile_type)
    wrap = (1 << dev.get_dma_bd_wrap_bits(col, row)) - 1
    tap = TensorAccessPattern((SPAN,), 0, [4 * wrap], [1])
    assert dev.dma_bd_fits(col, row, tap, np.int32)
    assert lowers(dev, col, row, [1, 1, 1, 4 * wrap], [0, 0, 0, 1])


@pytest.mark.parametrize("device", [NPU1Col1, NPU2Col1])
@pytest.mark.parametrize("tile_type", TILES, ids=lambda t: t.name)
def test_the_buffer_length_field_bounds_linear_and_nd_transfers(device, tile_type):
    # Each side of the core (2^14 words) and memtile (2^17) length fields.
    dev = device()
    col, row = tile_of(dev, tile_type)
    for words in (1 << 13, (1 << 14) - 1, 1 << 14, (1 << 17) - 1, 1 << 17):
        for sizes, strides in (
            ([1, 1, 1, words], [0, 0, 0, 1]),
            ([1, words // 256, 2, 128], [0, 256, 128, 1]),
        ):
            tap = TensorAccessPattern((SPAN,), 0, sizes, strides)
            fits = dev.dma_bd_fits(col, row, tap, np.int32)
            assert fits is lowers(dev, col, row, sizes, strides), (words, sizes)
    if tile_type is not AIETileType.ShimNOCTile:
        tap = TensorAccessPattern((SPAN,), 0, [1 << 17], [1])
        assert not dev.dma_bd_fits(col, row, tap, np.int32)


@pytest.mark.parametrize("device", [NPU1Col1, NPU2Col1])
@pytest.mark.parametrize("tile_type", TILES, ids=lambda t: t.name)
def test_a_contiguous_2d_transfer_is_linear_only_on_a_shim(device, tile_type):
    # Re-read twice, so the normalize pass leaves it as written: only a shim
    # takes a contiguous transfer in linear mode.
    dev = device()
    col, row = tile_of(dev, tile_type)
    wrap = (1 << dev.get_dma_bd_wrap_bits(col, row)) - 1
    shim = tile_type is AIETileType.ShimNOCTile
    sizes, strides = [2, 1, 2, wrap + 1], [0, 0, wrap + 1, 1]
    tap = TensorAccessPattern((SPAN,), 0, [2, 2, wrap + 1], [0, wrap + 1, 1])
    assert dev.dma_bd_fits(col, row, tap, np.int32) is shim
    assert lowers(dev, col, row, sizes, strides) is shim


@pytest.mark.parametrize("device", [NPU1Col1, NPU2Col1])
@pytest.mark.parametrize("tile_type", TILES, ids=lambda t: t.name)
def test_a_contiguous_2d_transfer_the_normalize_pass_flattens_fits(device, tile_type):
    # Its length is all of it, so aie-normalize-dma-bd-dims makes it one
    # linear transfer on every tile.
    dev = device()
    col, row = tile_of(dev, tile_type)
    wrap = (1 << dev.get_dma_bd_wrap_bits(col, row)) - 1
    tap = TensorAccessPattern((SPAN,), 0, [2, wrap + 1], [wrap + 1, 1])
    assert dev.dma_bd_fits(col, row, tap, np.int32)
    assert lowers(dev, col, row, [1, 1, 2, wrap + 1], [0, 0, wrap + 1, 1])


@pytest.mark.parametrize("device", [NPU1Col1, NPU2Col1])
def test_a_leading_re_read_stays_in_the_iteration_dimension(device):
    # As shim_dma_single_bd_task lays it out: [3, 1, 16, 512], the re-read
    # outside the length. Padded in front instead, the zero stride would land
    # in d2.
    dev = device()
    col, row = tile_of(dev, AIETileType.ShimNOCTile)
    tap = TensorAccessPattern((16 * 512,), 0, [3, 16, 512], [0, 512, 1])
    assert dev.dma_bd_fits(col, row, tap, np.int32)
    assert lowers(dev, col, row, [3, 1, 16, 512], [0, 0, 512, 1])


@pytest.mark.parametrize("device", [NPU1Col1, NPU2Col1])
@pytest.mark.parametrize("tile_type", TILES, ids=lambda t: t.name)
def test_dropping_a_unit_dimension_can_move_one_into_a_wrap(device, tile_type):
    # [1, 2000, 1, 512] is [2000, 512] once its unit dimensions go, and 2000
    # lands in d1's wrap.
    dev = device()
    col, row = tile_of(dev, tile_type)
    wrap = (1 << dev.get_dma_bd_wrap_bits(col, row)) - 1
    n = wrap + 977
    unit = TensorAccessPattern((SPAN,), 0, [1, n, 1, 512], [0, 1024, 0, 1])
    assert not dev.dma_bd_fits(col, row, unit, np.int32)
    assert not lowers(dev, col, row, [1, n, 1, 512], [0, 1024, 0, 1])


def test_offsets_and_element_sizes_are_whole_granules():
    dev = NPU2Col1()
    col, row = tile_of(dev, AIETileType.ShimNOCTile)
    assert dev.dma_bd_fits(col, row, TensorAccessPattern((64,), 2, [32], [1]), bfloat16)
    assert not dev.dma_bd_fits(
        col, row, TensorAccessPattern((64,), 1, [32], [1]), bfloat16
    )
    assert not dev.dma_bd_fits(
        col, row, TensorAccessPattern((64,), 0, [2, 3], [4, 1]), bfloat16
    )
    assert not dev.dma_bd_fits(
        col, row, TensorAccessPattern((64,), 0, [2, 4], [3, 1]), bfloat16
    )


def test_a_pattern_of_five_dimensions_does_not_fit():
    dev = NPU2Col1()
    tap = TensorAccessPattern((SPAN,), 0, [2, 2, 2, 2, 4], [1000, 300, 70, 9, 1])
    assert not dev.dma_bd_fits(0, 0, tap, np.int32)
    # On a shim, five dimensions that coalesce to fewer fit, as a fill
    # coalesces them; elsewhere a descriptor takes them as written.
    contiguous = TensorAccessPattern((SPAN,), 0, [2, 2, 2, 2, 4], [64, 32, 16, 8, 1])
    assert dev.dma_bd_fits(0, 0, contiguous, np.int32)
    col, row = tile_of(dev, AIETileType.MemTile)
    assert not dev.dma_bd_fits(col, row, contiguous, np.int32)


def test_coordinates_outside_the_device_raise():
    with pytest.raises(ValueError, match="out of range"):
        NPU2Col1().dma_bd_fits(5, 0, TensorAccessPattern((64,), 0, [64], [1]), np.int32)
