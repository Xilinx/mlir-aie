# test_moe_group.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""moe_group's factory and reference (no NPU required), the reference against
a layout built another way: a stable sort of the assignments by expert.
"""

from pathlib import Path

import numpy as np
import pytest
from aie.iron.device import from_name
from aie.iron.kernels import moe
from aie.iron.kernels._common import ARCH_TRAITS
from aie.utils import config
from aie.utils.compile.utils import compile_cxx_core_function
from aie.utils.hostruntime import set_current_device


def _sorted_layout(ids, valid, experts, tile, blocks):
    """The slots and blocks by sorting the routed assignments by expert."""
    chosen = (np.clip(ids[:valid], -experts, experts - 1) % experts).reshape(-1)
    order = np.argsort(chosen, kind="stable")
    where = np.zeros(ids.size, dtype=np.int64)
    layout, slot = [], 0
    for e in range(experts):
        mine = order[chosen[order] == e]
        where[mine] = slot + np.arange(mine.size)
        n = -(-mine.size // tile)
        layout += [e] * n
        slot += n * tile
    last = layout[-1] if layout else 0
    layout += [last] * (blocks - len(layout))
    return where.reshape(ids.shape), np.array(layout)


@pytest.mark.parametrize(
    "rows, top_k, experts, tile",
    [(2048, 8, 32, 64), (256, 6, 64, 32), (128, 4, 60, 16), (512, 8, 128, 64)],
)
@pytest.mark.parametrize("fraction", [1.0, 0.37, 0.0])
def test_the_layout_is_the_assignments_sorted_by_expert(
    rows, top_k, experts, tile, fraction
):
    rng = np.random.default_rng(rows + experts)
    ids = np.stack([rng.permutation(experts)[:top_k] for _ in range(rows)])
    valid = int(rows * fraction)
    blocks = moe.moe_group_blocks(rows=rows, top_k=top_k, experts=experts, tile=tile)
    where, layout = moe.moe_group_ref(
        ids, valid, experts=experts, tile=tile, blocks=blocks
    )
    want_where, want_layout = _sorted_layout(ids, valid, experts, tile, blocks)
    np.testing.assert_array_equal(where, want_where)
    np.testing.assert_array_equal(layout, want_layout)
    assert layout.shape == (blocks,)
    # Every slot a routed assignment takes lies in a block of its own expert.
    chosen = ids[:valid].reshape(-1)
    np.testing.assert_array_equal(layout[where[:valid].reshape(-1) // tile], chosen)


@pytest.mark.parametrize(
    "rows, top_k, experts, tile", [(2048, 8, 32, 64), (64, 2, 8, 128)]
)
def test_every_row_routed_alike_still_fits_the_blocks(rows, top_k, experts, tile):
    ids = np.tile(np.arange(top_k), (rows, 1))
    blocks = moe.moe_group_blocks(rows=rows, top_k=top_k, experts=experts, tile=tile)
    where, layout = moe.moe_group_ref(
        ids, rows, experts=experts, tile=tile, blocks=blocks
    )
    assert where.max() < blocks * tile
    np.testing.assert_array_equal(
        layout[: top_k * -(-rows // tile)],
        np.repeat(np.arange(top_k), -(-rows // tile)),
    )


def test_ids_outside_the_experts_are_clipped_into_them():
    ids = np.array([[-1, 8], [-9, 3]])
    blocks = moe.moe_group_blocks(rows=2, top_k=2, experts=8, tile=4)
    where, layout = moe.moe_group_ref(ids, 2, experts=8, tile=4, blocks=blocks)
    # -1 is expert 7, 8 clips to 7, -9 clips to -8, expert 0.
    np.testing.assert_array_equal(layout[:3], [0, 3, 7])
    np.testing.assert_array_equal(where, [[8, 9], [0, 4]])


def test_the_factory_binds_its_three_calls():
    fn = moe.moe_group(rows=256, call_rows=32, top_k=6, experts=64, tile=32)
    blocks = moe.moe_group_blocks(rows=256, top_k=6, experts=64, tile=32)
    assert fn.arg_shape(0) == (32 * 6,) and fn.arg_shape(1) == (2 * 64,)
    assert f"-DGROUP_BLOCKS={blocks}" in fn._compile_flags
    assert fn.blocks.object_file is fn.object_file
    assert fn.where.object_file is fn.object_file


def test_call_rows_must_divide_the_chunk():
    with pytest.raises(ValueError, match="must divide"):
        moe.moe_group(rows=100, call_rows=32, top_k=8, experts=32, tile=64)


def _peano_available() -> bool:
    try:
        return Path(config.peano_cxx_path()).is_file()
    except RuntimeError:
        return False


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
@pytest.mark.parametrize("arch", list(ARCH_TRAITS))
def test_it_builds_for_every_architecture(arch, tmp_path):
    with set_current_device(from_name(ARCH_TRAITS[arch].device, n_cols=1)):
        fn = moe.moe_group(rows=2048, call_rows=64, top_k=8, experts=32, tile=64)
        source = Path(fn._source_file)
        compile_cxx_core_function(
            str(source),
            arch,
            str(tmp_path / f"{fn.name}.o"),
            include_dirs=[*fn.include_dirs, str(source.parent)],
            compile_args=list(fn._compile_flags),
        )
    assert (tmp_path / f"{fn.name}.o").stat().st_size > 0
