# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Differential test: the tiling algebra reproduces TensorTiler2D.

For every tiler configuration below, the algebra spelling of the same tiling
must visit the same elements in the same order as the legacy tiler at every
step, and, with ``prune_step=False`` (the form nearly every design uses), it
must produce byte-identical offset/sizes/strides. The exact-match count is
CHECKed so a canonical-form change cannot pass silently.
"""

import itertools

from Inputs.legacy_tensortiler2d import TensorTiler2D
from aie.helpers.taplib import TensorAccessPattern
from util import construct_test

# RUN: %python %s | FileCheck %s


def algebra_tiler(
    tensor_dims,
    tile_dims,
    tile_group_repeats=(1, 1),
    tile_group_steps=(1, 1),
    tile_col_major=False,
    tile_group_col_major=False,
    iter_col_major=False,
    pattern_repeat=1,
    allow_partial=False,
):
    """TensorTiler2D.step_tiler spelled on the algebra."""
    grid = TensorAccessPattern.full(tensor_dims).tile(tile_dims)
    if tile_col_major:
        grid = grid.permute_tile((1, 0))
    grid = grid.order("col" if iter_col_major else "row")
    grid = grid.group(
        tile_group_repeats,
        steps=tile_group_steps,
        col_major=tile_group_col_major,
        partial=allow_partial,
    )
    if pattern_repeat != 1:
        grid = grid.repeat(pattern_repeat)
    return grid


def compare(legacy, grid):
    """Return (steps, exact matches); every step must be access-equivalent."""
    assert len(legacy) == len(grid), (len(legacy), len(grid))
    exact = 0
    for step, want in enumerate(legacy):
        got = grid[step]
        assert isinstance(got, TensorAccessPattern)
        if got == want:
            exact += 1
        else:
            assert got.compare_access_orders(want), (step, got, want)
    return len(legacy), exact


# CHECK-LABEL: simple_and_group_tilers
@construct_test
def simple_and_group_tilers():
    total = exact = 0
    cases = []
    for dims, tile in (
        ((3, 5), (3, 5)),
        ((8, 8), (2, 2)),
        ((16, 8), (4, 2)),
        ((12, 18), (3, 6)),
    ):
        for tcm, icm in itertools.product((False, True), repeat=2):
            for rep in (1, 3):
                cases.append((dims, tile, (1, 1), (1, 1), tcm, False, icm, rep))
    for dims, tile, group in (
        ((8, 8), (2, 2), (2, 2)),
        ((16, 24), (4, 3), (2, 4)),
        ((16, 8), (2, 2), (4, 1)),
        ((16, 8), (2, 2), (1, 4)),
    ):
        for tcm, gcm, icm in itertools.product((False, True), repeat=3):
            for rep in (1, 2):
                cases.append((dims, tile, group, (1, 1), tcm, gcm, icm, rep))
    for dims, tile, group, steps, tcm, gcm, icm, rep in cases:
        if rep != 1 and group[0] > 1 and group[1] > 1:
            continue  # the legacy tiler runs out of dimensions here, by design
        legacy = TensorTiler2D.step_tiler(
            dims,
            tile,
            tile_group_repeats=group,
            tile_group_steps=steps,
            tile_col_major=tcm,
            tile_group_col_major=gcm,
            iter_col_major=icm,
            pattern_repeat=rep,
            prune_step=False,
        )
        grid = algebra_tiler(dims, tile, group, steps, tcm, gcm, icm, rep)
        n, e = compare(legacy, grid)
        total += n
        exact += e
    print(f"simple/group steps={total} exact={exact}")
    # CHECK: simple/group steps=[[N:[0-9]+]] exact=[[N]]


# CHECK-LABEL: step_tilers
@construct_test
def step_tilers():
    total = exact = 0
    for dims, tile, group, steps in (
        ((32, 32), (2, 2), (2, 2), (1, 1)),
        ((32, 32), (2, 2), (2, 2), (2, 2)),
        ((32, 32), (2, 2), (2, 4), (4, 2)),
        ((32, 64), (4, 4), (1, 4), (1, 4)),
        ((64, 32), (4, 4), (4, 1), (4, 1)),
        ((24, 48), (2, 4), (3, 2), (1, 3)),
    ):
        for tcm, gcm, icm in itertools.product((False, True), repeat=3):
            for rep in (1, 2):
                if rep != 1 and group[0] > 1 and group[1] > 1:
                    continue
                legacy = TensorTiler2D.step_tiler(
                    dims,
                    tile,
                    tile_group_repeats=group,
                    tile_group_steps=steps,
                    tile_col_major=tcm,
                    tile_group_col_major=gcm,
                    iter_col_major=icm,
                    pattern_repeat=rep,
                    prune_step=False,
                )
                grid = algebra_tiler(dims, tile, group, steps, tcm, gcm, icm, rep)
                n, e = compare(legacy, grid)
                total += n
                exact += e
    print(f"step steps={total} exact={exact}")
    # CHECK: step steps=[[N:[0-9]+]] exact=[[N]]


# CHECK-LABEL: whole_array_matmul
@construct_test
def whole_array_matmul():
    """Build the whole-array GEMM's three tilers over the sweep the design supports."""
    total = exact = 0
    n_aie_rows = 4
    tb_n_rows = 2
    for M, K, N in ((256, 256, 256), (512, 256, 1024), (1024, 512, 512)):
        for m, k, n in ((32, 32, 32), (64, 64, 64), (64, 32, 64)):
            for n_aie_cols in (1, 2, 4):
                if M % (m * n_aie_rows) or N % (n * n_aie_cols) or K % k:
                    continue
                if (M // (m * n_aie_rows)) % tb_n_rows:
                    continue
                n_A_tiles_per_shim = n_aie_rows // n_aie_cols
                rep = N // n // n_aie_cols
                legacy_A = TensorTiler2D.group_tiler(
                    (M, K),
                    (m * n_A_tiles_per_shim, k),
                    (1, K // k),
                    pattern_repeat=rep,
                    prune_step=False,
                )
                grid_A = (
                    TensorAccessPattern.full((M, K))
                    .tile((m * n_A_tiles_per_shim, k))
                    .group((1, K // k))
                    .repeat(rep)
                )
                legacy_B = TensorTiler2D.step_tiler(
                    (K, N),
                    (k, n),
                    tile_group_repeats=(K // k, rep),
                    tile_group_steps=(1, n_aie_cols),
                    tile_group_col_major=True,
                    prune_step=False,
                )
                grid_B = (
                    TensorAccessPattern.full((K, N))
                    .tile((k, n))
                    .group((K // k, rep), steps=(1, n_aie_cols), col_major=True)
                )
                legacy_C = TensorTiler2D.step_tiler(
                    (M, N),
                    (m * n_aie_rows, n),
                    tile_group_repeats=(tb_n_rows, rep),
                    tile_group_steps=(1, n_aie_cols),
                    prune_step=False,
                )
                grid_C = (
                    TensorAccessPattern.full((M, N))
                    .tile((m * n_aie_rows, n))
                    .group((tb_n_rows, rep), steps=(1, n_aie_cols))
                )
                for legacy, grid in (
                    (legacy_A, grid_A),
                    (legacy_B, grid_B),
                    (legacy_C, grid_C),
                ):
                    n_steps, e = compare(legacy, grid)
                    total += n_steps
                    exact += e
    print(f"whole_array steps={total} exact={exact}")
    # CHECK: whole_array steps=[[N:[0-9]+]] exact=[[N]]


# CHECK-LABEL: prune_step_merges
@construct_test
def prune_step_merges():
    """prune_step=True merges a repeat into the tile in two col-major cases.

    The algebra keeps dimensions separate (coalesce() is explicit), so these
    are access-equivalent, not byte-identical; record how many differ.
    """
    total = exact = 0
    for dims, tile, group in (((8, 8), (2, 2), (2, 2)), ((16, 24), (4, 3), (2, 4))):
        for tcm, gcm, icm in itertools.product((False, True), repeat=3):
            legacy = TensorTiler2D.group_tiler(
                dims,
                tile,
                group,
                tile_col_major=tcm,
                tile_group_col_major=gcm,
                iter_col_major=icm,
            )
            grid = algebra_tiler(dims, tile, group, (1, 1), tcm, gcm, icm)
            n, e = compare(legacy, grid)
            total += n
            exact += e
    print(f"prune steps={total} exact={exact} merged={total - exact}")
    # CHECK: prune steps={{[0-9]+}} exact={{[0-9]+}} merged={{[1-9][0-9]*}}


# CHECK-LABEL: partial_tilers
@construct_test
def partial_tilers():
    """allow_partial=True: ragged edges, exact with prune_step=False."""
    total = exact = 0
    merged_total = merged_exact = 0
    configs = []
    for dims in ((45, 24), (36, 28), (36, 24), (45, 28)):
        for steps in ((1, 1), (3, 3), (1, 3), (2, 2), (2, 1), (2, 3)):
            for tcm, gcm, icm in itertools.product((False, True), repeat=3):
                for rep in (1, 2):
                    configs.append((dims, steps, tcm, gcm, icm, rep))
    for dims, steps, tcm, gcm, icm, rep in configs:
        kwargs = dict(
            tile_dims=(3, 2),
            tile_group_repeats=(5, 7),
            tile_group_steps=steps,
            tile_col_major=tcm,
            tile_group_col_major=gcm,
            iter_col_major=icm,
            pattern_repeat=rep,
            allow_partial=True,
        )
        try:
            legacy = TensorTiler2D.step_tiler(dims, prune_step=False, **kwargs)
        except ValueError:
            continue  # the legacy tiler runs out of dimensions for this combo
        grid = algebra_tiler(
            dims, (3, 2), (5, 7), steps, tcm, gcm, icm, rep, allow_partial=True
        )
        n, e = compare(legacy, grid)
        total += n
        exact += e
        # With the legacy default prune_step=True the col-major combos merge a
        # repeat into the tile; the algebra keeps them separate (coalesce() is
        # explicit), so those are access-equivalent only.
        legacy_pruned = TensorTiler2D.step_tiler(dims, **kwargs)
        n, e = compare(legacy_pruned, grid)
        merged_total += n
        merged_exact += e
    print(f"partial steps={total} exact={exact}")
    print(f"partial pruned steps={merged_total} exact={merged_exact}")
    # CHECK: partial steps=[[N:[0-9]+]] exact=[[N]]
    # CHECK: partial pruned steps={{[0-9]+}} exact={{[0-9]+}}
