from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence, TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: step_tiler_partial_row
@construct_test
def step_tiler_partial_row():
    tensor_dims = (3 * 5 * 3, 2 * 7 * 2)

    # all row major
    tiles = TensorTiler2D.step_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_repeats=(5, 7),
        tile_group_steps=(3, 3),
        allow_partial=True,
    )
    assert len(tiles) == 9
    reference_tiles = reference_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                tensor_dims, offset=0, sizes=[5, 5, 3, 2], strides=[252, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=2, sizes=[5, 5, 3, 2], strides=[252, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=4, sizes=[5, 4, 3, 2], strides=[252, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=84, sizes=[5, 5, 3, 2], strides=[252, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=86, sizes=[5, 5, 3, 2], strides=[252, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=88, sizes=[5, 4, 3, 2], strides=[252, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=168, sizes=[5, 5, 3, 2], strides=[252, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=170, sizes=[5, 5, 3, 2], strides=[252, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=172, sizes=[5, 4, 3, 2], strides=[252, 6, 28, 1]
            ),
        ]
    )
    assert tiles == reference_tiles
    assert tiles.compare_access_orders(reference_tiles)

    # tile col major
    tiles = TensorTiler2D.step_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_repeats=(5, 7),
        tile_group_steps=(3, 3),
        allow_partial=True,
        tile_col_major=True,
    )
    assert len(tiles) == 9
    reference_tiles = reference_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                tensor_dims, offset=0, sizes=[5, 5, 2, 3], strides=[252, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=2, sizes=[5, 5, 2, 3], strides=[252, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=4, sizes=[5, 4, 2, 3], strides=[252, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=84, sizes=[5, 5, 2, 3], strides=[252, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=86, sizes=[5, 5, 2, 3], strides=[252, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=88, sizes=[5, 4, 2, 3], strides=[252, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=168, sizes=[5, 5, 2, 3], strides=[252, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=170, sizes=[5, 5, 2, 3], strides=[252, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=172, sizes=[5, 4, 2, 3], strides=[252, 6, 1, 28]
            ),
        ]
    )
    assert tiles == reference_tiles
    assert tiles.compare_access_orders(reference_tiles)

    # tile group col major
    tiles = TensorTiler2D.step_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_repeats=(5, 7),
        tile_group_steps=(3, 3),
        allow_partial=True,
        tile_group_col_major=True,
    )
    assert len(tiles) == 9
    reference_tiles = reference_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                tensor_dims, offset=0, sizes=[5, 5, 3, 2], strides=[6, 252, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=2, sizes=[5, 5, 3, 2], strides=[6, 252, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=4, sizes=[4, 5, 3, 2], strides=[6, 252, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=84, sizes=[5, 5, 3, 2], strides=[6, 252, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=86, sizes=[5, 5, 3, 2], strides=[6, 252, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=88, sizes=[4, 5, 3, 2], strides=[6, 252, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=168, sizes=[5, 5, 3, 2], strides=[6, 252, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=170, sizes=[5, 5, 3, 2], strides=[6, 252, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=172, sizes=[4, 5, 3, 2], strides=[6, 252, 28, 1]
            ),
        ]
    )
    assert tiles == reference_tiles
    assert tiles.compare_access_orders(reference_tiles)

    # iter col major
    tiles = TensorTiler2D.step_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_repeats=(5, 7),
        tile_group_steps=(3, 3),
        allow_partial=True,
        iter_col_major=True,
    )
    assert len(tiles) == 9
    reference_tiles = reference_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                tensor_dims, offset=0, sizes=[5, 5, 3, 2], strides=[252, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=84, sizes=[5, 5, 3, 2], strides=[252, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=168, sizes=[5, 5, 3, 2], strides=[252, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=2, sizes=[5, 5, 3, 2], strides=[252, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=86, sizes=[5, 5, 3, 2], strides=[252, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=170, sizes=[5, 5, 3, 2], strides=[252, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=4, sizes=[5, 4, 3, 2], strides=[252, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=88, sizes=[5, 4, 3, 2], strides=[252, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=172, sizes=[5, 4, 3, 2], strides=[252, 6, 28, 1]
            ),
        ]
    )
    assert tiles == reference_tiles
    assert tiles.compare_access_orders(reference_tiles)

    # all col major
    tiles = TensorTiler2D.step_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_repeats=(5, 7),
        tile_group_steps=(3, 3),
        allow_partial=True,
        tile_col_major=True,
        tile_group_col_major=True,
        iter_col_major=True,
    )
    assert len(tiles) == 9
    reference_tiles = reference_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                tensor_dims, offset=0, sizes=[5, 5, 2, 3], strides=[6, 252, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=84, sizes=[5, 5, 2, 3], strides=[6, 252, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=168, sizes=[5, 5, 2, 3], strides=[6, 252, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=2, sizes=[5, 5, 2, 3], strides=[6, 252, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=86, sizes=[5, 5, 2, 3], strides=[6, 252, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=170, sizes=[5, 5, 2, 3], strides=[6, 252, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=4, sizes=[4, 5, 2, 3], strides=[6, 252, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=88, sizes=[4, 5, 2, 3], strides=[6, 252, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=172, sizes=[4, 5, 2, 3], strides=[6, 252, 1, 28]
            ),
        ]
    )
    assert tiles == reference_tiles
    assert tiles.compare_access_orders(reference_tiles)

    # pattern repeat
    tiles = TensorTiler2D.step_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_repeats=(5, 7),
        tile_group_steps=(1, 3),
        allow_partial=True,
        pattern_repeat=4,
        tile_group_col_major=True,
    )
    assert len(tiles) == 9
    reference_tiles = reference_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                tensor_dims, offset=0, sizes=[4, 5, 15, 2], strides=[0, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=2, sizes=[4, 5, 15, 2], strides=[0, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=4, sizes=[4, 4, 15, 2], strides=[0, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=420, sizes=[4, 5, 15, 2], strides=[0, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=422, sizes=[4, 5, 15, 2], strides=[0, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=424, sizes=[4, 4, 15, 2], strides=[0, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=840, sizes=[4, 5, 15, 2], strides=[0, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=842, sizes=[4, 5, 15, 2], strides=[0, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=844, sizes=[4, 4, 15, 2], strides=[0, 6, 28, 1]
            ),
        ]
    )
    assert tiles == reference_tiles
    assert tiles.compare_access_orders(reference_tiles)

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: step_tiler_partial_col
@construct_test
def step_tiler_partial_col():

    # all row major
    tensor_dims = (3 * 5 * 3, 2 * 7 * 2)
    tiles = TensorTiler2D.step_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_repeats=(5, 7),
        tile_group_steps=(2, 2),
        allow_partial=True,
    )
    assert len(tiles) == 8
    reference_tiles = reference_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                tensor_dims, offset=0, sizes=[5, 7, 3, 2], strides=[168, 4, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=2, sizes=[5, 7, 3, 2], strides=[168, 4, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=84, sizes=[5, 7, 3, 2], strides=[168, 4, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=86, sizes=[5, 7, 3, 2], strides=[168, 4, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=840, sizes=[3, 7, 3, 2], strides=[168, 4, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=842, sizes=[3, 7, 3, 2], strides=[168, 4, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=924, sizes=[2, 7, 3, 2], strides=[168, 4, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=926, sizes=[2, 7, 3, 2], strides=[168, 4, 28, 1]
            ),
        ]
    )
    assert tiles == reference_tiles
    assert tiles.compare_access_orders(reference_tiles)

    # tile col major
    tensor_dims = (3 * 5 * 3, 2 * 7 * 2)
    tiles = TensorTiler2D.step_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_repeats=(5, 7),
        tile_group_steps=(2, 2),
        allow_partial=True,
        tile_col_major=True,
    )
    assert len(tiles) == 8
    reference_tiles = reference_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                tensor_dims, offset=0, sizes=[5, 7, 2, 3], strides=[168, 4, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=2, sizes=[5, 7, 2, 3], strides=[168, 4, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=84, sizes=[5, 7, 2, 3], strides=[168, 4, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=86, sizes=[5, 7, 2, 3], strides=[168, 4, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=840, sizes=[3, 7, 2, 3], strides=[168, 4, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=842, sizes=[3, 7, 2, 3], strides=[168, 4, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=924, sizes=[2, 7, 2, 3], strides=[168, 4, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=926, sizes=[2, 7, 2, 3], strides=[168, 4, 1, 28]
            ),
        ]
    )
    assert tiles == reference_tiles
    assert tiles.compare_access_orders(reference_tiles)

    # tile group col major
    tensor_dims = (3 * 5 * 3, 2 * 7 * 2)
    tiles = TensorTiler2D.step_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_repeats=(5, 7),
        tile_group_steps=(2, 2),
        tile_group_col_major=True,
        allow_partial=True,
    )
    assert len(tiles) == 8
    reference_tiles = reference_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                tensor_dims, offset=0, sizes=[7, 5, 3, 2], strides=[4, 168, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=2, sizes=[7, 5, 3, 2], strides=[4, 168, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=84, sizes=[7, 5, 3, 2], strides=[4, 168, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=86, sizes=[7, 5, 3, 2], strides=[4, 168, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=840, sizes=[7, 3, 3, 2], strides=[4, 168, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=842, sizes=[7, 3, 3, 2], strides=[4, 168, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=924, sizes=[7, 2, 3, 2], strides=[4, 168, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=926, sizes=[7, 2, 3, 2], strides=[4, 168, 28, 1]
            ),
        ]
    )
    assert tiles == reference_tiles
    assert tiles.compare_access_orders(reference_tiles)

    # iter col major
    tensor_dims = (3 * 5 * 3, 2 * 7 * 2)
    tiles = TensorTiler2D.step_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_repeats=(5, 7),
        tile_group_steps=(2, 2),
        iter_col_major=True,
        allow_partial=True,
    )
    assert len(tiles) == 8
    reference_tiles = reference_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                tensor_dims, offset=0, sizes=[5, 7, 3, 2], strides=[168, 4, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=84, sizes=[5, 7, 3, 2], strides=[168, 4, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=840, sizes=[3, 7, 3, 2], strides=[168, 4, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=924, sizes=[2, 7, 3, 2], strides=[168, 4, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=2, sizes=[5, 7, 3, 2], strides=[168, 4, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=86, sizes=[5, 7, 3, 2], strides=[168, 4, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=842, sizes=[3, 7, 3, 2], strides=[168, 4, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=926, sizes=[2, 7, 3, 2], strides=[168, 4, 28, 1]
            ),
        ]
    )
    assert tiles == reference_tiles
    assert tiles.compare_access_orders(reference_tiles)

    # all col major
    tensor_dims = (3 * 5 * 3, 2 * 7 * 2)
    tiles = TensorTiler2D.step_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_repeats=(5, 7),
        tile_group_steps=(2, 2),
        allow_partial=True,
        iter_col_major=True,
        tile_col_major=True,
        tile_group_col_major=True,
    )
    assert len(tiles) == 8
    reference_tiles = reference_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                tensor_dims, offset=0, sizes=[7, 5, 2, 3], strides=[4, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=84, sizes=[7, 5, 2, 3], strides=[4, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=840, sizes=[7, 3, 2, 3], strides=[4, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=924, sizes=[7, 2, 2, 3], strides=[4, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=2, sizes=[7, 5, 2, 3], strides=[4, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=86, sizes=[7, 5, 2, 3], strides=[4, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=842, sizes=[7, 3, 2, 3], strides=[4, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=926, sizes=[7, 2, 2, 3], strides=[4, 168, 1, 28]
            ),
        ]
    )
    assert tiles == reference_tiles
    assert tiles.compare_access_orders(reference_tiles)

    # pattern repeat
    tensor_dims = (3 * 5 * 3, 2 * 7 * 2)
    tiles = TensorTiler2D.step_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_repeats=(5, 7),
        tile_group_steps=(2, 1),
        allow_partial=True,
        pattern_repeat=2,
        tile_col_major=True,
    )
    assert len(tiles) == 8
    reference_tiles = reference_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                tensor_dims, offset=0, sizes=[2, 5, 14, 3], strides=[0, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=14, sizes=[2, 5, 14, 3], strides=[0, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=84, sizes=[2, 5, 14, 3], strides=[0, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=98, sizes=[2, 5, 14, 3], strides=[0, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=840, sizes=[2, 3, 14, 3], strides=[0, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=854, sizes=[2, 3, 14, 3], strides=[0, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=924, sizes=[2, 2, 14, 3], strides=[0, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=938, sizes=[2, 2, 14, 3], strides=[0, 168, 1, 28]
            ),
        ]
    )
    assert tiles == reference_tiles
    assert tiles.compare_access_orders(reference_tiles)

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: step_tiler_partial_both
@construct_test
def step_tiler_partial_both():
    tensor_dims = (3 * 5 * 3, 2 * 7 * 2)

    # all row major
    tiles = TensorTiler2D.step_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_repeats=(5, 7),
        tile_group_steps=(2, 3),
        allow_partial=True,
    )
    assert len(tiles) == 12
    reference_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                tensor_dims, offset=0, sizes=[5, 5, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=2, sizes=[5, 5, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=4, sizes=[5, 4, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=84, sizes=[5, 5, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=86, sizes=[5, 5, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=88, sizes=[5, 4, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=840, sizes=[3, 5, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=842, sizes=[3, 5, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=844, sizes=[3, 4, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=924, sizes=[2, 5, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=926, sizes=[2, 5, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=928, sizes=[2, 4, 3, 2], strides=[168, 6, 28, 1]
            ),
        ]
    )
    assert tiles == reference_tiles
    assert tiles.compare_access_orders(reference_tiles)

    # tile col major
    tiles = TensorTiler2D.step_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_repeats=(5, 7),
        tile_group_steps=(2, 3),
        allow_partial=True,
        tile_col_major=True,
    )
    assert len(tiles) == 12
    reference_tiles = reference_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                tensor_dims, offset=0, sizes=[5, 5, 2, 3], strides=[168, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=2, sizes=[5, 5, 2, 3], strides=[168, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=4, sizes=[5, 4, 2, 3], strides=[168, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=84, sizes=[5, 5, 2, 3], strides=[168, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=86, sizes=[5, 5, 2, 3], strides=[168, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=88, sizes=[5, 4, 2, 3], strides=[168, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=840, sizes=[3, 5, 2, 3], strides=[168, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=842, sizes=[3, 5, 2, 3], strides=[168, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=844, sizes=[3, 4, 2, 3], strides=[168, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=924, sizes=[2, 5, 2, 3], strides=[168, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=926, sizes=[2, 5, 2, 3], strides=[168, 6, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=928, sizes=[2, 4, 2, 3], strides=[168, 6, 1, 28]
            ),
        ]
    )
    assert tiles == reference_tiles
    assert tiles.compare_access_orders(reference_tiles)

    # tile group col major
    tiles = TensorTiler2D.step_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_repeats=(5, 7),
        tile_group_steps=(2, 3),
        allow_partial=True,
        tile_group_col_major=True,
    )
    assert len(tiles) == 12
    reference_tiles = reference_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                tensor_dims, offset=0, sizes=[5, 5, 3, 2], strides=[6, 168, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=2, sizes=[5, 5, 3, 2], strides=[6, 168, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=4, sizes=[4, 5, 3, 2], strides=[6, 168, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=84, sizes=[5, 5, 3, 2], strides=[6, 168, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=86, sizes=[5, 5, 3, 2], strides=[6, 168, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=88, sizes=[4, 5, 3, 2], strides=[6, 168, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=840, sizes=[5, 3, 3, 2], strides=[6, 168, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=842, sizes=[5, 3, 3, 2], strides=[6, 168, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=844, sizes=[4, 3, 3, 2], strides=[6, 168, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=924, sizes=[5, 2, 3, 2], strides=[6, 168, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=926, sizes=[5, 2, 3, 2], strides=[6, 168, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=928, sizes=[4, 2, 3, 2], strides=[6, 168, 28, 1]
            ),
        ]
    )
    assert tiles == reference_tiles
    assert tiles.compare_access_orders(reference_tiles)

    # iter col major
    tiles = TensorTiler2D.step_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_repeats=(5, 7),
        tile_group_steps=(2, 3),
        allow_partial=True,
        iter_col_major=True,
    )
    assert len(tiles) == 12
    reference_tiles = reference_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                tensor_dims, offset=0, sizes=[5, 5, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=84, sizes=[5, 5, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=840, sizes=[3, 5, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=924, sizes=[2, 5, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=2, sizes=[5, 5, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=86, sizes=[5, 5, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=842, sizes=[3, 5, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=926, sizes=[2, 5, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=4, sizes=[5, 4, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=88, sizes=[5, 4, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=844, sizes=[3, 4, 3, 2], strides=[168, 6, 28, 1]
            ),
            TensorAccessPattern(
                tensor_dims, offset=928, sizes=[2, 4, 3, 2], strides=[168, 6, 28, 1]
            ),
        ]
    )
    assert tiles == reference_tiles
    assert tiles.compare_access_orders(reference_tiles)

    # all col major
    tiles = TensorTiler2D.step_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_repeats=(5, 7),
        tile_group_steps=(2, 3),
        allow_partial=True,
        tile_col_major=True,
        tile_group_col_major=True,
        iter_col_major=True,
    )
    assert len(tiles) == 12
    reference_tiles = reference_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                tensor_dims, offset=0, sizes=[5, 5, 2, 3], strides=[6, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=84, sizes=[5, 5, 2, 3], strides=[6, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=840, sizes=[5, 3, 2, 3], strides=[6, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=924, sizes=[5, 2, 2, 3], strides=[6, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=2, sizes=[5, 5, 2, 3], strides=[6, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=86, sizes=[5, 5, 2, 3], strides=[6, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=842, sizes=[5, 3, 2, 3], strides=[6, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=926, sizes=[5, 2, 2, 3], strides=[6, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=4, sizes=[4, 5, 2, 3], strides=[6, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=88, sizes=[4, 5, 2, 3], strides=[6, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=844, sizes=[4, 3, 2, 3], strides=[6, 168, 1, 28]
            ),
            TensorAccessPattern(
                tensor_dims, offset=928, sizes=[4, 2, 2, 3], strides=[6, 168, 1, 28]
            ),
        ]
    )
    assert tiles == reference_tiles
    assert tiles.compare_access_orders(reference_tiles)

    # CHECK: Pass!
    print("Pass!")
