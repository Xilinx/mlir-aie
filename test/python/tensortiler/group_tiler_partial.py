import numpy as np

from aie.helpers.tensortiler import TensorTile, TensorTileSequence, TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: group_tiler_partial_row
@construct_test
def group_tiler_partial_row():

    tensor_dims = (3 * 5 * 3, 2 * 6 * 2)

    # All row major
    tiles = TensorTiler2D.group_tiler(
        tensor_dims, tile_dims=(3, 2), tile_group_dims=(5, 7), allow_partial=True
    )
    reference_tiles = TensorTileSequence.from_tiles(
        [
            TensorTile(
                tensor_dims, offset=0, sizes=[5, 7, 3, 2], strides=[72, 2, 24, 1]
            ),
            TensorTile(
                tensor_dims, offset=14, sizes=[5, 5, 3, 2], strides=[72, 2, 24, 1]
            ),
            TensorTile(
                tensor_dims, offset=360, sizes=[5, 7, 3, 2], strides=[72, 2, 24, 1]
            ),
            TensorTile(
                tensor_dims, offset=374, sizes=[5, 5, 3, 2], strides=[72, 2, 24, 1]
            ),
            TensorTile(
                tensor_dims, offset=720, sizes=[5, 7, 3, 2], strides=[72, 2, 24, 1]
            ),
            TensorTile(
                tensor_dims, offset=734, sizes=[5, 5, 3, 2], strides=[72, 2, 24, 1]
            ),
        ]
    )
    assert tiles == reference_tiles

    # Tile col major
    tiles = TensorTiler2D.group_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_dims=(5, 7),
        tile_col_major=True,
        allow_partial=True,
    )
    reference_tiles = TensorTileSequence.from_tiles(
        [
            TensorTile(
                tensor_dims, offset=0, sizes=[1, 5, 14, 3], strides=[0, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=14, sizes=[1, 5, 10, 3], strides=[0, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=360, sizes=[1, 5, 14, 3], strides=[0, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=374, sizes=[1, 5, 10, 3], strides=[0, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=720, sizes=[1, 5, 14, 3], strides=[0, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=734, sizes=[1, 5, 10, 3], strides=[0, 72, 1, 24]
            ),
        ]
    )
    assert tiles == reference_tiles

    # Tile group col major
    tiles = TensorTiler2D.group_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_dims=(5, 7),
        tile_group_col_major=True,
        allow_partial=True,
    )
    reference_tiles = TensorTileSequence.from_tiles(
        [
            TensorTile(
                tensor_dims, offset=0, sizes=[1, 7, 15, 2], strides=[0, 2, 24, 1]
            ),
            TensorTile(
                tensor_dims, offset=14, sizes=[1, 5, 15, 2], strides=[0, 2, 24, 1]
            ),
            TensorTile(
                tensor_dims, offset=360, sizes=[1, 7, 15, 2], strides=[0, 2, 24, 1]
            ),
            TensorTile(
                tensor_dims, offset=374, sizes=[1, 5, 15, 2], strides=[0, 2, 24, 1]
            ),
            TensorTile(
                tensor_dims, offset=720, sizes=[1, 7, 15, 2], strides=[0, 2, 24, 1]
            ),
            TensorTile(
                tensor_dims, offset=734, sizes=[1, 5, 15, 2], strides=[0, 2, 24, 1]
            ),
        ]
    )
    assert tiles == reference_tiles

    # iter col major
    tiles = TensorTiler2D.group_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_dims=(5, 7),
        iter_col_major=True,
        allow_partial=True,
    )
    reference_tiles = TensorTileSequence.from_tiles(
        [
            TensorTile(
                tensor_dims, offset=0, sizes=[5, 7, 3, 2], strides=[72, 2, 24, 1]
            ),
            TensorTile(
                tensor_dims, offset=360, sizes=[5, 7, 3, 2], strides=[72, 2, 24, 1]
            ),
            TensorTile(
                tensor_dims, offset=720, sizes=[5, 7, 3, 2], strides=[72, 2, 24, 1]
            ),
            TensorTile(
                tensor_dims, offset=14, sizes=[5, 5, 3, 2], strides=[72, 2, 24, 1]
            ),
            TensorTile(
                tensor_dims, offset=374, sizes=[5, 5, 3, 2], strides=[72, 2, 24, 1]
            ),
            TensorTile(
                tensor_dims, offset=734, sizes=[5, 5, 3, 2], strides=[72, 2, 24, 1]
            ),
        ]
    )
    assert tiles == reference_tiles

    # all col major
    tiles = TensorTiler2D.group_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_dims=(5, 7),
        tile_col_major=True,
        tile_group_col_major=True,
        iter_col_major=True,
        allow_partial=True,
    )
    reference_tiles = TensorTileSequence.from_tiles(
        [
            TensorTile(
                tensor_dims, offset=0, sizes=[7, 5, 2, 3], strides=[2, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=360, sizes=[7, 5, 2, 3], strides=[2, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=720, sizes=[7, 5, 2, 3], strides=[2, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=14, sizes=[5, 5, 2, 3], strides=[2, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=374, sizes=[5, 5, 2, 3], strides=[2, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=734, sizes=[5, 5, 2, 3], strides=[2, 72, 1, 24]
            ),
        ]
    )
    assert tiles == reference_tiles

    # pattern repeat
    tiles = TensorTiler2D.group_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_dims=(5, 7),
        tile_col_major=True,
        allow_partial=True,
        pattern_repeat=4,
    )
    reference_tiles = TensorTileSequence.from_tiles(
        [
            TensorTile(
                tensor_dims, offset=0, sizes=[4, 5, 14, 3], strides=[0, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=14, sizes=[4, 5, 10, 3], strides=[0, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=360, sizes=[4, 5, 14, 3], strides=[0, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=374, sizes=[4, 5, 10, 3], strides=[0, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=720, sizes=[4, 5, 14, 3], strides=[0, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=734, sizes=[4, 5, 10, 3], strides=[0, 72, 1, 24]
            ),
        ]
    )
    assert tiles == reference_tiles

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: group_tiler_partial_col
@construct_test
def group_tiler_partial_col():
    """
    tensor_dims = (3 * 4 * 3, 2 * 6 * 2)
    tiles = TensorTiler2D.group_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_dims=(5, 7),
        tile_col_major=True,
        allow_partial=True
    )
    reference_tiles = TensorTileSequence.from_tiles([
        TensorTile(tensor_dims, offset=0, sizes=[1, 5, 14, 3], strides=[0, 84, 1, 28]),
        TensorTile(tensor_dims, offset=14, sizes=[1, 5, 14, 3], strides=[0, 84, 1, 28]),
        TensorTile(tensor_dims, offset=420, sizes=[1, 5, 14, 3], strides=[0, 84, 1, 28]),
        TensorTile(tensor_dims, offset=434, sizes=[1, 5, 14, 3], strides=[0, 84, 1, 28]),
        TensorTile(tensor_dims, offset=840, sizes=[1, 2, 14, 3], strides=[0, 84, 1, 28]),
        TensorTile(tensor_dims, offset=854, sizes=[1, 2, 14, 3], strides=[0, 84, 1, 28]),
    ])
    assert tiles == reference_tiles
    """

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: group_tiler_partial_both
@construct_test
def group_tiler_partial_both():

    tensor_dims = (3 * 4 * 3, 2 * 6 * 2)
    tiles = TensorTiler2D.group_tiler(
        tensor_dims,
        tile_dims=(3, 2),
        tile_group_dims=(5, 7),
        tile_col_major=True,
        allow_partial=True,
    )

    reference_tiles = TensorTileSequence.from_tiles(
        [
            TensorTile(
                tensor_dims, offset=0, sizes=[1, 5, 14, 3], strides=[0, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=14, sizes=[1, 5, 10, 3], strides=[0, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=360, sizes=[1, 5, 14, 3], strides=[0, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=374, sizes=[1, 5, 10, 3], strides=[0, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=720, sizes=[1, 2, 14, 3], strides=[0, 72, 1, 24]
            ),
            TensorTile(
                tensor_dims, offset=734, sizes=[1, 2, 10, 3], strides=[0, 72, 1, 24]
            ),
        ]
    )
    assert tiles == reference_tiles

    # CHECK: Pass!
    print("Pass!")
