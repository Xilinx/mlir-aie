import numpy as np

from aie.helpers.tensortiler import TensorTile, TensorTileSequence, TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: step_tiler
@construct_test
def step_tiler():
    # Start with Step == (1, 1)
    tiles = TensorTiler2D.step_tiler(
        (32, 32),
        tile_dims=(2, 2),
        tile_group_repeats=(2, 2),
        tile_group_steps=(1, 1),
        allow_partial=True,
    )
    assert len(tiles) == (32 // (2 * 2)) * (32 // (2 * 2))
    assert tiles[0] == TensorTile(
        (32, 32), offset=0, sizes=[2, 2, 2, 2], strides=[64, 2, 32, 1]
    )
    assert tiles[1] == TensorTile(
        (32, 32), offset=4, sizes=[2, 2, 2, 2], strides=[64, 2, 32, 1]
    )
    assert tiles[26] == TensorTile(
        (32, 32), offset=392, sizes=[2, 2, 2, 2], strides=[64, 2, 32, 1]
    )

    # Step == (2, 1)
    tiles = TensorTiler2D.step_tiler(
        (32, 32),
        tile_dims=(2, 2),
        tile_group_repeats=(2, 2),
        tile_group_steps=(2, 1),
        allow_partial=True,
    )
    assert len(tiles) == (32 // (2 * 2)) * (32 // (2 * 2))
    assert tiles[0] == TensorTile(
        (32, 32), offset=0, sizes=[2, 2, 2, 2], strides=[128, 2, 32, 1]
    )
    assert tiles[1] == TensorTile(
        (32, 32), offset=4, sizes=[2, 2, 2, 2], strides=[128, 2, 32, 1]
    )
    assert tiles[26] == TensorTile(
        (32, 32), offset=328, sizes=[2, 2, 2, 2], strides=[128, 2, 32, 1]
    )
    assert tiles[-1] == TensorTile(
        (32, 32), offset=860, sizes=[2, 2, 2, 2], strides=[128, 2, 32, 1]
    )

    # Step == (1, 2)
    tiles = TensorTiler2D.step_tiler(
        (32, 32),
        tile_dims=(2, 2),
        tile_group_repeats=(2, 2),
        tile_group_steps=(1, 2),
        allow_partial=True,
    )
    assert len(tiles) == (32 // (2 * 2)) * (32 // (2 * 2))
    assert tiles[0] == TensorTile(
        (32, 32), offset=0, sizes=[2, 2, 2, 2], strides=[64, 4, 32, 1]
    )
    assert tiles[1] == TensorTile(
        (32, 32), offset=2, sizes=[2, 2, 2, 2], strides=[64, 4, 32, 1]
    )
    assert tiles[26] == TensorTile(
        (32, 32), offset=392, sizes=[2, 2, 2, 2], strides=[64, 4, 32, 1]
    )
    assert tiles[-1] == TensorTile(
        (32, 32), offset=922, sizes=[2, 2, 2, 2], strides=[64, 4, 32, 1]
    )

    # Step == (2, 2)
    tiles = TensorTiler2D.step_tiler(
        (32, 32),
        tile_dims=(2, 2),
        tile_group_repeats=(2, 2),
        tile_group_steps=(2, 2),
        allow_partial=True,
    )
    assert len(tiles) == (32 // (2 * 2)) * (32 // (2 * 2))
    assert tiles[0] == TensorTile(
        (32, 32), offset=0, sizes=[2, 2, 2, 2], strides=[128, 4, 32, 1]
    )
    assert tiles[1] == TensorTile(
        (32, 32), offset=2, sizes=[2, 2, 2, 2], strides=[128, 4, 32, 1]
    )
    assert tiles[26] == TensorTile(
        (32, 32), offset=328, sizes=[2, 2, 2, 2], strides=[128, 4, 32, 1]
    )
    assert tiles[-1] == TensorTile(
        (32, 32), offset=858, sizes=[2, 2, 2, 2], strides=[128, 4, 32, 1]
    )

    # Step == (2, 2)
    tiles = TensorTiler2D.step_tiler(
        (32, 32),
        tile_dims=(2, 2),
        tile_group_repeats=(2, 2),
        tile_group_steps=(2, 2),
        allow_partial=True,
    )
    assert len(tiles) == (32 // (2 * 2)) * (32 // (2 * 2))
    assert tiles[0] == TensorTile(
        (32, 32), offset=0, sizes=[2, 2, 2, 2], strides=[128, 4, 32, 1]
    )
    assert tiles[1] == TensorTile(
        (32, 32), offset=2, sizes=[2, 2, 2, 2], strides=[128, 4, 32, 1]
    )
    assert tiles[26] == TensorTile(
        (32, 32), offset=328, sizes=[2, 2, 2, 2], strides=[128, 4, 32, 1]
    )
    assert tiles[-1] == TensorTile(
        (32, 32), offset=858, sizes=[2, 2, 2, 2], strides=[128, 4, 32, 1]
    )

    # Repeat across column/row
    tiles = TensorTiler2D.step_tiler(
        (32, 32),
        tile_dims=(2, 2),
        tile_group_repeats=(32 // 4, 32 // 4),
        tile_group_steps=(2, 2),
        allow_partial=True,
    )
    assert len(tiles) == 4  # (32//(2*(32//4))) * (32//(2*(32//4)))
    assert tiles[0] == TensorTile(
        (32, 32), offset=0, sizes=[8, 8, 2, 2], strides=[128, 4, 32, 1]
    )
    assert tiles[1] == TensorTile(
        (32, 32), offset=2, sizes=[8, 8, 2, 2], strides=[128, 4, 32, 1]
    )
    assert tiles[2] == TensorTile(
        (32, 32), offset=64, sizes=[8, 8, 2, 2], strides=[128, 4, 32, 1]
    )
    assert tiles[3] == TensorTile(
        (32, 32), offset=66, sizes=[8, 8, 2, 2], strides=[128, 4, 32, 1]
    )

    # Repeat one dimension
    tiles = TensorTiler2D.step_tiler(
        (32, 32),
        tile_dims=(2, 2),
        tile_group_repeats=(1, 32 // 4),
        tile_group_steps=(2, 2),
        allow_partial=True,
    )
    assert len(tiles) == (32 // (2 * 1)) * (32 // (2 * (32 // 4)))
    assert tiles[0] == TensorTile(
        (32, 32), offset=0, sizes=[1, 8, 2, 2], strides=[0, 4, 32, 1]
    )
    assert tiles[1] == TensorTile(
        (32, 32), offset=2, sizes=[1, 8, 2, 2], strides=[0, 4, 32, 1]
    )
    assert tiles[26] == TensorTile(
        (32, 32), offset=832, sizes=[1, 8, 2, 2], strides=[0, 4, 32, 1]
    )
    assert tiles[-1] == TensorTile(
        (32, 32), offset=962, sizes=[1, 8, 2, 2], strides=[0, 4, 32, 1]
    )

    # Repeat other dimension
    tiles = TensorTiler2D.step_tiler(
        (32, 32),
        tile_dims=(2, 2),
        tile_group_repeats=(32 // 4, 1),
        tile_group_steps=(2, 2),
        allow_partial=True,
    )
    assert len(tiles) == (32 // (2 * 1)) * (32 // (2 * (32 // 4)))
    assert tiles[0] == TensorTile(
        (32, 32), offset=0, sizes=[1, 8, 2, 2], strides=[0, 128, 32, 1]
    )
    assert tiles[1] == TensorTile(
        (32, 32), offset=2, sizes=[1, 8, 2, 2], strides=[0, 128, 32, 1]
    )
    assert tiles[26] == TensorTile(
        (32, 32), offset=84, sizes=[1, 8, 2, 2], strides=[0, 128, 32, 1]
    )
    assert tiles[-1] == TensorTile(
        (32, 32), offset=94, sizes=[1, 8, 2, 2], strides=[0, 128, 32, 1]
    )

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: step_tiler_invalid
@construct_test
def step_tiler_invalid():
    try:
        tiles = TensorTiler2D.step_tiler(
            (), (3, 2), (1, 1), (1, 1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Bad tensor dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.step_tiler(
            (10, 9, 4), (3, 2), (1, 1), (1, 1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Too many tensor dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.step_tiler(
            (9, 4), (3, -1), (1, 1), (1, 1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Bad tile dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.step_tiler(
            (9, 4), (3,), (1, 1), (1, 1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Too few tile dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.step_tiler(
            (9, 4), (1, 1, 1), (1, 1), (1, 1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Too many tile dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.step_tiler(
            (9, 4), (3, 2), (1, 1), (1, 1), tile_col_major=True, pattern_repeat=0
        )
        raise ValueError("Invalid repeat.")
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.step_tiler(
            (9, 4), (4, 2), (1, 1), (1, 1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Indivisible tile (height)")
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.step_tiler(
            (9, 4), (3, 3), (1, 1), (1, 1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Indivisible tile (width)")
    except ValueError:
        # good
        pass

    try:
        tiles = TensorTiler2D.step_tiler(
            (9, 4), (3, 2), (1,), (1, 1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Too few tile group dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.step_tiler(
            (9, 4), (3, 2), (1, -1), (1, 1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Bad tile group dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.step_tiler(
            (9, 4), (3, 2), (1, 1, 1), (1, 1), tile_col_major=True
        )
        raise ValueError("Too many tile group dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.step_tiler(
            (18, 8), (3, 2), (2, 3), (1, 1), tile_col_major=True
        )
        raise ValueError(
            "Indivisible by tile repeat width (but without allow_partial)."
        )
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.step_tiler(
            (18, 8), (3, 2), (4, 2), (1, 1), tile_col_major=True
        )
        raise ValueError(
            "Indivisible by tile repeat height (but without allow_partial)."
        )
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.step_tiler(
            (18, 8), (3, 2), (4, 2), (1, -1), tile_col_major=True
        )
        raise ValueError("Bad tile step dims")
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.step_tiler(
            (18, 8), (3, 2), (4, 2), (1,), tile_col_major=True
        )
        raise ValueError("Too few tile step dims")
    except ValueError:
        # good
        pass

    # CHECK: Pass!
    print("Pass!")
