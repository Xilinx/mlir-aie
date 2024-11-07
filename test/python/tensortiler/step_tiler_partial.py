import numpy as np

from aie.helpers.tensortiler import TensorTile, TensorTileSequence, TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: step_tiler_partial_row
@construct_test
def step_tiler_partial_row():

    # all row major
    # tile col major
    # tile group col major
    # iter col major
    # all col major
    # pattern repeat

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: step_tiler_partial_col
@construct_test
def step_tiler_partial_col():
    """
    THIS IS BUGGY:
    tensor_dims = (3 * 5 * 3, 2 * 6 * 2)

    # All row major
    tiles = TensorTiler2D.step_tiler(
        tensor_dims, tile_dims=(3, 2), tile_group_repeats=(5, 7), tile_group_steps=(2, 3), allow_partial=True
    )
    print(len(tiles))
    for t in tiles:
        print(t)
    print(tiles[0])
    print(tiles[1])
    print(tiles[3])
    print(tiles[-1])
    tiles.visualize(plot_access_count=True)
    anim = tiles.animate()
    HTML(anim.to_jshtml())
    """

    # all row major
    # tile col major
    # tile group col major
    # iter col major
    # all col major
    # pattern repeat

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: step_tiler_partial_both
@construct_test
def step_tiler_partial_both():

    # all row major
    # tile col major
    # tile group col major
    # iter col major
    # all col major
    # pattern repeat

    # CHECK: Pass!
    print("Pass!")
