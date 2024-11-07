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
