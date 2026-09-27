# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Test fill/drain(size_parameters={dim: parameter}).

A shim BD bounds only D2 (dimension 1 of the four outermost-first sizes): its
length ends D2. Dimension 1 reaches the descriptor as given; dimension 2 is
laid out on D2 when dimension 0 is 1, the dimension outside it becoming the
iteration. Any other dimension is rejected where the transfer is written.
"""

import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2Col1
from aie.iron.scratchpad_parameter import ScratchpadParameter
from aie.helpers.taplib import TensorAccessPattern

N = 4096
buf_ty = np.ndarray[(N,), np.dtype[np.int32]]
tile_ty = np.ndarray[(64,), np.dtype[np.int32]]


def build(sizes, strides, dims):
    n = ScratchpadParameter("n", np.int32)
    of_in = ObjectFifo(tile_ty, name="of_in")
    of_out = ObjectFifo(tile_ty, name="of_out")
    worker = Worker(lambda *_: None, [of_in.cons(), of_out.prod()], while_true=False)
    tap = TensorAccessPattern((N,), 0, sizes, strides)

    def sequence(a, c, in_h, out_h):
        in_h.fill(a, tap=tap, size_parameters={d: n for d in dims})
        out_h.drain(c, wait=True)

    rt = Runtime(sequence, [buf_ty, buf_ty, of_in.prod(), of_out.cons()])
    return Program(NPU2Col1(), rt, workers=[worker]).resolve_program()


print("\nTEST: d2_as_given")
print(build([1, 16, 2, 32], [0, 256, 64, 1], [1]))
# CHECK-LABEL: d2_as_given
# CHECK: aie.dma_bd({{.*}} sizes = [1, 16, 2, 32] strides = [0, 256, 64, 1]) {size_parameter = @n}

print("\nTEST: dimension_2_moves_to_d2")
print(build([4, 16, 64], [1024, 64, 1], [2]))
# CHECK-LABEL: dimension_2_moves_to_d2
# CHECK: aie.dma_bd({{.*}} sizes = [4, 16, 1, 64] strides = [1024, 64, 0, 1]) {size_parameter = @n}

print("\nTEST: rejects_what_d2_cannot_bound")
for sizes, strides, dims in [
    ([2, 4, 16, 64], [2048, 1024, 64, 1], [2]),
    ([4, 16, 64], [1024, 64, 1], [3]),
    ([4, 16, 64], [1024, 64, 1], [0]),
    ([4, 16, 64], [1024, 64, 1], [1, 2]),
]:
    try:
        build(sizes, strides, dims)
    except ValueError as e:
        print(f"{dims}: {e}")
    else:
        raise AssertionError(f"Expected {dims} of {sizes} to be rejected")
# CHECK-LABEL: rejects_what_d2_cannot_bound
# CHECK: [2]: size_parameters patches dimension 2 of sizes [2, 4, 16, 64]: a shim descriptor bounds D2
# CHECK: [3]: size_parameters patches dimension 3 of sizes [1, 4, 16, 64]
# CHECK: [0]: size_parameters patches dimension 0 of sizes [1, 4, 16, 64]
# CHECK: [1, 2]: size_parameters patches one dimension of a transfer; got dimensions [1, 2]
