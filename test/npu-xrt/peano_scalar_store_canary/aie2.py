# peano_scalar_store_canary/aie2.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# One core fills a tile with a byte-store loop; the host checks every byte.

import argparse

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import AnyShimTile, Tile, from_name

N = 1024


def build(dev):
    tile_ty = np.ndarray[(N,), np.dtype[np.int8]]
    fill_tile = Kernel("fill_tile", "kernel.o", [tile_ty])
    of_out = ObjectFifo(tile_ty, name="out")

    def core_fn(out_prod, fill_tile):
        c = out_prod.acquire(1)
        fill_tile(c)
        out_prod.release(1)

    worker = Worker(
        core_fn, [of_out.prod(), fill_tile], tile=Tile(0, 2), while_true=False
    )

    def sequence(host_out, out_cons):
        out_cons.drain(host_out, wait=True)

    rt = Runtime(sequence, [tile_ty, of_out.cons(tile=AnyShimTile)])
    return Program(dev, rt, workers=[worker]).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("--dev", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
with open(a.out, "w") as f:
    print(build(from_name(a.dev, n_cols=1)), file=f)
