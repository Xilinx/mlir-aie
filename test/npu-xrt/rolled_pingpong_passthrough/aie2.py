# rolled_pingpong_passthrough/aie2.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# REQUIRES: ryzen_ai_npu1, peano
#
# RUN: %python %S/aie2.py > ./aie2.mlir
# RUN: %aiecc %backend_flags --get-npu-insts \
# RUN:        --get-xclbin \
# RUN:        --xclbin-name=final.xclbin --npu-insts-name=insts.bin ./aie2.mlir
# RUN: %host_clang %S/test.cpp -o test.exe -std=c++17 -Wall \
# RUN:        %xrt_flags %host_link_flags %test_utils_flags
# RUN: %run_on_npu1% ./test.exe

# Rolled ping-pong passthrough, written in the high-level IRON API.
#
# The compute structure (ObjectFifos + a passthrough Worker) is expressed with
# IRON's dataflow API.  The runtime sequence itself is a rolled ping-pong: it
# issues N_TILES copies of a single input tile using a depth-1 ping-pong, where
# an scf.for carries the in-flight input task handle as an iter_arg and frees the
# previous iteration's task while the next is in flight.
#
# The rolled loop is written directly with fill/drain in "unmanaged" mode
# (managed=False): each returns a Task the body owns, carried across scf.for
# iterations as a range_ iter_arg and freed by hand while the next is in flight.
#
# The sequence is additionally wrapped in a constant-true scf.if to exercise the
# full static-path control-flow story: the loop is unrolled (the unroll pass
# descends into the scf.if arm), then the constant-predicate if is folded by
# canonicalize, leaving straight-line IR for allocation.
#
# Golden: output[i] == input[i % TILE_LEN]  (same tile repeated N_TILES times).
#
# This validates the full constant-trip ping-pong stack:
#   aie-unroll-runtime-sequence-loops   -> unrolls the loop to straight-line BDs
#   canonicalize                        -> folds the constant-predicate scf.if
#   aie-assign-runtime-sequence-bd-ids  -> colors them by liveness (ids alternate)
#   aie-dma-tasks-to-npu                -> push_queue with concrete bd_ids

import sys

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_, yield_
from aie.iron.device import NPU1Col1, NPU2

from aie.extras.dialects.arith import constant
from aie.helpers.dialects.scf import if_

DTYPE = np.int32
TILE_LEN = 256  # elements per tile
N_TILES = 8  # number of tiles streamed; must be >= 2 to exercise ping-pong
TOTAL_OUT = TILE_LEN * N_TILES


def design(dev):
    tile_ty = np.ndarray[(TILE_LEN,), np.dtype[DTYPE]]
    in_ty = np.ndarray[(TILE_LEN,), np.dtype[DTYPE]]
    out_ty = np.ndarray[(TOTAL_OUT,), np.dtype[DTYPE]]

    # depth=2 for double-buffering on the input side
    of_in = ObjectFifo(tile_ty, depth=2, name="of_in")
    of_out = ObjectFifo(tile_ty, depth=2, name="of_out")

    def core_fn(of_in, of_out):
        # Worker wraps this in `while True` by default.
        elem_out = of_out.acquire(1)
        elem_in = of_in.acquire(1)
        for i in range_(TILE_LEN):
            elem_out[i] = elem_in[i]
        of_in.release(1)
        of_out.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

    # The rolled ping-pong runtime sequence. This is a hand-rolled software
    # pipeline: fill/drain run in "unmanaged" mode (managed=False) so the body
    # owns each transfer's lifetime via the returned Task, carrying the in-flight
    # input task across scf.for iterations as a range_ iter_arg and freeing the
    # previous one while the next is in flight.
    def ping_pong(A, B):
        # Wrap the whole sequence in a constant-true scf.if. This exercises the
        # other half of the static-path invariant: the loop below is unrolled
        # (the unroll pass descends into the scf.if arm, since it runs before the
        # fold), then --canonicalize folds this constant-predicate if away,
        # leaving straight-line IR for BD-ID allocation -- same result as if the
        # if were not here.
        cond = constant(True)
        with if_(cond):
            # Prologue: first input tile.
            init_in = of_in.prod().fill(
                A,
                sizes=[1, 1, 1, TILE_LEN],
                strides=[0, 0, 0, 1],
                wait=True,
                managed=False,
            )

            # Output task: collect N_TILES output tiles contiguously. The third
            # dim walks the N_TILES tiles with stride TILE_LEN so tile i lands at
            # offset i*TILE_LEN (a size-N stride-0 wrap dim would overwrite one
            # tile).
            out_task = of_out.cons().drain(
                B,
                sizes=[1, 1, N_TILES, TILE_LEN],
                strides=[0, 0, TILE_LEN, 1],
                wait=True,
                managed=False,
            )

            # Rolled ping-pong: issue N_TILES-1 more input BDs while the previous
            # is in flight. The Task flows as iter_arg. All input BDs read from
            # the same tile (offset 0), so the output collects N_TILES copies.
            result = init_in
            for _iv, prev, result in range_(
                1, N_TILES, iter_args=[init_in], insert_yield=False
            ):
                tile_in = of_in.prod().fill(
                    A,
                    sizes=[1, 1, 1, TILE_LEN],
                    strides=[0, 0, 0, 1],
                    wait=True,
                    managed=False,
                )
                prev.free()
                yield_([tile_in])

            result.await_()
            out_task.await_()
            result.free()

    rt = Runtime()
    rt.sequence(ping_pong, [in_ty, out_ty])

    return Program(dev, rt, workers=[worker]).resolve_program()


def main():
    dev = NPU1Col1()
    if len(sys.argv) > 1 and sys.argv[1] == "npu2":
        dev = NPU2()
    print(design(dev))


if __name__ == "__main__":
    main()
