# dispatch_overhead_bisector.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
#
# AIE2P per-launch dispatch-overhead bisector example.
#
# Goal: provide an IRON topology that exposes a TRIVIAL kernel (pure
# passthrough, no arithmetic, no per-element compute) but still goes
# through the entire xrt::run() -> xrt::run.wait() dispatch path.
# The four diagnostic suspects below can be isolated by varying ONE
# knob each at the host-runner level.
#
# Diagnostic suspects under bisection:
#   (a) per-launch xrt::run.wait() return-path overhead
#   (b) per-launch instruction-stream upload cost
#   (c) per-chunk shim-DMA setup overhead × N_CHUNKS
#   (d) AIE2P firmware dispatcher per-launch handshake
#
# This file emits the topology for ONE (n_chunks, dense_bytes) variant.
# A driver script can build N variants by parameterising N_CHUNKS and
# DENSE_BYTES via CLI flag and rebuilding the xclbin per variant; a
# simple linear regression across the (n_chunks, wall_time) and
# (dense_bytes, wall_time) sweeps separates suspect (c) per-chunk
# shim-DMA setup from the per-byte payload-DMA throughput, leaving
# the (a)+(b)+(d) fixed per-launch floor as the regression intercept.
#
# Topology:
#   shim(0,0) --(of_in)--> Tile (0,2) passThroughLine --(of_out)--> shim(0,0)
#
# Single compute tile, single fifo each direction, no memtiles, no
# cross-column routing. The simplest possible IRON topology that still
# emits a real shim<->compute<->shim DMA chain per launch — the only
# way to attribute per-launch wall cost to the dispatch layer rather
# than kernel compute.

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.helpers.taplib import TensorAccessPattern


# Default per-chunk payload size. 4096 bytes is large enough to be
# above the small-payload firmware-flush behaviour observed at
# ``dense_bytes <= 256`` on AIE2P silicon, and is a multiple of 64
# (the kernel's vector inner-loop width).
DEFAULT_DENSE_BYTES: int = 4096


def my_dispatch_overhead_topology(dev, dense_bytes: int, n_chunks: int):
    """Single-tile passthrough topology with N_CHUNKS chunked transfers.

    Per launch, the host runtime issues ``n_chunks`` separate
    fill/drain pairs of ``dense_bytes`` bytes each, all going through
    the same single compute-tile passthrough. The total per-launch
    work is ``n_chunks * dense_bytes`` bytes copied, but the per-chunk
    shim-DMA setup cost (suspect c) accumulates linearly with
    ``n_chunks``.

    Two ORTHOGONAL knobs the driver can sweep:

      shim-chunks knob: N in {1, 2, 4}
        -> isolates suspect (c) per-chunk shim-DMA setup
      dense-bytes knob: B in e.g. {512, 4096, 8192}
        -> isolates per-byte payload-DMA throughput from the per-launch
           fixed cost (a)+(b)+(d)

    The AIE2P shim BD pool is finite (the lowering pass complains
    about unassigned BD chain IDs once the per-channel BD count
    exceeds the auto-chain threshold; empirically this is hit around
    n_chunks=16 -> 32 total shim BDs). The shim-chunks knob is
    therefore typically capped at 4.
    """
    line_type = np.ndarray[(dense_bytes,), np.dtype[np.uint8]]
    total_bytes = dense_bytes * n_chunks
    vector_type = np.ndarray[(total_bytes,), np.dtype[np.uint8]]

    of_in = ObjectFifo(line_type, name="bisector_in", depth=2)
    of_out = ObjectFifo(line_type, name="bisector_out", depth=2)

    passthrough_fn = Kernel(
        "passThroughLine",
        "passthrough.cc.o",
        [line_type, line_type, np.int32],
    )

    def core_fn(of_in_h, of_out_h, passThroughLine):
        # Run n_chunks iterations on the compute tile so the work
        # matches the host's n_chunks fill/drain pairs.
        for _ in range(n_chunks):
            elem_in = of_in_h.acquire(1)
            elem_out = of_out_h.acquire(1)
            passThroughLine(elem_in, elem_out, dense_bytes)
            of_in_h.release(1)
            of_out_h.release(1)

    worker = Worker(
        core_fn,
        [of_in.cons(), of_out.prod(), passthrough_fn],
        tile=Tile(0, 2),
    )

    rt = Runtime()
    with rt.sequence(vector_type, vector_type, vector_type) as (a_in, b_out, _):
        rt.start(worker)
        # Emit n_chunks SEPARATE fill/drain tasks in the runtime_sequence
        # so each chunk lowers to its own shim DMA descriptor (BD).
        # This is what makes N_CHUNKS the bisector knob for suspect (c)
        # per-chunk shim-DMA setup overhead. Without separate tasks,
        # IRON collapses the linear access pattern into a single big-BD
        # fill which makes N_CHUNKS only vary the on-tile compute-loop
        # count and the total transfer size, not the per-chunk setup
        # count.
        for k in range(n_chunks):
            offset = k * dense_bytes
            tap_in = TensorAccessPattern(
                tensor_dims=(n_chunks * dense_bytes,),
                offset=offset,
                sizes=[1, 1, 1, dense_bytes],
                strides=[0, 0, 0, 1],
            )
            tap_out = TensorAccessPattern(
                tensor_dims=(n_chunks * dense_bytes,),
                offset=offset,
                sizes=[1, 1, 1, dense_bytes],
                strides=[0, 0, 0, 1],
            )
            rt.fill(of_in.prod(), a_in, tap=tap_in)
            rt.drain(of_out.cons(), b_out, tap=tap_out,
                     wait=(k == n_chunks - 1))

    return Program(dev, rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dev", default="npu2",
                   help="AIE Device (npu2 only)")
    p.add_argument("--dense-bytes", type=int, default=DEFAULT_DENSE_BYTES,
                   help="Bytes per chunk")
    p.add_argument("--n-chunks", type=int, default=1,
                   help="Number of chunks per launch (bisector knob for "
                        "suspect c)")
    args = p.parse_args(sys.argv[1:])

    if args.dev != "npu2":
        raise ValueError(
            "dispatch_overhead_bisector targets AIE2P silicon (-d npu2)."
        )

    if args.dense_bytes % 64 != 0:
        raise ValueError(
            f"dense_bytes must be a multiple of 64; got {args.dense_bytes}"
        )

    if args.n_chunks < 1:
        raise ValueError(f"n_chunks must be >= 1; got {args.n_chunks}")

    print(my_dispatch_overhead_topology(NPU2(), args.dense_bytes,
                                        args.n_chunks))


if __name__ == "__main__":
    main()
