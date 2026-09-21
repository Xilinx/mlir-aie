#!/usr/bin/env python3
# inline_kernel/inline_kernel.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Test invocation lives in run.lit (npu1) / run_strix.lit (npu2); lit only
# collects `.lit` files under programming_examples/.

"""Microbenchmark: object-linked ``func.call`` vs inlined kernel (issue #3396).

IRON keeps control loops in Python and kernels in C++, so a compute core makes a
``func.call`` into the kernel once per tile.  In a tight loop that per-call
overhead adds up.  ``ExternalFunction(inline=True)`` emits the kernel as
``alwaysinline`` LLVM IR that aiecc ``llvm-link``s into the core module and
inlines -- removing the call boundary (and the separate kernel ``.o``).

This runs a deliberately call-heavy design -- a tiny 16-element ``add_one``
kernel invoked once per tile over a large tensor, so call overhead dominates the
compute -- two ways and reports what each variant costs:

  * object-linked : ``ExternalFunction("add_one")``               -> a call per tile
  * inlined       : ``ExternalFunction("add_one", inline=True)``   -> body folded in

Both variants run over the same input and are compared for exact equality, so
this doubles as a correctness check of the inline path.

Timing comes from ``aie.utils.benchmark.run_iters``, which reports on-NPU time
(captured around ``kernel.wait()``) separately from end-to-end host latency.
The NPU figure is the one quoted below: it excludes launch overhead, so the
remaining delta between the two variants is dominated by the per-tile calls.
It is still not a cycle count -- both variants move identical data, so the DMA
cost cancels in the *difference* but is present in each number.  For
cycle-accurate call overhead, bracket the kernel loop with the AIE trace
(event0/event1).
"""

from __future__ import annotations

import argparse

import aie.iron as iron
import numpy as np
from aie.iron import (
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    jit,
)
from aie.iron.controlflow import range_
from aie.utils.benchmark import run_iters
from aie.utils.verify import assert_pass

_SRC = """extern "C" {
    void add_one(int* input, int* output, int tile_size) {
        for (int i = 0; i < tile_size; i++) {
            output[i] = input[i] + 1;
        }
    }
}"""


@jit
def transform(
    input: In,
    output: Out,
    *,
    func: CompileTime[ExternalFunction],
    num_elements: CompileTime[int],
):
    tile_size = func.tile_size(0)
    num_tiles = num_elements // tile_size

    tensor_ty = np.ndarray[(num_elements,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]

    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in, of_out, fn):
        for _ in range_(num_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            fn(elem_in, elem_out, fn.tile_size(0))
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), func])

    def sequence(a, b, in_h, out_h):
        in_h.fill(a)
        out_h.drain(b, wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])

    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _add_one(inline: bool) -> ExternalFunction:
    return ExternalFunction(
        "add_one",
        source_string=_SRC,
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.dtype(np.int32),
        ],
        inline=inline,
    )


def _bench(label: str, inline: bool, x, num_elements: int, iters: int):
    """Build one variant and time it. Returns (timings, output)."""
    # Independent build per variant.
    transform._kernel_cache.clear()
    ExternalFunction._instances.clear()

    y = iron.zeros((num_elements,), dtype=np.int32, device="npu")

    # warmup=1 absorbs the JIT compile and cache population.
    bench = run_iters(
        transform,
        x,
        y,
        func=_add_one(inline=inline),
        num_elements=num_elements,
        warmup=1,
        iters=iters,
    )

    # Prefer on-NPU time: it excludes host launch overhead, which is what makes
    # the per-call delta legible.  e2e is the fallback if the runtime did not
    # report npu_time.
    stats = bench.npu if bench.npu is not None else bench.e2e
    scope = "NPU" if bench.npu is not None else "e2e"
    print(
        f"  {label:<13} {scope} (avg/min/max us): "
        f"{stats.avg_us:.1f} / {stats.min_us:.1f} / {stats.max_us:.1f}"
    )
    # Copy, don't alias: Tensor.numpy() hands back a view over the XRT buffer's
    # mapped memory (np.frombuffer(bo.map(), ...)).  `y` dies when this function
    # returns, its __del__ drops the bo, and the view would dangle -- a segfault
    # in whatever reads it next.
    return stats, y.numpy().copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-elements", type=int, default=16384)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    calls = args.num_elements // 16
    print(
        f"add_one microbench: {args.num_elements} elems, tile=16, "
        f"{calls} calls/iter, {args.iters} timed iters"
    )

    # One input for both variants, so the outputs are directly comparable.
    x = iron.randint(0, 100, (args.num_elements,), dtype=np.int32, device="npu")
    # Snapshot the host copy up front.  Reading an NPU tensor syncs its XRT
    # buffer from the device, and _bench tears down each variant's kernel
    # between runs -- so the only safe time to touch device memory is while
    # that variant's kernel is still alive.
    expected = x.numpy() + 1

    obj, obj_out = _bench("object-link", False, x, args.num_elements, args.iters)
    inl, inl_out = _bench("inline", True, x, args.num_elements, args.iters)
    if inl.avg_us > 0.0:
        print(f"  speedup (object/inline): {obj.avg_us / inl.avg_us:.3f}x")

    assert_pass(
        obj_out,
        expected,
        fail_msg="object-linked output mismatch",
        print_pass=False,
    )
    # The example's actual claim: inlining the kernel changes nothing but speed.
    # This is the PASS! that run.lit / run_strix.lit FileCheck for.
    assert_pass(inl_out, obj_out, fail_msg="inline output differs from object-linked")


if __name__ == "__main__":
    main()
