#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# REQUIRES: ryzen_ai
#
# RUN: %run_on_npu1% %python %s --iters 50 | FileCheck %s
# RUN: %run_on_npu2% %python %s --iters 50 | FileCheck %s
# CHECK: PASS!

"""Microbenchmark: object-linked ``func.call`` vs inlined kernel (issue #3396).

IRON keeps control loops in Python and kernels in C++, so a compute core makes a
``func.call`` into the kernel once per tile.  In a tight loop that per-call
overhead adds up.  ``ExternalFunction(inline=True)`` emits the kernel as
``alwaysinline`` LLVM IR that aiecc ``llvm-link``s into the core module and
inlines -- removing the call boundary (and the separate kernel ``.o``).

This runs a deliberately call-heavy design -- a tiny 16-element ``add_one``
kernel invoked once per tile over a large tensor, so call overhead dominates the
compute -- two ways and reports each variant's host-visible latency:

  * object-linked : ``ExternalFunction("add_one")``               -> a call per tile
  * inlined       : ``ExternalFunction("add_one", inline=True)``   -> body folded in

Both are validated for numerical equality, so this doubles as a correctness
check of the inline path.

Caveat: this measures end-to-end host latency (launch + DMA + compute), not
isolated on-core cycles.  For cycle-accurate call overhead, bracket the kernel
loop with the AIE trace (event0/event1).  This example targets a quick, portable
A/B signal that the inline path is wired up and faster on a call-bound workload.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, ExternalFunction, In, Out, jit
from aie.iron import ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_

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
    func: CompileTime[object],
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

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, B):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), B, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _add_one(inline: bool) -> ExternalFunction:
    return ExternalFunction(
        "add_one",
        source_string=_SRC,
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.int32,
        ],
        inline=inline,
    )


def _bench(label: str, inline: bool, num_elements: int, iters: int) -> float:
    # Independent build per variant.
    transform._kernel_cache.clear()
    ExternalFunction._instances.clear()

    x = iron.randint(0, 100, (num_elements,), dtype=np.int32, device="npu")
    y = iron.zeros((num_elements,), dtype=np.int32, device="npu")
    expected = x.numpy() + 1

    func = _add_one(inline=inline)
    transform(x, y, func=func, num_elements=num_elements)  # warm up / compile
    np.testing.assert_array_equal(y.numpy(), expected)

    start = time.perf_counter()
    for _ in range(iters):
        transform(x, y, func=func, num_elements=num_elements)
    per_iter = (time.perf_counter() - start) / iters

    calls = num_elements // 16
    print(f"  {label:<13} {per_iter * 1e3:8.3f} ms/iter   ({calls} calls/iter)")
    return per_iter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-elements", type=int, default=16384)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    print(
        f"add_one microbench: {args.num_elements} elems, tile=16, "
        f"{args.iters} timed iters"
    )
    try:
        obj = _bench("object-link", False, args.num_elements, args.iters)
        inl = _bench("inline", True, args.num_elements, args.iters)
        if inl > 0.0:
            print(f"  speedup (object/inline): {obj / inl:.3f}x")
        print("PASS!")
        return 0
    except Exception as exc:  # noqa: BLE001 - example-level reporting
        import traceback

        traceback.print_exc()
        print(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
