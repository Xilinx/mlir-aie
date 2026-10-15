# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IRON multi-core element-wise design skeleton (explicit topology).

Before adapting this: if your op is in `aie.iron.kernels` and the topology is
"same op over every tile", you don't need any of this — a few lines of
`aie.iron.algorithms.transform_parallel` inside an `@iron.jit` function does the
same job with no C++ and no hand-built fifos. See references/builtin_kernels.md.

Use this skeleton when the topology is genuinely custom, or when no built-in
kernel covers your op/dtype.

Adapt:
  - TENSOR_SIZE, N_WORKERS
  - DTYPE
  - kernel name + .o file in `Kernel(...)` — the placeholder `my_unary_kernel`
    assumes a unary `(in, out, n_elements)` signature; if you're wiring in a
    binary kernel like `eltwise_add_bf16_vector` (see kernel_intrinsics.md),
    drop the trailing size arg and adjust `core_fn` to acquire/pass two inputs
  - the body of `core_fn` if you need a different kernel-call pattern
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, Kernel, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2


def build_design(
    tensor_size: int = 4096,
    n_workers: int = 4,
    kernel_fn_name: str = "my_unary_kernel",
    kernel_obj_file: str = "kernel.o",
):
    """Build a data-parallel element-wise design and return the MLIR module."""

    assert tensor_size % n_workers == 0, "tensor_size must be divisible by n_workers"

    DTYPE = bfloat16
    tensor_ty = np.ndarray[(tensor_size,), np.dtype[DTYPE]]

    # Top-level FIFOs, split/joined across n_workers
    of_in_top = ObjectFifo(tensor_ty, name="in")
    of_out_top = ObjectFifo(tensor_ty, name="out")

    chunk = tensor_size // n_workers
    offsets = [chunk * i for i in range(n_workers)]
    chunk_ty = np.ndarray[(chunk,), np.dtype[DTYPE]]

    sub_ins = of_in_top.cons().split(
        offsets,
        obj_types=[chunk_ty] * n_workers,
        names=[f"in{i}" for i in range(n_workers)],
    )
    sub_outs = of_out_top.prod().join(
        offsets,
        obj_types=[chunk_ty] * n_workers,
        names=[f"out{i}" for i in range(n_workers)],
    )

    # Each worker consumes its chunk directly from sub_ins / produces to sub_outs.
    # If a chunk is too large for L1, add a per-worker inner ObjectFifo (depth=2)
    # and forward the chunk through it in smaller pieces.

    kernel_fn = Kernel(
        kernel_fn_name,
        kernel_obj_file,
        [
            chunk_ty,
            chunk_ty,
            chunk_ty,
        ],  # signature: (in0, in1, out) — this skeleton self-adds (in + in)
    )

    def core_fn(of_in, of_out, kfn):
        e_in = of_in.acquire(1)
        e_out = of_out.acquire(1)
        kfn(e_in, e_in, e_out)
        of_in.release(1)
        of_out.release(1)

    workers = [
        Worker(core_fn, [sub_ins[i].cons(), sub_outs[i].prod(), kernel_fn])
        for i in range(n_workers)
    ]

    def sequence(a_in, c_out, in_h, out_h):
        in_h.fill(a_in)
        out_h.drain(c_out, wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in_top.prod(), of_out_top.cons()])

    return Program(NPU2(), rt, workers=workers).resolve_program()


@iron.jit
def my_op(
    input_tensor: In,
    output_tensor: Out,
    *,
    tensor_size: CompileTime[int] = 8192,
    n_workers: CompileTime[int] = 4,
):
    """JIT entry point — wires `build_design` into a runnable kernel."""
    return build_design(tensor_size=tensor_size, n_workers=n_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-mlir", action="store_true")
    args = parser.parse_args()

    if args.print_mlir:
        print(build_design())
    else:
        N = 8192
        inp = iron.rand((N,), dtype=np.dtype(bfloat16), device="npu")
        outp = iron.zeros_like(inp)
        my_op(inp, outp, tensor_size=N)
        print("first 8 outputs:", outp.numpy()[:8])
