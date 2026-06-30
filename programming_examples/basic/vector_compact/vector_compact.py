# vector_compact/vector_compact.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
"""Vectorized stream-compaction (left-pack) — IRON + ``@iron.jit``

The C++ kernel (``vector_compact_kernel.cc``) left-packs the survivors of a
threshold test (here ``x >= 0``) from an N-element ``bfloat16`` input into the
output, followed by zeros, using a butterfly / prefix-scan network (see the
kernel header comment).  Survivors keep their relative order.

A single AIE core processes the whole N=1024 tensor in one call, because the
left-pack carries a running write offset across the 32-element tiles.

  * standalone:   ``python3 vector_compact.py``
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``  (NPU Makefile)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

_KERNEL_SRC = str(Path(__file__).parent / "vector_compact_kernel.cc")
_N = 1024  # fixed for this version (matches N_ELEMS default in the kernel)


def _build_design(kernel_name):
    """Return an @iron.jit design that invokes the named C kernel symbol.

    Both ``bf16_vector_compact`` (butterfly/prefix-scan) and
    ``bf16_scalar_compact`` (scalar baseline) share the same two-pointer
    signature and ObjectFifo plumbing, so the design is identical apart from
    the external-function symbol it calls.
    """

    @iron.jit
    def design(a: In, c: Out):
        N = _N

        tensor_ty = np.ndarray[(N,), np.dtype[bfloat16]]

        compact_fn = ExternalFunction(
            kernel_name,
            source_file=_KERNEL_SRC,
            arg_types=[tensor_ty, tensor_ty],
            compile_flags=[f"-DN_ELEMS={N}", "-DTILE=32"],
        )

        in_fifo = ObjectFifo(tensor_ty, name="in_fifo")
        out_fifo = ObjectFifo(tensor_ty, name="out_fifo")

        def core_fn(in_fifo, out_fifo, compact_fn):
            for _ in range_(1):
                elem_in = in_fifo.acquire(1)
                elem_out = out_fifo.acquire(1)
                compact_fn(elem_in, elem_out)
                in_fifo.release(1)
                out_fifo.release(1)

        worker = Worker(
            core_fn, fn_args=[in_fifo.cons(), out_fifo.prod(), compact_fn]
        )

        rt = Runtime()
        with rt.sequence(tensor_ty, tensor_ty) as (a_in, c_out):
            rt.start(worker)
            rt.fill(in_fifo.prod(), a_in)
            rt.drain(out_fifo.cons(), c_out, wait=True)

        return Program(iron.get_current_device(), rt).resolve_program()

    return design


# The butterfly (vector) kernel is the primary design used by run_design_cli /
# the NPU compile-only Makefile path.  The scalar baseline is built on demand by
# the benchmark in _run_and_verify.
vector_compact = _build_design("bf16_vector_compact")


def _reference(x_np):
    """numpy reference: left-pack survivors (x >= 0), zero-pad the rest."""
    tau = 0.0
    survivors = x_np[x_np >= tau]
    k = len(survivors)
    ref = np.zeros(_N, dtype=bfloat16)
    ref[:k] = survivors
    return ref, k


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Vector Stream Compaction")
    add_compile_args(p)
    return p


def _compile_kwargs(opts):
    return dict()


def _run_kernel(design, x_np):
    a_t = iron.tensor(x_np, dtype=bfloat16, device="npu")
    c_t = iron.zeros(_N, dtype=bfloat16, device="npu")
    design(a_t, c_t)
    return np.array(c_t.numpy(), copy=True)


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    x_np = rng.uniform(-1.0, 1.0, size=(_N,)).astype(bfloat16)
    ref, k = _reference(x_np)

    kernels = [
        ("butterfly", vector_compact),
        ("scalar",    _build_design("bf16_scalar_compact")),
    ]
    for name, design in kernels:
        out = _run_kernel(design, x_np)
        ok_survivors = np.array_equal(out[:k], ref[:k])
        ok_positive  = bool(np.all(out[:k].astype(np.float32) >= 0.0))
        ok_tail      = bool(np.all(out[k:].astype(np.float32) == 0.0))
        if not (ok_survivors and ok_positive and ok_tail):
            nbad = int(np.sum(out[:k] != ref[:k]))
            print(
                f"FAIL ({name}): survivors_match={ok_survivors} "
                f"(mismatches={nbad}), all_positive={ok_positive}, "
                f"tail_zero={ok_tail}"
            )
            sys.exit(1)

    print("PASS")


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        vector_compact,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
