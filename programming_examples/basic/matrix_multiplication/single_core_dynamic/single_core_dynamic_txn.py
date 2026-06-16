#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Dynamic single-core GEMM TXN generation, driven through high-level IRON.

The problem size ``(M, K, N)`` is supplied to the runtime sequence as SSA
values rather than compile-time constants, so one compiled design serves any
multiple-of-tile shape.  This reuses the same ``@iron.jit`` design body as the
normal flow (``single_core_dynamic``).

Because the sequence sizes are runtime values, the NPU program can't be frozen
to a static ``insts.bin``; the host rebuilds the instruction stream at runtime
from the emitted C++ TXN header (``generate_txn_sequence(M, K, N)``).  So this
drives ``compile_mlir_module`` to produce the XCLBIN + the C++ TXN header (no
insts), letting IRON build the kernel ``.o`` itself.
"""

import argparse
from pathlib import Path

import aie.utils as utils
from aie.iron.device import NPU1Col1, NPU2
from aie.utils.compile import compile_mlir_module

from single_core_dynamic import single_core_dynamic

TXN_CPP_NAME = "generated_gemm_txn.h"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--device", choices=["npu", "npu2"], default="npu2")
    p.add_argument("-m", type=int, default=32)
    p.add_argument("-k", type=int, default=32)
    p.add_argument("-n", type=int, default=32)
    p.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="bf16"
    )
    p.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="f32",
    )
    p.add_argument("--max-m", type=int, default=4096)
    p.add_argument("--max-k", type=int, default=4096)
    p.add_argument("--max-n", type=int, default=4096)
    p.add_argument("--xclbin-path", type=str, default=None)
    args = p.parse_args()

    dev = NPU2() if args.device == "npu2" else NPU1Col1()
    utils.set_current_device(dev)

    mlir_module = single_core_dynamic.as_mlir(
        None,
        None,
        None,
        m=args.m,
        k=args.k,
        n=args.n,
        dtype_in_str=args.dtype_in,
        dtype_out_str=args.dtype_out,
        max_m=args.max_m,
        max_k=args.max_k,
        max_n=args.max_n,
    )

    if args.xclbin_path is None:
        print(mlir_module)
    else:
        xclbin_path = Path(args.xclbin_path).resolve()
        work_dir = xclbin_path.parent
        work_dir.mkdir(parents=True, exist_ok=True)
        txn_header = work_dir / TXN_CPP_NAME
        compile_mlir_module(
            mlir_module=mlir_module,
            xclbin_path=str(xclbin_path),
            work_dir=str(work_dir),
            device=dev,
            options=[
                "--aie-generate-txn-cpp",
                f"--txn-cpp-name={txn_header}",
            ],
        )
