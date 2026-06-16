# passthrough_kernel/passthrough_kernel_dynamic.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
#
"""Dynamic passthrough TXN generation, driven entirely through high-level IRON.

The transfer length is an SSA argument of the runtime sequence rather than a
compile-time constant, so one compiled design serves any size up to the
compiled-in maximum.  This reuses the same ``@iron.jit`` design body used for
the normal run (``my_passthrough_kernel`` with ``dynamic_txn=True``).

Because the sequence sizes are runtime values, the NPU instructions can't be
lowered to a static ``insts.bin``; the host builds them at runtime from the
emitted C++ TXN header (``generate_txn_sequence``).  So this drives
``compile_mlir_module`` to produce the XCLBIN + the C++ TXN header (no insts),
letting IRON build the (content-hashed, symbol-prefixed) kernel ``.o`` itself —
no hand-built ``.o`` and no hand-rolled aiecc.
"""

import argparse
from pathlib import Path

import aie.utils as utils
from aie.iron.device import NPU1Col1, NPU2
from aie.utils.compile import compile_mlir_module

from passthrough_kernel import my_passthrough_kernel

TXN_CPP_NAME = "generated_txn.h"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--device", choices=["npu", "npu2"], default="npu2")
    p.add_argument("-i1s", "--in1_size", type=int, default=4096)
    p.add_argument("--xclbin-path", type=str, default=None)
    args = p.parse_args()

    # The design body reads the device via iron.get_current_device(), so set it
    # here.  A single column for "npu" (this is a one-column passthrough design).
    dev = NPU2() if args.device == "npu2" else NPU1Col1()
    utils.set_current_device(dev)

    n_elems = args.in1_size  # uint8 kernel: byte size == element count
    mlir_module = my_passthrough_kernel.as_mlir(
        None, None, n=n_elems, dynamic_txn=True
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
