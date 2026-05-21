# vector_vector_add_BDs_init_values/vector_vector_add.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Vector + vector with a buffer-descriptor pre-initialized constant.

The design body intentionally uses low-level placed IRON (``@device`` /
``@mem`` / ``@core`` / ``flow`` / raw lock + buffer) because the
pedagogical point is the ``initial_value=`` parameter on ``buffer``
that bakes ``np.arange(N)`` into the second operand at compile time
(no shim DMA needed for it).

Three invocation modes (mirrors matrix_scalar_add):

  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``  (NPU Makefile)
  * emit-MLIR:    ``... -d xcvc1902 --emit-mlir``               (vck5000)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_
from aie.utils.compile import compile_mlir_module


def _device_for(dev_str: str):
    if dev_str == "npu":
        return AIEDevice.npu1_1col
    if dev_str == "npu2":
        return AIEDevice.npu2_1col
    if dev_str == "xcvc1902":
        return AIEDevice.xcvc1902
    raise ValueError(f"[ERROR] Device name {dev_str!r} is unknown")


def _build_module(dev, col: int):
    N = 256
    n = 16
    N_div_n = N // n

    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
            tile_ty = np.ndarray[(n,), np.dtype[np.int32]]

            ShimTile = tile(col, 0)
            ComputeTile2 = tile(col, 2)

            in1_cons_prod_lock = lock(ComputeTile2, lock_id=0, init=1)
            in1_cons_cons_lock = lock(ComputeTile2, lock_id=1, init=0)
            in1_cons_buff_0 = buffer(
                tile=ComputeTile2, datatype=tile_ty, name="in1_cons_buff_0"
            )
            in2_cons_prod_lock = lock(ComputeTile2, lock_id=2, init=0)
            in2_cons_cons_lock = lock(ComputeTile2, lock_id=3, init=1)
            in2_cons_buff_0 = buffer(
                tile=ComputeTile2,
                datatype=tensor_ty,
                name="in2_cons_buff_0",
                initial_value=np.arange(N, dtype=np.int32),
            )

            out_prod_lock = lock(ComputeTile2, lock_id=4, init=1)
            out_cons_lock = lock(ComputeTile2, lock_id=5, init=0)
            out_buff_0 = buffer(tile=ComputeTile2, datatype=tile_ty, name="out_buff_0")

            flow(ShimTile, WireBundle.DMA, 0, ComputeTile2, WireBundle.DMA, 0)
            flow(ComputeTile2, WireBundle.DMA, 0, ShimTile, WireBundle.DMA, 0)

            @mem(ComputeTile2)
            def m(block):
                s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
                with block[1]:
                    use_lock(in1_cons_prod_lock, LockAction.AcquireGreaterEqual)
                    dma_bd(in1_cons_buff_0)
                    use_lock(in1_cons_cons_lock, LockAction.Release)
                    next_bd(block[1])
                with block[2]:
                    s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
                with block[3]:
                    use_lock(out_cons_lock, LockAction.AcquireGreaterEqual)
                    dma_bd(out_buff_0)
                    use_lock(out_prod_lock, LockAction.Release)
                    next_bd(block[3])
                with block[4]:
                    EndOp()

            @core(ComputeTile2)
            def core_body():
                for _ in range_(sys.maxsize):
                    use_lock(in2_cons_cons_lock, LockAction.AcquireGreaterEqual)
                    for j in range_(N_div_n):
                        use_lock(in1_cons_cons_lock, LockAction.AcquireGreaterEqual)
                        use_lock(out_prod_lock, LockAction.AcquireGreaterEqual)
                        for i in range_(n):
                            out_buff_0[i] = (
                                in2_cons_buff_0[j * N_div_n + i] + in1_cons_buff_0[i]
                            )
                        use_lock(in1_cons_prod_lock, LockAction.Release)
                        use_lock(out_cons_lock, LockAction.Release)
                    use_lock(in2_cons_prod_lock, LockAction.Release)

            shim_dma_allocation("of_in1", ShimTile, DMAChannelDir.MM2S, 0)
            shim_dma_allocation("of_out", ShimTile, DMAChannelDir.S2MM, 0)

            @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
            def sequence(A, B, C):
                in1_task = shim_dma_single_bd_task("of_in1", A, sizes=[1, 1, 1, N])
                out_task = shim_dma_single_bd_task(
                    "of_out", C, sizes=[1, 1, 1, N], issue_token=True
                )
                dma_start_task(in1_task, out_task)
                dma_await_task(out_task)
                dma_free_task(in1_task)

        res = ctx.module.operation.verify()
        if res is not True:
            raise RuntimeError(f"MLIR verify failed: {res}")
        return ctx.module


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Vector Vector Add (BDs init values)")
    p.add_argument(
        "-d", "--dev", type=str, choices=["npu", "npu2", "xcvc1902"], default="npu"
    )
    p.add_argument("-c", "--col", type=int, default=0)
    p.add_argument(
        "--emit-mlir",
        action="store_true",
        help="print the resolved MLIR module to stdout (legacy aiecc / vck5000 path)",
    )
    p.add_argument("--xclbin-path", type=str, default=None)
    p.add_argument("--insts-path", type=str, default=None)
    return p


def main():
    opts = _make_argparser().parse_args()
    module = _build_module(dev=_device_for(opts.dev), col=opts.col)
    if opts.emit_mlir:
        print(module)
        return
    if opts.xclbin_path:
        if not opts.insts_path:
            sys.exit("--xclbin-path requires --insts-path (must be set together)")
        compile_mlir_module(
            mlir_module=str(module),
            xclbin_path=opts.xclbin_path,
            insts_path=opts.insts_path,
            work_dir=str(Path(opts.xclbin_path).resolve().parent),
        )
        return
    print(module)


if __name__ == "__main__":
    main()
