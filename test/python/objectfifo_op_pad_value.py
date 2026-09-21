# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# The object_fifo() builder plumbs padDimensions (geometry) and padValue (the
# constant fill) onto aie.objectfifo. Targets npu2, which has the
# CONSTANT_PAD_VALUE register.

import numpy as np
from aie.dialects.aie import AIEDevice, device, object_fifo, tile
from aie.extras.context import mlir_mod_ctx

with mlir_mod_ctx() as ctx:

    @device(AIEDevice.npu2_1col)
    def device_body():
        shim = tile(0, 0)
        mem = tile(0, 1)

        # CHECK: aie.objectfifo @of{{.*}}padValue = 7 : i32
        object_fifo(
            "of",
            mem,  # producer memtile: padding is on its MM2S channel
            shim,
            2,
            np.ndarray[(64,), np.dtype[np.int8]],
            dimensionsToStream=[(8, 8), (8, 1)],
            padDimensions=[(0, 1), (0, 0)],
            padValue=7,
        )

    print(ctx.module)
