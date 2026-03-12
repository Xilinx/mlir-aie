# passthrough_kernel/passthrough_kernel_dynamic.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
#
# Dynamic runtime_sequence that takes buffer_length as a runtime parameter.
# Uses BLOCKWRITE for BD configuration (required for address_patch to work),
# then overwrites BD word[0] with a dynamic write32 for the buffer_length.
#
# The buffer_length argument is in units of 32-bit words (bytes / 4).
# For example, to transfer 4096 bytes, pass buffer_length = 1024.
#
# The core loops forever processing fixed-size ObjectFIFO elements.
# As long as total transfer is a multiple of the element size, it works.

import numpy as np
import sys

from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects import memref
import aie.extras.types as T
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    device,
    shim_dma_allocation,
    tile,
)
from aie.dialects.aiex import (
    npu_address_patch,
    npu_blockwrite,
    npu_maskwrite32,
    npu_sync,
    npu_write32,
    npu_write32_dynamic,
    runtime_sequence,
)


def passthrough_kernel_dynamic():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2)
        def device_body():
            tile_0_0 = tile(0, 0)
            shim_dma_allocation("in_shim_alloc", tile_0_0, DMAChannelDir.MM2S, 0)
            shim_dma_allocation("out_shim_alloc", tile_0_0, DMAChannelDir.S2MM, 0)

            # Trace BD data (tile 0,0 BD 15) - size-independent
            memref.global_(
                "blockwrite_data_trace",
                T.memref(8, T.i32()),
                initial_value=np.array(
                    [0, 0, 1073741824, 0, 0, 33554432, 0, 33554432],
                    dtype=np.int32,
                ),
                constant=True,
                visibility="private",
            )

            # Input BD (address 0x1D000 = 118784)
            memref.global_(
                "blockwrite_data_in_bd",
                T.memref(8, T.i32()),
                initial_value=np.array(
                    [0, 0, 0, 0, -1073741824, 33554432, 0, 33554432],
                    dtype=np.int32,
                ),
                constant=True,
                visibility="private",
            )

            # Output BD (address 0x1D020 = 118816)
            memref.global_(
                "blockwrite_data_out_bd",
                T.memref(8, T.i32()),
                initial_value=np.array(
                    [0, 0, 0, 0, -1073741824, 33554432, 0, 33554432],
                    dtype=np.int32,
                ),
                constant=True,
                visibility="private",
            )

            @runtime_sequence(
                np.ndarray[(4096,), np.dtype[np.uint8]],
                np.ndarray[(4096,), np.dtype[np.uint8]],
                np.ndarray[(4096,), np.dtype[np.uint8]],
                T.i32(),
                T.i32(),
                T.i32(),
            )
            def sequence(
                input_buf,
                output_buf,
                trace_buf,
                buffer_length,
                input_bd_addr,
                output_bd_addr,
            ):
                # Trace configuration (tile 0,2) - all static
                npu_write32(address=213200, value=2038038528, column=0, row=2)
                npu_write32(address=213204, value=1, column=0, row=2)
                npu_write32(address=213216, value=1260724769, column=0, row=2)
                npu_write32(address=213220, value=439168079, column=0, row=2)
                npu_write32(address=261888, value=289, column=0, row=2)
                npu_write32(address=261892, value=0, column=0, row=2)
                npu_write32(address=212992, value=31232, column=0, row=2)

                # Trace BD (tile 0,0 BD 15)
                trace_data = memref.get_global(
                    T.memref(8, T.i32()), "blockwrite_data_trace"
                )
                npu_blockwrite(address=119264, data=trace_data)
                npu_address_patch(addr=119268, arg_idx=4, arg_plus=0)
                npu_maskwrite32(
                    address=119304, value=3840, mask=7936, column=0, row=0
                )
                npu_write32(address=119308, value=2147483663, column=0, row=0)

                # DMA queue configuration (tile 0,0) - static
                npu_write32(address=212992, value=32512, column=0, row=0)
                npu_write32(address=213068, value=127, column=0, row=0)
                npu_write32(address=213000, value=127, column=0, row=0)

                # Input BD (0x1D000 = 118784, BD 0)
                in_bd_data = memref.get_global(
                    T.memref(8, T.i32()), "blockwrite_data_in_bd"
                )
                npu_blockwrite(address=118784, data=in_bd_data)
                npu_address_patch(addr=118788, arg_idx=0, arg_plus=0)
                npu_write32_dynamic(input_bd_addr, buffer_length)

                # Start input DMA queue
                npu_write32(address=119316, value=0)

                # Output BD (0x1D020 = 118816, BD 1)
                out_bd_data = memref.get_global(
                    T.memref(8, T.i32()), "blockwrite_data_out_bd"
                )
                npu_blockwrite(address=118816, data=out_bd_data)
                npu_address_patch(addr=118820, arg_idx=1, arg_plus=0)
                npu_write32_dynamic(output_bd_addr, buffer_length)

                # Start output DMA and configure for sync token
                npu_maskwrite32(address=119296, value=3840, mask=7936)
                npu_write32(address=119300, value=2147483649)

                # Wait for output DMA completion
                npu_sync(column=0, row=0, direction=0, channel=0)

                # Cleanup (disable trace)
                npu_write32(address=213064, value=126, column=0, row=0)
                npu_write32(address=213000, value=126, column=0, row=0)

        print(ctx.module)


if __name__ == "__main__":
    passthrough_kernel_dynamic()
