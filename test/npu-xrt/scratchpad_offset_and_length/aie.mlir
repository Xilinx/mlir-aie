// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Test: one scratchpad parameter as both the offset and the length of a
// transfer.
//
// The input holds [0, 1, ..., 1023]. The shim MM2S starts @n values in and
// moves 16 + 16 * @n values to a second shim, whose S2MM writes them
// contiguously with the same length:
//
//   n = 0 → 16 values: 0..15
//   n = 3 → 64 values: 3..66
//
module {
    aiex.scratchpad_parameter @n : i32

    aie.device(npu2) @empty { }

    aie.device(npu2) @test {
        %t00 = aie.tile(0, 0)
        %t10 = aie.tile(1, 0)

        aie.flow(%t00, DMA : 0, %t10, DMA : 0)
        aie.shim_dma_allocation @in(%t00, MM2S, 0)
        aie.shim_dma_allocation @out(%t10, S2MM, 0)

        aie.runtime_sequence @sequence(%in : memref<1024xi32>, %out : memref<1024xi32>) {

            aiex.npu.load_pdi { device_ref = @empty }
            aiex.npu.load_pdi { device_ref = @test }

            %t_in = aiex.dma_configure_task_for @in {
                aie.dma_bd(%in : memref<1024xi32> offset = 0 len = 16) {offset_parameter = @n, length_parameter = @n, length_unit = 16 : i32}
                aie.end
            }

            %t_out = aiex.dma_configure_task_for @out {
                aie.dma_bd(%out : memref<1024xi32> offset = 0 len = 16) {length_parameter = @n, length_unit = 16 : i32}
                aie.end
            } {issue_token = true}

            aiex.dma_start_task(%t_in)
            aiex.dma_start_task(%t_out)
            aiex.dma_await_task(%t_out)
        }
    }
}
