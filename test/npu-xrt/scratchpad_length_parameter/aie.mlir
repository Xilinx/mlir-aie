// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Test: DMA transfer length set at runtime via length_parameter.
//
// The input holds rows of 16 i32 values [0, 1, ..., 255]. The shim MM2S reads
// the first 8 values of each row and streams them to a second shim, whose
// S2MM writes them contiguously. Both BDs have a static length of 0, so they
// move exactly @rows rows, one row (8 values) per unit:
//
//   rows = 0 → nothing
//   rows = 1 → 8 values: 0..7
//   rows = 2 → 16 values: 0..7, 16..23
//
// The MM2S pattern is strided: its sizes give the shape of each row and the
// runtime length counts steps of its third dimension. A second channel pair
// reads the last 8 values of each row with a two-dimensional pattern, whose
// rows the lowering steps in the third dimension, into %out2:
//
//   rows = 2 → 16 values: 8..15, 24..31
//
module {
    aiex.scratchpad_parameter @rows : i32

    aie.device(npu2) @empty { }

    aie.device(npu2) @test {
        %t00 = aie.tile(0, 0)
        %t10 = aie.tile(1, 0)

        aie.flow(%t00, DMA : 0, %t10, DMA : 0)
        aie.flow(%t00, DMA : 1, %t10, DMA : 1)
        aie.shim_dma_allocation @in(%t00, MM2S, 0)
        aie.shim_dma_allocation @out(%t10, S2MM, 0)
        aie.shim_dma_allocation @in2(%t00, MM2S, 1)
        aie.shim_dma_allocation @out2(%t10, S2MM, 1)

        aie.runtime_sequence @sequence(%in : memref<256xi32>, %out : memref<256xi32>, %out2 : memref<256xi32>) {

            aiex.npu.load_pdi { device_ref = @empty }
            aiex.npu.load_pdi { device_ref = @test }

            %t_in = aiex.dma_configure_task_for @in {
                aie.dma_bd(%in : memref<256xi32> offset = 0 len = 0 sizes = [2, 2, 4] strides = [16, 4, 1]) {length_parameter = @rows, length_unit = 8 : i32}
                aie.end
            }

            %t_out = aiex.dma_configure_task_for @out {
                aie.dma_bd(%out : memref<256xi32> offset = 0 len = 0) {length_parameter = @rows, length_unit = 8 : i32}
                aie.end
            } {issue_token = true}

            %t_in2 = aiex.dma_configure_task_for @in2 {
                aie.dma_bd(%in : memref<256xi32> offset = 8 len = 0 sizes = [2, 8] strides = [16, 1]) {length_parameter = @rows, length_unit = 8 : i32}
                aie.end
            }

            %t_out2 = aiex.dma_configure_task_for @out2 {
                aie.dma_bd(%out2 : memref<256xi32> offset = 0 len = 0) {length_parameter = @rows, length_unit = 8 : i32}
                aie.end
            } {issue_token = true}

            aiex.dma_start_task(%t_in)
            aiex.dma_start_task(%t_out)
            aiex.dma_start_task(%t_in2)
            aiex.dma_start_task(%t_out2)
            aiex.dma_await_task(%t_out)
            aiex.dma_await_task(%t_out2)
        }
    }
}
