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
// A third pair streams the first 8 * @rows values twice, and its S2MM scatters
// them into %out3 as rows of 8 at a stride of 16, with an iteration dimension
// placing the second pass 256 values after the first:
//
//   rows = 2 → out3[0..7] = out3[256..263] = 0..7,
//              out3[16..23] = out3[272..279] = 8..15
//
module {
    aiex.scratchpad_parameter @rows : i32

    aie.device(npu2) @empty { }

    aie.device(npu2) @test {
        %t00 = aie.tile(0, 0)
        %t10 = aie.tile(1, 0)
        %t20 = aie.tile(2, 0)
        %t30 = aie.tile(3, 0)

        aie.flow(%t00, DMA : 0, %t10, DMA : 0)
        aie.flow(%t00, DMA : 1, %t10, DMA : 1)
        aie.shim_dma_allocation @in(%t00, MM2S, 0)
        aie.shim_dma_allocation @out(%t10, S2MM, 0)
        aie.shim_dma_allocation @in2(%t00, MM2S, 1)
        aie.shim_dma_allocation @out2(%t10, S2MM, 1)
        aie.flow(%t20, DMA : 0, %t30, DMA : 0)
        aie.shim_dma_allocation @in3(%t20, MM2S, 0)
        aie.shim_dma_allocation @out3(%t30, S2MM, 0)

        aie.runtime_sequence @sequence(%in : memref<256xi32>, %out : memref<256xi32>, %out2 : memref<256xi32>, %out3 : memref<512xi32>) {

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

            %t_in3 = aiex.dma_configure_task_for @in3 {
                aie.dma_bd(%in : memref<256xi32> offset = 0 len = 0) {length_parameter = @rows, length_unit = 8 : i32}
                aie.end
            } {repeat_count = 1 : i32}

            %t_out3 = aiex.dma_configure_task_for @out3 {
                aie.dma_bd(%out3 : memref<512xi32> offset = 0 len = 0 sizes = [2, 1, 2, 8] strides = [256, 0, 16, 1]) {length_parameter = @rows, length_unit = 8 : i32}
                aie.end
            } {issue_token = true, repeat_count = 1 : i32}

            aiex.dma_start_task(%t_in)
            aiex.dma_start_task(%t_out)
            aiex.dma_start_task(%t_in2)
            aiex.dma_start_task(%t_out2)
            aiex.dma_start_task(%t_in3)
            aiex.dma_start_task(%t_out3)
            aiex.dma_await_task(%t_out)
            aiex.dma_await_task(%t_out2)
            aiex.dma_await_task(%t_out3)
        }
    }
}
