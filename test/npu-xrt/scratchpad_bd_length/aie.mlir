// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Test: DMA transfer length set at runtime via length_parameter.
//
// The input shim streams to the output shim over a circuit flow. Both BDs
// have a static length of 16 words; length_parameter @extra adds the host's
// scratchpad value to each, so a dispatch moves 16 + extra words.
//
module {
    aiex.scratchpad_parameter @extra : i32

    aie.device(npu2) @empty { }

    aie.device(npu2) @test {
        %t00 = aie.tile(0, 0)
        %t10 = aie.tile(1, 0)
        aie.flow(%t00, DMA : 0, %t10, DMA : 0)
        aie.shim_dma_allocation @in0 (%t00, MM2S, 0)
        aie.shim_dma_allocation @out0 (%t10, S2MM, 0)

        aie.runtime_sequence @sequence(%in : memref<64xi32>, %out : memref<64xi32>) {
            aiex.npu.load_pdi { device_ref = @empty }
            aiex.npu.load_pdi { device_ref = @test }

            %t_in = aiex.dma_configure_task_for @in0 {
                aie.dma_bd(%in : memref<64xi32> offset = 0 len = 16) {length_parameter = @extra}
                aie.end
            }
            %t_out = aiex.dma_configure_task_for @out0 {
                aie.dma_bd(%out : memref<64xi32> offset = 0 len = 16) {length_parameter = @extra}
                aie.end
            } {issue_token = true}

            aiex.dma_start_task(%t_in)
            aiex.dma_start_task(%t_out)
            aiex.dma_await_task(%t_out)
        }
    }
}
