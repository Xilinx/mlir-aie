// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Test: a DMA task's repeat count set at runtime via repeat_parameter.
//
// The input holds 1024 i32 values [0, 1, ..., 1023]. Two shim MM2S tasks
// stream to a second shim, each started with the repeat count @reps:
//
//   pair 0: a 64-value BD repeated, so out[0 : 64 (reps + 1)] holds reps + 1
//           copies of 0..63;
//   pair 1: a 64-value BD whose iteration steps 64 values per run (wrap 16),
//           so out2[0 : 64 (reps + 1)] = 0 .. 64 (reps + 1) - 1: the runtime
//           count stops the iteration early.
//
// Each task's repeat count, 15, is the parameter's maximum. Each S2MM moves
// @rows units of 64 i32 (length_parameter); the host writes rows = reps + 1.
//
// The push adds the host's count to what the task-queue register reads back,
// so both queues first take a plain push in the same run, with a constant
// repeat count of 1, into out[1024 : 1152] and out2[1024 : 1152]: if the
// register read back that push, the parameter push after it would start the
// wrong BD or repeat the wrong number of times.
//
module {
    aiex.scratchpad_parameter @rows : i32
    aiex.scratchpad_parameter @reps : i32

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

        aie.runtime_sequence @sequence(%in : memref<1024xi32>, %out : memref<2048xi32>, %out2 : memref<2048xi32>) {

            aiex.npu.load_pdi { device_ref = @empty }
            aiex.npu.load_pdi { device_ref = @test }

            %p_out = aiex.dma_configure_task_for @out {
                aie.dma_bd(%out : memref<2048xi32> offset = 1024 len = 128)
                aie.end
            } {issue_token = true}

            %p_out2 = aiex.dma_configure_task_for @out2 {
                aie.dma_bd(%out2 : memref<2048xi32> offset = 1024 len = 128)
                aie.end
            } {issue_token = true}

            %p_in = aiex.dma_configure_task_for @in {
                aie.dma_bd(%in : memref<1024xi32> offset = 0 len = 64)
                aie.end
            } {repeat_count = 1 : i32}

            %p_in2 = aiex.dma_configure_task_for @in2 {
                aie.dma_bd(%in : memref<1024xi32> offset = 0 len = 64) {iteration = #aie.bd_iteration<size = 16, stride = 64, current = 0>}
                aie.end
            } {repeat_count = 1 : i32}

            aiex.dma_start_task(%p_out)
            aiex.dma_start_task(%p_out2)
            aiex.dma_start_task(%p_in)
            aiex.dma_start_task(%p_in2)
            aiex.dma_await_task(%p_out)
            aiex.dma_await_task(%p_out2)

            %t_out = aiex.dma_configure_task_for @out {
                aie.dma_bd(%out : memref<2048xi32> offset = 0 len = 0) {length_parameter = @rows, length_unit = 64 : i32}
                aie.end
            } {issue_token = true}

            %t_out2 = aiex.dma_configure_task_for @out2 {
                aie.dma_bd(%out2 : memref<2048xi32> offset = 0 len = 0) {length_parameter = @rows, length_unit = 64 : i32}
                aie.end
            } {issue_token = true}

            %t_in = aiex.dma_configure_task_for @in {
                aie.dma_bd(%in : memref<1024xi32> offset = 0 len = 64)
                aie.end
            } {repeat_count = 15 : i32}

            %t_in2 = aiex.dma_configure_task_for @in2 {
                aie.dma_bd(%in : memref<1024xi32> offset = 0 len = 64) {iteration = #aie.bd_iteration<size = 16, stride = 64, current = 0>}
                aie.end
            } {repeat_count = 15 : i32}

            aiex.dma_start_task(%t_out)
            aiex.dma_start_task(%t_out2)
            aiex.dma_start_task(%t_in) {repeat_parameter = @reps}
            aiex.dma_start_task(%t_in2) {repeat_parameter = @reps}
            aiex.dma_await_task(%t_out)
            aiex.dma_await_task(%t_out2)
        }
    }
}
