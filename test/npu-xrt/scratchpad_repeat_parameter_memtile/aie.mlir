// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A two-slot ring on mem tile (0, 1) relays 64 i32 a run from shim MM2S 0 to
// shim S2MM 0. Each call runs three phases on the same channels: A pushes
// every task with a constant repeat count of 1 (the parameter pushes after it
// add the host's count to what each queue register reads back), B runs
// @long + 1 times from in[0], C @short + 1 times from in[1024], C's drain
// transposing each run's 8x8 block. A B task that ran past its count would
// pass a C block untransposed; one that stopped short would hang.

module {
    aiex.scratchpad_parameter @long : i32
    aiex.scratchpad_parameter @long_rows : i32
    aiex.scratchpad_parameter @short : i32
    aiex.scratchpad_parameter @short_rows : i32

    aie.device(npu2) @empty { }

    aie.device(npu2) @test {
        %t00 = aie.tile(0, 0)
        %t01 = aie.tile(0, 1)

        aie.lock @prod(%t01, 0) {init = 2 : i32}
        aie.lock @cons(%t01, 1) {init = 0 : i32}
        aie.buffer @ring(%t01) : memref<128xi32>

        aie.flow(%t00, DMA : 0, %t01, DMA : 0)
        aie.flow(%t01, DMA : 0, %t00, DMA : 0)

        aie.runtime_sequence @sequence(%in : memref<2048xi32>, %out : memref<4096xi32>) {

            aiex.npu.load_pdi { device_ref = @empty }
            aiex.npu.load_pdi { device_ref = @test }

            %a_out = aiex.dma_configure_task(%t00, S2MM, 0) {
                aie.dma_bd(%out : memref<4096xi32> offset = 0 len = 128)
                aie.end
            } {issue_token = true}
            %a_drain = aiex.dma_configure_task(%t01, MM2S, 0) {
                %c1 = arith.constant 1 : i32
                aie.use_lock(@cons, AcquireGreaterEqual, %c1)
                %ring = aie.get_buffer @ring : memref<128xi32>
                aie.dma_bd(%ring : memref<128xi32> offset = 0 len = 64) {iteration = #aie.bd_iteration<size = 2, stride = 64, current = 0>}
                %c2 = arith.constant 1 : i32
                aie.use_lock(@prod, Release, %c2)
                aie.end
            } {repeat_count = 1 : i32}
            %a_fill = aiex.dma_configure_task(%t01, S2MM, 0) {
                %c3 = arith.constant 1 : i32
                aie.use_lock(@prod, AcquireGreaterEqual, %c3)
                %ring_1 = aie.get_buffer @ring : memref<128xi32>
                aie.dma_bd(%ring_1 : memref<128xi32> offset = 0 len = 64) {iteration = #aie.bd_iteration<size = 2, stride = 64, current = 0>}
                %c4 = arith.constant 1 : i32
                aie.use_lock(@cons, Release, %c4)
                aie.end
            } {repeat_count = 1 : i32}
            %a_in = aiex.dma_configure_task(%t00, MM2S, 0) {
                aie.dma_bd(%in : memref<2048xi32> offset = 0 len = 64) {iteration = #aie.bd_iteration<size = 16, stride = 64, current = 0>}
                aie.end
            } {repeat_count = 1 : i32}
            aiex.dma_start_task(%a_out)
            aiex.dma_start_task(%a_drain)
            aiex.dma_start_task(%a_fill)
            aiex.dma_start_task(%a_in)
            aiex.dma_await_task(%a_out)

            %b_out = aiex.dma_configure_task(%t00, S2MM, 0) {
                aie.dma_bd(%out : memref<4096xi32> offset = 1024 len = 0) {length_parameter = @long_rows, length_unit = 64 : i32}
                aie.end
            } {issue_token = true}
            %b_drain = aiex.dma_configure_task(%t01, MM2S, 0) {
                %c5 = arith.constant 1 : i32
                aie.use_lock(@cons, AcquireGreaterEqual, %c5)
                %ring_2 = aie.get_buffer @ring : memref<128xi32>
                aie.dma_bd(%ring_2 : memref<128xi32> offset = 0 len = 64) {iteration = #aie.bd_iteration<size = 2, stride = 64, current = 0>}
                %c6 = arith.constant 1 : i32
                aie.use_lock(@prod, Release, %c6)
                aie.end
            } {repeat_count = 15 : i32}
            %b_fill = aiex.dma_configure_task(%t01, S2MM, 0) {
                %c7 = arith.constant 1 : i32
                aie.use_lock(@prod, AcquireGreaterEqual, %c7)
                %ring_3 = aie.get_buffer @ring : memref<128xi32>
                aie.dma_bd(%ring_3 : memref<128xi32> offset = 0 len = 64) {iteration = #aie.bd_iteration<size = 2, stride = 64, current = 0>}
                %c8 = arith.constant 1 : i32
                aie.use_lock(@cons, Release, %c8)
                aie.end
            } {repeat_count = 15 : i32}
            %b_in = aiex.dma_configure_task(%t00, MM2S, 0) {
                aie.dma_bd(%in : memref<2048xi32> offset = 0 len = 64) {iteration = #aie.bd_iteration<size = 16, stride = 64, current = 0>}
                aie.end
            } {repeat_count = 15 : i32}
            aiex.dma_start_task(%b_out)
            aiex.dma_start_task(%b_drain) {repeat_parameter = @long}
            aiex.dma_start_task(%b_fill) {repeat_parameter = @long}
            aiex.dma_start_task(%b_in) {repeat_parameter = @long}
            aiex.dma_await_task(%b_out)

            %c_out = aiex.dma_configure_task(%t00, S2MM, 0) {
                aie.dma_bd(%out : memref<4096xi32> offset = 2048 len = 0) {length_parameter = @short_rows, length_unit = 64 : i32}
                aie.end
            } {issue_token = true}
            %c_drain = aiex.dma_configure_task(%t01, MM2S, 0) {
                %c9 = arith.constant 1 : i32
                aie.use_lock(@cons, AcquireGreaterEqual, %c9)
                %ring_4 = aie.get_buffer @ring : memref<128xi32>
                aie.dma_bd(%ring_4 : memref<128xi32> offset = 0 len = 64 sizes = [8, 8] strides = [1, 8]) {iteration = #aie.bd_iteration<size = 2, stride = 64, current = 0>}
                %c10 = arith.constant 1 : i32
                aie.use_lock(@prod, Release, %c10)
                aie.end
            } {repeat_count = 15 : i32}
            %c_fill = aiex.dma_configure_task(%t01, S2MM, 0) {
                %c11 = arith.constant 1 : i32
                aie.use_lock(@prod, AcquireGreaterEqual, %c11)
                %ring_5 = aie.get_buffer @ring : memref<128xi32>
                aie.dma_bd(%ring_5 : memref<128xi32> offset = 0 len = 64) {iteration = #aie.bd_iteration<size = 2, stride = 64, current = 0>}
                %c12 = arith.constant 1 : i32
                aie.use_lock(@cons, Release, %c12)
                aie.end
            } {repeat_count = 15 : i32}
            %c_in = aiex.dma_configure_task(%t00, MM2S, 0) {
                aie.dma_bd(%in : memref<2048xi32> offset = 1024 len = 64) {iteration = #aie.bd_iteration<size = 16, stride = 64, current = 0>}
                aie.end
            } {repeat_count = 15 : i32}
            aiex.dma_start_task(%c_out)
            aiex.dma_start_task(%c_drain) {repeat_parameter = @short}
            aiex.dma_start_task(%c_fill) {repeat_parameter = @short}
            aiex.dma_start_task(%c_in) {repeat_parameter = @short}
            aiex.dma_await_task(%c_out)
        }
    }
}
