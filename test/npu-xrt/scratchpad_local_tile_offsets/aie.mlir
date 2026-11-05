// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Runtime offsets on a mem tile and a core tile buffer descriptor. The shim
// fills a 64-word table into a buffer on each tile; each tile's MM2S then
// reads it back from a runtime offset: 8 words from the mem tile at @m_off,
// 13 from the core tile at @c_off. npu.update_from_scratchpad clears bits
// [1:0] of the register it adds into: on the mem tile those are the address's
// own low bits, so @m_off moves in 4 words; on the core tile they are the low
// bits of the BD's length (13), which the lowering writes back after.

module {
    aiex.scratchpad_parameter @m_off : i32
    aiex.scratchpad_parameter @c_off : i32

    aie.device(npu2) @empty { }

    aie.device(npu2) @test {
        %t00 = aie.tile(0, 0)
        %t01 = aie.tile(0, 1)
        %t02 = aie.tile(0, 2)

        %m_empty = aie.lock(%t01, 0) {init = 1 : i32, sym_name = "m_empty"}
        %m_full = aie.lock(%t01, 1) {init = 0 : i32, sym_name = "m_full"}
        %c_empty = aie.lock(%t02, 0) {init = 1 : i32, sym_name = "c_empty"}
        %c_full = aie.lock(%t02, 1) {init = 0 : i32, sym_name = "c_full"}
        %m_buf = aie.buffer(%t01) {address = 0 : i32, sym_name = "m_buf"} : memref<64xi32>
        %c_buf = aie.buffer(%t02) {address = 4096 : i32, sym_name = "c_buf"} : memref<64xi32>

        aie.flow(%t00, DMA : 0, %t01, DMA : 0)
        aie.flow(%t01, DMA : 0, %t00, DMA : 0)
        aie.flow(%t00, DMA : 1, %t02, DMA : 0)
        aie.flow(%t02, DMA : 0, %t00, DMA : 1)

        aie.runtime_sequence @sequence(%in : memref<64xi32>, %m_out : memref<16xi32>, %c_out : memref<16xi32>) {

            aiex.npu.load_pdi { device_ref = @empty }
            aiex.npu.load_pdi { device_ref = @test }

            %m_fill = aiex.dma_configure_task(%t01, S2MM, 0) {
                %one = arith.constant 1 : i32
                aie.use_lock(%m_empty, AcquireGreaterEqual, %one)
                aie.dma_bd(%m_buf : memref<64xi32> offset = 0 len = 64)
                aie.use_lock(%m_full, Release, %one)
                aie.end
            }
            %m_read = aiex.dma_configure_task(%t01, MM2S, 0) {
                %one = arith.constant 1 : i32
                aie.use_lock(%m_full, AcquireGreaterEqual, %one)
                aie.dma_bd(%m_buf : memref<64xi32> offset = 0 len = 8) {offset_parameter = @m_off}
                aie.use_lock(%m_empty, Release, %one)
                aie.end
            }
            %c_fill = aiex.dma_configure_task(%t02, S2MM, 0) {
                %one = arith.constant 1 : i32
                aie.use_lock(%c_empty, AcquireGreaterEqual, %one)
                aie.dma_bd(%c_buf : memref<64xi32> offset = 0 len = 64)
                aie.use_lock(%c_full, Release, %one)
                aie.end
            }
            %c_read = aiex.dma_configure_task(%t02, MM2S, 0) {
                %one = arith.constant 1 : i32
                aie.use_lock(%c_full, AcquireGreaterEqual, %one)
                aie.dma_bd(%c_buf : memref<64xi32> offset = 0 len = 13) {offset_parameter = @c_off}
                aie.use_lock(%c_empty, Release, %one)
                aie.end
            }
            %m_in = aiex.dma_configure_task(%t00, MM2S, 0) {
                aie.dma_bd(%in : memref<64xi32> offset = 0 len = 64)
                aie.end
            }
            %c_in = aiex.dma_configure_task(%t00, MM2S, 1) {
                aie.dma_bd(%in : memref<64xi32> offset = 0 len = 64)
                aie.end
            }
            %m_res = aiex.dma_configure_task(%t00, S2MM, 0) {
                aie.dma_bd(%m_out : memref<16xi32> offset = 0 len = 8)
                aie.end
            } {issue_token = true}
            %c_res = aiex.dma_configure_task(%t00, S2MM, 1) {
                aie.dma_bd(%c_out : memref<16xi32> offset = 0 len = 13)
                aie.end
            } {issue_token = true}
            aiex.dma_start_task(%m_res)
            aiex.dma_start_task(%c_res)
            aiex.dma_start_task(%m_read)
            aiex.dma_start_task(%c_read)
            aiex.dma_start_task(%m_fill)
            aiex.dma_start_task(%c_fill)
            aiex.dma_start_task(%m_in)
            aiex.dma_start_task(%c_in)
            aiex.dma_await_task(%m_res)
            aiex.dma_await_task(%c_res)
        }
    }
}
