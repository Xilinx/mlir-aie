// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %s2 = aie.tile(0, 2)
    %s3 = aie.tile(0, 3)

    %buf2 = aie.buffer(%s2) {sym_name = "buf2"} : memref<8xi32>
    %tok2 = aie.buffer(%s2) {sym_name = "tok2"} : memref<8xi32>
    %buf3 = aie.buffer(%s3) {sym_name = "buf3"} : memref<8xi32>

    %prod2 = aie.lock(%s2, 0) {init = 1 : i32, sym_name = "prod2"}
    %full2 = aie.lock(%s2, 1) {init = 0 : i32, sym_name = "full2"}
    %prod3 = aie.lock(%s3, 0) {init = 1 : i32, sym_name = "prod3"}
    %full3 = aie.lock(%s3, 1) {init = 0 : i32, sym_name = "full3"}

    %cons = aie.lock(%shim, 0) {init = 0 : i32, sym_name = "cons"}
    %cons_done = aie.lock(%shim, 1) {init = 0 : i32, sym_name = "cons_done"}

    aie.packet_flow(0x0) {
      aie.packet_source<%s2, DMA : 0>
      aie.packet_dest<%shim, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(0x1) {
      aie.packet_source<%s3, DMA : 0>
      aie.packet_dest<%shim, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(0x2) {
      aie.packet_source<%s2, DMA : 0>
      aie.packet_dest<%shim, DMA : 1>
    }

    %core2 = aie.core(%s2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c1_i32 = arith.constant 1 : i32
      %c42_i32 = arith.constant 42 : i32
      aie.use_lock(%prod2, AcquireGreaterEqual, %c1_i32)
      scf.for %i = %c0 to %c8 step %c1 {
        %i_i32 = arith.index_cast %i : index to i32
        %val = arith.addi %i_i32, %c1_i32 : i32
        memref.store %val, %buf2[%i] : memref<8xi32>
      }
      memref.store %c42_i32, %tok2[%c0] : memref<8xi32>
      aie.use_lock(%full2, Release, %c1_i32)
      aie.end
    }

    %core3 = aie.core(%s3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c1_i32 = arith.constant 1 : i32
      %c101_i32 = arith.constant 101 : i32
      aie.use_lock(%prod3, AcquireGreaterEqual, %c1_i32)
      scf.for %i = %c0 to %c8 step %c1 {
        %i_i32 = arith.index_cast %i : index to i32
        %val = arith.addi %i_i32, %c101_i32 : i32
        memref.store %val, %buf3[%i] : memref<8xi32>
      }
      aie.use_lock(%full3, Release, %c1_i32)
      aie.end
    }

    %mem2 = aie.mem(%s2) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      %c1a = arith.constant 1 : i32
      aie.use_lock(%full2, AcquireGreaterEqual, %c1a)
      aie.dma_bd(%buf2 : memref<8xi32> offset = 0 len = 8) {packet = #aie.packet_info<pkt_id = 0, pkt_type = 0>, out_of_order_id = 0 : i32}
      %c1b = arith.constant 1 : i32
      aie.use_lock(%prod2, Release, %c1b)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%tok2 : memref<8xi32> offset = 0 len = 8) {packet = #aie.packet_info<pkt_id = 2, pkt_type = 0>}
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    %mem3 = aie.mem(%s3) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      %c1c = arith.constant 1 : i32
      aie.use_lock(%full3, AcquireGreaterEqual, %c1c)
      aie.dma_bd(%buf3 : memref<8xi32> offset = 0 len = 8) {packet = #aie.packet_info<pkt_id = 1, pkt_type = 0>, out_of_order_id = 1 : i32}
      %c1d = arith.constant 1 : i32
      aie.use_lock(%prod3, Release, %c1d)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    aie.runtime_sequence(%out : memref<16xi32>, %tok : memref<8xi32>) {
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %bda = aiex.dma_bd_pool_pop(0, 0) : i32
      %t0 = aiex.dma_configure_task(%shim, S2MM, 0, <pkt_type = 0, pkt_id = 0>) {
        aie.dma_bd(%out : memref<16xi32> offset = 8 len = 8) bd_id_val %bda : i32
        aie.use_lock(%cons, Release, %c1)
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%t0)
      %bdb = aiex.dma_bd_pool_pop(0, 0) : i32
      %t1 = aiex.dma_configure_task(%shim, S2MM, 0, <pkt_type = 0, pkt_id = 0>) {
        aie.dma_bd(%out : memref<16xi32> offset = 0 len = 8) bd_id_val %bdb : i32
        aie.use_lock(%cons, Release, %c1)
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%t1)

      %bdc = aiex.dma_bd_pool_pop(0, 0) : i32
      %tc = aiex.dma_configure_task(%shim, S2MM, 1, <pkt_type = 0, pkt_id = 2>) {
        aie.use_lock(%cons, AcquireGreaterEqual, %c2)
        aie.dma_bd(%tok : memref<8xi32> offset = 0 len = 8) bd_id_val %bdc : i32
        aie.use_lock(%cons_done, Release, %c1)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tc)
      aiex.dma_await_task(%tc)
      aiex.dma_bd_pool_push(0, 0) bd_id %bdc : i32
      aiex.dma_bd_pool_push(0, 0) bd_id %bdb : i32
      aiex.dma_bd_pool_push(0, 0) bd_id %bda : i32
    }
  }
}
