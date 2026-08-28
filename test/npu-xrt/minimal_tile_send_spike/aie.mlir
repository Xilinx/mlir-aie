//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Smallest possible repro: one shim, one compute tile with no aie.core doing
// real work, one packet-switched send of a two-word constant marker into a
// shim S2MM channel, one host read. Nothing else in the design at all.
//
// This exists to answer, with zero other variables in play: does a freshly
// declared shim S2MM channel, fed only by a compute tile's one-shot/looping
// packet-switched DMA (never touched by any circuit-switched aie.flow or
// IRON-generated machinery), ever signal completion back to the host at all?
module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    %payload = aie.buffer(%tile_0_2) {sym_name = "payload"} : memref<2xi32>
    // Real producer/consumer locks, same idiom as every working example
    // (packet_flow_fanin, add_one_ctrl_packet_4_cores): the core writes and
    // releases; the DMA acquires and sends. No self-looping lock hack.
    %prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "prod_lock"}
    %cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "cons_lock"}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cdead = arith.constant 0xDEADBEEF : i32
      %ccafe = arith.constant 0xCAFEF00D : i32
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%prod_lock, AcquireGreaterEqual, %c1_i32)
      memref.store %cdead, %payload[%c0] : memref<2xi32>
      memref.store %ccafe, %payload[%c1] : memref<2xi32>
      aie.use_lock(%cons_lock, Release, %c1_i32)
      aie.end
    }

    aie.packet_flow(0x1) {
      aie.packet_source<%tile_0_2, DMA : 0>
      aie.packet_dest<%tile_0_0, DMA : 0>
    }

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%cons_lock, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%payload : memref<2xi32> offset = 0 len = 2) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.use_lock(%prod_lock, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    aie.shim_dma_allocation @result (%tile_0_0, S2MM, 0)

    aie.runtime_sequence @seq(%arg0: memref<2xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @result} : memref<2xi32>
      aiex.npu.dma_wait {symbol = @result}
    }
  }
}
