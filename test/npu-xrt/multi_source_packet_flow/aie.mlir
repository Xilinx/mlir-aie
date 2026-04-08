//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Regression test for the multi-source packet flow pathfinder bug:
// When a packet_flow has multiple aie.packet_source ops, the pathfinder must
// route ALL sources to the destination. Before the fix, only the last source
// encountered in block order was routed ("last source wins"), silently dropping
// all other sources.
//
// Topology:
//   tile(0,2) core writes [1..8]     to buf_2, then DMA MM2S sends via pkt flow
//   tile(0,3) core writes [101..108] to buf_3, then DMA MM2S sends via pkt flow
//   Both DMAs share a single aie.packet_flow(0x0) with two aie.packet_source ops
//   Shim S2MM:0 receives 16 elements total
//
// Expected output (sorted): [1, 2, 3, 4, 5, 6, 7, 8, 101, 102, ..., 108]
// With bug: tile(0,2) has no route, its DMA stalls, test times out.
//
// Lock protocol per tile (standard producer-consumer, one-shot):
//   prod_lock (init=1): core acquires to confirm buffer is writable before writing
//   cons_lock (init=0): core releases after writing; DMA acquires before sending
//   After the DMA sends, it releases prod_lock. Since the core calls aie.end
//   and never re-acquires prod_lock, the DMA loops back but blocks — one-shot.

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    %buf_2 = aie.buffer(%tile_0_2) {sym_name = "buf_2"} : memref<8xi32>
    %buf_3 = aie.buffer(%tile_0_3) {sym_name = "buf_3"} : memref<8xi32>

    // Two locks per tile implement a one-shot producer (core) / consumer (DMA):
    //   prod_lock: init=1 means buffer is writable; DMA acquires, buffer becomes
    //              busy while DMA reads, DMA releases after send.
    //   cons_lock: init=0 means buffer is not yet full; core releases once
    //              data is written; DMA acquires before reading.
    %prod_lock_2 = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "prod_lock_2"}
    %cons_lock_2 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "cons_lock_2"}
    %prod_lock_3 = aie.lock(%tile_0_3, 0) {init = 1 : i32, sym_name = "prod_lock_3"}
    %cons_lock_3 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "cons_lock_3"}

    // Multi-source packet flow: both tile DMAs fan-in to the shim.
    // This is the feature under test. Both tiles must be routed by the
    // pathfinder, not just the last one.
    aie.packet_flow(0x0) {
      aie.packet_source<%tile_0_2, DMA : 0>
      aie.packet_source<%tile_0_3, DMA : 0>
      aie.packet_dest<%tile_0_0, DMA : 0>
    }

    // Core for tile(0,2): acquires prod_lock (confirms buffer writable),
    // writes [1..8] to buf_2, releases cons_lock (signals DMA).
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%prod_lock_2, AcquireGreaterEqual, 1)
      scf.for %i = %c0 to %c8 step %c1 {
        %i_i32 = arith.index_cast %i : index to i32
        %val = arith.addi %i_i32, %c1_i32 : i32
        memref.store %val, %buf_2[%i] : memref<8xi32>
      }
      aie.use_lock(%cons_lock_2, Release, 1)
      aie.end
    }

    // Core for tile(0,3): same pattern, writes [101..108] to buf_3.
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c101_i32 = arith.constant 101 : i32
      aie.use_lock(%prod_lock_3, AcquireGreaterEqual, 1)
      scf.for %i = %c0 to %c8 step %c1 {
        %i_i32 = arith.index_cast %i : index to i32
        %val = arith.addi %i_i32, %c101_i32 : i32
        memref.store %val, %buf_3[%i] : memref<8xi32>
      }
      aie.use_lock(%cons_lock_3, Release, 1)
      aie.end
    }

    // DMA for tile(0,2): acquires cons_lock (waits for core), sends buf_2 via
    // packet flow, releases prod_lock. Loops back but prod_lock stays 0 after
    // one transfer (core never re-acquires), so the DMA is effectively one-shot.
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_2 : memref<8xi32>, 0, 8) {packet = #aie.packet_info<pkt_id = 0, pkt_type = 0>}
      aie.use_lock(%prod_lock_2, Release, 1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    // DMA for tile(0,3): same pattern for buf_3.
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_3 : memref<8xi32>, 0, 8) {packet = #aie.packet_info<pkt_id = 0, pkt_type = 0>}
      aie.use_lock(%prod_lock_3, Release, 1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    aie.shim_dma_allocation @out (%tile_0_0, S2MM, 0)

    // Sequence: start receiving 16 elements (8 from each tile), then wait.
    // The cores run automatically on device startup; no host input needed.
    aie.runtime_sequence @seq(%dummy_a : memref<8xi32>, %dummy_b : memref<8xi32>,
                              %out : memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd (%out[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c16]
                              [%c0, %c0, %c0, %c1])
          {metadata = @out, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}
