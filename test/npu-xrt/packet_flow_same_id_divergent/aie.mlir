//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// POSITIVE test: two DIVERGENT flows sharing packet id 0 must not cross-deliver.
//
// This is the shape of the bug fixed by "[router] Fix packet flow channel
// sharing" (PR #3472): two distinct packet flows carry the same packet id but
// have DIFFERENT destinations. A packet id is the only thing a downstream
// switchbox can demultiplex on, so if the router ever lets these two share a
// link they can no longer be told apart and must not be split again.
//
// The upstream regression test for that fix, test/create-flows/
// same_id_shared_link.mlir, is FileCheck-only and targets npu2. This test runs
// the same property on real npu1 silicon.
//
// tile(0,2) -- id 0 --> shim DMA:0  (must contain ONLY sentinel 1)
// tile(1,2) -- id 0 --> shim DMA:1  (must contain ONLY sentinel 2)
//
// The router gives them disjoint paths into the shim (North:0 -> DMA:0 and
// North:2 -> DMA:1, on separate amsels), so each destination sees only its own
// source. Unlike the fan-in tests there is no merging here, so the expected
// result is fully deterministic and checked element-by-element.

module {
  aie.device(npu1) {
    %shim0 = aie.tile(0, 0)
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(1, 2)

    %buf0 = aie.buffer(%t0) {sym_name = "buf0"} : memref<256xi8>
    %prod0 = aie.lock(%t0, 0) {init = 1 : i32, sym_name = "prod0"}
    %cons0 = aie.lock(%t0, 1) {init = 0 : i32, sym_name = "cons0"}

    %buf1 = aie.buffer(%t1) {sym_name = "buf1"} : memref<256xi8>
    %prod1 = aie.lock(%t1, 0) {init = 1 : i32, sym_name = "prod1"}
    %cons1 = aie.lock(%t1, 1) {init = 0 : i32, sym_name = "cons1"}

    // Same packet id, different destinations.
    aie.packet_flow(0) {
      aie.packet_source<%t0, DMA : 0>
      aie.packet_dest<%shim0, DMA : 0>
    }
    aie.packet_flow(0) {
      aie.packet_source<%t1, DMA : 0>
      aie.packet_dest<%shim0, DMA : 1>
    }

    %core0 = aie.core(%t0) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %v = arith.constant 1 : i8
      %one = arith.constant 1 : i32
      aie.use_lock(%prod0, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c256 step %c1 {
        memref.store %v, %buf0[%i] : memref<256xi8>
      }
      aie.use_lock(%cons0, Release, %one)
      aie.end
    }
    %mem0 = aie.mem(%t0) {
      %dma = aie.dma(MM2S, 0) [{
        %one = arith.constant 1 : i32
        aie.use_lock(%cons0, AcquireGreaterEqual, %one)
        aie.dma_bd(%buf0 : memref<256xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%prod0, Release, %one)
      }]
      aie.end
    }

    %core1 = aie.core(%t1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %v = arith.constant 2 : i8
      %one = arith.constant 1 : i32
      aie.use_lock(%prod1, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c256 step %c1 {
        memref.store %v, %buf1[%i] : memref<256xi8>
      }
      aie.use_lock(%cons1, Release, %one)
      aie.end
    }
    %mem1 = aie.mem(%t1) {
      %dma = aie.dma(MM2S, 0) [{
        %one = arith.constant 1 : i32
        aie.use_lock(%cons1, AcquireGreaterEqual, %one)
        aie.dma_bd(%buf1 : memref<256xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%prod1, Release, %one)
      }]
      aie.end
    }

    aie.shim_dma_allocation @outA (%shim0, S2MM, 0)
    aie.shim_dma_allocation @outB (%shim0, S2MM, 1)

    aie.runtime_sequence(%argA: memref<256xi8>, %argB: memref<256xi8>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c256_i64 = arith.constant 256 : i64
      aiex.npu.dma_memcpy_nd (%argA[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c256_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, metadata = @outA, issue_token = true} : memref<256xi8>
      aiex.npu.dma_memcpy_nd (%argB[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c256_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, metadata = @outB, issue_token = true} : memref<256xi8>
      aiex.npu.dma_wait { symbol = @outA }
      aiex.npu.dma_wait { symbol = @outB }
    }
  }
}
