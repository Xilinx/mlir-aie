//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// CONTROL: 3 sources with DISTINCT packet ids 0/1/2 into one shim S2MM channel.
//
// Identical geometry and data path to packet_flow_same_id_merge_3way; only the
// packet ids differ. This isolates the variable under test: if the same-id
// version were to fail while this passes, the fault is the shared id, not the
// fan-in width or the 3-tile topology.

module {
  aie.device(npu1) {
    %shim = aie.tile(0, 0)
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(0, 3)
    %t2 = aie.tile(0, 4)
    %buf0 = aie.buffer(%t0) {sym_name = "buf0"} : memref<256xi8>
    %prod0 = aie.lock(%t0, 0) {init = 1 : i32, sym_name = "prod0"}
    %cons0 = aie.lock(%t0, 1) {init = 0 : i32, sym_name = "cons0"}
    %buf1 = aie.buffer(%t1) {sym_name = "buf1"} : memref<256xi8>
    %prod1 = aie.lock(%t1, 0) {init = 1 : i32, sym_name = "prod1"}
    %cons1 = aie.lock(%t1, 1) {init = 0 : i32, sym_name = "cons1"}
    %buf2 = aie.buffer(%t2) {sym_name = "buf2"} : memref<256xi8>
    %prod2 = aie.lock(%t2, 0) {init = 1 : i32, sym_name = "prod2"}
    %cons2 = aie.lock(%t2, 1) {init = 0 : i32, sym_name = "cons2"}
    aie.packet_flow(0) {
      aie.packet_source<%t0, DMA : 0>
      aie.packet_dest<%shim, DMA : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%t1, DMA : 0>
      aie.packet_dest<%shim, DMA : 0>
    }
    aie.packet_flow(2) {
      aie.packet_source<%t2, DMA : 0>
      aie.packet_dest<%shim, DMA : 0>
    }
    %core0 = aie.core(%t0) {
      %c0_0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c256_0 = arith.constant 256 : index
      %v0 = arith.constant 1 : i8
      %one0 = arith.constant 1 : i32
      aie.use_lock(%prod0, AcquireGreaterEqual, %one0)
      scf.for %idx0 = %c0_0 to %c256_0 step %c1_0 {
        memref.store %v0, %buf0[%idx0] : memref<256xi8>
      }
      aie.use_lock(%cons0, Release, %one0)
      aie.end
    }

    %mem0 = aie.mem(%t0) {
      %dma0 = aie.dma(MM2S, 0) [{
        %onem0 = arith.constant 1 : i32
        aie.use_lock(%cons0, AcquireGreaterEqual, %onem0)
        aie.dma_bd(%buf0 : memref<256xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%prod0, Release, %onem0)
      }]
      aie.end
    }

    %core1 = aie.core(%t1) {
      %c0_1 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %c256_1 = arith.constant 256 : index
      %v1 = arith.constant 2 : i8
      %one1 = arith.constant 1 : i32
      aie.use_lock(%prod1, AcquireGreaterEqual, %one1)
      scf.for %idx1 = %c0_1 to %c256_1 step %c1_1 {
        memref.store %v1, %buf1[%idx1] : memref<256xi8>
      }
      aie.use_lock(%cons1, Release, %one1)
      aie.end
    }

    %mem1 = aie.mem(%t1) {
      %dma1 = aie.dma(MM2S, 0) [{
        %onem1 = arith.constant 1 : i32
        aie.use_lock(%cons1, AcquireGreaterEqual, %onem1)
        aie.dma_bd(%buf1 : memref<256xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%prod1, Release, %onem1)
      }]
      aie.end
    }

    %core2 = aie.core(%t2) {
      %c0_2 = arith.constant 0 : index
      %c1_2 = arith.constant 1 : index
      %c256_2 = arith.constant 256 : index
      %v2 = arith.constant 3 : i8
      %one2 = arith.constant 1 : i32
      aie.use_lock(%prod2, AcquireGreaterEqual, %one2)
      scf.for %idx2 = %c0_2 to %c256_2 step %c1_2 {
        memref.store %v2, %buf2[%idx2] : memref<256xi8>
      }
      aie.use_lock(%cons2, Release, %one2)
      aie.end
    }

    %mem2 = aie.mem(%t2) {
      %dma2 = aie.dma(MM2S, 0) [{
        %onem2 = arith.constant 1 : i32
        aie.use_lock(%cons2, AcquireGreaterEqual, %onem2)
        aie.dma_bd(%buf2 : memref<256xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(%prod2, Release, %onem2)
      }]
      aie.end
    }

    aie.shim_dma_allocation @out0 (%shim, S2MM, 0)

    aie.runtime_sequence(%arg0: memref<768xi8>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %ctot_i64 = arith.constant 768 : i64
      aiex.npu.dma_memcpy_nd (%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %ctot_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, metadata = @out0, issue_token = true} : memref<768xi8>
      aiex.npu.dma_wait { symbol = @out0 }
    }
  }
}
