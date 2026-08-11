//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// POSITIVE test: 2 sources sharing packet id 0 merge into one shim S2MM channel.
//
// Two core tiles each emit one 256-byte packet with packet id 0, both destined
// for shim DMA:0. The two flows are co-terminal -- they never need to diverge --
// so sharing links is legal and the merged stream is delivered exactly once per
// source. Arbitration order between the two is NOT deterministic, so the host
// checks a histogram of sentinel values rather than positions.
//
// The router gives each id-0 stream its own path into the shim (North:0 and
// North:1), with a single amsel driving a single master port. That is what
// #3472 guarantees: same-id flows never share a link they would later have to
// be split apart on.
//
// Before #3472 the router could merge two same-id streams onto one amsel and
// then fan that amsel out to two master ports, delivering a payload twice.
// packet_flow_same_id_merge_3way scales this shape to three sources; it also
// routes to disjoint paths and is likewise expected to pass.

module {
  aie.device(npu1) {
    %shim = aie.tile(0, 0)
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(0, 3)

    %buf0 = aie.buffer(%t0) {sym_name = "buf0"} : memref<256xi8>
    %prod0 = aie.lock(%t0, 0) {init = 1 : i32, sym_name = "prod0"}
    %cons0 = aie.lock(%t0, 1) {init = 0 : i32, sym_name = "cons0"}

    %buf1 = aie.buffer(%t1) {sym_name = "buf1"} : memref<256xi8>
    %prod1 = aie.lock(%t1, 0) {init = 1 : i32, sym_name = "prod1"}
    %cons1 = aie.lock(%t1, 1) {init = 0 : i32, sym_name = "cons1"}

    // Both flows carry packet id 0 and share the destination.
    aie.packet_flow(0) {
      aie.packet_source<%t0, DMA : 0>
      aie.packet_dest<%shim, DMA : 0>
    }
    aie.packet_flow(0) {
      aie.packet_source<%t1, DMA : 0>
      aie.packet_dest<%shim, DMA : 0>
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
      %0 = aie.dma(MM2S, 0) [{
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
      %0 = aie.dma(MM2S, 0) [{
        %one = arith.constant 1 : i32
        aie.use_lock(%cons1, AcquireGreaterEqual, %one)
        aie.dma_bd(%buf1 : memref<256xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%prod1, Release, %one)
      }]
      aie.end
    }

    aie.shim_dma_allocation @out0 (%shim, S2MM, 0)

    aie.runtime_sequence(%arg0: memref<512xi8>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c512_i64 = arith.constant 512 : i64
      aiex.npu.dma_memcpy_nd (%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c512_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, metadata = @out0, issue_token = true} : memref<512xi8>
      aiex.npu.dma_wait { symbol = @out0 }
    }
  }
}
