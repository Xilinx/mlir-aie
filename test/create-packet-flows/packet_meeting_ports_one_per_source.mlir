//===- packet_meeting_ports_one_per_source.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOERR --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOERR --allow-empty

// NOERR-NOT: {{warning|error}}

// Cores (0, 2) and (0, 4) send id 3 and core (0, 3) id 12 into the memtile and
// core (0, 5), so the three trees meet at (0, 2), whose tree leaves by South:3
// and North:2. Both others are cheapest into (0, 2) by North:3, but there the
// second only shares the first's way in, not meeting it; it takes another
// slave port, by way of column 1.

module {
  aie.device(npu2) {
    %t0_0 = aie.tile(0, 0)
    %t0_1 = aie.tile(0, 1)
    %t0_2 = aie.tile(0, 2)
    %t0_3 = aie.tile(0, 3)
    %t0_4 = aie.tile(0, 4)
    %t0_5 = aie.tile(0, 5)
    aie.packet_flow(3) { aie.packet_source<%t0_2, DMA : 0> aie.packet_source<%t0_4, DMA : 0> aie.packet_dest<%t0_1, DMA : 0> aie.packet_dest<%t0_5, DMA : 0> }
    aie.packet_flow(12) { aie.packet_source<%t0_3, DMA : 0> aie.packet_dest<%t0_1, DMA : 0> aie.packet_dest<%t0_5, DMA : 0> }
    aie.flow(%t0_1, DMA : 0, %t0_0, DMA : 0)
    aie.flow(%t0_5, DMA : 0, %t0_0, DMA : 1)
    %buf2 = aie.buffer(%t0_2) {sym_name = "buf2"} : memref<64xi32>
    %full2 = aie.lock(%t0_2, 0) {init = 0 : i32, sym_name = "full2"}
    %sent2 = aie.lock(%t0_2, 1) {init = 0 : i32, sym_name = "sent2"}
    %core_t0_2 = aie.core(%t0_2) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 64 : index
      %base = arith.constant 1000 : i32
      %k = arith.constant 7 : i32
      scf.for %i = %c0 to %cn step %c1 {
        %i32 = arith.index_cast %i : index to i32
        %t = arith.muli %i32, %k : i32
        %v = arith.addi %base, %t : i32
        memref.store %v, %buf2[%i] : memref<64xi32>
      }
      aie.use_lock(%full2, Release, %one)
      aie.end
    }
    %dma_t0_2 = aie.mem(%t0_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^b0, ^end)
      ^b0:
        aie.use_lock(%full2, AcquireGreaterEqual, %one)
        aie.dma_bd(%buf2 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
        aie.use_lock(%sent2, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %buf3 = aie.buffer(%t0_3) {sym_name = "buf3"} : memref<64xi32>
    %full3 = aie.lock(%t0_3, 0) {init = 0 : i32, sym_name = "full3"}
    %sent3 = aie.lock(%t0_3, 1) {init = 0 : i32, sym_name = "sent3"}
    %core_t0_3 = aie.core(%t0_3) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 64 : index
      %base = arith.constant 3000 : i32
      %k = arith.constant 5 : i32
      scf.for %i = %c0 to %cn step %c1 {
        %i32 = arith.index_cast %i : index to i32
        %t = arith.muli %i32, %k : i32
        %v = arith.addi %base, %t : i32
        memref.store %v, %buf3[%i] : memref<64xi32>
      }
      aie.use_lock(%full3, Release, %one)
      aie.end
    }
    %dma_t0_3 = aie.mem(%t0_3) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^b0, ^end)
      ^b0:
        aie.use_lock(%full3, AcquireGreaterEqual, %one)
        aie.dma_bd(%buf3 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 12>}
        aie.use_lock(%sent3, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %buf4 = aie.buffer(%t0_4) {sym_name = "buf4"} : memref<64xi32>
    %full4 = aie.lock(%t0_4, 0) {init = 0 : i32, sym_name = "full4"}
    %sent4 = aie.lock(%t0_4, 1) {init = 0 : i32, sym_name = "sent4"}
    %core_t0_4 = aie.core(%t0_4) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 64 : index
      %base = arith.constant 5000 : i32
      %k = arith.constant 11 : i32
      scf.for %i = %c0 to %cn step %c1 {
        %i32 = arith.index_cast %i : index to i32
        %t = arith.muli %i32, %k : i32
        %v = arith.addi %base, %t : i32
        memref.store %v, %buf4[%i] : memref<64xi32>
      }
      aie.use_lock(%full4, Release, %one)
      aie.end
    }
    %dma_t0_4 = aie.mem(%t0_4) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^b0, ^end)
      ^b0:
        aie.use_lock(%full4, AcquireGreaterEqual, %one)
        aie.dma_bd(%buf4 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
        aie.use_lock(%sent4, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %mt = aie.buffer(%t0_1) {sym_name = "mt"} : memref<192xi32>
    %mt_free = aie.lock(%t0_1, 0) {init = 1 : i32, sym_name = "mt_free"}
    %mt_full = aie.lock(%t0_1, 1) {init = 0 : i32, sym_name = "mt_full"}
    %dma_t0_1 = aie.memtile_dma(%t0_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^r0, ^s1)
      ^r0:
        aie.use_lock(%mt_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt : memref<192xi32> offset = 0 len = 192)
        aie.use_lock(%mt_full, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^w0, ^end)
      ^w0:
        aie.use_lock(%mt_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt : memref<192xi32> offset = 0 len = 192)
        aie.use_lock(%mt_free, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %d5 = aie.buffer(%t0_5) {sym_name = "d5"} : memref<192xi32>
    %d5_free = aie.lock(%t0_5, 0) {init = 1 : i32, sym_name = "d5_free"}
    %d5_full = aie.lock(%t0_5, 1) {init = 0 : i32, sym_name = "d5_full"}
    %dma_t0_5 = aie.mem(%t0_5) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^r0, ^s1)
      ^r0:
        aie.use_lock(%d5_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%d5 : memref<192xi32> offset = 0 len = 192)
        aie.use_lock(%d5_full, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^w0, ^end)
      ^w0:
        aie.use_lock(%d5_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%d5 : memref<192xi32> offset = 0 len = 192)
        aie.use_lock(%d5_free, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @out0(%t0_0, S2MM, 0)
    aie.shim_dma_allocation @out1(%t0_0, S2MM, 1)
    aie.runtime_sequence(%o0: memref<192xi32>, %o1: memref<192xi32>) {
      aiex.npu.dma_memcpy_nd(%o0[0, 0, 0, 0][1, 1, 1, 192][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<192xi32>
      aiex.npu.dma_memcpy_nd(%o1[0, 0, 0, 0][1, 1, 1, 192][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @out1} : memref<192xi32>
      aiex.npu.dma_wait {symbol = @out0}
      aiex.npu.dma_wait {symbol = @out1}
    }
  }
}
