//===- wait_chain_pipeline.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s

// Twenty stages, each sending to the next over two flows, so a wait chain from
// the head can take either receiver at every stage: 2^20 ways to reach the
// last core, none of which reaches (7, 2), which receives id 2. Each receiver
// has one sender, so the search need not remember which ones it took, and it
// rules the chain out in one pass over the stages.

// CHECK-NOT: {{warning|error}}
// CHECK-LABEL: aie.switchbox(%tile_7_2)
// CHECK:         aie.masterset(DMA : 0
// CHECK:         aie.rule(31, 2,

module {
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    %t15 = aie.tile(1, 5)
    %t14 = aie.tile(1, 4)
    %t13 = aie.tile(1, 3)
    %t12 = aie.tile(1, 2)
    %t22 = aie.tile(2, 2)
    %t23 = aie.tile(2, 3)
    %t24 = aie.tile(2, 4)
    %t25 = aie.tile(2, 5)
    %t35 = aie.tile(3, 5)
    %t34 = aie.tile(3, 4)
    %t33 = aie.tile(3, 3)
    %t32 = aie.tile(3, 2)
    %t42 = aie.tile(4, 2)
    %t43 = aie.tile(4, 3)
    %t44 = aie.tile(4, 4)
    %t45 = aie.tile(4, 5)
    %t55 = aie.tile(5, 5)
    %t72 = aie.tile(7, 2)
    aie.packet_flow(1) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t03, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t72, DMA : 0> }
    aie.flow(%t03, DMA : 0, %t04, DMA : 0)
    aie.flow(%t03, DMA : 1, %t04, DMA : 1)
    aie.flow(%t04, DMA : 0, %t05, DMA : 0)
    aie.flow(%t04, DMA : 1, %t05, DMA : 1)
    aie.flow(%t05, DMA : 0, %t15, DMA : 0)
    aie.flow(%t05, DMA : 1, %t15, DMA : 1)
    aie.flow(%t15, DMA : 0, %t14, DMA : 0)
    aie.flow(%t15, DMA : 1, %t14, DMA : 1)
    aie.flow(%t14, DMA : 0, %t13, DMA : 0)
    aie.flow(%t14, DMA : 1, %t13, DMA : 1)
    aie.flow(%t13, DMA : 0, %t12, DMA : 0)
    aie.flow(%t13, DMA : 1, %t12, DMA : 1)
    aie.flow(%t12, DMA : 0, %t22, DMA : 0)
    aie.flow(%t12, DMA : 1, %t22, DMA : 1)
    aie.flow(%t22, DMA : 0, %t23, DMA : 0)
    aie.flow(%t22, DMA : 1, %t23, DMA : 1)
    aie.flow(%t23, DMA : 0, %t24, DMA : 0)
    aie.flow(%t23, DMA : 1, %t24, DMA : 1)
    aie.flow(%t24, DMA : 0, %t25, DMA : 0)
    aie.flow(%t24, DMA : 1, %t25, DMA : 1)
    aie.flow(%t25, DMA : 0, %t35, DMA : 0)
    aie.flow(%t25, DMA : 1, %t35, DMA : 1)
    aie.flow(%t35, DMA : 0, %t34, DMA : 0)
    aie.flow(%t35, DMA : 1, %t34, DMA : 1)
    aie.flow(%t34, DMA : 0, %t33, DMA : 0)
    aie.flow(%t34, DMA : 1, %t33, DMA : 1)
    aie.flow(%t33, DMA : 0, %t32, DMA : 0)
    aie.flow(%t33, DMA : 1, %t32, DMA : 1)
    aie.flow(%t32, DMA : 0, %t42, DMA : 0)
    aie.flow(%t32, DMA : 1, %t42, DMA : 1)
    aie.flow(%t42, DMA : 0, %t43, DMA : 0)
    aie.flow(%t42, DMA : 1, %t43, DMA : 1)
    aie.flow(%t43, DMA : 0, %t44, DMA : 0)
    aie.flow(%t43, DMA : 1, %t44, DMA : 1)
    aie.flow(%t44, DMA : 0, %t45, DMA : 0)
    aie.flow(%t44, DMA : 1, %t45, DMA : 1)
    aie.flow(%t45, DMA : 0, %t55, DMA : 0)
    aie.flow(%t45, DMA : 1, %t55, DMA : 1)
    %src = aie.buffer(%t02) : memref<16xi32>
    aie.mem(%t02) {
      %0 = aie.dma_start(MM2S, 0, ^first, ^end)
    ^first:
      aie.dma_bd(%src : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.next_bd ^second
    ^second:
      aie.dma_bd(%src : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %s0_in0_free = aie.lock(%t03, 0) {init = 1 : i32}
    %s0_in0_full = aie.lock(%t03, 1) {init = 0 : i32}
    %s0_in0 = aie.buffer(%t03) : memref<8xi32>
    %s0_out0_free = aie.lock(%t03, 4) {init = 1 : i32}
    %s0_out0_full = aie.lock(%t03, 5) {init = 0 : i32}
    %s0_out0 = aie.buffer(%t03) : memref<16xi32>
    %s0_out1_free = aie.lock(%t03, 6) {init = 1 : i32}
    %s0_out1_full = aie.lock(%t03, 7) {init = 0 : i32}
    %s0_out1 = aie.buffer(%t03) : memref<16xi32>
    aie.core(%t03) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s0_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s0_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s0_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s0_in0_free, Release, %one)
      aie.use_lock(%s0_out0_full, Release, %one)
      aie.use_lock(%s0_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t03) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s0_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s0_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s0_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(MM2S, 0, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s0_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s0_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s0_out0_free, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 1, ^bd2, ^end)
    ^bd2:
      aie.use_lock(%s0_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s0_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s0_out1_free, Release, %one)
      aie.next_bd ^bd2
    ^end:
      aie.end
    }
    %s1_in0_free = aie.lock(%t04, 0) {init = 1 : i32}
    %s1_in0_full = aie.lock(%t04, 1) {init = 0 : i32}
    %s1_in0 = aie.buffer(%t04) : memref<8xi32>
    %s1_in1_free = aie.lock(%t04, 2) {init = 1 : i32}
    %s1_in1_full = aie.lock(%t04, 3) {init = 0 : i32}
    %s1_in1 = aie.buffer(%t04) : memref<8xi32>
    %s1_out0_free = aie.lock(%t04, 4) {init = 1 : i32}
    %s1_out0_full = aie.lock(%t04, 5) {init = 0 : i32}
    %s1_out0 = aie.buffer(%t04) : memref<16xi32>
    %s1_out1_free = aie.lock(%t04, 6) {init = 1 : i32}
    %s1_out1_full = aie.lock(%t04, 7) {init = 0 : i32}
    %s1_out1 = aie.buffer(%t04) : memref<16xi32>
    aie.core(%t04) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s1_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s1_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s1_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s1_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s1_in0_free, Release, %one)
      aie.use_lock(%s1_in1_free, Release, %one)
      aie.use_lock(%s1_out0_full, Release, %one)
      aie.use_lock(%s1_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t04) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s1_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s1_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s1_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s1_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s1_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s1_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 0, ^bd2, ^start3)
    ^bd2:
      aie.use_lock(%s1_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s1_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s1_out0_free, Release, %one)
      aie.next_bd ^bd2
    ^start3:
      %3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      aie.use_lock(%s1_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s1_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s1_out1_free, Release, %one)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
    %s2_in0_free = aie.lock(%t05, 0) {init = 1 : i32}
    %s2_in0_full = aie.lock(%t05, 1) {init = 0 : i32}
    %s2_in0 = aie.buffer(%t05) : memref<8xi32>
    %s2_in1_free = aie.lock(%t05, 2) {init = 1 : i32}
    %s2_in1_full = aie.lock(%t05, 3) {init = 0 : i32}
    %s2_in1 = aie.buffer(%t05) : memref<8xi32>
    %s2_out0_free = aie.lock(%t05, 4) {init = 1 : i32}
    %s2_out0_full = aie.lock(%t05, 5) {init = 0 : i32}
    %s2_out0 = aie.buffer(%t05) : memref<16xi32>
    %s2_out1_free = aie.lock(%t05, 6) {init = 1 : i32}
    %s2_out1_full = aie.lock(%t05, 7) {init = 0 : i32}
    %s2_out1 = aie.buffer(%t05) : memref<16xi32>
    aie.core(%t05) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s2_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s2_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s2_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s2_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s2_in0_free, Release, %one)
      aie.use_lock(%s2_in1_free, Release, %one)
      aie.use_lock(%s2_out0_full, Release, %one)
      aie.use_lock(%s2_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t05) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s2_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s2_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s2_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s2_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s2_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s2_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 0, ^bd2, ^start3)
    ^bd2:
      aie.use_lock(%s2_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s2_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s2_out0_free, Release, %one)
      aie.next_bd ^bd2
    ^start3:
      %3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      aie.use_lock(%s2_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s2_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s2_out1_free, Release, %one)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
    %s3_in0_free = aie.lock(%t15, 0) {init = 1 : i32}
    %s3_in0_full = aie.lock(%t15, 1) {init = 0 : i32}
    %s3_in0 = aie.buffer(%t15) : memref<8xi32>
    %s3_in1_free = aie.lock(%t15, 2) {init = 1 : i32}
    %s3_in1_full = aie.lock(%t15, 3) {init = 0 : i32}
    %s3_in1 = aie.buffer(%t15) : memref<8xi32>
    %s3_out0_free = aie.lock(%t15, 4) {init = 1 : i32}
    %s3_out0_full = aie.lock(%t15, 5) {init = 0 : i32}
    %s3_out0 = aie.buffer(%t15) : memref<16xi32>
    %s3_out1_free = aie.lock(%t15, 6) {init = 1 : i32}
    %s3_out1_full = aie.lock(%t15, 7) {init = 0 : i32}
    %s3_out1 = aie.buffer(%t15) : memref<16xi32>
    aie.core(%t15) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s3_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s3_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s3_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s3_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s3_in0_free, Release, %one)
      aie.use_lock(%s3_in1_free, Release, %one)
      aie.use_lock(%s3_out0_full, Release, %one)
      aie.use_lock(%s3_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t15) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s3_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s3_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s3_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s3_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s3_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s3_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 0, ^bd2, ^start3)
    ^bd2:
      aie.use_lock(%s3_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s3_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s3_out0_free, Release, %one)
      aie.next_bd ^bd2
    ^start3:
      %3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      aie.use_lock(%s3_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s3_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s3_out1_free, Release, %one)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
    %s4_in0_free = aie.lock(%t14, 0) {init = 1 : i32}
    %s4_in0_full = aie.lock(%t14, 1) {init = 0 : i32}
    %s4_in0 = aie.buffer(%t14) : memref<8xi32>
    %s4_in1_free = aie.lock(%t14, 2) {init = 1 : i32}
    %s4_in1_full = aie.lock(%t14, 3) {init = 0 : i32}
    %s4_in1 = aie.buffer(%t14) : memref<8xi32>
    %s4_out0_free = aie.lock(%t14, 4) {init = 1 : i32}
    %s4_out0_full = aie.lock(%t14, 5) {init = 0 : i32}
    %s4_out0 = aie.buffer(%t14) : memref<16xi32>
    %s4_out1_free = aie.lock(%t14, 6) {init = 1 : i32}
    %s4_out1_full = aie.lock(%t14, 7) {init = 0 : i32}
    %s4_out1 = aie.buffer(%t14) : memref<16xi32>
    aie.core(%t14) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s4_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s4_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s4_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s4_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s4_in0_free, Release, %one)
      aie.use_lock(%s4_in1_free, Release, %one)
      aie.use_lock(%s4_out0_full, Release, %one)
      aie.use_lock(%s4_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t14) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s4_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s4_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s4_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s4_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s4_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s4_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 0, ^bd2, ^start3)
    ^bd2:
      aie.use_lock(%s4_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s4_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s4_out0_free, Release, %one)
      aie.next_bd ^bd2
    ^start3:
      %3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      aie.use_lock(%s4_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s4_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s4_out1_free, Release, %one)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
    %s5_in0_free = aie.lock(%t13, 0) {init = 1 : i32}
    %s5_in0_full = aie.lock(%t13, 1) {init = 0 : i32}
    %s5_in0 = aie.buffer(%t13) : memref<8xi32>
    %s5_in1_free = aie.lock(%t13, 2) {init = 1 : i32}
    %s5_in1_full = aie.lock(%t13, 3) {init = 0 : i32}
    %s5_in1 = aie.buffer(%t13) : memref<8xi32>
    %s5_out0_free = aie.lock(%t13, 4) {init = 1 : i32}
    %s5_out0_full = aie.lock(%t13, 5) {init = 0 : i32}
    %s5_out0 = aie.buffer(%t13) : memref<16xi32>
    %s5_out1_free = aie.lock(%t13, 6) {init = 1 : i32}
    %s5_out1_full = aie.lock(%t13, 7) {init = 0 : i32}
    %s5_out1 = aie.buffer(%t13) : memref<16xi32>
    aie.core(%t13) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s5_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s5_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s5_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s5_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s5_in0_free, Release, %one)
      aie.use_lock(%s5_in1_free, Release, %one)
      aie.use_lock(%s5_out0_full, Release, %one)
      aie.use_lock(%s5_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t13) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s5_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s5_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s5_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s5_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s5_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s5_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 0, ^bd2, ^start3)
    ^bd2:
      aie.use_lock(%s5_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s5_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s5_out0_free, Release, %one)
      aie.next_bd ^bd2
    ^start3:
      %3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      aie.use_lock(%s5_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s5_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s5_out1_free, Release, %one)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
    %s6_in0_free = aie.lock(%t12, 0) {init = 1 : i32}
    %s6_in0_full = aie.lock(%t12, 1) {init = 0 : i32}
    %s6_in0 = aie.buffer(%t12) : memref<8xi32>
    %s6_in1_free = aie.lock(%t12, 2) {init = 1 : i32}
    %s6_in1_full = aie.lock(%t12, 3) {init = 0 : i32}
    %s6_in1 = aie.buffer(%t12) : memref<8xi32>
    %s6_out0_free = aie.lock(%t12, 4) {init = 1 : i32}
    %s6_out0_full = aie.lock(%t12, 5) {init = 0 : i32}
    %s6_out0 = aie.buffer(%t12) : memref<16xi32>
    %s6_out1_free = aie.lock(%t12, 6) {init = 1 : i32}
    %s6_out1_full = aie.lock(%t12, 7) {init = 0 : i32}
    %s6_out1 = aie.buffer(%t12) : memref<16xi32>
    aie.core(%t12) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s6_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s6_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s6_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s6_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s6_in0_free, Release, %one)
      aie.use_lock(%s6_in1_free, Release, %one)
      aie.use_lock(%s6_out0_full, Release, %one)
      aie.use_lock(%s6_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t12) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s6_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s6_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s6_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s6_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s6_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s6_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 0, ^bd2, ^start3)
    ^bd2:
      aie.use_lock(%s6_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s6_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s6_out0_free, Release, %one)
      aie.next_bd ^bd2
    ^start3:
      %3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      aie.use_lock(%s6_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s6_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s6_out1_free, Release, %one)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
    %s7_in0_free = aie.lock(%t22, 0) {init = 1 : i32}
    %s7_in0_full = aie.lock(%t22, 1) {init = 0 : i32}
    %s7_in0 = aie.buffer(%t22) : memref<8xi32>
    %s7_in1_free = aie.lock(%t22, 2) {init = 1 : i32}
    %s7_in1_full = aie.lock(%t22, 3) {init = 0 : i32}
    %s7_in1 = aie.buffer(%t22) : memref<8xi32>
    %s7_out0_free = aie.lock(%t22, 4) {init = 1 : i32}
    %s7_out0_full = aie.lock(%t22, 5) {init = 0 : i32}
    %s7_out0 = aie.buffer(%t22) : memref<16xi32>
    %s7_out1_free = aie.lock(%t22, 6) {init = 1 : i32}
    %s7_out1_full = aie.lock(%t22, 7) {init = 0 : i32}
    %s7_out1 = aie.buffer(%t22) : memref<16xi32>
    aie.core(%t22) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s7_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s7_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s7_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s7_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s7_in0_free, Release, %one)
      aie.use_lock(%s7_in1_free, Release, %one)
      aie.use_lock(%s7_out0_full, Release, %one)
      aie.use_lock(%s7_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t22) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s7_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s7_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s7_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s7_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s7_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s7_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 0, ^bd2, ^start3)
    ^bd2:
      aie.use_lock(%s7_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s7_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s7_out0_free, Release, %one)
      aie.next_bd ^bd2
    ^start3:
      %3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      aie.use_lock(%s7_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s7_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s7_out1_free, Release, %one)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
    %s8_in0_free = aie.lock(%t23, 0) {init = 1 : i32}
    %s8_in0_full = aie.lock(%t23, 1) {init = 0 : i32}
    %s8_in0 = aie.buffer(%t23) : memref<8xi32>
    %s8_in1_free = aie.lock(%t23, 2) {init = 1 : i32}
    %s8_in1_full = aie.lock(%t23, 3) {init = 0 : i32}
    %s8_in1 = aie.buffer(%t23) : memref<8xi32>
    %s8_out0_free = aie.lock(%t23, 4) {init = 1 : i32}
    %s8_out0_full = aie.lock(%t23, 5) {init = 0 : i32}
    %s8_out0 = aie.buffer(%t23) : memref<16xi32>
    %s8_out1_free = aie.lock(%t23, 6) {init = 1 : i32}
    %s8_out1_full = aie.lock(%t23, 7) {init = 0 : i32}
    %s8_out1 = aie.buffer(%t23) : memref<16xi32>
    aie.core(%t23) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s8_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s8_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s8_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s8_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s8_in0_free, Release, %one)
      aie.use_lock(%s8_in1_free, Release, %one)
      aie.use_lock(%s8_out0_full, Release, %one)
      aie.use_lock(%s8_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t23) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s8_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s8_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s8_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s8_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s8_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s8_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 0, ^bd2, ^start3)
    ^bd2:
      aie.use_lock(%s8_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s8_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s8_out0_free, Release, %one)
      aie.next_bd ^bd2
    ^start3:
      %3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      aie.use_lock(%s8_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s8_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s8_out1_free, Release, %one)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
    %s9_in0_free = aie.lock(%t24, 0) {init = 1 : i32}
    %s9_in0_full = aie.lock(%t24, 1) {init = 0 : i32}
    %s9_in0 = aie.buffer(%t24) : memref<8xi32>
    %s9_in1_free = aie.lock(%t24, 2) {init = 1 : i32}
    %s9_in1_full = aie.lock(%t24, 3) {init = 0 : i32}
    %s9_in1 = aie.buffer(%t24) : memref<8xi32>
    %s9_out0_free = aie.lock(%t24, 4) {init = 1 : i32}
    %s9_out0_full = aie.lock(%t24, 5) {init = 0 : i32}
    %s9_out0 = aie.buffer(%t24) : memref<16xi32>
    %s9_out1_free = aie.lock(%t24, 6) {init = 1 : i32}
    %s9_out1_full = aie.lock(%t24, 7) {init = 0 : i32}
    %s9_out1 = aie.buffer(%t24) : memref<16xi32>
    aie.core(%t24) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s9_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s9_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s9_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s9_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s9_in0_free, Release, %one)
      aie.use_lock(%s9_in1_free, Release, %one)
      aie.use_lock(%s9_out0_full, Release, %one)
      aie.use_lock(%s9_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t24) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s9_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s9_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s9_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s9_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s9_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s9_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 0, ^bd2, ^start3)
    ^bd2:
      aie.use_lock(%s9_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s9_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s9_out0_free, Release, %one)
      aie.next_bd ^bd2
    ^start3:
      %3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      aie.use_lock(%s9_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s9_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s9_out1_free, Release, %one)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
    %s10_in0_free = aie.lock(%t25, 0) {init = 1 : i32}
    %s10_in0_full = aie.lock(%t25, 1) {init = 0 : i32}
    %s10_in0 = aie.buffer(%t25) : memref<8xi32>
    %s10_in1_free = aie.lock(%t25, 2) {init = 1 : i32}
    %s10_in1_full = aie.lock(%t25, 3) {init = 0 : i32}
    %s10_in1 = aie.buffer(%t25) : memref<8xi32>
    %s10_out0_free = aie.lock(%t25, 4) {init = 1 : i32}
    %s10_out0_full = aie.lock(%t25, 5) {init = 0 : i32}
    %s10_out0 = aie.buffer(%t25) : memref<16xi32>
    %s10_out1_free = aie.lock(%t25, 6) {init = 1 : i32}
    %s10_out1_full = aie.lock(%t25, 7) {init = 0 : i32}
    %s10_out1 = aie.buffer(%t25) : memref<16xi32>
    aie.core(%t25) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s10_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s10_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s10_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s10_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s10_in0_free, Release, %one)
      aie.use_lock(%s10_in1_free, Release, %one)
      aie.use_lock(%s10_out0_full, Release, %one)
      aie.use_lock(%s10_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t25) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s10_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s10_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s10_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s10_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s10_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s10_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 0, ^bd2, ^start3)
    ^bd2:
      aie.use_lock(%s10_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s10_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s10_out0_free, Release, %one)
      aie.next_bd ^bd2
    ^start3:
      %3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      aie.use_lock(%s10_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s10_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s10_out1_free, Release, %one)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
    %s11_in0_free = aie.lock(%t35, 0) {init = 1 : i32}
    %s11_in0_full = aie.lock(%t35, 1) {init = 0 : i32}
    %s11_in0 = aie.buffer(%t35) : memref<8xi32>
    %s11_in1_free = aie.lock(%t35, 2) {init = 1 : i32}
    %s11_in1_full = aie.lock(%t35, 3) {init = 0 : i32}
    %s11_in1 = aie.buffer(%t35) : memref<8xi32>
    %s11_out0_free = aie.lock(%t35, 4) {init = 1 : i32}
    %s11_out0_full = aie.lock(%t35, 5) {init = 0 : i32}
    %s11_out0 = aie.buffer(%t35) : memref<16xi32>
    %s11_out1_free = aie.lock(%t35, 6) {init = 1 : i32}
    %s11_out1_full = aie.lock(%t35, 7) {init = 0 : i32}
    %s11_out1 = aie.buffer(%t35) : memref<16xi32>
    aie.core(%t35) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s11_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s11_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s11_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s11_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s11_in0_free, Release, %one)
      aie.use_lock(%s11_in1_free, Release, %one)
      aie.use_lock(%s11_out0_full, Release, %one)
      aie.use_lock(%s11_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t35) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s11_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s11_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s11_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s11_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s11_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s11_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 0, ^bd2, ^start3)
    ^bd2:
      aie.use_lock(%s11_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s11_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s11_out0_free, Release, %one)
      aie.next_bd ^bd2
    ^start3:
      %3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      aie.use_lock(%s11_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s11_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s11_out1_free, Release, %one)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
    %s12_in0_free = aie.lock(%t34, 0) {init = 1 : i32}
    %s12_in0_full = aie.lock(%t34, 1) {init = 0 : i32}
    %s12_in0 = aie.buffer(%t34) : memref<8xi32>
    %s12_in1_free = aie.lock(%t34, 2) {init = 1 : i32}
    %s12_in1_full = aie.lock(%t34, 3) {init = 0 : i32}
    %s12_in1 = aie.buffer(%t34) : memref<8xi32>
    %s12_out0_free = aie.lock(%t34, 4) {init = 1 : i32}
    %s12_out0_full = aie.lock(%t34, 5) {init = 0 : i32}
    %s12_out0 = aie.buffer(%t34) : memref<16xi32>
    %s12_out1_free = aie.lock(%t34, 6) {init = 1 : i32}
    %s12_out1_full = aie.lock(%t34, 7) {init = 0 : i32}
    %s12_out1 = aie.buffer(%t34) : memref<16xi32>
    aie.core(%t34) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s12_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s12_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s12_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s12_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s12_in0_free, Release, %one)
      aie.use_lock(%s12_in1_free, Release, %one)
      aie.use_lock(%s12_out0_full, Release, %one)
      aie.use_lock(%s12_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t34) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s12_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s12_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s12_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s12_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s12_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s12_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 0, ^bd2, ^start3)
    ^bd2:
      aie.use_lock(%s12_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s12_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s12_out0_free, Release, %one)
      aie.next_bd ^bd2
    ^start3:
      %3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      aie.use_lock(%s12_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s12_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s12_out1_free, Release, %one)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
    %s13_in0_free = aie.lock(%t33, 0) {init = 1 : i32}
    %s13_in0_full = aie.lock(%t33, 1) {init = 0 : i32}
    %s13_in0 = aie.buffer(%t33) : memref<8xi32>
    %s13_in1_free = aie.lock(%t33, 2) {init = 1 : i32}
    %s13_in1_full = aie.lock(%t33, 3) {init = 0 : i32}
    %s13_in1 = aie.buffer(%t33) : memref<8xi32>
    %s13_out0_free = aie.lock(%t33, 4) {init = 1 : i32}
    %s13_out0_full = aie.lock(%t33, 5) {init = 0 : i32}
    %s13_out0 = aie.buffer(%t33) : memref<16xi32>
    %s13_out1_free = aie.lock(%t33, 6) {init = 1 : i32}
    %s13_out1_full = aie.lock(%t33, 7) {init = 0 : i32}
    %s13_out1 = aie.buffer(%t33) : memref<16xi32>
    aie.core(%t33) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s13_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s13_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s13_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s13_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s13_in0_free, Release, %one)
      aie.use_lock(%s13_in1_free, Release, %one)
      aie.use_lock(%s13_out0_full, Release, %one)
      aie.use_lock(%s13_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t33) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s13_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s13_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s13_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s13_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s13_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s13_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 0, ^bd2, ^start3)
    ^bd2:
      aie.use_lock(%s13_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s13_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s13_out0_free, Release, %one)
      aie.next_bd ^bd2
    ^start3:
      %3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      aie.use_lock(%s13_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s13_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s13_out1_free, Release, %one)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
    %s14_in0_free = aie.lock(%t32, 0) {init = 1 : i32}
    %s14_in0_full = aie.lock(%t32, 1) {init = 0 : i32}
    %s14_in0 = aie.buffer(%t32) : memref<8xi32>
    %s14_in1_free = aie.lock(%t32, 2) {init = 1 : i32}
    %s14_in1_full = aie.lock(%t32, 3) {init = 0 : i32}
    %s14_in1 = aie.buffer(%t32) : memref<8xi32>
    %s14_out0_free = aie.lock(%t32, 4) {init = 1 : i32}
    %s14_out0_full = aie.lock(%t32, 5) {init = 0 : i32}
    %s14_out0 = aie.buffer(%t32) : memref<16xi32>
    %s14_out1_free = aie.lock(%t32, 6) {init = 1 : i32}
    %s14_out1_full = aie.lock(%t32, 7) {init = 0 : i32}
    %s14_out1 = aie.buffer(%t32) : memref<16xi32>
    aie.core(%t32) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s14_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s14_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s14_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s14_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s14_in0_free, Release, %one)
      aie.use_lock(%s14_in1_free, Release, %one)
      aie.use_lock(%s14_out0_full, Release, %one)
      aie.use_lock(%s14_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t32) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s14_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s14_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s14_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s14_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s14_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s14_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 0, ^bd2, ^start3)
    ^bd2:
      aie.use_lock(%s14_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s14_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s14_out0_free, Release, %one)
      aie.next_bd ^bd2
    ^start3:
      %3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      aie.use_lock(%s14_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s14_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s14_out1_free, Release, %one)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
    %s15_in0_free = aie.lock(%t42, 0) {init = 1 : i32}
    %s15_in0_full = aie.lock(%t42, 1) {init = 0 : i32}
    %s15_in0 = aie.buffer(%t42) : memref<8xi32>
    %s15_in1_free = aie.lock(%t42, 2) {init = 1 : i32}
    %s15_in1_full = aie.lock(%t42, 3) {init = 0 : i32}
    %s15_in1 = aie.buffer(%t42) : memref<8xi32>
    %s15_out0_free = aie.lock(%t42, 4) {init = 1 : i32}
    %s15_out0_full = aie.lock(%t42, 5) {init = 0 : i32}
    %s15_out0 = aie.buffer(%t42) : memref<16xi32>
    %s15_out1_free = aie.lock(%t42, 6) {init = 1 : i32}
    %s15_out1_full = aie.lock(%t42, 7) {init = 0 : i32}
    %s15_out1 = aie.buffer(%t42) : memref<16xi32>
    aie.core(%t42) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s15_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s15_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s15_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s15_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s15_in0_free, Release, %one)
      aie.use_lock(%s15_in1_free, Release, %one)
      aie.use_lock(%s15_out0_full, Release, %one)
      aie.use_lock(%s15_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t42) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s15_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s15_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s15_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s15_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s15_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s15_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 0, ^bd2, ^start3)
    ^bd2:
      aie.use_lock(%s15_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s15_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s15_out0_free, Release, %one)
      aie.next_bd ^bd2
    ^start3:
      %3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      aie.use_lock(%s15_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s15_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s15_out1_free, Release, %one)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
    %s16_in0_free = aie.lock(%t43, 0) {init = 1 : i32}
    %s16_in0_full = aie.lock(%t43, 1) {init = 0 : i32}
    %s16_in0 = aie.buffer(%t43) : memref<8xi32>
    %s16_in1_free = aie.lock(%t43, 2) {init = 1 : i32}
    %s16_in1_full = aie.lock(%t43, 3) {init = 0 : i32}
    %s16_in1 = aie.buffer(%t43) : memref<8xi32>
    %s16_out0_free = aie.lock(%t43, 4) {init = 1 : i32}
    %s16_out0_full = aie.lock(%t43, 5) {init = 0 : i32}
    %s16_out0 = aie.buffer(%t43) : memref<16xi32>
    %s16_out1_free = aie.lock(%t43, 6) {init = 1 : i32}
    %s16_out1_full = aie.lock(%t43, 7) {init = 0 : i32}
    %s16_out1 = aie.buffer(%t43) : memref<16xi32>
    aie.core(%t43) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s16_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s16_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s16_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s16_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s16_in0_free, Release, %one)
      aie.use_lock(%s16_in1_free, Release, %one)
      aie.use_lock(%s16_out0_full, Release, %one)
      aie.use_lock(%s16_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t43) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s16_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s16_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s16_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s16_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s16_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s16_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 0, ^bd2, ^start3)
    ^bd2:
      aie.use_lock(%s16_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s16_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s16_out0_free, Release, %one)
      aie.next_bd ^bd2
    ^start3:
      %3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      aie.use_lock(%s16_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s16_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s16_out1_free, Release, %one)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
    %s17_in0_free = aie.lock(%t44, 0) {init = 1 : i32}
    %s17_in0_full = aie.lock(%t44, 1) {init = 0 : i32}
    %s17_in0 = aie.buffer(%t44) : memref<8xi32>
    %s17_in1_free = aie.lock(%t44, 2) {init = 1 : i32}
    %s17_in1_full = aie.lock(%t44, 3) {init = 0 : i32}
    %s17_in1 = aie.buffer(%t44) : memref<8xi32>
    %s17_out0_free = aie.lock(%t44, 4) {init = 1 : i32}
    %s17_out0_full = aie.lock(%t44, 5) {init = 0 : i32}
    %s17_out0 = aie.buffer(%t44) : memref<16xi32>
    %s17_out1_free = aie.lock(%t44, 6) {init = 1 : i32}
    %s17_out1_full = aie.lock(%t44, 7) {init = 0 : i32}
    %s17_out1 = aie.buffer(%t44) : memref<16xi32>
    aie.core(%t44) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s17_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s17_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s17_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s17_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s17_in0_free, Release, %one)
      aie.use_lock(%s17_in1_free, Release, %one)
      aie.use_lock(%s17_out0_full, Release, %one)
      aie.use_lock(%s17_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t44) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s17_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s17_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s17_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s17_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s17_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s17_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 0, ^bd2, ^start3)
    ^bd2:
      aie.use_lock(%s17_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s17_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s17_out0_free, Release, %one)
      aie.next_bd ^bd2
    ^start3:
      %3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      aie.use_lock(%s17_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s17_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s17_out1_free, Release, %one)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
    %s18_in0_free = aie.lock(%t45, 0) {init = 1 : i32}
    %s18_in0_full = aie.lock(%t45, 1) {init = 0 : i32}
    %s18_in0 = aie.buffer(%t45) : memref<8xi32>
    %s18_in1_free = aie.lock(%t45, 2) {init = 1 : i32}
    %s18_in1_full = aie.lock(%t45, 3) {init = 0 : i32}
    %s18_in1 = aie.buffer(%t45) : memref<8xi32>
    %s18_out0_free = aie.lock(%t45, 4) {init = 1 : i32}
    %s18_out0_full = aie.lock(%t45, 5) {init = 0 : i32}
    %s18_out0 = aie.buffer(%t45) : memref<16xi32>
    %s18_out1_free = aie.lock(%t45, 6) {init = 1 : i32}
    %s18_out1_full = aie.lock(%t45, 7) {init = 0 : i32}
    %s18_out1 = aie.buffer(%t45) : memref<16xi32>
    aie.core(%t45) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s18_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s18_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s18_out0_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s18_out1_free, AcquireGreaterEqual, %one)
      aie.use_lock(%s18_in0_free, Release, %one)
      aie.use_lock(%s18_in1_free, Release, %one)
      aie.use_lock(%s18_out0_full, Release, %one)
      aie.use_lock(%s18_out1_full, Release, %one)
      aie.end
    }
    aie.mem(%t45) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s18_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s18_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s18_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^start2)
    ^bd1:
      aie.use_lock(%s18_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s18_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s18_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^start2:
      %2 = aie.dma_start(MM2S, 0, ^bd2, ^start3)
    ^bd2:
      aie.use_lock(%s18_out0_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s18_out0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s18_out0_free, Release, %one)
      aie.next_bd ^bd2
    ^start3:
      %3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      aie.use_lock(%s18_out1_full, AcquireGreaterEqual, %one)
      aie.dma_bd(%s18_out1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%s18_out1_free, Release, %one)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
    %s19_in0_free = aie.lock(%t55, 0) {init = 1 : i32}
    %s19_in0_full = aie.lock(%t55, 1) {init = 0 : i32}
    %s19_in0 = aie.buffer(%t55) : memref<8xi32>
    %s19_in1_free = aie.lock(%t55, 2) {init = 1 : i32}
    %s19_in1_full = aie.lock(%t55, 3) {init = 0 : i32}
    %s19_in1 = aie.buffer(%t55) : memref<8xi32>
    aie.core(%t55) {
      %one = arith.constant 1 : i32
      aie.use_lock(%s19_in0_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s19_in1_full, AcquireGreaterEqual, %one)
      aie.use_lock(%s19_in0_free, Release, %one)
      aie.use_lock(%s19_in1_free, Release, %one)
      aie.end
    }
    aie.mem(%t55) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^start1)
    ^bd0:
      aie.use_lock(%s19_in0_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s19_in0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s19_in0_full, Release, %one)
      aie.next_bd ^bd0
    ^start1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^end)
    ^bd1:
      aie.use_lock(%s19_in1_free, AcquireGreaterEqual, %one)
      aie.dma_bd(%s19_in1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%s19_in1_full, Release, %one)
      aie.next_bd ^bd1
    ^end:
      aie.end
    }
  }
}
