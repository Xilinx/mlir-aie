//===- arbiter_silent_stream.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// A flow whose source never sends its id takes no arbiter it could hold, and
// never waits for one.

// NOWARN-NOT: {{warning|error}}

// Flow 6 overruns memtile (0,1)'s buffer, and draining it waits on MM2S 0,
// the source of flows 4 and 5. Every BD there carries id 4, so flow 5 carries
// nothing, and it can share the one arbiter left at shim (0,0) with flow 6.

// CHECK-LABEL: aie.switchbox(%shim_noc_tile_0_0)
// CHECK-DAG:     aie.amsel<5> (0)
// CHECK-DAG:     aie.amsel<5> (1)

module {
  aie.device(npu1_1col) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %c2 = aie.tile(0, 2)
    %sb = aie.switchbox(%s0) {
      %a00 = aie.amsel<0> (0)
      %a01 = aie.amsel<0> (1)
      %a02 = aie.amsel<0> (2)
      %a03 = aie.amsel<0> (3)
      aie.masterset(North : 0, %a00, %a01, %a02, %a03)
      %a10 = aie.amsel<1> (0)
      %a11 = aie.amsel<1> (1)
      %a12 = aie.amsel<1> (2)
      %a13 = aie.amsel<1> (3)
      aie.masterset(North : 1, %a10, %a11, %a12, %a13)
      %a20 = aie.amsel<2> (0)
      %a21 = aie.amsel<2> (1)
      %a22 = aie.amsel<2> (2)
      %a23 = aie.amsel<2> (3)
      aie.masterset(North : 2, %a20, %a21, %a22, %a23)
      %a30 = aie.amsel<3> (0)
      %a31 = aie.amsel<3> (1)
      %a32 = aie.amsel<3> (2)
      %a33 = aie.amsel<3> (3)
      aie.masterset(North : 3, %a30, %a31, %a32, %a33)
      %a40 = aie.amsel<4> (0)
      %a41 = aie.amsel<4> (1)
      %a42 = aie.amsel<4> (2)
      %a43 = aie.amsel<4> (3)
      aie.masterset(North : 4, %a40, %a41, %a42, %a43)
    }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%c2, DMA : 0> }
    aie.packet_flow(5) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%s0, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
    %prod = aie.lock(%m, 0) {init = 1 : i32}
    %cons = aie.lock(%m, 1) {init = 0 : i32}
    %buf = aie.buffer(%m) : memref<16xi32>
    aie.memtile_dma(%m) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^in, ^ch1)
    ^in:
      aie.use_lock(%prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %one)
      aie.next_bd ^in
    ^ch1:
      %1 = aie.dma_start(MM2S, 0, ^out, ^end)
    ^out:
      aie.use_lock(%cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
      aie.use_lock(%prod, Release, %one)
      aie.next_bd ^out
    ^end:
      aie.end
    }
    aie.shim_dma_allocation @in6(%s0, MM2S, 0)
    aie.runtime_sequence(%a: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 32][0, 0, 0, 1]) { metadata = @in6, id = 0 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @in6}
    }
  }
}

// -----

// Flows 0 and 1 cross between cores (0,3) and (0,4), and each switchbox there
// has one arbiter left, which would let each hold the arbiter the other
// needs. Core (0,5) sends only id 2 on DMA:0, so flow 1 holds nothing.

// CHECK-LABEL: aie.switchbox(%tile_0_3)
// CHECK-DAG:     aie.amsel<0> (0)
// CHECK-DAG:     aie.amsel<0> (1)
// CHECK-LABEL: aie.switchbox(%tile_0_4)
// CHECK-DAG:     aie.amsel<0> (0)
// CHECK-DAG:     aie.amsel<0> (1)

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    %sb03 = aie.switchbox(%t03) {
      %a1_0 = aie.amsel<1> (0)  %a1_1 = aie.amsel<1> (1)  %a1_2 = aie.amsel<1> (2)  %a1_3 = aie.amsel<1> (3)
      %a2_0 = aie.amsel<2> (0)  %a2_1 = aie.amsel<2> (1)  %a2_2 = aie.amsel<2> (2)  %a2_3 = aie.amsel<2> (3)
      %a3_0 = aie.amsel<3> (0)  %a3_1 = aie.amsel<3> (1)  %a3_2 = aie.amsel<3> (2)  %a3_3 = aie.amsel<3> (3)
      %a4_0 = aie.amsel<4> (0)  %a4_1 = aie.amsel<4> (1)  %a4_2 = aie.amsel<4> (2)  %a4_3 = aie.amsel<4> (3)
      %a5_0 = aie.amsel<5> (0)  %a5_1 = aie.amsel<5> (1)  %a5_2 = aie.amsel<5> (2)  %a5_3 = aie.amsel<5> (3)
      %m1 = aie.masterset(North : 0, %a1_0, %a1_1, %a1_2, %a1_3)
      %m2 = aie.masterset(North : 1, %a2_0, %a2_1, %a2_2, %a2_3)
      %m3 = aie.masterset(North : 2, %a3_0, %a3_1, %a3_2, %a3_3)
      %m4 = aie.masterset(North : 3, %a4_0, %a4_1, %a4_2, %a4_3)
      %m5 = aie.masterset(North : 4, %a5_0, %a5_1, %a5_2, %a5_3)
    }
    %sb04 = aie.switchbox(%t04) {
      %a1_0 = aie.amsel<1> (0)  %a1_1 = aie.amsel<1> (1)  %a1_2 = aie.amsel<1> (2)  %a1_3 = aie.amsel<1> (3)
      %a2_0 = aie.amsel<2> (0)  %a2_1 = aie.amsel<2> (1)  %a2_2 = aie.amsel<2> (2)  %a2_3 = aie.amsel<2> (3)
      %a3_0 = aie.amsel<3> (0)  %a3_1 = aie.amsel<3> (1)  %a3_2 = aie.amsel<3> (2)  %a3_3 = aie.amsel<3> (3)
      %a4_0 = aie.amsel<4> (0)  %a4_1 = aie.amsel<4> (1)  %a4_2 = aie.amsel<4> (2)  %a4_3 = aie.amsel<4> (3)
      %a5_0 = aie.amsel<5> (0)  %a5_1 = aie.amsel<5> (1)  %a5_2 = aie.amsel<5> (2)  %a5_3 = aie.amsel<5> (3)
      %m1 = aie.masterset(North : 0, %a1_0, %a1_1, %a1_2, %a1_3)
      %m2 = aie.masterset(North : 1, %a2_0, %a2_1, %a2_2, %a2_3)
      %m3 = aie.masterset(North : 2, %a3_0, %a3_1, %a3_2, %a3_3)
      %m4 = aie.masterset(North : 3, %a4_0, %a4_1, %a4_2, %a4_3)
      %m5 = aie.masterset(North : 4, %a5_0, %a5_1, %a5_2, %a5_3)
    }
    aie.packet_flow(0) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t04, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%t05, DMA : 0> aie.packet_dest<%t03, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%t05, DMA : 0> aie.packet_dest<%t05, DMA : 0> }
    %buf = aie.buffer(%t05) : memref<16xi32>
    aie.mem(%t05) {
      %0 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.next_bd ^send
    ^end:
      aie.end
    }
  }
}
