//===- arbiter_hold_cycle_one_holder.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

//   flow 0: memtile (0,1) --> core (0,2) S2MM 0
//   flow 1: memtile (0,1) --> core (0,2) S2MM 1
//   flow 2: memtile (0,1) --> core (0,3) S2MM 0
//   flow 3: memtile (0,1) --> core (0,3) S2MM 1

// Each 64-word packet fills its 16-word receiver, and each core drains only
// once both of its receivers fill, so two flows into one core cannot share an
// arbiter. With arbiters 3-5 taken at (0,1), four flows share three arbiters,
// so one flow into (0,2) shares with one into (0,3). A hold cycle through that
// arbiter would need both flows to hold it at once, but a packet holds its
// arbiter until tlast, so only one can.

// NOWARN-NOT: {{warning|error}}
// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK-DAG:   %[[SHARED:.*]] = aie.amsel<{{[0-2]}}> (1)
// CHECK-DAG:   aie.rule(31, 0, %{{.*}})
// CHECK-DAG:   aie.rule(31, 1, %{{.*}})
// CHECK-DAG:   aie.rule(31, 2, %{{.*}})
// CHECK-DAG:   aie.rule(31, 3, %{{.*}})

module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.packet_flow(0) { aie.packet_source<%t01, DMA : 0> aie.packet_dest<%t02, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%t01, DMA : 1> aie.packet_dest<%t02, DMA : 1> }
    aie.packet_flow(2) { aie.packet_source<%t01, DMA : 2> aie.packet_dest<%t03, DMA : 0> }
    aie.packet_flow(3) { aie.packet_source<%t01, DMA : 3> aie.packet_dest<%t03, DMA : 1> }
    // Arbiters 3-5 taken at (0,1).
    %sb01 = aie.switchbox(%t01) {
      %a3_0 = aie.amsel<3> (0)  %a3_1 = aie.amsel<3> (1)  %a3_2 = aie.amsel<3> (2)  %a3_3 = aie.amsel<3> (3)
      %a4_0 = aie.amsel<4> (0)  %a4_1 = aie.amsel<4> (1)  %a4_2 = aie.amsel<4> (2)  %a4_3 = aie.amsel<4> (3)
      %a5_0 = aie.amsel<5> (0)  %a5_1 = aie.amsel<5> (1)  %a5_2 = aie.amsel<5> (2)  %a5_3 = aie.amsel<5> (3)
      aie.masterset(South : 1, %a3_0, %a3_1, %a3_2, %a3_3)
      aie.masterset(South : 2, %a4_0, %a4_1, %a4_2, %a4_3)
      aie.masterset(South : 3, %a5_0, %a5_1, %a5_2, %a5_3)
    }

    %m0 = aie.buffer(%t01) : memref<64xi32>
    %m1 = aie.buffer(%t01) : memref<64xi32>
    %m2 = aie.buffer(%t01) : memref<64xi32>
    %m3 = aie.buffer(%t01) : memref<64xi32>
    aie.memtile_dma(%t01) {
      %0 = aie.dma_start(MM2S, 0, ^s0, ^c1)
    ^s0:
      aie.dma_bd(%m0 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.next_bd ^end
    ^c1:
      %1 = aie.dma_start(MM2S, 1, ^s1, ^c2)
    ^s1:
      aie.dma_bd(%m1 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.next_bd ^end
    ^c2:
      %2 = aie.dma_start(MM2S, 2, ^s2, ^c3)
    ^s2:
      aie.dma_bd(%m2 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.next_bd ^end
    ^c3:
      %3 = aie.dma_start(MM2S, 3, ^s3, ^end)
    ^s3:
      aie.dma_bd(%m3 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
      aie.next_bd ^end
    ^end:
      aie.end
    }

    %a02 = aie.buffer(%t02) : memref<16xi32>
    %b02 = aie.buffer(%t02) : memref<16xi32>
    %pa02 = aie.lock(%t02, 0) {init = 1 : i32}
    %ca02 = aie.lock(%t02, 1) {init = 0 : i32}
    %pb02 = aie.lock(%t02, 2) {init = 1 : i32}
    %cb02 = aie.lock(%t02, 3) {init = 0 : i32}
    aie.core(%t02) {
      %one = arith.constant 1 : i32
      aie.use_lock(%ca02, AcquireGreaterEqual, %one)
      aie.use_lock(%cb02, AcquireGreaterEqual, %one)
      aie.use_lock(%pa02, Release, %one)
      aie.use_lock(%pb02, Release, %one)
      aie.end
    }
    aie.mem(%t02) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^a, ^c1)
    ^a:
      aie.use_lock(%pa02, AcquireGreaterEqual, %one)
      aie.dma_bd(%a02 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%ca02, Release, %one)
      aie.next_bd ^a
    ^c1:
      %1 = aie.dma_start(S2MM, 1, ^b, ^end)
    ^b:
      aie.use_lock(%pb02, AcquireGreaterEqual, %one)
      aie.dma_bd(%b02 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cb02, Release, %one)
      aie.next_bd ^b
    ^end:
      aie.end
    }

    %a03 = aie.buffer(%t03) : memref<16xi32>
    %b03 = aie.buffer(%t03) : memref<16xi32>
    %pa03 = aie.lock(%t03, 0) {init = 1 : i32}
    %ca03 = aie.lock(%t03, 1) {init = 0 : i32}
    %pb03 = aie.lock(%t03, 2) {init = 1 : i32}
    %cb03 = aie.lock(%t03, 3) {init = 0 : i32}
    aie.core(%t03) {
      %one = arith.constant 1 : i32
      aie.use_lock(%ca03, AcquireGreaterEqual, %one)
      aie.use_lock(%cb03, AcquireGreaterEqual, %one)
      aie.use_lock(%pa03, Release, %one)
      aie.use_lock(%pb03, Release, %one)
      aie.end
    }
    aie.mem(%t03) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^a, ^c1)
    ^a:
      aie.use_lock(%pa03, AcquireGreaterEqual, %one)
      aie.dma_bd(%a03 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%ca03, Release, %one)
      aie.next_bd ^a
    ^c1:
      %1 = aie.dma_start(S2MM, 1, ^b, ^end)
    ^b:
      aie.use_lock(%pb03, AcquireGreaterEqual, %one)
      aie.dma_bd(%b03 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cb03, Release, %one)
      aie.next_bd ^b
    ^end:
      aie.end
    }
  }
}
