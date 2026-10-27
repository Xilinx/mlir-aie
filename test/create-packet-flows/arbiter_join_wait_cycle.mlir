//===- arbiter_join_wait_cycle.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s --check-prefix=CIRCUIT

// Memtile (0,1) joins S2MM 0..4, fed by cores (0..4, 2), onto one MM2S out to
// core (0,3). Core (5,2) sends one buffer to S2MM 5, which has room for it.
// That is seven master ports for six arbiters. Any two joined channels on one
// arbiter deadlock once one runs ahead, and so does a joined channel with the
// join's own output; neither shows in the tile-level flow graph. S2MM 5 waits
// on nobody, so it is the one master that can share; it lands on the join's
// output arbiter.

// CHECK-NOT:     warning
// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[OUT:.*]] = aie.amsel<0> (0)
// CHECK:         %[[SHARED:.*]] = aie.amsel<0> (1)
// CHECK:         aie.masterset(DMA : 5, %[[SHARED]])
// CHECK:         aie.masterset(North : 1, %[[OUT]])

// CIRCUIT-LABEL: aie.switchbox(%mem_tile_0_1)
// CIRCUIT-NEXT:    aie.connect<DMA : 0, North : 1>
// CIRCUIT-NOT:     aie.amsel<{{[0-9]}}> (1)
// CIRCUIT:       aie.switchbox(%tile_0_2)

module {
  aie.device(npu2) {
    %m  = aie.tile(0, 1)
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(1, 2)
    %t2 = aie.tile(2, 2)
    %t3 = aie.tile(3, 2)
    %t4 = aie.tile(4, 2)
    %t5 = aie.tile(5, 2)
    %t6 = aie.tile(0, 3)
    aie.packet_flow(0) { aie.packet_source<%t0, DMA : 0> aie.packet_dest<%m, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%t0, DMA : 0> aie.packet_dest<%m, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%t0, DMA : 0> aie.packet_dest<%m, DMA : 0> }
    aie.packet_flow(0) { aie.packet_source<%t1, DMA : 0> aie.packet_dest<%m, DMA : 1> }
    aie.packet_flow(1) { aie.packet_source<%t1, DMA : 0> aie.packet_dest<%m, DMA : 1> }
    aie.packet_flow(2) { aie.packet_source<%t1, DMA : 0> aie.packet_dest<%m, DMA : 1> }
    aie.packet_flow(0) { aie.packet_source<%t2, DMA : 0> aie.packet_dest<%m, DMA : 2> }
    aie.packet_flow(1) { aie.packet_source<%t2, DMA : 0> aie.packet_dest<%m, DMA : 2> }
    aie.packet_flow(2) { aie.packet_source<%t2, DMA : 0> aie.packet_dest<%m, DMA : 2> }
    aie.packet_flow(0) { aie.packet_source<%t3, DMA : 0> aie.packet_dest<%m, DMA : 3> }
    aie.packet_flow(1) { aie.packet_source<%t3, DMA : 0> aie.packet_dest<%m, DMA : 3> }
    aie.packet_flow(2) { aie.packet_source<%t3, DMA : 0> aie.packet_dest<%m, DMA : 3> }
    aie.packet_flow(0) { aie.packet_source<%t4, DMA : 0> aie.packet_dest<%m, DMA : 4> }
    aie.packet_flow(1) { aie.packet_source<%t4, DMA : 0> aie.packet_dest<%m, DMA : 4> }
    aie.packet_flow(2) { aie.packet_source<%t4, DMA : 0> aie.packet_dest<%m, DMA : 4> }
    aie.packet_flow(3) { aie.packet_source<%t5, DMA : 0> aie.packet_dest<%m, DMA : 5> }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 0> aie.packet_dest<%t6, DMA : 1> }
    aie.packet_flow(5) { aie.packet_source<%m, DMA : 0> aie.packet_dest<%t6, DMA : 1> }
    aie.packet_flow(6) { aie.packet_source<%m, DMA : 0> aie.packet_dest<%t6, DMA : 1> }

    aie.buffer @b0(%t0) : memref<256xi32>
    aie.lock @p0(%t0, 0) {init = 1 : i32}
    aie.lock @q0(%t0, 1) {init = 0 : i32}
    %core0 = aie.core(%t0) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %it = %c0 to %c3 step %c1 {
        aie.use_lock(@p0, AcquireGreaterEqual, %one)
        %b0 = aie.get_buffer @b0 : memref<256xi32>
        memref.store %one, %b0[%c0] : memref<256xi32>
        aie.use_lock(@q0, Release, %one)
      }
      aie.end
    }
    %mem0 = aie.mem(%t0) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^s0, ^end)
    ^s0:
      aie.use_lock(@q0, AcquireGreaterEqual, %one)
      %b0_1 = aie.get_buffer @b0 : memref<256xi32>
      aie.dma_bd(%b0_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(@p0, Release, %one)
      aie.next_bd ^s1
    ^s1:
      aie.use_lock(@q0, AcquireGreaterEqual, %one)
      %b0_2 = aie.get_buffer @b0 : memref<256xi32>
      aie.dma_bd(%b0_2 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.use_lock(@p0, Release, %one)
      aie.next_bd ^s2
    ^s2:
      aie.use_lock(@q0, AcquireGreaterEqual, %one)
      %b0_3 = aie.get_buffer @b0 : memref<256xi32>
      aie.dma_bd(%b0_3 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.use_lock(@p0, Release, %one)
      aie.next_bd ^s0
    ^end:
      aie.end
    }

    aie.buffer @b1(%t1) : memref<256xi32>
    aie.lock @p1(%t1, 0) {init = 1 : i32}
    aie.lock @q1(%t1, 1) {init = 0 : i32}
    %core1 = aie.core(%t1) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %it = %c0 to %c3 step %c1 {
        aie.use_lock(@p1, AcquireGreaterEqual, %one)
        %b1 = aie.get_buffer @b1 : memref<256xi32>
        memref.store %one, %b1[%c0] : memref<256xi32>
        aie.use_lock(@q1, Release, %one)
      }
      aie.end
    }
    %mem1 = aie.mem(%t1) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^s0, ^end)
    ^s0:
      aie.use_lock(@q1, AcquireGreaterEqual, %one)
      %b1_1 = aie.get_buffer @b1 : memref<256xi32>
      aie.dma_bd(%b1_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(@p1, Release, %one)
      aie.next_bd ^s1
    ^s1:
      aie.use_lock(@q1, AcquireGreaterEqual, %one)
      %b1_2 = aie.get_buffer @b1 : memref<256xi32>
      aie.dma_bd(%b1_2 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.use_lock(@p1, Release, %one)
      aie.next_bd ^s2
    ^s2:
      aie.use_lock(@q1, AcquireGreaterEqual, %one)
      %b1_3 = aie.get_buffer @b1 : memref<256xi32>
      aie.dma_bd(%b1_3 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.use_lock(@p1, Release, %one)
      aie.next_bd ^s0
    ^end:
      aie.end
    }

    aie.buffer @b2(%t2) : memref<256xi32>
    aie.lock @p2(%t2, 0) {init = 1 : i32}
    aie.lock @q2(%t2, 1) {init = 0 : i32}
    %core2 = aie.core(%t2) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %it = %c0 to %c3 step %c1 {
        aie.use_lock(@p2, AcquireGreaterEqual, %one)
        %b2 = aie.get_buffer @b2 : memref<256xi32>
        memref.store %one, %b2[%c0] : memref<256xi32>
        aie.use_lock(@q2, Release, %one)
      }
      aie.end
    }
    %mem2 = aie.mem(%t2) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^s0, ^end)
    ^s0:
      aie.use_lock(@q2, AcquireGreaterEqual, %one)
      %b2_1 = aie.get_buffer @b2 : memref<256xi32>
      aie.dma_bd(%b2_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(@p2, Release, %one)
      aie.next_bd ^s1
    ^s1:
      aie.use_lock(@q2, AcquireGreaterEqual, %one)
      %b2_2 = aie.get_buffer @b2 : memref<256xi32>
      aie.dma_bd(%b2_2 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.use_lock(@p2, Release, %one)
      aie.next_bd ^s2
    ^s2:
      aie.use_lock(@q2, AcquireGreaterEqual, %one)
      %b2_3 = aie.get_buffer @b2 : memref<256xi32>
      aie.dma_bd(%b2_3 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.use_lock(@p2, Release, %one)
      aie.next_bd ^s0
    ^end:
      aie.end
    }

    aie.buffer @b3(%t3) : memref<256xi32>
    aie.lock @p3(%t3, 0) {init = 1 : i32}
    aie.lock @q3(%t3, 1) {init = 0 : i32}
    %core3 = aie.core(%t3) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %it = %c0 to %c3 step %c1 {
        aie.use_lock(@p3, AcquireGreaterEqual, %one)
        %b3 = aie.get_buffer @b3 : memref<256xi32>
        memref.store %one, %b3[%c0] : memref<256xi32>
        aie.use_lock(@q3, Release, %one)
      }
      aie.end
    }
    %mem3 = aie.mem(%t3) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^s0, ^end)
    ^s0:
      aie.use_lock(@q3, AcquireGreaterEqual, %one)
      %b3_1 = aie.get_buffer @b3 : memref<256xi32>
      aie.dma_bd(%b3_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(@p3, Release, %one)
      aie.next_bd ^s1
    ^s1:
      aie.use_lock(@q3, AcquireGreaterEqual, %one)
      %b3_2 = aie.get_buffer @b3 : memref<256xi32>
      aie.dma_bd(%b3_2 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.use_lock(@p3, Release, %one)
      aie.next_bd ^s2
    ^s2:
      aie.use_lock(@q3, AcquireGreaterEqual, %one)
      %b3_3 = aie.get_buffer @b3 : memref<256xi32>
      aie.dma_bd(%b3_3 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.use_lock(@p3, Release, %one)
      aie.next_bd ^s0
    ^end:
      aie.end
    }

    aie.buffer @b4(%t4) : memref<256xi32>
    aie.lock @p4(%t4, 0) {init = 1 : i32}
    aie.lock @q4(%t4, 1) {init = 0 : i32}
    %core4 = aie.core(%t4) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %it = %c0 to %c3 step %c1 {
        aie.use_lock(@p4, AcquireGreaterEqual, %one)
        %b4 = aie.get_buffer @b4 : memref<256xi32>
        memref.store %one, %b4[%c0] : memref<256xi32>
        aie.use_lock(@q4, Release, %one)
      }
      aie.end
    }
    %mem4 = aie.mem(%t4) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^s0, ^end)
    ^s0:
      aie.use_lock(@q4, AcquireGreaterEqual, %one)
      %b4_1 = aie.get_buffer @b4 : memref<256xi32>
      aie.dma_bd(%b4_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(@p4, Release, %one)
      aie.next_bd ^s1
    ^s1:
      aie.use_lock(@q4, AcquireGreaterEqual, %one)
      %b4_2 = aie.get_buffer @b4 : memref<256xi32>
      aie.dma_bd(%b4_2 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.use_lock(@p4, Release, %one)
      aie.next_bd ^s2
    ^s2:
      aie.use_lock(@q4, AcquireGreaterEqual, %one)
      %b4_3 = aie.get_buffer @b4 : memref<256xi32>
      aie.dma_bd(%b4_3 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.use_lock(@p4, Release, %one)
      aie.next_bd ^s0
    ^end:
      aie.end
    }

    aie.buffer @b5(%t5) : memref<256xi32>
    aie.lock @p5(%t5, 0) {init = 0 : i32}
    aie.lock @q5(%t5, 1) {init = 0 : i32}
    %core5 = aie.core(%t5) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %b5 = aie.get_buffer @b5 : memref<256xi32>
      memref.store %one, %b5[%c0] : memref<256xi32>
      aie.use_lock(@q5, Release, %one)
      aie.end
    }
    %mem5 = aie.mem(%t5) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^s0, ^end)
    ^s0:
      aie.use_lock(@q5, AcquireGreaterEqual, %one)
      %b5_1 = aie.get_buffer @b5 : memref<256xi32>
      aie.dma_bd(%b5_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
      aie.use_lock(@p5, Release, %one)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.buffer @mb0(%m) : memref<256xi32>
    aie.lock @mp0(%m, 0) {init = 1 : i32}
    aie.lock @mq0(%m, 1) {init = 0 : i32}
    aie.buffer @mb1(%m) : memref<256xi32>
    aie.lock @mp1(%m, 2) {init = 1 : i32}
    aie.lock @mq1(%m, 3) {init = 0 : i32}
    aie.buffer @mb2(%m) : memref<256xi32>
    aie.lock @mp2(%m, 4) {init = 1 : i32}
    aie.lock @mq2(%m, 5) {init = 0 : i32}
    aie.buffer @mb3(%m) : memref<256xi32>
    aie.lock @mp3(%m, 6) {init = 1 : i32}
    aie.lock @mq3(%m, 7) {init = 0 : i32}
    aie.buffer @mb4(%m) : memref<256xi32>
    aie.lock @mp4(%m, 8) {init = 1 : i32}
    aie.lock @mq4(%m, 9) {init = 0 : i32}
    aie.buffer @mb5(%m) : memref<256xi32>
    aie.lock @mp5(%m, 10) {init = 1 : i32}
    aie.lock @mq5(%m, 11) {init = 0 : i32}
    %mdma = aie.memtile_dma(%m) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^r0, ^in1)
    ^r0:
      aie.use_lock(@mp0, AcquireGreaterEqual, %one)
      %mb0 = aie.get_buffer @mb0 : memref<256xi32>
      aie.dma_bd(%mb0 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(@mq0, Release, %one)
      aie.next_bd ^r0
    ^in1:
      %1 = aie.dma_start(S2MM, 1, ^r1, ^in2)
    ^r1:
      aie.use_lock(@mp1, AcquireGreaterEqual, %one)
      %mb1 = aie.get_buffer @mb1 : memref<256xi32>
      aie.dma_bd(%mb1 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(@mq1, Release, %one)
      aie.next_bd ^r1
    ^in2:
      %2 = aie.dma_start(S2MM, 2, ^r2, ^in3)
    ^r2:
      aie.use_lock(@mp2, AcquireGreaterEqual, %one)
      %mb2 = aie.get_buffer @mb2 : memref<256xi32>
      aie.dma_bd(%mb2 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(@mq2, Release, %one)
      aie.next_bd ^r2
    ^in3:
      %3 = aie.dma_start(S2MM, 3, ^r3, ^in4)
    ^r3:
      aie.use_lock(@mp3, AcquireGreaterEqual, %one)
      %mb3 = aie.get_buffer @mb3 : memref<256xi32>
      aie.dma_bd(%mb3 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(@mq3, Release, %one)
      aie.next_bd ^r3
    ^in4:
      %4 = aie.dma_start(S2MM, 4, ^r4, ^in5)
    ^r4:
      aie.use_lock(@mp4, AcquireGreaterEqual, %one)
      %mb4 = aie.get_buffer @mb4 : memref<256xi32>
      aie.dma_bd(%mb4 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(@mq4, Release, %one)
      aie.next_bd ^r4
    ^in5:
      %5 = aie.dma_start(S2MM, 5, ^r5, ^out)
    ^r5:
      aie.use_lock(@mp5, AcquireGreaterEqual, %one)
      %mb5 = aie.get_buffer @mb5 : memref<256xi32>
      aie.dma_bd(%mb5 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(@mq5, Release, %one)
      aie.next_bd ^end
    ^out:
      %6 = aie.dma_start(MM2S, 0, ^j0, ^end)
    ^j0:
      aie.use_lock(@mq0, AcquireGreaterEqual, %one)
      %mb0_1 = aie.get_buffer @mb0 : memref<256xi32>
      aie.dma_bd(%mb0_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
      aie.use_lock(@mp0, Release, %one)
      aie.next_bd ^j1
    ^j1:
      aie.use_lock(@mq1, AcquireGreaterEqual, %one)
      %mb1_1 = aie.get_buffer @mb1 : memref<256xi32>
      aie.dma_bd(%mb1_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
      aie.use_lock(@mp1, Release, %one)
      aie.next_bd ^j2
    ^j2:
      aie.use_lock(@mq2, AcquireGreaterEqual, %one)
      %mb2_1 = aie.get_buffer @mb2 : memref<256xi32>
      aie.dma_bd(%mb2_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
      aie.use_lock(@mp2, Release, %one)
      aie.next_bd ^j3
    ^j3:
      aie.use_lock(@mq3, AcquireGreaterEqual, %one)
      %mb3_1 = aie.get_buffer @mb3 : memref<256xi32>
      aie.dma_bd(%mb3_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
      aie.use_lock(@mp3, Release, %one)
      aie.next_bd ^j4
    ^j4:
      aie.use_lock(@mq4, AcquireGreaterEqual, %one)
      %mb4_1 = aie.get_buffer @mb4 : memref<256xi32>
      aie.dma_bd(%mb4_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
      aie.use_lock(@mp4, Release, %one)
      aie.next_bd ^j0
    ^end:
      aie.end
    }

    aie.buffer @ob(%t6) : memref<256xi32>
    aie.lock @op(%t6, 0) {init = 1 : i32}
    aie.lock @oq(%t6, 1) {init = 0 : i32}
    %core6 = aie.core(%t6) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c15 = arith.constant 15 : index
      scf.for %it = %c0 to %c15 step %c1 {
        aie.use_lock(@oq, AcquireGreaterEqual, %one)
        %ob = aie.get_buffer @ob : memref<256xi32>
        %v = memref.load %ob[%c0] : memref<256xi32>
        aie.use_lock(@op, Release, %one)
      }
      aie.end
    }
    %mem6 = aie.mem(%t6) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 1, ^r, ^end)
    ^r:
      aie.use_lock(@op, AcquireGreaterEqual, %one)
      %ob_1 = aie.get_buffer @ob : memref<256xi32>
      aie.dma_bd(%ob_1 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(@oq, Release, %one)
      aie.next_bd ^r
    ^end:
      aie.end
    }
  }
}
