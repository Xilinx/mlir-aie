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

    %b0 = aie.buffer(%t0) {sym_name = "b0"} : memref<256xi32>
    %p0 = aie.lock(%t0, 0) {init = 1 : i32, sym_name = "p0"}
    %q0 = aie.lock(%t0, 1) {init = 0 : i32, sym_name = "q0"}
    %core0 = aie.core(%t0) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %it = %c0 to %c3 step %c1 {
        aie.use_lock(%p0, AcquireGreaterEqual, %one)
        memref.store %one, %b0[%c0] : memref<256xi32>
        aie.use_lock(%q0, Release, %one)
      }
      aie.end
    }
    %mem0 = aie.mem(%t0) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^s0, ^end)
    ^s0:
      aie.use_lock(%q0, AcquireGreaterEqual, %one)
      aie.dma_bd(%b0 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(%p0, Release, %one)
      aie.next_bd ^s1
    ^s1:
      aie.use_lock(%q0, AcquireGreaterEqual, %one)
      aie.dma_bd(%b0 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.use_lock(%p0, Release, %one)
      aie.next_bd ^s2
    ^s2:
      aie.use_lock(%q0, AcquireGreaterEqual, %one)
      aie.dma_bd(%b0 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.use_lock(%p0, Release, %one)
      aie.next_bd ^s0
    ^end:
      aie.end
    }

    %b1 = aie.buffer(%t1) {sym_name = "b1"} : memref<256xi32>
    %p1 = aie.lock(%t1, 0) {init = 1 : i32, sym_name = "p1"}
    %q1 = aie.lock(%t1, 1) {init = 0 : i32, sym_name = "q1"}
    %core1 = aie.core(%t1) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %it = %c0 to %c3 step %c1 {
        aie.use_lock(%p1, AcquireGreaterEqual, %one)
        memref.store %one, %b1[%c0] : memref<256xi32>
        aie.use_lock(%q1, Release, %one)
      }
      aie.end
    }
    %mem1 = aie.mem(%t1) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^s0, ^end)
    ^s0:
      aie.use_lock(%q1, AcquireGreaterEqual, %one)
      aie.dma_bd(%b1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(%p1, Release, %one)
      aie.next_bd ^s1
    ^s1:
      aie.use_lock(%q1, AcquireGreaterEqual, %one)
      aie.dma_bd(%b1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.use_lock(%p1, Release, %one)
      aie.next_bd ^s2
    ^s2:
      aie.use_lock(%q1, AcquireGreaterEqual, %one)
      aie.dma_bd(%b1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.use_lock(%p1, Release, %one)
      aie.next_bd ^s0
    ^end:
      aie.end
    }

    %b2 = aie.buffer(%t2) {sym_name = "b2"} : memref<256xi32>
    %p2 = aie.lock(%t2, 0) {init = 1 : i32, sym_name = "p2"}
    %q2 = aie.lock(%t2, 1) {init = 0 : i32, sym_name = "q2"}
    %core2 = aie.core(%t2) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %it = %c0 to %c3 step %c1 {
        aie.use_lock(%p2, AcquireGreaterEqual, %one)
        memref.store %one, %b2[%c0] : memref<256xi32>
        aie.use_lock(%q2, Release, %one)
      }
      aie.end
    }
    %mem2 = aie.mem(%t2) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^s0, ^end)
    ^s0:
      aie.use_lock(%q2, AcquireGreaterEqual, %one)
      aie.dma_bd(%b2 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(%p2, Release, %one)
      aie.next_bd ^s1
    ^s1:
      aie.use_lock(%q2, AcquireGreaterEqual, %one)
      aie.dma_bd(%b2 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.use_lock(%p2, Release, %one)
      aie.next_bd ^s2
    ^s2:
      aie.use_lock(%q2, AcquireGreaterEqual, %one)
      aie.dma_bd(%b2 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.use_lock(%p2, Release, %one)
      aie.next_bd ^s0
    ^end:
      aie.end
    }

    %b3 = aie.buffer(%t3) {sym_name = "b3"} : memref<256xi32>
    %p3 = aie.lock(%t3, 0) {init = 1 : i32, sym_name = "p3"}
    %q3 = aie.lock(%t3, 1) {init = 0 : i32, sym_name = "q3"}
    %core3 = aie.core(%t3) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %it = %c0 to %c3 step %c1 {
        aie.use_lock(%p3, AcquireGreaterEqual, %one)
        memref.store %one, %b3[%c0] : memref<256xi32>
        aie.use_lock(%q3, Release, %one)
      }
      aie.end
    }
    %mem3 = aie.mem(%t3) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^s0, ^end)
    ^s0:
      aie.use_lock(%q3, AcquireGreaterEqual, %one)
      aie.dma_bd(%b3 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(%p3, Release, %one)
      aie.next_bd ^s1
    ^s1:
      aie.use_lock(%q3, AcquireGreaterEqual, %one)
      aie.dma_bd(%b3 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.use_lock(%p3, Release, %one)
      aie.next_bd ^s2
    ^s2:
      aie.use_lock(%q3, AcquireGreaterEqual, %one)
      aie.dma_bd(%b3 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.use_lock(%p3, Release, %one)
      aie.next_bd ^s0
    ^end:
      aie.end
    }

    %b4 = aie.buffer(%t4) {sym_name = "b4"} : memref<256xi32>
    %p4 = aie.lock(%t4, 0) {init = 1 : i32, sym_name = "p4"}
    %q4 = aie.lock(%t4, 1) {init = 0 : i32, sym_name = "q4"}
    %core4 = aie.core(%t4) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %it = %c0 to %c3 step %c1 {
        aie.use_lock(%p4, AcquireGreaterEqual, %one)
        memref.store %one, %b4[%c0] : memref<256xi32>
        aie.use_lock(%q4, Release, %one)
      }
      aie.end
    }
    %mem4 = aie.mem(%t4) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^s0, ^end)
    ^s0:
      aie.use_lock(%q4, AcquireGreaterEqual, %one)
      aie.dma_bd(%b4 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(%p4, Release, %one)
      aie.next_bd ^s1
    ^s1:
      aie.use_lock(%q4, AcquireGreaterEqual, %one)
      aie.dma_bd(%b4 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.use_lock(%p4, Release, %one)
      aie.next_bd ^s2
    ^s2:
      aie.use_lock(%q4, AcquireGreaterEqual, %one)
      aie.dma_bd(%b4 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.use_lock(%p4, Release, %one)
      aie.next_bd ^s0
    ^end:
      aie.end
    }

    %b5 = aie.buffer(%t5) {sym_name = "b5"} : memref<256xi32>
    %p5 = aie.lock(%t5, 0) {init = 0 : i32, sym_name = "p5"}
    %q5 = aie.lock(%t5, 1) {init = 0 : i32, sym_name = "q5"}
    %core5 = aie.core(%t5) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      memref.store %one, %b5[%c0] : memref<256xi32>
      aie.use_lock(%q5, Release, %one)
      aie.end
    }
    %mem5 = aie.mem(%t5) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^s0, ^end)
    ^s0:
      aie.use_lock(%q5, AcquireGreaterEqual, %one)
      aie.dma_bd(%b5 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
      aie.use_lock(%p5, Release, %one)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %mb0 = aie.buffer(%m) {sym_name = "mb0"} : memref<256xi32>
    %mp0 = aie.lock(%m, 0) {init = 1 : i32, sym_name = "mp0"}
    %mq0 = aie.lock(%m, 1) {init = 0 : i32, sym_name = "mq0"}
    %mb1 = aie.buffer(%m) {sym_name = "mb1"} : memref<256xi32>
    %mp1 = aie.lock(%m, 2) {init = 1 : i32, sym_name = "mp1"}
    %mq1 = aie.lock(%m, 3) {init = 0 : i32, sym_name = "mq1"}
    %mb2 = aie.buffer(%m) {sym_name = "mb2"} : memref<256xi32>
    %mp2 = aie.lock(%m, 4) {init = 1 : i32, sym_name = "mp2"}
    %mq2 = aie.lock(%m, 5) {init = 0 : i32, sym_name = "mq2"}
    %mb3 = aie.buffer(%m) {sym_name = "mb3"} : memref<256xi32>
    %mp3 = aie.lock(%m, 6) {init = 1 : i32, sym_name = "mp3"}
    %mq3 = aie.lock(%m, 7) {init = 0 : i32, sym_name = "mq3"}
    %mb4 = aie.buffer(%m) {sym_name = "mb4"} : memref<256xi32>
    %mp4 = aie.lock(%m, 8) {init = 1 : i32, sym_name = "mp4"}
    %mq4 = aie.lock(%m, 9) {init = 0 : i32, sym_name = "mq4"}
    %mb5 = aie.buffer(%m) {sym_name = "mb5"} : memref<256xi32>
    %mp5 = aie.lock(%m, 10) {init = 1 : i32, sym_name = "mp5"}
    %mq5 = aie.lock(%m, 11) {init = 0 : i32, sym_name = "mq5"}
    %mdma = aie.memtile_dma(%m) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^r0, ^in1)
    ^r0:
      aie.use_lock(%mp0, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb0 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%mq0, Release, %one)
      aie.next_bd ^r0
    ^in1:
      %1 = aie.dma_start(S2MM, 1, ^r1, ^in2)
    ^r1:
      aie.use_lock(%mp1, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb1 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%mq1, Release, %one)
      aie.next_bd ^r1
    ^in2:
      %2 = aie.dma_start(S2MM, 2, ^r2, ^in3)
    ^r2:
      aie.use_lock(%mp2, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb2 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%mq2, Release, %one)
      aie.next_bd ^r2
    ^in3:
      %3 = aie.dma_start(S2MM, 3, ^r3, ^in4)
    ^r3:
      aie.use_lock(%mp3, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb3 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%mq3, Release, %one)
      aie.next_bd ^r3
    ^in4:
      %4 = aie.dma_start(S2MM, 4, ^r4, ^in5)
    ^r4:
      aie.use_lock(%mp4, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb4 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%mq4, Release, %one)
      aie.next_bd ^r4
    ^in5:
      %5 = aie.dma_start(S2MM, 5, ^r5, ^out)
    ^r5:
      aie.use_lock(%mp5, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb5 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%mq5, Release, %one)
      aie.next_bd ^out
    ^out:
      %6 = aie.dma_start(MM2S, 0, ^j0, ^end)
    ^j0:
      aie.use_lock(%mq0, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb0 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
      aie.use_lock(%mp0, Release, %one)
      aie.next_bd ^j1
    ^j1:
      aie.use_lock(%mq1, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
      aie.use_lock(%mp1, Release, %one)
      aie.next_bd ^j2
    ^j2:
      aie.use_lock(%mq2, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb2 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
      aie.use_lock(%mp2, Release, %one)
      aie.next_bd ^j3
    ^j3:
      aie.use_lock(%mq3, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb3 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
      aie.use_lock(%mp3, Release, %one)
      aie.next_bd ^j4
    ^j4:
      aie.use_lock(%mq4, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb4 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
      aie.use_lock(%mp4, Release, %one)
      aie.next_bd ^j0
    ^end:
      aie.end
    }

    %ob = aie.buffer(%t6) {sym_name = "ob"} : memref<256xi32>
    %op = aie.lock(%t6, 0) {init = 1 : i32, sym_name = "op"}
    %oq = aie.lock(%t6, 1) {init = 0 : i32, sym_name = "oq"}
    %core6 = aie.core(%t6) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c15 = arith.constant 15 : index
      scf.for %it = %c0 to %c15 step %c1 {
        aie.use_lock(%oq, AcquireGreaterEqual, %one)
        %v = memref.load %ob[%c0] : memref<256xi32>
        aie.use_lock(%op, Release, %one)
      }
      aie.end
    }
    %mem6 = aie.mem(%t6) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 1, ^r, ^end)
    ^r:
      aie.use_lock(%op, AcquireGreaterEqual, %one)
      aie.dma_bd(%ob : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%oq, Release, %one)
      aie.next_bd ^r
    ^end:
      aie.end
    }
  }
}
