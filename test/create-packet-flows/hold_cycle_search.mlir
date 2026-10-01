//===- hold_cycle_search.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s

// The pre-routed streams around memtile (0,1) leave 32 alternative holders on
// one candidate hold cycle for the new flow's direct route, and none of them
// closes it. Proving that takes 65 constraint searches; a search budget below
// that would give up and push id 2 onto a detour.

// CHECK-NOT:     warning
// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         aie.connect<DMA : 0, North : 5>
// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK:         %[[A:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(DMA : 0, %[[A]])
// CHECK:         aie.packet_rules(South : 5) {
// CHECK-NEXT:      aie.rule(31, 2, %[[A]])

module {
  aie.device(npu1_2col) {
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_1_2 = aie.tile(1, 2)
    %t_1_3 = aie.tile(1, 3)
    %t_1_4 = aie.tile(1, 4)
    %l_0_2_0 = aie.lock(%t_0_2, 0) {init = 1 : i32, sym_name = "l_0_2_0"}
    %l_0_2_1 = aie.lock(%t_0_2, 1) {init = 0 : i32, sym_name = "l_0_2_1"}
    %l_0_2_2 = aie.lock(%t_0_2, 2) {init = 1 : i32, sym_name = "l_0_2_2"}
    %l_0_2_3 = aie.lock(%t_0_2, 3) {init = 0 : i32, sym_name = "l_0_2_3"}
    %l_0_3_0 = aie.lock(%t_0_3, 0) {init = 1 : i32, sym_name = "l_0_3_0"}
    %l_0_3_1 = aie.lock(%t_0_3, 1) {init = 0 : i32, sym_name = "l_0_3_1"}
    %l_0_3_2 = aie.lock(%t_0_3, 2) {init = 1 : i32, sym_name = "l_0_3_2"}
    %l_0_3_3 = aie.lock(%t_0_3, 3) {init = 0 : i32, sym_name = "l_0_3_3"}
    %b_0_1_0 = aie.buffer(%t_0_1) {sym_name = "b_0_1_0"} : memref<64xi32>
    %b_0_1_1 = aie.buffer(%t_0_1) {sym_name = "b_0_1_1"} : memref<64xi32>
    %b_0_1_2 = aie.buffer(%t_0_1) {sym_name = "b_0_1_2"} : memref<64xi32>
    %b_0_1_3 = aie.buffer(%t_0_1) {sym_name = "b_0_1_3"} : memref<64xi32>
    %b_0_1_4 = aie.buffer(%t_0_1) {sym_name = "b_0_1_4"} : memref<64xi32>
    %b_0_1_5 = aie.buffer(%t_0_1) {sym_name = "b_0_1_5"} : memref<64xi32>
    %b_0_1_6 = aie.buffer(%t_0_1) {sym_name = "b_0_1_6"} : memref<64xi32>
    %b_0_1_7 = aie.buffer(%t_0_1) {sym_name = "b_0_1_7"} : memref<64xi32>
    %b_0_1_8 = aie.buffer(%t_0_1) {sym_name = "b_0_1_8"} : memref<64xi32>
    %b_0_1_9 = aie.buffer(%t_0_1) {sym_name = "b_0_1_9"} : memref<64xi32>
    %b_0_1_10 = aie.buffer(%t_0_1) {sym_name = "b_0_1_10"} : memref<64xi32>
    %b_0_1_11 = aie.buffer(%t_0_1) {sym_name = "b_0_1_11"} : memref<64xi32>
    %b_0_1_12 = aie.buffer(%t_0_1) {sym_name = "b_0_1_12"} : memref<64xi32>
    %b_0_1_13 = aie.buffer(%t_0_1) {sym_name = "b_0_1_13"} : memref<64xi32>
    %b_0_1_14 = aie.buffer(%t_0_1) {sym_name = "b_0_1_14"} : memref<64xi32>
    %b_0_1_15 = aie.buffer(%t_0_1) {sym_name = "b_0_1_15"} : memref<64xi32>
    %b_0_1_16 = aie.buffer(%t_0_1) {sym_name = "b_0_1_16"} : memref<64xi32>
    %b_0_1_17 = aie.buffer(%t_0_1) {sym_name = "b_0_1_17"} : memref<64xi32>
    %b_0_1_18 = aie.buffer(%t_0_1) {sym_name = "b_0_1_18"} : memref<64xi32>
    %b_0_1_19 = aie.buffer(%t_0_1) {sym_name = "b_0_1_19"} : memref<64xi32>
    %b_0_1_20 = aie.buffer(%t_0_1) {sym_name = "b_0_1_20"} : memref<64xi32>
    %b_0_1_21 = aie.buffer(%t_0_1) {sym_name = "b_0_1_21"} : memref<64xi32>
    %b_0_1_22 = aie.buffer(%t_0_1) {sym_name = "b_0_1_22"} : memref<64xi32>
    %b_0_1_23 = aie.buffer(%t_0_1) {sym_name = "b_0_1_23"} : memref<64xi32>
    %b_0_1_24 = aie.buffer(%t_0_1) {sym_name = "b_0_1_24"} : memref<64xi32>
    %b_0_1_25 = aie.buffer(%t_0_1) {sym_name = "b_0_1_25"} : memref<64xi32>
    %b_0_1_26 = aie.buffer(%t_0_1) {sym_name = "b_0_1_26"} : memref<64xi32>
    %b_0_1_27 = aie.buffer(%t_0_1) {sym_name = "b_0_1_27"} : memref<64xi32>
    %b_0_1_28 = aie.buffer(%t_0_1) {sym_name = "b_0_1_28"} : memref<64xi32>
    %b_0_1_29 = aie.buffer(%t_0_1) {sym_name = "b_0_1_29"} : memref<64xi32>
    %b_0_1_30 = aie.buffer(%t_0_1) {sym_name = "b_0_1_30"} : memref<64xi32>
    %b_0_1_31 = aie.buffer(%t_0_1) {sym_name = "b_0_1_31"} : memref<64xi32>
    %b_0_1_32 = aie.buffer(%t_0_1) {sym_name = "b_0_1_32"} : memref<64xi32>
    %b_0_1_33 = aie.buffer(%t_0_1) {sym_name = "b_0_1_33"} : memref<64xi32>
    %b_0_1_34 = aie.buffer(%t_0_1) {sym_name = "b_0_1_34"} : memref<64xi32>
    %b_0_1_35 = aie.buffer(%t_0_1) {sym_name = "b_0_1_35"} : memref<64xi32>
    %dma_0_1 = aie.memtile_dma(%t_0_1) {
      %d0 = aie.dma_start(MM2S, 0, ^p0b0, ^p1)
    ^p0b0:
      aie.dma_bd(%b_0_1_0 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.next_bd ^end
    ^p1:
      %d1 = aie.dma_start(MM2S, 1, ^p1b0, ^p2)
    ^p1b0:
      aie.dma_bd(%b_0_1_1 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.next_bd ^p1b1
    ^p1b1:
      aie.dma_bd(%b_0_1_2 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.next_bd ^end
    ^p2:
      %d2 = aie.dma_start(MM2S, 2, ^p2b0, ^p3)
    ^p2b0:
      aie.dma_bd(%b_0_1_3 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.next_bd ^end
    ^p3:
      %d3 = aie.dma_start(MM2S, 3, ^p3b0, ^p4)
    ^p3b0:
      aie.dma_bd(%b_0_1_4 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.next_bd ^end
    ^p4:
      %d4 = aie.dma_start(MM2S, 4, ^p4b0, ^end)
    ^p4b0:
      aie.dma_bd(%b_0_1_5 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.next_bd ^p4b1
    ^p4b1:
      aie.dma_bd(%b_0_1_6 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.next_bd ^p4b2
    ^p4b2:
      aie.dma_bd(%b_0_1_7 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.next_bd ^p4b3
    ^p4b3:
      aie.dma_bd(%b_0_1_8 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
      aie.next_bd ^p4b4
    ^p4b4:
      aie.dma_bd(%b_0_1_9 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
      aie.next_bd ^p4b5
    ^p4b5:
      aie.dma_bd(%b_0_1_10 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
      aie.next_bd ^p4b6
    ^p4b6:
      aie.dma_bd(%b_0_1_11 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
      aie.next_bd ^p4b7
    ^p4b7:
      aie.dma_bd(%b_0_1_12 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
      aie.next_bd ^p4b8
    ^p4b8:
      aie.dma_bd(%b_0_1_13 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 8>}
      aie.next_bd ^p4b9
    ^p4b9:
      aie.dma_bd(%b_0_1_14 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 9>}
      aie.next_bd ^p4b10
    ^p4b10:
      aie.dma_bd(%b_0_1_15 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 10>}
      aie.next_bd ^p4b11
    ^p4b11:
      aie.dma_bd(%b_0_1_16 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 11>}
      aie.next_bd ^p4b12
    ^p4b12:
      aie.dma_bd(%b_0_1_17 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 12>}
      aie.next_bd ^p4b13
    ^p4b13:
      aie.dma_bd(%b_0_1_18 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 13>}
      aie.next_bd ^p4b14
    ^p4b14:
      aie.dma_bd(%b_0_1_19 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 14>}
      aie.next_bd ^p4b15
    ^p4b15:
      aie.dma_bd(%b_0_1_20 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
      aie.next_bd ^p4b16
    ^p4b16:
      aie.dma_bd(%b_0_1_21 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 16>}
      aie.next_bd ^p4b17
    ^p4b17:
      aie.dma_bd(%b_0_1_22 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 17>}
      aie.next_bd ^p4b18
    ^p4b18:
      aie.dma_bd(%b_0_1_23 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 18>}
      aie.next_bd ^p4b19
    ^p4b19:
      aie.dma_bd(%b_0_1_24 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 19>}
      aie.next_bd ^p4b20
    ^p4b20:
      aie.dma_bd(%b_0_1_25 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 20>}
      aie.next_bd ^p4b21
    ^p4b21:
      aie.dma_bd(%b_0_1_26 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 21>}
      aie.next_bd ^p4b22
    ^p4b22:
      aie.dma_bd(%b_0_1_27 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 22>}
      aie.next_bd ^p4b23
    ^p4b23:
      aie.dma_bd(%b_0_1_28 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 23>}
      aie.next_bd ^p4b24
    ^p4b24:
      aie.dma_bd(%b_0_1_29 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 24>}
      aie.next_bd ^p4b25
    ^p4b25:
      aie.dma_bd(%b_0_1_30 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 25>}
      aie.next_bd ^p4b26
    ^p4b26:
      aie.dma_bd(%b_0_1_31 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
      aie.next_bd ^p4b27
    ^p4b27:
      aie.dma_bd(%b_0_1_32 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
      aie.next_bd ^p4b28
    ^p4b28:
      aie.dma_bd(%b_0_1_33 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 28>}
      aie.next_bd ^p4b29
    ^p4b29:
      aie.dma_bd(%b_0_1_34 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
      aie.next_bd ^p4b30
    ^p4b30:
      aie.dma_bd(%b_0_1_35 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %b_0_2_36 = aie.buffer(%t_0_2) {sym_name = "b_0_2_36"} : memref<16xi32>
    %b_0_2_37 = aie.buffer(%t_0_2) {sym_name = "b_0_2_37"} : memref<16xi32>
    %dma_0_2 = aie.mem(%t_0_2) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^p1)
    ^p0b0:
      aie.dma_bd(%b_0_2_36 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%l_0_2_1, Release, %c1)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(S2MM, 1, ^p1b0, ^end)
    ^p1b0:
      aie.use_lock(%l_0_2_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_0_2_37 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%l_0_2_3, Release, %c1)
      aie.next_bd ^p1b0
    ^end:
      aie.end
    }
    %b_0_3_38 = aie.buffer(%t_0_3) {sym_name = "b_0_3_38"} : memref<16xi32>
    %b_0_3_39 = aie.buffer(%t_0_3) {sym_name = "b_0_3_39"} : memref<16xi32>
    %dma_0_3 = aie.mem(%t_0_3) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^p1)
    ^p0b0:
      aie.use_lock(%l_0_3_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_0_3_38 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%l_0_3_1, Release, %c1)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(S2MM, 1, ^p1b0, ^end)
    ^p1b0:
      aie.use_lock(%l_0_3_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_0_3_39 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%l_0_3_3, Release, %c1)
      aie.next_bd ^p1b0
    ^end:
      aie.end
    }
    %b_0_4_40 = aie.buffer(%t_0_4) {sym_name = "b_0_4_40"} : memref<64xi32>
    %dma_0_4 = aie.mem(%t_0_4) {
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_0_4_40 : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %core_0_2 = aie.core(%t_0_2) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%l_0_2_1, AcquireGreaterEqual, %c1)
      aie.use_lock(%l_0_2_3, AcquireGreaterEqual, %c1)
      aie.use_lock(%l_0_2_0, Release, %c1)
      aie.use_lock(%l_0_2_2, Release, %c1)
      aie.end
    }
    %core_0_3 = aie.core(%t_0_3) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%l_0_3_1, AcquireGreaterEqual, %c1)
      aie.use_lock(%l_0_3_3, AcquireGreaterEqual, %c1)
      aie.use_lock(%l_0_3_0, Release, %c1)
      aie.use_lock(%l_0_3_2, Release, %c1)
      aie.end
    }
    %sb_0_1 = aie.switchbox(%t_0_1) {
      %reserved0_0 = aie.amsel<0> (0)
      %reserved0_1 = aie.amsel<0> (1)
      %reserved0_2 = aie.amsel<0> (2)
      %reserved0_3 = aie.amsel<0> (3)
      %y = aie.amsel<1> (0)
      %dj = aie.amsel<5> (0)
      %xi = aie.amsel<2> (0)
      %z = aie.amsel<3> (0)
      %di = aie.amsel<4> (0)
      %m_North_0 = aie.masterset(North : 0, %y)
      %m_North_1 = aie.masterset(North : 1, %dj)
      %m_North_2 = aie.masterset(North : 2, %xi)
      %m_North_3 = aie.masterset(North : 3, %z)
      %m_North_4 = aie.masterset(North : 4, %di)
      aie.packet_rules(DMA : 1) {
        aie.rule(31, 0, %y)
        aie.rule(31, 1, %dj)
      }
      aie.packet_rules(DMA : 2) {
        aie.rule(31, 0, %xi)
      }
      aie.packet_rules(DMA : 3) {
        aie.rule(31, 0, %z)
      }
      aie.packet_rules(DMA : 4) {
        aie.rule(0, 0, %di)
      }
    }
    %sb_0_2 = aie.switchbox(%t_0_2) {
      %y = aie.amsel<1> (0)
      %dj = aie.amsel<2> (0)
      %m_East_0 = aie.masterset(East : 0, %y)
      %m_DMA_1 = aie.masterset(DMA : 1, %dj)
      aie.packet_rules(South : 0) {
        aie.rule(31, 0, %y)
      }
      aie.packet_rules(East : 0) {
        aie.rule(31, 1, %dj)
      }
      aie.connect<South : 1, North : 1>
      aie.connect<South : 2, East : 2>
      aie.connect<South : 3, North : 0>
      aie.connect<South : 4, East : 3>
    }
    %sb_0_3 = aie.switchbox(%t_0_3) {
      %xi = aie.amsel<0> (0)
      %z = aie.amsel<1> (0)
      %m_DMA_0 = aie.masterset(DMA : 0, %xi)
      %m_DMA_1 = aie.masterset(DMA : 1, %z)
      aie.packet_rules(East : 0) {
        aie.rule(0, 0, %xi)
      }
      aie.packet_rules(South : 0) {
        aie.rule(31, 0, %z)
      }
      aie.connect<South : 1, East : 1>
    }
    %sb_0_4 = aie.switchbox(%t_0_4) {
      %y = aie.amsel<0> (0)
      %m_DMA_0 = aie.masterset(DMA : 0, %y)
      aie.packet_rules(East : 0) {
        aie.rule(31, 0, %y)
      }
    }
    %sb_1_2 = aie.switchbox(%t_1_2) {
      %xi = aie.amsel<0> (0)
      %y = aie.amsel<0> (1)
      %dj = aie.amsel<0> (2)
      %m_North_0 = aie.masterset(North : 0, %xi)
      %m_North_1 = aie.masterset(North : 1, %y)
      %m_West_0 = aie.masterset(West : 0, %dj)
      aie.packet_rules(West : 0) {
        aie.rule(31, 0, %y)
      }
      aie.packet_rules(North : 1) {
        aie.rule(31, 1, %dj)
      }
      aie.packet_rules(West : 2) {
        aie.rule(31, 0, %xi)
      }
      aie.packet_rules(West : 3) {
        aie.rule(0, 0, %xi)
      }
    }
    %sb_1_3 = aie.switchbox(%t_1_3) {
      aie.connect<South : 0, West : 0>
      aie.connect<South : 1, North : 0>
      aie.connect<West : 1, South : 1>
    }
    %sb_1_4 = aie.switchbox(%t_1_4) {
      aie.connect<South : 0, West : 0>
    }
    aie.packet_flow(2) { aie.packet_source<%t_0_1, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> }
  }
}
