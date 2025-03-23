//===- test_create_packet_flows0.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --split-input-file %s | FileCheck %s

// one-to-many, single arbiter
module @test_create_packet_flows0 {
 aie.device(xcvc1902) {
// CHECK-LABEL:   module @test_create_packet_flows0 {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_1:.*]] = aie.switchbox(%[[VAL_0]]) {
// The actual indices used for the amsel arguments is unimportant.
// CHECK:           %[[VAL_6:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_7:.*]] = aie.amsel<1> (0)
// CHECK:           %[[VAL_4:.*]] = aie.masterset(Core : 0, %[[VAL_2:.*]])
// CHECK:           %[[VAL_5:.*]] = aie.masterset(Core : 1, %[[VAL_3:.*]])
// CHECK:           aie.packet_rules(West : 0) {
// CHECK:             aie.rule(31, 1, %[[VAL_3]])
// CHECK:             aie.rule(31, 0, %[[VAL_2]])
// CHECK:           }
// CHECK:         }
// CHECK:       }
  %t11 = aie.tile(1, 1)

  aie.packet_flow(0x0) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 0>
  }

  aie.packet_flow(0x1) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 1>
  }
 }
}

// -----

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_1:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:             %[[VAL_2:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_3:.*]] = aie.masterset(Core : 0, %[[VAL_2]])
// CHECK:             aie.packet_rules(West : 1) {
// CHECK:               aie.rule(31, 1, %[[VAL_2]])
// CHECK:             }
// CHECK:             aie.packet_rules(West : 0) {
// CHECK:               aie.rule(31, 0, %[[VAL_2]])
// CHECK:             }
// CHECK:           }
// CHECK:         }

// many-to-one, single arbiter
module @test_create_packet_flows1 {
 aie.device(xcvc1902) {
  %t11 = aie.tile(1, 1)

  aie.packet_flow(0x0) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 0>
  }

  aie.packet_flow(0x1) {
    aie.packet_source<%t11, West : 1>
    aie.packet_dest<%t11, Core : 0>
  }
 }
}

// -----

// partial multicast
module @test_create_packet_flows2 {
 aie.device(xcvc1902) {
// CHECK-LABEL: module @test_create_packet_flows2 {
// CHECK:         %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:         %[[VAL_1:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:           %[[VAL_6:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_7:.*]] = aie.amsel<0> (1)
// CHECK:           %[[VAL_4:.*]] = aie.masterset(Core : 0, %[[VAL_2:.*]])
// VAL_3 should also appear here, but it's difficult to filecheck.
// CHECK:           %[[VAL_5:.*]] = aie.masterset(Core : 1,
// CHECK-SACore :      %[[VAL_2]]
// CHECK:           aie.packet_rules(West : 0) {
// CHECK:             aie.rule(31, 1, %[[VAL_3:.*]])
// CHECK:             aie.rule(31, 0, %[[VAL_2]])
// CHECK:           }
// CHECK:         }
// CHECK:       }
  %t11 = aie.tile(1, 1)

  aie.packet_flow(0x0) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 0>
    aie.packet_dest<%t11, Core : 1>
  }

  aie.packet_flow(0x1) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 1>
  }
 }
}

// -----

// many-to-many, 2 streams
module @test_create_packet_flows3 {
 aie.device(xcvc1902) {
// CHECK-LABEL: module @test_create_packet_flows3 {
// CHECK:         %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:         %[[VAL_1:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:           %[[VAL_6:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_7:.*]] = aie.amsel<0> (1)
// CHECK:           %[[VAL_4:.*]] = aie.masterset(Core : 0, %[[VAL_2:.*]])
// CHECK:           %[[VAL_5:.*]] = aie.masterset(Core : 1,
// CHECK-SACore :      %[[VAL_2]]
// CHECK:           aie.packet_rules(West : 1) {
// CHECK:             aie.rule(31, 1, %[[VAL_3:.*]])
// CHECK:           }
// CHECK:           aie.packet_rules(West : 0) {
// CHECK:             aie.rule(31, 0, %[[VAL_2]])
// CHECK:           }
// CHECK:         }
// CHECK:       }
  %t11 = aie.tile(1, 1)

  aie.packet_flow(0x0) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 0>
    aie.packet_dest<%t11, Core : 1>
  }

  aie.packet_flow(0x1) {
    aie.packet_source<%t11, West : 1>
    aie.packet_dest<%t11, Core : 1>
  }
 }
}

// -----

module @test_create_packet_flows4 {
 aie.device(xcvc1902) {
// CHECK-LABEL: module @test_create_packet_flows4 {
// CHECK:         %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:         %[[VAL_1:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:           %[[VAL_6:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_7:.*]] = aie.amsel<1> (0)
// CHECK:           %[[VAL_4:.*]] = aie.masterset(Core : 0, %[[VAL_3:.*]])
// CHECK:           %[[VAL_5:.*]] = aie.masterset(Core : 1, %[[VAL_2:.*]])
// CHECK:           aie.packet_rules(West : 1) {
// CHECK:             aie.rule(31, 0, %[[VAL_2]])
// CHECK:           }
// CHECK:           aie.packet_rules(West : 0) {
// CHECK:             aie.rule(31, 1, %[[VAL_2]])
// CHECK:             aie.rule(31, 0, %[[VAL_3]])
// CHECK:           }
// CHECK:         }
// CHECK:       }
  %t11 = aie.tile(1, 1)

  aie.packet_flow(0x0) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 0>
  }

  aie.packet_flow(0x1) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 1>
  }

  aie.packet_flow(0x0) {
    aie.packet_source<%t11, West : 1>
    aie.packet_dest<%t11, Core : 1>
  }
 }
}

// -----

// many-to-one, 3 streams
module @test_create_packet_flows5 {
 aie.device(xcvc1902) {
  %t11 = aie.tile(1, 1)

  aie.packet_flow(0x0) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 0>
  }

  aie.packet_flow(0x1) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 0>
  }

  aie.packet_flow(0x2) {
    aie.packet_source<%t11, West : 1>
    aie.packet_dest<%t11, Core : 0>
  }
 }
}

// -----

// generating tile op declarations for tiles on the packetflow
// CHECK-LABEL:   module @test_create_packet_flows6 {
// CHECK:        %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:        %{{.*}} = aie.switchbox(%[[TILE_1_2]]) {
// CHECK:          %[[VAL0:.*]] = aie.amsel<0> (0)
// CHECK:          %{{.*}} = aie.masterset(South : 3, %[[VAL0]])
// CHECK:          aie.packet_rules(Trace : 0) {
// CHECK:            aie.rule(31, 1, %[[VAL0]])
// CHECK:          }
// CHECK:        }
// CHECK:        %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK:        %{{.*}} = aie.shim_mux(%[[TILE_0_0]]) {
// CHECK:          aie.connect<North : 3, DMA : 1>
// CHECK:        }
// CHECK:        %{{.*}} = aie.switchbox(%[[TILE_0_0]]) {
// CHECK:          %[[VAL1:.*]] = aie.amsel<0> (0)
// CHECK:          %{{.*}} = aie.masterset(South : 3, %[[VAL1]]) {keep_pkt_header = true}
// CHECK:          aie.packet_rules(East : 0) {
// CHECK:            aie.rule(31, 1, %[[VAL1]])
// CHECK:          }
// CHECK:        }
// CHECK:        aie.packet_flow(1) {
// CHECK:          aie.packet_source<%[[TILE_1_2]], Trace : 0>
// CHECK:          aie.packet_dest<%[[TILE_0_0]], DMA : 1>
// CHECK:        } {keep_pkt_header = true}
// CHECK:        %[[TILE_1_0:.*]] = aie.tile(1, 0)
// CHECK:        %{{.*}} = aie.switchbox(%[[TILE_1_0]]) {
// CHECK:          %[[VAL2:.*]] = aie.amsel<0> (0)
// CHECK:          %{{.*}} = aie.masterset(West : 0, %[[VAL2]])
// CHECK:          aie.packet_rules(North : 3) {
// CHECK:            aie.rule(31, 1, %[[VAL2]])
// CHECK:          }
// CHECK:        }
// CHECK:        %[[TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK:        %{{.*}} = aie.switchbox(%[[TILE_1_1]]) {
// CHECK:          %[[VAL3:.*]] = aie.amsel<0> (0)
// CHECK:          %{{.*}} = aie.masterset(South : 3, %[[VAL3]])
// CHECK:          aie.packet_rules(North : 3) {
// CHECK:            aie.rule(31, 1, %[[VAL3]])
// CHECK:          }
// CHECK:        }
module @test_create_packet_flows6 {
  aie.device(npu2) {
    %tile_1_2 = aie.tile(1, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.packet_flow(1) {
      aie.packet_source<%tile_1_2, Trace : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
  }
}

