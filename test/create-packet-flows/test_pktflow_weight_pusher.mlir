//===- test_pktflow_weight_pusher.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: stephenn
// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// FIXCore : nondeterministic

// This setup follows one of Phil's hand-crafted aie examples where packet-switched routing is used
// to stream data to the herd. Packet-switched is necessary since we have to distribute the data to
// quite many tiles.
// Check out "sixteen_tiles_*" examples.
// Two AIE tiles are used to push data to all the compute tiles in the
// herd. What is missing here is the DMA configuration to set up the DMA in PacketSwitch mode.
// Here we just concern with generating the packet-switched configuration automatically.
module @test_pktflow_weight_pusher {
 aie.device(xcvc1902) {
  // Herd "compute"
// CHECK:       module @test_pktflow_weight_pusher {

// CHECK:    %[[VAL_0:.*]] = aie.tile(2, 2)
// CHECK-NEXT:    %[[VAL_1:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:           %[[VAL_2:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_3:.*]] = aie.masterset(DMA : 1, %[[VAL_2]])
// CHECK:           aie.packet_rules(North : 0) {
// CHECK:             aie.rule(31, 0, %[[VAL_2]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_4:.*]] = aie.tile(3, 2)
// CHECK-NEXT:    %[[VAL_5:.*]] = aie.switchbox(%[[VAL_4]]) {
// CHECK:           %[[VAL_6:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_7:.*]] = aie.masterset(DMA : 1, %[[VAL_6]])
// CHECK:           aie.packet_rules(North : 0) {
// CHECK:             aie.rule(31, 4, %[[VAL_6]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_8:.*]] = aie.tile(4, 2)
// CHECK-NEXT:    %[[VAL_9:.*]] = aie.switchbox(%[[VAL_8]]) {
// CHECK:           %[[VAL_10:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_11:.*]] = aie.masterset(DMA : 1, %[[VAL_10]])
// CHECK:           aie.packet_rules(North : 0) {
// CHECK:             aie.rule(31, 8, %[[VAL_10]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_12:.*]] = aie.tile(5, 2)
// CHECK-NEXT:    %[[VAL_13:.*]] = aie.switchbox(%[[VAL_12]]) {
// CHECK:           %[[VAL_14:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_15:.*]] = aie.masterset(DMA : 1, %[[VAL_14]])
// CHECK:           aie.packet_rules(North : 0) {
// CHECK:             aie.rule(31, 12, %[[VAL_14]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_16:.*]] = aie.tile(2, 3)
// CHECK-NEXT:    %[[VAL_17:.*]] = aie.switchbox(%[[VAL_16]]) {
// CHECK:           %[[VAL_18:.*]] = aie.amsel<0> (1)
// CHECK:           %[[VAL_19:.*]] = aie.masterset(DMA : 1, %[[VAL_18]])
// CHECK:           %[[VAL_20:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_21:.*]] = aie.masterset(South : 0, %[[VAL_20]])
// CHECK:           aie.packet_rules(North : 0) {
// CHECK:             aie.rule(31, 1, %[[VAL_18]])
// CHECK:             aie.rule(31, 0, %[[VAL_20]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_22:.*]] = aie.tile(3, 3)
// CHECK-NEXT:    %[[VAL_23:.*]] = aie.switchbox(%[[VAL_22]]) {
// CHECK:           %[[VAL_24:.*]] = aie.amsel<0> (1)
// CHECK:           %[[VAL_25:.*]] = aie.masterset(DMA : 1, %[[VAL_24]])
// CHECK:           %[[VAL_26:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_27:.*]] = aie.masterset(South : 0, %[[VAL_26]])
// CHECK:           aie.packet_rules(North : 0) {
// CHECK:             aie.rule(31, 5, %[[VAL_24]])
// CHECK:             aie.rule(31, 4, %[[VAL_26]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_28:.*]] = aie.tile(4, 3)
// CHECK-NEXT:    %[[VAL_29:.*]] = aie.switchbox(%[[VAL_28]]) {
// CHECK:           %[[VAL_30:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_31:.*]] = aie.masterset(South : 0, %[[VAL_30]])
// CHECK:           %[[VAL_32:.*]] = aie.amsel<0> (1)
// CHECK:           %[[VAL_33:.*]] = aie.masterset(DMA : 1, %[[VAL_32]])
// CHECK:           aie.packet_rules(North : 0) {
// CHECK:             aie.rule(31, 9, %[[VAL_32]])
// CHECK:             aie.rule(31, 8, %[[VAL_30]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_34:.*]] = aie.tile(5, 3)
// CHECK-NEXT:    %[[VAL_35:.*]] = aie.switchbox(%[[VAL_34]]) {
// CHECK:           %[[VAL_36:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_37:.*]] = aie.masterset(DMA : 1, %[[VAL_36]])
// CHECK:           %[[VAL_38:.*]] = aie.amsel<0> (1)
// CHECK:           %[[VAL_39:.*]] = aie.masterset(South : 0, %[[VAL_38]])
// CHECK:           aie.packet_rules(North : 0) {
// CHECK:             aie.rule(31, 13, %[[VAL_36]])
// CHECK:             aie.rule(31, 12, %[[VAL_38]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_40:.*]] = aie.tile(2, 4)
// CHECK-NEXT:    %[[VAL_41:.*]] = aie.switchbox(%[[VAL_40]]) {
// CHECK:           %[[VAL_42:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_43:.*]] = aie.masterset(DMA : 1, %[[VAL_42]])
// CHECK:           %[[VAL_44:.*]] = aie.amsel<0> (1)
// CHECK:           %[[VAL_45:.*]] = aie.masterset(South : 0, %[[VAL_44]])
// CHECK:           aie.packet_rules(North : 0) {
// CHECK:             aie.rule(31, 2, %[[VAL_42]])
// CHECK:             aie.rule(30, 1, %[[VAL_44]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_46:.*]] = aie.tile(3, 4)
// CHECK-NEXT:    %[[VAL_47:.*]] = aie.switchbox(%[[VAL_46]]) {
// CHECK:           %[[VAL_48:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_49:.*]] = aie.masterset(South : 0, %[[VAL_48]])
// CHECK:           %[[VAL_50:.*]] = aie.amsel<0> (1)
// CHECK:           %[[VAL_51:.*]] = aie.masterset(DMA : 1, %[[VAL_50]])
// CHECK:           aie.packet_rules(North : 0) {
// CHECK:             aie.rule(31, 6, %[[VAL_50]])
// CHECK:             aie.rule(30, 5, %[[VAL_48]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_52:.*]] = aie.tile(4, 4)
// CHECK-NEXT:    %[[VAL_53:.*]] = aie.switchbox(%[[VAL_52]]) {
// CHECK:           %[[VAL_54:.*]] = aie.amsel<0> (1)
// CHECK:           %[[VAL_55:.*]] = aie.masterset(DMA : 1, %[[VAL_54]])
// CHECK:           %[[VAL_56:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_57:.*]] = aie.masterset(South : 0, %[[VAL_56]])
// CHECK:           aie.packet_rules(North : 0) {
// CHECK:             aie.rule(31, 10, %[[VAL_54]])
// CHECK:             aie.rule(30, 9, %[[VAL_56]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_58:.*]] = aie.tile(5, 4)
// CHECK-NEXT:    %[[VAL_59:.*]] = aie.switchbox(%[[VAL_58]]) {
// CHECK:           %[[VAL_60:.*]] = aie.amsel<0> (1)
// CHECK:           %[[VAL_61:.*]] = aie.masterset(DMA : 1, %[[VAL_60]])
// CHECK:           %[[VAL_62:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_63:.*]] = aie.masterset(South : 0, %[[VAL_62]])
// CHECK:           aie.packet_rules(North : 0) {
// CHECK:             aie.rule(31, 14, %[[VAL_60]])
// CHECK:             aie.rule(30, 13, %[[VAL_62]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_64:.*]] = aie.tile(2, 5)
// CHECK-NEXT:    %[[VAL_65:.*]] = aie.switchbox(%[[VAL_64]]) {
// CHECK:           %[[VAL_66:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_67:.*]] = aie.masterset(DMA : 1, %[[VAL_66]])
// CHECK:           %[[VAL_68:.*]] = aie.amsel<0> (1)
// CHECK:           %[[VAL_69:.*]] = aie.masterset(South : 0, %[[VAL_68]])
// CHECK:           aie.packet_rules(East : 0) {
// CHECK:             aie.rule(31, 3, %[[VAL_66]])
// CHECK:             aie.rule(28, 2, %[[VAL_68]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_70:.*]] = aie.tile(3, 5)
// CHECK-NEXT:    %[[VAL_71:.*]] = aie.switchbox(%[[VAL_70]]) {
// CHECK:           %[[VAL_72:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_73:.*]] = aie.masterset(West : 0, %[[VAL_72]])
// CHECK:           %[[VAL_74:.*]] = aie.amsel<0> (2)
// CHECK:           %[[VAL_75:.*]] = aie.masterset(South : 0, %[[VAL_74]])
// CHECK:           %[[VAL_76:.*]] = aie.amsel<0> (1)
// CHECK:           %[[VAL_77:.*]] = aie.masterset(DMA : 1, %[[VAL_76]])
// CHECK:           aie.packet_rules(East : 0) {
// CHECK:             aie.rule(31, 7, %[[VAL_76]])
// CHECK:             aie.rule(28, 6, %[[VAL_74]])
// CHECK:             aie.rule(28, 3, %[[VAL_72]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_78:.*]] = aie.tile(4, 5)
// CHECK-NEXT:    %[[VAL_79:.*]] = aie.switchbox(%[[VAL_78]]) {
// CHECK:           %[[VAL_80:.*]] = aie.amsel<0> (2)
// CHECK:           %[[VAL_81:.*]] = aie.masterset(South : 0, %[[VAL_80]])
// CHECK:           %[[VAL_82:.*]] = aie.amsel<0> (1)
// CHECK:           %[[VAL_83:.*]] = aie.masterset(DMA : 1, %[[VAL_82]])
// CHECK:           %[[VAL_84:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_85:.*]] = aie.masterset(West : 0, %[[VAL_84]])
// CHECK:           aie.packet_rules(East : 0) {
// CHECK:             aie.rule(31, 11, %[[VAL_82]])
// CHECK:             aie.rule(28, 10, %[[VAL_80]])
// CHECK:             aie.rule(24, 7, %[[VAL_84]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_86:.*]] = aie.tile(5, 5)
// CHECK-NEXT:    %[[VAL_87:.*]] = aie.switchbox(%[[VAL_86]]) {
// CHECK:           %[[VAL_88:.*]] = aie.amsel<0> (1)
// CHECK:           %[[VAL_89:.*]] = aie.masterset(West : 0, %[[VAL_88]])
// CHECK:           %[[VAL_90:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_91:.*]] = aie.masterset(DMA : 1, %[[VAL_90]])
// CHECK:           %[[VAL_92:.*]] = aie.amsel<0> (2)
// CHECK:           %[[VAL_93:.*]] = aie.masterset(South : 0, %[[VAL_92]])
// CHECK:           aie.packet_rules(East : 0) {
// CHECK:             aie.rule(31, 15, %[[VAL_90]])
// CHECK:             aie.rule(28, 14, %[[VAL_92]])
// CHECK:             aie.rule(16, 11, %[[VAL_88]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_94:.*]] = aie.tile(6, 5)
// CHECK-NEXT:    %[[VAL_95:.*]] = aie.switchbox(%[[VAL_94]]) {
// CHECK:           %[[VAL_96:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_97:.*]] = aie.masterset(West : 0, %[[VAL_96]])
// CHECK:           aie.packet_rules(East : 0) {
// CHECK:             aie.rule(24, 15, %[[VAL_96]])
// CHECK:           }
// CHECK:           aie.packet_rules(DMA : 0) {
// CHECK:             aie.rule(24, 7, %[[VAL_96]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_98:.*]] = aie.tile(7, 5)
// CHECK-NEXT:    %[[VAL_99:.*]] = aie.switchbox(%[[VAL_98]]) {
// CHECK:           %[[VAL_100:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_101:.*]] = aie.masterset(West : 0, %[[VAL_100]])
// CHECK:           aie.packet_rules(DMA : 0) {
// CHECK:             aie.rule(24, 15, %[[VAL_100]])
// CHECK:           }
// CHECK:         }
// CHECK:       }
  %tile22 = aie.tile(2, 2) // 5'b0_0000
  %tile32 = aie.tile(3, 2) // 5'b0_0100
  %tile42 = aie.tile(4, 2) // 5'b0_1000
  %tile52 = aie.tile(5, 2) // 5'b0_1100

  %tile23 = aie.tile(2, 3) // 5'b0_0001
  %tile33 = aie.tile(3, 3) // 5'b0_0101
  %tile43 = aie.tile(4, 3) // 5'b0_1001
  %tile53 = aie.tile(5, 3) // 5'b0_1101

  %tile24 = aie.tile(2, 4) // 5'b0_0010
  %tile34 = aie.tile(3, 4) // 5'b0_0110
  %tile44 = aie.tile(4, 4) // 5'b0_1010
  %tile54 = aie.tile(5, 4) // 5'b0_1110

  %tile25 = aie.tile(2, 5) // 5'b0_0011
  %tile35 = aie.tile(3, 5) // 5'b0_0111
  %tile45 = aie.tile(4, 5) // 5'b0_1011
  %tile55 = aie.tile(5, 5) // 5'b0_1111

  // Herd "weight"
  %tile65 = aie.tile(6, 5)
  %tile75 = aie.tile(7, 5)


  // Tile (6, 5) streams data to the first two columns of herd "compute"
  // Tile (7, 5) streams data to the next two columns of herd "compute"
  //
  //  (2, 5)--(3, 5)--(4, 5)--(5, 5) < --(6, 5) <-- (7, 5)
  //    |       |       |       |
  //  (2, 4)--(3, 4)--(4, 4)--(5, 4)
  //    |       |       |       |
  //  (2, 3)--(3, 3)--(4, 3)--(5, 3)
  //    |       |       |       |
  //  (2, 2)--(3, 2)--(4, 2)--(5, 2)
  //

  // weight[0]: 0 - 7
  aie.packet_flow(0x0) {
    aie.packet_source<%tile65, DMA : 0>
    aie.packet_dest<%tile22, DMA : 1>
  }

  aie.packet_flow(0x1) {
    aie.packet_source<%tile65, DMA : 0>
    aie.packet_dest<%tile23, DMA : 1>
  }

  aie.packet_flow(0x2) {
    aie.packet_source<%tile65, DMA : 0>
    aie.packet_dest<%tile24, DMA : 1>
  }

  aie.packet_flow(0x3) {
    aie.packet_source<%tile65, DMA : 0>
    aie.packet_dest<%tile25, DMA : 1>
  }

  aie.packet_flow(0x4) {
    aie.packet_source<%tile65, DMA : 0>
    aie.packet_dest<%tile32, DMA : 1>
  }

  aie.packet_flow(0x5) {
    aie.packet_source<%tile65, DMA : 0>
    aie.packet_dest<%tile33, DMA : 1>
  }

  aie.packet_flow(0x6) {
    aie.packet_source<%tile65, DMA : 0>
    aie.packet_dest<%tile34, DMA : 1>
  }

  aie.packet_flow(0x7) {
    aie.packet_source<%tile65, DMA : 0>
    aie.packet_dest<%tile35, DMA : 1>
  }

  // weight[1]: 8 - 15
  aie.packet_flow(0x8) {
    aie.packet_source<%tile75, DMA : 0>
    aie.packet_dest<%tile42, DMA : 1>
  }

  aie.packet_flow(0x9) {
    aie.packet_source<%tile75, DMA : 0>
    aie.packet_dest<%tile43, DMA : 1>
  }

  aie.packet_flow(0xa) {
    aie.packet_source<%tile75, DMA : 0>
    aie.packet_dest<%tile44, DMA : 1>
  }

  aie.packet_flow(0xb) {
    aie.packet_source<%tile75, DMA : 0>
    aie.packet_dest<%tile45, DMA : 1>
  }

  aie.packet_flow(0xc) {
    aie.packet_source<%tile75, DMA : 0>
    aie.packet_dest<%tile52, DMA : 1>
  }

  aie.packet_flow(0xd) {
    aie.packet_source<%tile75, DMA : 0>
    aie.packet_dest<%tile53, DMA : 1>
  }

  aie.packet_flow(0xe) {
    aie.packet_source<%tile75, DMA : 0>
    aie.packet_dest<%tile54, DMA : 1>
  }

  aie.packet_flow(0xf) {
    aie.packet_source<%tile75, DMA : 0>
    aie.packet_dest<%tile55, DMA : 1>
  }

 }
}
