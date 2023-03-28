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
// RUN: aie-opt --aie-create-packet-flows %s | FileCheck %s

// FIXCore : nondeterministic

// This setup follows one of Phil's hand-crafted aie examples where packet-switched routing is used
// to stream data to the herd. Packet-switched is necessary since we have to distribute the data to
// quite many tiles.
// Check out "sixteen_tiles_*" examples.
// Two AIE tiles are used to push data to all the compute tiles in the
// herd. What is missing here is the DMA configuration to set up the DMA in PacketSwitch mode.
// Here we just concern with generating the packet-switched configuration automatically.
module @test_pktflow_weight_pusher {
 AIE.device(xcvc1902) {
  // Herd "compute"
// CHECK:       module @test_pktflow_weight_pusher {

// CHECK:    %[[VAL_0:.*]] = AIE.tile(2, 2)
// CHECK-NEXT:    %[[VAL_1:.*]] = AIE.switchbox(%[[VAL_0]]) {
// CHECK:           %[[VAL_2:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_3:.*]] = AIE.masterset(DMA : 1, %[[VAL_2]])
// CHECK:           AIE.packetrules(North : 0) {
// CHECK:             AIE.rule(31, 0, %[[VAL_2]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_4:.*]] = AIE.tile(3, 2)
// CHECK-NEXT:    %[[VAL_5:.*]] = AIE.switchbox(%[[VAL_4]]) {
// CHECK:           %[[VAL_6:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_7:.*]] = AIE.masterset(DMA : 1, %[[VAL_6]])
// CHECK:           AIE.packetrules(North : 0) {
// CHECK:             AIE.rule(31, 4, %[[VAL_6]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_8:.*]] = AIE.tile(4, 2)
// CHECK-NEXT:    %[[VAL_9:.*]] = AIE.switchbox(%[[VAL_8]]) {
// CHECK:           %[[VAL_10:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_11:.*]] = AIE.masterset(DMA : 1, %[[VAL_10]])
// CHECK:           AIE.packetrules(North : 0) {
// CHECK:             AIE.rule(31, 8, %[[VAL_10]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_12:.*]] = AIE.tile(5, 2)
// CHECK-NEXT:    %[[VAL_13:.*]] = AIE.switchbox(%[[VAL_12]]) {
// CHECK:           %[[VAL_14:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_15:.*]] = AIE.masterset(DMA : 1, %[[VAL_14]])
// CHECK:           AIE.packetrules(North : 0) {
// CHECK:             AIE.rule(31, 12, %[[VAL_14]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_16:.*]] = AIE.tile(2, 3)
// CHECK-NEXT:    %[[VAL_17:.*]] = AIE.switchbox(%[[VAL_16]]) {
// CHECK:           %[[VAL_18:.*]] = AIE.amsel<0> (1)
// CHECK:           %[[VAL_19:.*]] = AIE.masterset(DMA : 1, %[[VAL_18]])
// CHECK:           %[[VAL_20:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_21:.*]] = AIE.masterset(South : 0, %[[VAL_20]])
// CHECK:           AIE.packetrules(North : 0) {
// CHECK:             AIE.rule(31, 1, %[[VAL_18]])
// CHECK:             AIE.rule(31, 0, %[[VAL_20]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_22:.*]] = AIE.tile(3, 3)
// CHECK-NEXT:    %[[VAL_23:.*]] = AIE.switchbox(%[[VAL_22]]) {
// CHECK:           %[[VAL_24:.*]] = AIE.amsel<0> (1)
// CHECK:           %[[VAL_25:.*]] = AIE.masterset(DMA : 1, %[[VAL_24]])
// CHECK:           %[[VAL_26:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_27:.*]] = AIE.masterset(South : 0, %[[VAL_26]])
// CHECK:           AIE.packetrules(North : 0) {
// CHECK:             AIE.rule(31, 5, %[[VAL_24]])
// CHECK:             AIE.rule(31, 4, %[[VAL_26]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_28:.*]] = AIE.tile(4, 3)
// CHECK-NEXT:    %[[VAL_29:.*]] = AIE.switchbox(%[[VAL_28]]) {
// CHECK:           %[[VAL_30:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_31:.*]] = AIE.masterset(South : 0, %[[VAL_30]])
// CHECK:           %[[VAL_32:.*]] = AIE.amsel<0> (1)
// CHECK:           %[[VAL_33:.*]] = AIE.masterset(DMA : 1, %[[VAL_32]])
// CHECK:           AIE.packetrules(North : 0) {
// CHECK:             AIE.rule(31, 9, %[[VAL_32]])
// CHECK:             AIE.rule(31, 8, %[[VAL_30]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_34:.*]] = AIE.tile(5, 3)
// CHECK-NEXT:    %[[VAL_35:.*]] = AIE.switchbox(%[[VAL_34]]) {
// CHECK:           %[[VAL_36:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_37:.*]] = AIE.masterset(DMA : 1, %[[VAL_36]])
// CHECK:           %[[VAL_38:.*]] = AIE.amsel<0> (1)
// CHECK:           %[[VAL_39:.*]] = AIE.masterset(South : 0, %[[VAL_38]])
// CHECK:           AIE.packetrules(North : 0) {
// CHECK:             AIE.rule(31, 13, %[[VAL_36]])
// CHECK:             AIE.rule(31, 12, %[[VAL_38]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_40:.*]] = AIE.tile(2, 4)
// CHECK-NEXT:    %[[VAL_41:.*]] = AIE.switchbox(%[[VAL_40]]) {
// CHECK:           %[[VAL_42:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_43:.*]] = AIE.masterset(DMA : 1, %[[VAL_42]])
// CHECK:           %[[VAL_44:.*]] = AIE.amsel<0> (1)
// CHECK:           %[[VAL_45:.*]] = AIE.masterset(South : 0, %[[VAL_44]])
// CHECK:           AIE.packetrules(North : 0) {
// CHECK:             AIE.rule(31, 2, %[[VAL_42]])
// CHECK:             AIE.rule(30, 1, %[[VAL_44]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_46:.*]] = AIE.tile(3, 4)
// CHECK-NEXT:    %[[VAL_47:.*]] = AIE.switchbox(%[[VAL_46]]) {
// CHECK:           %[[VAL_48:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_49:.*]] = AIE.masterset(South : 0, %[[VAL_48]])
// CHECK:           %[[VAL_50:.*]] = AIE.amsel<0> (1)
// CHECK:           %[[VAL_51:.*]] = AIE.masterset(DMA : 1, %[[VAL_50]])
// CHECK:           AIE.packetrules(North : 0) {
// CHECK:             AIE.rule(31, 6, %[[VAL_50]])
// CHECK:             AIE.rule(30, 5, %[[VAL_48]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_52:.*]] = AIE.tile(4, 4)
// CHECK-NEXT:    %[[VAL_53:.*]] = AIE.switchbox(%[[VAL_52]]) {
// CHECK:           %[[VAL_54:.*]] = AIE.amsel<0> (1)
// CHECK:           %[[VAL_55:.*]] = AIE.masterset(DMA : 1, %[[VAL_54]])
// CHECK:           %[[VAL_56:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_57:.*]] = AIE.masterset(South : 0, %[[VAL_56]])
// CHECK:           AIE.packetrules(North : 0) {
// CHECK:             AIE.rule(31, 10, %[[VAL_54]])
// CHECK:             AIE.rule(30, 9, %[[VAL_56]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_58:.*]] = AIE.tile(5, 4)
// CHECK-NEXT:    %[[VAL_59:.*]] = AIE.switchbox(%[[VAL_58]]) {
// CHECK:           %[[VAL_60:.*]] = AIE.amsel<0> (1)
// CHECK:           %[[VAL_61:.*]] = AIE.masterset(DMA : 1, %[[VAL_60]])
// CHECK:           %[[VAL_62:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_63:.*]] = AIE.masterset(South : 0, %[[VAL_62]])
// CHECK:           AIE.packetrules(North : 0) {
// CHECK:             AIE.rule(31, 14, %[[VAL_60]])
// CHECK:             AIE.rule(30, 13, %[[VAL_62]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_64:.*]] = AIE.tile(2, 5)
// CHECK-NEXT:    %[[VAL_65:.*]] = AIE.switchbox(%[[VAL_64]]) {
// CHECK:           %[[VAL_66:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_67:.*]] = AIE.masterset(DMA : 1, %[[VAL_66]])
// CHECK:           %[[VAL_68:.*]] = AIE.amsel<0> (1)
// CHECK:           %[[VAL_69:.*]] = AIE.masterset(South : 0, %[[VAL_68]])
// CHECK:           AIE.packetrules(East : 0) {
// CHECK:             AIE.rule(31, 3, %[[VAL_66]])
// CHECK:             AIE.rule(28, 2, %[[VAL_68]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_70:.*]] = AIE.tile(3, 5)
// CHECK-NEXT:    %[[VAL_71:.*]] = AIE.switchbox(%[[VAL_70]]) {
// CHECK:           %[[VAL_72:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_73:.*]] = AIE.masterset(West : 0, %[[VAL_72]])
// CHECK:           %[[VAL_74:.*]] = AIE.amsel<0> (2)
// CHECK:           %[[VAL_75:.*]] = AIE.masterset(South : 0, %[[VAL_74]])
// CHECK:           %[[VAL_76:.*]] = AIE.amsel<0> (1)
// CHECK:           %[[VAL_77:.*]] = AIE.masterset(DMA : 1, %[[VAL_76]])
// CHECK:           AIE.packetrules(East : 0) {
// CHECK:             AIE.rule(31, 7, %[[VAL_76]])
// CHECK:             AIE.rule(28, 6, %[[VAL_74]])
// CHECK:             AIE.rule(28, 3, %[[VAL_72]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_78:.*]] = AIE.tile(4, 5)
// CHECK-NEXT:    %[[VAL_79:.*]] = AIE.switchbox(%[[VAL_78]]) {
// CHECK:           %[[VAL_80:.*]] = AIE.amsel<0> (2)
// CHECK:           %[[VAL_81:.*]] = AIE.masterset(South : 0, %[[VAL_80]])
// CHECK:           %[[VAL_82:.*]] = AIE.amsel<0> (1)
// CHECK:           %[[VAL_83:.*]] = AIE.masterset(DMA : 1, %[[VAL_82]])
// CHECK:           %[[VAL_84:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_85:.*]] = AIE.masterset(West : 0, %[[VAL_84]])
// CHECK:           AIE.packetrules(East : 0) {
// CHECK:             AIE.rule(31, 11, %[[VAL_82]])
// CHECK:             AIE.rule(28, 10, %[[VAL_80]])
// CHECK:             AIE.rule(24, 7, %[[VAL_84]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_86:.*]] = AIE.tile(5, 5)
// CHECK-NEXT:    %[[VAL_87:.*]] = AIE.switchbox(%[[VAL_86]]) {
// CHECK:           %[[VAL_88:.*]] = AIE.amsel<0> (1)
// CHECK:           %[[VAL_89:.*]] = AIE.masterset(West : 0, %[[VAL_88]])
// CHECK:           %[[VAL_90:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_91:.*]] = AIE.masterset(DMA : 1, %[[VAL_90]])
// CHECK:           %[[VAL_92:.*]] = AIE.amsel<0> (2)
// CHECK:           %[[VAL_93:.*]] = AIE.masterset(South : 0, %[[VAL_92]])
// CHECK:           AIE.packetrules(East : 0) {
// CHECK:             AIE.rule(31, 15, %[[VAL_90]])
// CHECK:             AIE.rule(28, 14, %[[VAL_92]])
// CHECK:             AIE.rule(16, 11, %[[VAL_88]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_94:.*]] = AIE.tile(6, 5)
// CHECK-NEXT:    %[[VAL_95:.*]] = AIE.switchbox(%[[VAL_94]]) {
// CHECK:           %[[VAL_96:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_97:.*]] = AIE.masterset(West : 0, %[[VAL_96]])
// CHECK:           AIE.packetrules(East : 0) {
// CHECK:             AIE.rule(24, 15, %[[VAL_96]])
// CHECK:           }
// CHECK:           AIE.packetrules(DMA : 0) {
// CHECK:             AIE.rule(24, 7, %[[VAL_96]])
// CHECK:           }
// CHECK:         }

// CHECK:    %[[VAL_98:.*]] = AIE.tile(7, 5)
// CHECK-NEXT:    %[[VAL_99:.*]] = AIE.switchbox(%[[VAL_98]]) {
// CHECK:           %[[VAL_100:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_101:.*]] = AIE.masterset(West : 0, %[[VAL_100]])
// CHECK:           AIE.packetrules(DMA : 0) {
// CHECK:             AIE.rule(24, 15, %[[VAL_100]])
// CHECK:           }
// CHECK:         }
// CHECK:       }
  %tile22 = AIE.tile(2, 2) // 5'b0_0000
  %tile32 = AIE.tile(3, 2) // 5'b0_0100
  %tile42 = AIE.tile(4, 2) // 5'b0_1000
  %tile52 = AIE.tile(5, 2) // 5'b0_1100

  %tile23 = AIE.tile(2, 3) // 5'b0_0001
  %tile33 = AIE.tile(3, 3) // 5'b0_0101
  %tile43 = AIE.tile(4, 3) // 5'b0_1001
  %tile53 = AIE.tile(5, 3) // 5'b0_1101

  %tile24 = AIE.tile(2, 4) // 5'b0_0010
  %tile34 = AIE.tile(3, 4) // 5'b0_0110
  %tile44 = AIE.tile(4, 4) // 5'b0_1010
  %tile54 = AIE.tile(5, 4) // 5'b0_1110

  %tile25 = AIE.tile(2, 5) // 5'b0_0011
  %tile35 = AIE.tile(3, 5) // 5'b0_0111
  %tile45 = AIE.tile(4, 5) // 5'b0_1011
  %tile55 = AIE.tile(5, 5) // 5'b0_1111

  // Herd "weight"
  %tile65 = AIE.tile(6, 5)
  %tile75 = AIE.tile(7, 5)


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
  AIE.packet_flow(0x0) {
    AIE.packet_source<%tile65, DMA : 0>
    AIE.packet_dest<%tile22, DMA : 1>
  }

  AIE.packet_flow(0x1) {
    AIE.packet_source<%tile65, DMA : 0>
    AIE.packet_dest<%tile23, DMA : 1>
  }

  AIE.packet_flow(0x2) {
    AIE.packet_source<%tile65, DMA : 0>
    AIE.packet_dest<%tile24, DMA : 1>
  }

  AIE.packet_flow(0x3) {
    AIE.packet_source<%tile65, DMA : 0>
    AIE.packet_dest<%tile25, DMA : 1>
  }

  AIE.packet_flow(0x4) {
    AIE.packet_source<%tile65, DMA : 0>
    AIE.packet_dest<%tile32, DMA : 1>
  }

  AIE.packet_flow(0x5) {
    AIE.packet_source<%tile65, DMA : 0>
    AIE.packet_dest<%tile33, DMA : 1>
  }

  AIE.packet_flow(0x6) {
    AIE.packet_source<%tile65, DMA : 0>
    AIE.packet_dest<%tile34, DMA : 1>
  }

  AIE.packet_flow(0x7) {
    AIE.packet_source<%tile65, DMA : 0>
    AIE.packet_dest<%tile35, DMA : 1>
  }

  // weight[1]: 8 - 15
  AIE.packet_flow(0x8) {
    AIE.packet_source<%tile75, DMA : 0>
    AIE.packet_dest<%tile42, DMA : 1>
  }

  AIE.packet_flow(0x9) {
    AIE.packet_source<%tile75, DMA : 0>
    AIE.packet_dest<%tile43, DMA : 1>
  }

  AIE.packet_flow(0xa) {
    AIE.packet_source<%tile75, DMA : 0>
    AIE.packet_dest<%tile44, DMA : 1>
  }

  AIE.packet_flow(0xb) {
    AIE.packet_source<%tile75, DMA : 0>
    AIE.packet_dest<%tile45, DMA : 1>
  }

  AIE.packet_flow(0xc) {
    AIE.packet_source<%tile75, DMA : 0>
    AIE.packet_dest<%tile52, DMA : 1>
  }

  AIE.packet_flow(0xd) {
    AIE.packet_source<%tile75, DMA : 0>
    AIE.packet_dest<%tile53, DMA : 1>
  }

  AIE.packet_flow(0xe) {
    AIE.packet_source<%tile75, DMA : 0>
    AIE.packet_dest<%tile54, DMA : 1>
  }

  AIE.packet_flow(0xf) {
    AIE.packet_source<%tile75, DMA : 0>
    AIE.packet_dest<%tile55, DMA : 1>
  }

 }
}
