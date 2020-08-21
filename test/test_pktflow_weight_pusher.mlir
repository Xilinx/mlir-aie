// RUN: aie-opt --aie-create-packet-flows %s | FileCheck %s

// CHECK-LABEL: module @test_pktflow_weight_pusher {
// CHECK:   %0 = AIE.tile(2, 2)
// CHECK:   %1 = AIE.switchbox(%0) {
// CHECK:     %36 = AIE.amsel<0> (0)
// CHECK:     %37 = AIE.masterset("DMA" : 1, %36)
// CHECK:     AIE.packetrules("North" : 0) {
// CHECK:       AIE.rule(31, 0, %36)
// CHECK:     }
// CHECK:   }
// CHECK:   %2 = AIE.tile(3, 2)
// CHECK:   %3 = AIE.switchbox(%2) {
// CHECK:     %36 = AIE.amsel<0> (0)
// CHECK:     %37 = AIE.masterset("DMA" : 1, %36)
// CHECK:     AIE.packetrules("North" : 0) {
// CHECK:       AIE.rule(31, 4, %36)
// CHECK:     }
// CHECK:   }
// CHECK:   %4 = AIE.tile(4, 2)
// CHECK:   %5 = AIE.switchbox(%4) {
// CHECK:     %36 = AIE.amsel<0> (0)
// CHECK:     %37 = AIE.masterset("DMA" : 1, %36)
// CHECK:     AIE.packetrules("North" : 0) {
// CHECK:       AIE.rule(31, 8, %36)
// CHECK:     }
// CHECK:   }
// CHECK:   %6 = AIE.tile(5, 2)
// CHECK:   %7 = AIE.switchbox(%6) {
// CHECK:     %36 = AIE.amsel<0> (0)
// CHECK:     %37 = AIE.masterset("DMA" : 1, %36)
// CHECK:     AIE.packetrules("North" : 0) {
// CHECK:       AIE.rule(31, 12, %36)
// CHECK:     }
// CHECK:   }
// CHECK:   %8 = AIE.tile(2, 3)
// CHECK:   %9 = AIE.switchbox(%8) {
// CHECK:     %36 = AIE.amsel<0> (0)
// CHECK:     %37 = AIE.masterset("South" : 0, %36)
// CHECK:     %38 = AIE.amsel<0> (1)
// CHECK:     %39 = AIE.masterset("DMA" : 1, %38)
// CHECK:     AIE.packetrules("North" : 0) {
// CHECK:       AIE.rule(31, 1, %38)
// CHECK:       AIE.rule(31, 0, %36)
// CHECK:     }
// CHECK:   }
// CHECK:   %10 = AIE.tile(3, 3)
// CHECK:   %11 = AIE.switchbox(%10) {
// CHECK:     %36 = AIE.amsel<0> (0)
// CHECK:     %37 = AIE.masterset("South" : 0, %36)
// CHECK:     %38 = AIE.amsel<0> (1)
// CHECK:     %39 = AIE.masterset("DMA" : 1, %38)
// CHECK:     AIE.packetrules("North" : 0) {
// CHECK:       AIE.rule(31, 5, %38)
// CHECK:       AIE.rule(31, 4, %36)
// CHECK:     }
// CHECK:   }
// CHECK:   %12 = AIE.tile(4, 3)
// CHECK:   %13 = AIE.switchbox(%12) {
// CHECK:     %36 = AIE.amsel<0> (0)
// CHECK:     %37 = AIE.masterset("South" : 0, %36)
// CHECK:     %38 = AIE.amsel<0> (1)
// CHECK:     %39 = AIE.masterset("DMA" : 1, %38)
// CHECK:     AIE.packetrules("North" : 0) {
// CHECK:       AIE.rule(31, 9, %38)
// CHECK:       AIE.rule(31, 8, %36)
// CHECK:     }
// CHECK:   }
// CHECK:   %14 = AIE.tile(5, 3)
// CHECK:   %15 = AIE.switchbox(%14) {
// CHECK:     %36 = AIE.amsel<0> (0)
// CHECK:     %37 = AIE.masterset("South" : 0, %36)
// CHECK:     %38 = AIE.amsel<0> (1)
// CHECK:     %39 = AIE.masterset("DMA" : 1, %38)
// CHECK:     AIE.packetrules("North" : 0) {
// CHECK:       AIE.rule(31, 13, %38)
// CHECK:       AIE.rule(31, 12, %36)
// CHECK:     }
// CHECK:   }
// CHECK:   %16 = AIE.tile(2, 4)
// CHECK:   %17 = AIE.switchbox(%16) {
// CHECK:     %36 = AIE.amsel<0> (1)
// CHECK:     %37 = AIE.masterset("DMA" : 1, %36)
// CHECK:     %38 = AIE.amsel<0> (0)
// CHECK:     %39 = AIE.masterset("South" : 0, %38)
// CHECK:     AIE.packetrules("North" : 0) {
// CHECK:       AIE.rule(31, 2, %36)
// CHECK:       AIE.rule(30, 1, %38)
// CHECK:     }
// CHECK:   }
// CHECK:   %18 = AIE.tile(3, 4)
// CHECK:   %19 = AIE.switchbox(%18) {
// CHECK:     %36 = AIE.amsel<0> (1)
// CHECK:     %37 = AIE.masterset("DMA" : 1, %36)
// CHECK:     %38 = AIE.amsel<0> (0)
// CHECK:     %39 = AIE.masterset("South" : 0, %38)
// CHECK:     AIE.packetrules("North" : 0) {
// CHECK:       AIE.rule(31, 6, %36)
// CHECK:       AIE.rule(30, 5, %38)
// CHECK:     }
// CHECK:   }
// CHECK:   %20 = AIE.tile(4, 4)
// CHECK:   %21 = AIE.switchbox(%20) {
// CHECK:     %36 = AIE.amsel<0> (0)
// CHECK:     %37 = AIE.masterset("South" : 0, %36)
// CHECK:     %38 = AIE.amsel<0> (1)
// CHECK:     %39 = AIE.masterset("DMA" : 1, %38)
// CHECK:     AIE.packetrules("North" : 0) {
// CHECK:       AIE.rule(31, 10, %38)
// CHECK:       AIE.rule(30, 9, %36)
// CHECK:     }
// CHECK:   }
// CHECK:   %22 = AIE.tile(5, 4)
// CHECK:   %23 = AIE.switchbox(%22) {
// CHECK:     %36 = AIE.amsel<0> (0)
// CHECK:     %37 = AIE.masterset("DMA" : 1, %36)
// CHECK:     %38 = AIE.amsel<0> (1)
// CHECK:     %39 = AIE.masterset("South" : 0, %38)
// CHECK:     AIE.packetrules("North" : 0) {
// CHECK:       AIE.rule(31, 14, %36)
// CHECK:       AIE.rule(30, 13, %38)
// CHECK:     }
// CHECK:   }
// CHECK:   %24 = AIE.tile(2, 5)
// CHECK:   %25 = AIE.switchbox(%24) {
// CHECK:     %36 = AIE.amsel<0> (0)
// CHECK:     %37 = AIE.masterset("South" : 0, %36)
// CHECK:     %38 = AIE.amsel<0> (1)
// CHECK:     %39 = AIE.masterset("DMA" : 1, %38)
// CHECK:     AIE.packetrules("East" : 0) {
// CHECK:       AIE.rule(31, 3, %38)
// CHECK:       AIE.rule(28, 2, %36)
// CHECK:     }
// CHECK:   }
// CHECK:   %26 = AIE.tile(3, 5)
// CHECK:   %27 = AIE.switchbox(%26) {
// CHECK:     %36 = AIE.amsel<0> (2)
// CHECK:     %37 = AIE.masterset("DMA" : 1, %36)
// CHECK:     %38 = AIE.amsel<0> (0)
// CHECK:     %39 = AIE.masterset("South" : 0, %38)
// CHECK:     %40 = AIE.amsel<0> (1)
// CHECK:     %41 = AIE.masterset("West" : 0, %40)
// CHECK:     AIE.packetrules("East" : 0) {
// CHECK:       AIE.rule(31, 7, %36)
// CHECK:       AIE.rule(28, 6, %38)
// CHECK:       AIE.rule(28, 3, %40)
// CHECK:     }
// CHECK:   }
// CHECK:   %28 = AIE.tile(4, 5)
// CHECK:   %29 = AIE.switchbox(%28) {
// CHECK:     %36 = AIE.amsel<0> (2)
// CHECK:     %37 = AIE.masterset("DMA" : 1, %36)
// CHECK:     %38 = AIE.amsel<0> (0)
// CHECK:     %39 = AIE.masterset("West" : 0, %38)
// CHECK:     %40 = AIE.amsel<0> (1)
// CHECK:     %41 = AIE.masterset("South" : 0, %40)
// CHECK:     AIE.packetrules("East" : 0) {
// CHECK:       AIE.rule(31, 11, %36)
// CHECK:       AIE.rule(28, 10, %40)
// CHECK:       AIE.rule(24, 7, %38)
// CHECK:     }
// CHECK:   }
// CHECK:   %30 = AIE.tile(5, 5)
// CHECK:   %31 = AIE.switchbox(%30) {
// CHECK:     %36 = AIE.amsel<0> (2)
// CHECK:     %37 = AIE.masterset("DMA" : 1, %36)
// CHECK:     %38 = AIE.amsel<0> (1)
// CHECK:     %39 = AIE.masterset("South" : 0, %38)
// CHECK:     %40 = AIE.amsel<0> (0)
// CHECK:     %41 = AIE.masterset("West" : 0, %40)
// CHECK:     AIE.packetrules("East" : 0) {
// CHECK:       AIE.rule(31, 15, %36)
// CHECK:       AIE.rule(28, 14, %38)
// CHECK:       AIE.rule(16, 11, %40)
// CHECK:     }
// CHECK:   }
// CHECK:   %32 = AIE.tile(6, 5)
// CHECK:   %33 = AIE.switchbox(%32) {
// CHECK:     %36 = AIE.amsel<0> (0)
// CHECK:     %37 = AIE.masterset("West" : 0, %36)
// CHECK:     AIE.packetrules("East" : 0) {
// CHECK:       AIE.rule(24, 15, %36)
// CHECK:     }
// CHECK:     AIE.packetrules("DMA" : 0) {
// CHECK:       AIE.rule(24, 7, %36)
// CHECK:     }
// CHECK:   }
// CHECK:   %34 = AIE.tile(7, 5)
// CHECK:   %35 = AIE.switchbox(%34) {
// CHECK:     %36 = AIE.amsel<0> (0)
// CHECK:     %37 = AIE.masterset("West" : 0, %36)
// CHECK:     AIE.packetrules("DMA" : 0) {
// CHECK:       AIE.rule(24, 15, %36)
// CHECK:     }
// CHECK:   }
// CHECK: }

// This setup follows one of Phil's hand-crafted aie examples where packet-switched routing is used
// to stream data to the herd. Packet-switched is necessary since we have to distribute the data to
// quite many tiles.
// Check out "sixteen_tiles_*" examples.
// Two AIE tiles are used to push data to all the compute tiles in the
// herd. What is missing here is the DMA configuration to set up the DMA in PacketSwitch mode.
// Here we just concern with generating the packet-switched configuration automatically.
module @test_pktflow_weight_pusher {

  // Herd "compute"
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
    AIE.packet_source<%tile65, "DMA" : 0>
    AIE.packet_dest<%tile22, "DMA" : 1>
  }

  AIE.packet_flow(0x1) {
    AIE.packet_source<%tile65, "DMA" : 0>
    AIE.packet_dest<%tile23, "DMA" : 1>
  }

  AIE.packet_flow(0x2) {
    AIE.packet_source<%tile65, "DMA" : 0>
    AIE.packet_dest<%tile24, "DMA" : 1>
  }

  AIE.packet_flow(0x3) {
    AIE.packet_source<%tile65, "DMA" : 0>
    AIE.packet_dest<%tile25, "DMA" : 1>
  }

  AIE.packet_flow(0x4) {
    AIE.packet_source<%tile65, "DMA" : 0>
    AIE.packet_dest<%tile32, "DMA" : 1>
  }

  AIE.packet_flow(0x5) {
    AIE.packet_source<%tile65, "DMA" : 0>
    AIE.packet_dest<%tile33, "DMA" : 1>
  }

  AIE.packet_flow(0x6) {
    AIE.packet_source<%tile65, "DMA" : 0>
    AIE.packet_dest<%tile34, "DMA" : 1>
  }

  AIE.packet_flow(0x7) {
    AIE.packet_source<%tile65, "DMA" : 0>
    AIE.packet_dest<%tile35, "DMA" : 1>
  }

  // weight[1]: 8 - 15
  AIE.packet_flow(0x8) {
    AIE.packet_source<%tile75, "DMA" : 0>
    AIE.packet_dest<%tile42, "DMA" : 1>
  }

  AIE.packet_flow(0x9) {
    AIE.packet_source<%tile75, "DMA" : 0>
    AIE.packet_dest<%tile43, "DMA" : 1>
  }

  AIE.packet_flow(0xa) {
    AIE.packet_source<%tile75, "DMA" : 0>
    AIE.packet_dest<%tile44, "DMA" : 1>
  }

  AIE.packet_flow(0xb) {
    AIE.packet_source<%tile75, "DMA" : 0>
    AIE.packet_dest<%tile45, "DMA" : 1>
  }

  AIE.packet_flow(0xc) {
    AIE.packet_source<%tile75, "DMA" : 0>
    AIE.packet_dest<%tile52, "DMA" : 1>
  }

  AIE.packet_flow(0xd) {
    AIE.packet_source<%tile75, "DMA" : 0>
    AIE.packet_dest<%tile53, "DMA" : 1>
  }

  AIE.packet_flow(0xe) {
    AIE.packet_source<%tile75, "DMA" : 0>
    AIE.packet_dest<%tile54, "DMA" : 1>
  }

  AIE.packet_flow(0xf) {
    AIE.packet_source<%tile75, "DMA" : 0>
    AIE.packet_dest<%tile55, "DMA" : 1>
  }

}
