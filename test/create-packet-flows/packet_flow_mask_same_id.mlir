//===- packet_flow_mask_same_id.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two packet flows from one source may state different masks for one id. The
// switchbox routes on the id alone, so both flows go everywhere either does,
// and the rules must match every id either mask claims. Keeping only the mask
// of the flow routed last would drop the ids only the other one claims.

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// The first flow claims 0x8 through 0xb and the second 0x8 and 0xc, so the
// rules match 0x8 through 0xc.
// CHECK: aie.switchbox(%{{.*}}tile_0_2)
// CHECK:   aie.packet_rules(DMA : 0)
// CHECK-NEXT:     aie.rule(27, 8, %{{.*}})
// CHECK-NEXT:     aie.rule(28, 8, %{{.*}})
// CHECK-NEXT:   }

// The same holds when the flow with the narrower mask comes first.
// CHECK: aie.switchbox(%{{.*}}tile_1_2)
// CHECK:   aie.packet_rules(DMA : 0)
// CHECK-NEXT:     aie.rule(27, 8, %{{.*}})
// CHECK-NEXT:     aie.rule(28, 8, %{{.*}})
// CHECK-NEXT:   }

module {
  aie.device(npu1_2col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t10 = aie.tile(1, 0)
    %t12 = aie.tile(1, 2)
    %t13 = aie.tile(1, 3)

    aie.packet_flow(8, mask = 28) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t00, DMA : 0>
    }
    aie.packet_flow(8, mask = 27) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
    }

    aie.packet_flow(8, mask = 27) {
      aie.packet_source<%t12, DMA : 0>
      aie.packet_dest<%t13, DMA : 0>
    }
    aie.packet_flow(8, mask = 28) {
      aie.packet_source<%t12, DMA : 0>
      aie.packet_dest<%t10, DMA : 0>
    }
  }
}
