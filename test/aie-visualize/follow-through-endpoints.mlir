//===- follow-through-endpoints.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Flow grouping crosses shared endpoints while retaining endpoint fan-out.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --only-flow=0 %s | FileCheck %s --check-prefix=FANOUT
// RUN: aie-visualize --emit-dot --only-flow=1 %s | FileCheck %s --check-prefix=OTHER
// RUN: aie-visualize --emit-dot --no-follow-buffers --only-flow=3 %s | FileCheck %s --check-prefix=UNGROUPED

// FANOUT: p_0_2_1_0_m -> p_0_2_5_5
// FANOUT: p_0_2_5_5 -> p_0_3_3_5 {{.*}} label=<<FONT COLOR="#d73027">F0</FONT>>
// FANOUT: p_0_3_3_5 -> p_0_3_4_2
// FANOUT: p_0_3_3_5 -> p_0_3_5_1
// FANOUT-NOT: p_0_4_0_1

// OTHER: p_0_4_0_0 -> p_0_4_3_0
// OTHER: p_0_4_3_0 -> p_0_3_5_0 {{.*}} label=<<FONT COLOR="#4575b4">F1</FONT>>

// UNGROUPED: F3

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)

    aie.flow(%t02, DMA : 0, %t03, South : 5)
      via (%t02 : DMA : 0 -> North : 5)
    aie.flow(%t03, South : 5, %t04, DMA : 0)
      via (%t03 : South : 5 -> North : 1,
           %t04 : South : 1 -> DMA : 0)
    aie.flow(%t03, South : 5, %t03, Core : 0)
      via (%t03 : South : 5 -> West : 2,
           %t03 : West : 2 -> Core : 0)
    aie.flow(%t04, Core : 0, %t03, DMA : 1)
      via (%t04 : Core : 0 -> South : 0,
           %t03 : North : 0 -> DMA : 1)
  }
}
