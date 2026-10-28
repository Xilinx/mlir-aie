//===- follow-through-endpoints.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A flow fanning out at a shared switchbox port keeps one origin on every
// branch, and the branches differ only in where they end up.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --only-flow=0 %s | FileCheck %s --check-prefix=FANOUT
// RUN: aie-visualize --emit-dot --only-flow=1 %s | FileCheck %s --check-prefix=OTHER

// FANOUT-DAG: p_0_2_1_0_m -> p_0_2_5_5_out_r0 {{.*}}flow src-0 dst-0{{.*}}"
// FANOUT-DAG: p_0_2_5_5_out_r0 -> p_0_3_3_5_in_r0 {{.*}}flow src-0 dst-0{{.*}}"
// FANOUT-DAG: p_0_3_3_5_in_r0 -> p_0_3_4_2_out_r0 {{.*}}flow src-0 dst-1{{.*}}"
// FANOUT-DAG: p_0_3_3_5_in_r0 -> p_0_3_5_1_out_r0 {{.*}}flow src-0 dst-2{{.*}}"
// FANOUT-NOT: p_0_4_0_1

// OTHER-DAG: p_0_4_0_0_in_r3 -> p_0_4_3_0_out_r3 {{.*}}flow src-1 dst-1{{.*}}"
// OTHER-DAG: p_0_4_3_0_out_r3 -> p_0_3_5_0_in_r3 {{.*}}flow src-1 dst-1{{.*}}"
// OTHER-NOT: p_0_2_1_0_m

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
