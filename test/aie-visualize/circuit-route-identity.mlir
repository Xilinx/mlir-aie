//===- circuit-route-identity.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two circuit routes may use the same physical port without carrying the same
// stream. Their semantic port IDs retain the route group and port direction.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --skip-verify %s | FileCheck %s

// CHECK-DAG: p_0_2_1_0_m -> p_0_2_5_0_out_r0 {{.*}}origin-p_0_2_1_0_m{{.*}}
// CHECK-DAG: p_0_2_1_1_m -> p_0_2_5_0_out_r1 {{.*}}origin-p_0_2_1_1_m{{.*}}
// CHECK-NOT: p_0_2_1_0_m -> p_0_2_5_0_out_r1
// CHECK-NOT: p_0_2_1_1_m -> p_0_2_5_0_out_r0

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)

    aie.flow(%t02, DMA : 0, %t03, Core : 0)
      via (%t02 : DMA : 0 -> North : 0,
           %t03 : South : 0 -> Core : 0)
    aie.flow(%t02, DMA : 1, %t03, Core : 1)
      via (%t02 : DMA : 1 -> North : 0,
           %t03 : South : 0 -> Core : 1)
  }
}
