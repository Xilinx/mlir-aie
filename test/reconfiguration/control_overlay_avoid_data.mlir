// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-create-control-overlay %s -o %t
// RUN: FileCheck %s < %t
// RUN: aie-opt --aie-create-pathfinder-flows --aie-import-overlay %t -o /dev/null

// CHECK-LABEL: aie.device(npu2_1col) @first
// CHECK: overlay = @ctrl_pkt_overlay
// CHECK-LABEL: aie.device(npu2_1col) @second
// CHECK: overlay = @ctrl_pkt_overlay
// CHECK-LABEL: aie.device(npu2_1col) @ctrl_pkt_overlay
// CHECK-NOT: aie.packet_flow
// CHECK: aie.masterset
// CHECK-NOT: ctrl_pkt_overlay_avoid

module {
  aie.device(npu2_1col) @first {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %tile = aie.tile(0, 2)
    aie.flow(%mem, DMA : 0, %tile, DMA : 0)
  }
  aie.device(npu2_1col) @second {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %tile = aie.tile(0, 3)
    aie.flow(%mem, DMA : 0, %tile, DMA : 1)
  }
}
