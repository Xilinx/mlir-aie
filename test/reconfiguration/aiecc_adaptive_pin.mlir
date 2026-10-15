//===- aiecc_adaptive_pin.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// --ctrlpkt-adaptive-pin in aiecc, under --load-pdi-to-ctrl-pkt. The overlay
// is routed around the data of @a and @b (control_overlay_avoid_data.mlir
// checks how), so it leaves (1, 4) on another channel, the same in every
// device; the router consumes aie.ctrl_pkt_overlay_avoid.

// RUN: rm -rf %t && mkdir -p %t/blind %t/adapt
// RUN: cd %t/blind && %aiecc --load-pdi-to-ctrl-pkt --get=input_physical.mlir %s
// RUN: FileCheck %s --check-prefix=BLIND < %t/blind/input_physical.mlir
// RUN: cd %t/adapt && %aiecc --load-pdi-to-ctrl-pkt --ctrlpkt-adaptive-pin --get=input_physical.mlir %s
// RUN: FileCheck %s --check-prefix=ADAPT < %t/adapt/input_physical.mlir

// BLIND-LABEL: aie.device(npu2) @a
// BLIND:       aie.switchbox(%tile_1_4)
// BLIND-NOT:   aie.switchbox
// BLIND:       aie.masterset(North : 3, %{{.+}}) {aie.is_ctrl_pkt_overlay}
// BLIND-LABEL: aie.device(npu2) @b
// BLIND:       aie.switchbox(%tile_1_4)
// BLIND-NOT:   aie.switchbox
// BLIND:       aie.masterset(North : 3, %{{.+}}) {aie.is_ctrl_pkt_overlay}
// BLIND-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// BLIND:       aie.switchbox(%tile_1_4)
// BLIND-NOT:   aie.switchbox
// BLIND:       aie.masterset(North : 3, %{{.+}}) {aie.is_ctrl_pkt_overlay}

// ADAPT-LABEL: aie.device(npu2) @a
// ADAPT:       aie.switchbox(%tile_1_4)
// ADAPT-NOT:   aie.switchbox
// ADAPT:       aie.masterset(North : 4, %{{.+}}) {aie.is_ctrl_pkt_overlay}
// ADAPT-LABEL: aie.device(npu2) @b
// ADAPT:       aie.switchbox(%tile_1_4)
// ADAPT-NOT:   aie.switchbox
// ADAPT:       aie.masterset(North : 4, %{{.+}}) {aie.is_ctrl_pkt_overlay}
// ADAPT-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// ADAPT:       aie.switchbox(%tile_1_4)
// ADAPT-NOT:   aie.switchbox
// ADAPT:       aie.masterset(North : 4, %{{.+}}) {aie.is_ctrl_pkt_overlay}

// ADAPT-NOT: aie.ctrl_pkt_overlay_avoid

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @seq(%x : memref<4xi32>) {
      aiex.configure @a {
        aiex.run @a_run(%x) : (memref<4xi32>)
      }
      aiex.configure @b {
        aiex.run @b_run(%x) : (memref<4xi32>)
      }
    }
  }
  aie.device(npu2) @a {
    %s00 = aie.tile(0, 0)
    %t14 = aie.tile(1, 4)
    aie.flow(%s00, DMA : 0, %t14, DMA : 1)
    aie.runtime_sequence @a_run(%x : memref<4xi32>) {
    }
  }
  aie.device(npu2) @b {
    %m11 = aie.tile(1, 1)
    %t03 = aie.tile(0, 3)
    aie.flow(%m11, DMA : 0, %t03, DMA : 0)
    aie.runtime_sequence @b_run(%x : memref<4xi32>) {
    }
  }
}
