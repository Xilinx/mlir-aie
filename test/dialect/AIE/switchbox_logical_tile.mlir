//===- switchbox_logical_tile.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aie.switchbox / aie.shim_mux may reference an unplaced aie.logical_tile.
// Their target-model-dependent checks (port bounds, legal-connection rules)
// are deferred until --aie-place-tiles resolves the tile, mirroring how
// FlowOp and the UsesAreAccessible trait treat logical tiles. This lets IRON
// emit manual routing before placement.

// RUN: aie-opt --split-input-file %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s --check-prefix=PLACED
// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles %s

// A switchbox on a logical tile passes the early verifier untouched...
// CHECK-LABEL: @switchbox_on_logical_tile
// CHECK: aie.switchbox
// ...and after placement its operand is a placed aie.tile.
// PLACED-LABEL: @switchbox_on_logical_tile
// PLACED: aie.tile(0, 2)
// PLACED: aie.switchbox
// PLACED-NOT: aie.logical_tile
module @switchbox_on_logical_tile {
  aie.device(npu2) {
    %t = aie.logical_tile<CoreTile>(0, 2)
    aie.switchbox(%t) {
      aie.connect<South : 1, DMA : 0>
      aie.end
    }
  }
}

// -----

// A shim_mux on a logical shim tile is likewise deferred + placed.
// CHECK-LABEL: @shim_mux_on_logical_tile
// CHECK: aie.shim_mux
// PLACED-LABEL: @shim_mux_on_logical_tile
// PLACED: aie.tile(0, 0)
// PLACED: aie.shim_mux
// PLACED-NOT: aie.logical_tile
module @shim_mux_on_logical_tile {
  aie.device(npu2) {
    %t = aie.logical_tile<ShimNOCTile>(0, 0)
    aie.shim_mux(%t) {
      aie.connect<DMA : 0, North : 3>
      aie.end
    }
  }
}
