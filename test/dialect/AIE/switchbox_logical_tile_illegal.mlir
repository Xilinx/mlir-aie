//===- switchbox_logical_tile_illegal.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Deferring the switchbox port-bound check for a logical tile does NOT drop it:
// an out-of-range connection passes the early verifier but is still caught once
// --aie-place-tiles resolves the tile and the target model becomes known.

// The early verifier accepts it (logical tile → checks deferred):
// RUN: aie-opt %s

// After placement the illegal port index is rejected:
// RUN: not aie-opt --aie-place-tiles %s 2>&1 | FileCheck %s
// CHECK: error{{.*}} 'aie.connect' op index 99 for source bundle DMA must be less than

aie.device(npu2) {
  %t = aie.logical_tile<CoreTile>(0, 2)
  aie.switchbox(%t) {
    aie.connect<DMA : 99, North : 0>
    aie.end
  }
}
