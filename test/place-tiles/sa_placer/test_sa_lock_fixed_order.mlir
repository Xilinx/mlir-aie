//===- test_sa_lock_fixed_order.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" --mlir-pass-statistics %s -o /dev/null 2>&1 | FileCheck %s

// Two logical shim tiles holding 9 and 8 locks share npu2_1col's one shim
// tile, which has 16. The SA charges the locks of the tile it finds no room
// for, taking the tiles in their structural order, so the same tile is
// charged whichever is declared first and whatever its locks are named.
// CHECK: warning: SA placer's resource estimate is over capacity (penalty=180)
// CHECK: 180 sa-lock-penalty
// CHECK: warning: SA placer's resource estimate is over capacity (penalty=180)
// CHECK: 180 sa-lock-penalty

module @nine_first {
  aie.device(npu2_1col) {
    %c = aie.logical_tile<CoreTile>(?, ?)
    %sa = aie.logical_tile<ShimNOCTile>(?, ?)
    %la0 = aie.lock(%sa) {sym_name = "la0"}
    %la1 = aie.lock(%sa) {sym_name = "la1"}
    %la2 = aie.lock(%sa) {sym_name = "la2"}
    %la3 = aie.lock(%sa) {sym_name = "la3"}
    %la4 = aie.lock(%sa) {sym_name = "la4"}
    %la5 = aie.lock(%sa) {sym_name = "la5"}
    %la6 = aie.lock(%sa) {sym_name = "la6"}
    %la7 = aie.lock(%sa) {sym_name = "la7"}
    %la8 = aie.lock(%sa) {sym_name = "la8"}
    %sb = aie.logical_tile<ShimNOCTile>(?, ?)
    %lb0 = aie.lock(%sb) {sym_name = "lb0"}
    %lb1 = aie.lock(%sb) {sym_name = "lb1"}
    %lb2 = aie.lock(%sb) {sym_name = "lb2"}
    %lb3 = aie.lock(%sb) {sym_name = "lb3"}
    %lb4 = aie.lock(%sb) {sym_name = "lb4"}
    %lb5 = aie.lock(%sb) {sym_name = "lb5"}
    %lb6 = aie.lock(%sb) {sym_name = "lb6"}
    %lb7 = aie.lock(%sb) {sym_name = "lb7"}
    aie.core(%c) {
      aie.end
    }
  }
}

// -----

module @eight_first {
  aie.device(npu2_1col) {
    %c = aie.logical_tile<CoreTile>(?, ?)
    %sb = aie.logical_tile<ShimNOCTile>(?, ?)
    %qb0 = aie.lock(%sb) {sym_name = "qb0"}
    %qb1 = aie.lock(%sb) {sym_name = "qb1"}
    %qb2 = aie.lock(%sb) {sym_name = "qb2"}
    %qb3 = aie.lock(%sb) {sym_name = "qb3"}
    %qb4 = aie.lock(%sb) {sym_name = "qb4"}
    %qb5 = aie.lock(%sb) {sym_name = "qb5"}
    %qb6 = aie.lock(%sb) {sym_name = "qb6"}
    %qb7 = aie.lock(%sb) {sym_name = "qb7"}
    %sa = aie.logical_tile<ShimNOCTile>(?, ?)
    %qa0 = aie.lock(%sa) {sym_name = "qa0"}
    %qa1 = aie.lock(%sa) {sym_name = "qa1"}
    %qa2 = aie.lock(%sa) {sym_name = "qa2"}
    %qa3 = aie.lock(%sa) {sym_name = "qa3"}
    %qa4 = aie.lock(%sa) {sym_name = "qa4"}
    %qa5 = aie.lock(%sa) {sym_name = "qa5"}
    %qa6 = aie.lock(%sa) {sym_name = "qa6"}
    %qa7 = aie.lock(%sa) {sym_name = "qa7"}
    %qa8 = aie.lock(%sa) {sym_name = "qa8"}
    aie.core(%c) {
      aie.end
    }
  }
}
