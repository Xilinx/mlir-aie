//===- test_place_cores_changed.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// @a and @b load one right after the other and run the same core program.
// Placing @b, both placers put its core where @a's runs, so switching
// between them rewrites no program; renamed and reordered, alike.

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=1' %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=2' %s | FileCheck %s

// CHECK-LABEL: module @cores_changed
// CHECK: aie.device(npu2) @a
// CHECK: aie.tile(2, 3)
// CHECK: aie.device(npu2) @b
// CHECK: aie.tile(2, 3)
// CHECK-LABEL: module @cores_changed_permuted
// CHECK: aie.device(npu2) @later
// CHECK: aie.tile(2, 3)
// CHECK: aie.device(npu2) @earlier
// CHECK: aie.tile(2, 3)

module @cores_changed {
  aie.device(npu2) @a {
    %t = aie.logical_tile<CoreTile>(2, 3)
    aie.core(%t) {
      %c = arith.constant 7 : i32
      aie.end
    }
  }
  aie.device(npu2) @b {
    %t = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%t) {
      %c = arith.constant 7 : i32
      aie.end
    }
  }
  aie.device(npu2) @main {
    aie.runtime_sequence @run() {
      aiex.npu.load_pdi {device_ref = @a}
      aiex.npu.load_pdi {device_ref = @b}
    }
  }
}

// -----

module @cores_changed_permuted {
  aie.device(npu2) @switches {
    aie.runtime_sequence @go() {
      aiex.npu.load_pdi {device_ref = @earlier}
      aiex.npu.load_pdi {device_ref = @later}
    }
  }
  aie.device(npu2) @later {
    %u = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%u) {
      %k = arith.constant 7 : i32
      aie.end
    }
  }
  aie.device(npu2) @earlier {
    %u = aie.logical_tile<CoreTile>(2, 3)
    aie.core(%u) {
      %k = arith.constant 7 : i32
      aie.end
    }
  }
}
