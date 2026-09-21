//===- cpp_peano_default.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Naming no backend selects Peano: cores go through llc and the Chess wrapper
// is never invoked. The counterpart on the Chess side is cpp_xchesscc_basic.

// REQUIRES: peano

// RUN: aiecc -v --output-dir=%t --tmpdir=%t.prj --get-core-elfs %s 2>&1 | FileCheck %s

// CHECK-NOT: xchesscc_wrapper
// CHECK: exec:{{.*}}llc
// CHECK-NOT: xchesscc_wrapper

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) {sym_name = "buf"} : memref<16xi32>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c1_i32, %buf[%c0] : memref<16xi32>
      aie.end
    }
  }
}
