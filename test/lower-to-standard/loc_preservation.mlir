//===- loc_preservation.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 AMD Inc.
//
//===----------------------------------------------------------------------===//

// Verifies that --aie-standard-lowering forwards the source op's location
// onto its lowered standard-dialect ops.

// RUN: aie-opt --aie-localize-locks --aie-standard-lowering --mlir-print-debuginfo %s | FileCheck %s

#user_loc = loc("user_design.py":42:4)
#buf_loc = loc("user_design.py":50:4)

module @loc_test {
 aie.device(xcve2302) {
  %tile13 = aie.tile(1, 3)
  %lock13 = aie.lock(%tile13, 0) loc(#user_loc)
  %buf13 = aie.buffer(%tile13) {sym_name = "my_buf"} : memref<16xi32> loc(#buf_loc)

  func.func private @kernel(%lock : index) {
    aie.use_lock(%lock, "Acquire", 0) loc(#user_loc)
    return
  }

  %core13 = aie.core(%tile13) {
    func.call @kernel(%lock13) : (index) -> ()
    aie.end
  }
 }
}

// The lowered call to llvm.aie2.acquire and the memref.global for the
// buffer must reference the user's source location, not loc(unknown).
// CHECK-DAG: call @llvm.aie2.acquire({{.*}}) : (i32, i32) -> () loc(#[[ULOC:loc[0-9]*]])
// CHECK-DAG: memref.global "public" @my_buf : memref<16xi32> loc(#[[BUFLOC:loc[0-9]*]])
// CHECK-DAG: #[[ULOC]] = loc("user_design.py":42:4)
// CHECK-DAG: #[[BUFLOC]] = loc("user_design.py":50:4)
