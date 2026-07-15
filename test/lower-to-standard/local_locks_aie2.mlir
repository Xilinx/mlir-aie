//===- local_locks_aie2.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-standard-lowering="tilecol=3 tilerow=3" %s | FileCheck --check-prefix=CHECK33 %s

// CHECK33:  func.func @core_3_3() {
// CHECK33:    %c56 = arith.constant 56 : index
// CHECK33:    %c0_i32 = arith.constant 0 : i32
// CHECK33:    %0 = arith.index_cast %c56 : index to i32
// CHECK33:    call @llvm.aie2.acquire(%0, %c0_i32) : (i32, i32) -> ()
// CHECK33:    %c1_i32 = arith.constant 1 : i32
// CHECK33:    %1 = arith.index_cast %c56 : index to i32
// CHECK33:    call @llvm.aie2.release(%1, %c1_i32) : (i32, i32) -> ()
// CHECK33:    return
// CHECK33:  }

module @local_locks {
 aie.device(xcve2302) {
  %3 = aie.tile(3, 3)
  %11 = aie.core(%3)  {
    %c56 = arith.constant 56 : index
    %c0_ul0 = arith.constant 0 : i32
    aie.use_lock(%c56, Acquire, %c0_ul0)
    %c1_ul1 = arith.constant 1 : i32
    aie.use_lock(%c56, Release, %c1_ul1)
    aie.end
  }
 }
}
