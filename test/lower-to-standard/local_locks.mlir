//===- local_locks.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-standard-lowering="tilecol=3 tilerow=3" %s | FileCheck --check-prefix=CHECK33 %s

// CHECK33:  func.func @core_3_3() {
// CHECK33:    %c56 = arith.constant 56 : index
// CHECK33:    %0 = arith.index_cast %c56 : index to i32
// CHECK33:    %c0_i32 = arith.constant 0 : i32
// CHECK33:    call @llvm.aie.lock.acquire.reg(%0, %c0_i32) : (i32, i32) -> ()
// CHECK33:    %1 = arith.index_cast %c56 : index to i32
// CHECK33:    %c1_i32 = arith.constant 1 : i32
// CHECK33:    call @llvm.aie.lock.release.reg(%1, %c1_i32) : (i32, i32) -> ()
// CHECK33:    return
// CHECK33:  }

module @local_locks {
 AIE.device(xcvc1902) {
  %3 = AIE.tile(3, 3)
  %11 = AIE.core(%3)  {
    %c56 = arith.constant 56 : index
    AIE.useLock(%c56, Acquire, 0)
    AIE.useLock(%c56, Release, 1)
    AIE.end
  }
 }
}

