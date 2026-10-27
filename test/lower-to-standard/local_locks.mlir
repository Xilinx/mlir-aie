//===- local_locks.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-standard-lowering="tilecol=3 tilerow=3" %s | FileCheck --check-prefix=CHECK33 %s

// CHECK33:  func.func @core_3_3() {
// CHECK33:    %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK33:    %[[ACQUIRE_LOCK:.*]] = arith.constant 56 : i32
// CHECK33:    call @llvm.aie.lock.acquire.reg(%[[ACQUIRE_LOCK]], %[[ZERO]]) : (i32, i32) -> ()
// CHECK33:    %[[ONE:.*]] = arith.constant 1 : i32
// CHECK33:    %[[RELEASE_LOCK:.*]] = arith.constant 56 : i32
// CHECK33:    call @llvm.aie.lock.release.reg(%[[RELEASE_LOCK]], %[[ONE]]) : (i32, i32) -> ()
// CHECK33:    return
// CHECK33:  }

module @local_locks {
 aie.device(xcvc1902) {
  %3 = aie.tile(3, 3)
  aie.lock @local(%3, 8)
  %11 = aie.core(%3)  {
    %c0_ul0 = arith.constant 0 : i32
    aie.use_lock(@local, Acquire, %c0_ul0)
    %c1_ul1 = arith.constant 1 : i32
    aie.use_lock(@local, Release, %c1_ul1)
    aie.end
  }
 }
}
