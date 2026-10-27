// Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-localize-locks --aie-standard-lowering="tilecol=1 tilerow=3" %s | FileCheck --check-prefix=CHECK %s

// CHECK: module @test attributes {llvm.target_triple = "aie"} {
// A function can compute the SSA lock value; the lock itself is a device symbol.
// CHECK:   func.func private @kernel() -> i32 {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK:   }
// CHECK:   func.func @core_1_3() {
// CHECK:     %[[VALUE:.*]] = call @kernel() : () -> i32
// CHECK:     %[[LOCK:.*]] = arith.constant 48 : i32
// CHECK:     call @llvm.aie.lock.acquire.reg(%[[LOCK]], %[[VALUE]]) : (i32, i32) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }

module @test {
 aie.device(xcvc1902) {
  %tile13 = aie.tile(1, 3)
  aie.lock @lock13_3(%tile13, 0)

  func.func private @kernel() -> i32 {
    %c0_ul0 = arith.constant 0 : i32
    return %c0_ul0 : i32
  }

  %core13 = aie.core(%tile13) {
    %value = func.call @kernel() : () -> i32
    aie.use_lock(@lock13_3, "Acquire", %value)
    aie.end
  }
 }
}
