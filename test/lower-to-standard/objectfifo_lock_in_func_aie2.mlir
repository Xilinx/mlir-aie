//===- objectfifo_lock_in_func_aie2.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Verify the full pipeline: objectfifo-stateful-transform resolves
// objectfifo.lock/buffer ops to concrete locks and buffers, then
// localize-locks + standard-lowering convert lock SSA values to constants
// suitable for passing to external C kernels.

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-localize-locks --aie-standard-lowering="tilecol=1 tilerow=2" %s | FileCheck --check-prefix=CHECK %s

// After the full pipeline, the core function should call @kernel with:
//   - buffer memref
//   - localized lock constant indices (prod_lock and cons_lock)
// CHECK: module @test attributes {llvm.target_triple = "aie2"} {
// CHECK:   func.func private @kernel(memref<256xi32>, index, index)
// CHECK:   func.func @core_1_2() {
// CHECK-DAG: %c{{[0-9]+}} = arith.constant
// CHECK-DAG: %c{{[0-9]+}} = arith.constant
// CHECK:     call @kernel
// CHECK:     return
// CHECK:   }

module @test {
 aie.device(xcve2302) {
  %tile12 = aie.tile(1, 2)
  %tile13 = aie.tile(1, 3)

  aie.objectfifo @of0(%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

  func.func private @kernel(%buf: memref<256xi32>,
                             %acq_lock: index, %rel_lock: index) -> ()

  %core12 = aie.core(%tile12) {
    %buf = aie.objectfifo.buffer @of0 (0) : memref<256xi32>
    %acq_lock, %rel_lock = aie.objectfifo.lock @of0 (Produce) : (index, index)
    func.call @kernel(%buf, %acq_lock, %rel_lock)
      : (memref<256xi32>, index, index) -> ()
    aie.end
  } { link_with = "kernel.o" }
 }
}
