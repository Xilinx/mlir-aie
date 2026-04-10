//===- useLock_in_func_external_aie2p.mlir ----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Verify that lock SSA values passed to external (body-less) function calls
// get correctly localized to constant indices by AIELocalizeLocks, and that
// the standard lowering preserves these index arguments for linking with
// precompiled C kernels.

// RUN: aie-opt --aie-localize-locks --aie-standard-lowering="tilecol=1 tilerow=3" %s | FileCheck --check-prefix=CHECK %s

// CHECK: module @test attributes {llvm.target_triple = "aie2p"} {
// CHECK:   func.func private @func(memref<256xi32>, memref<256xi32>, index, index)
// CHECK:   func.func @core_1_3() {
// CHECK:     %c48 = arith.constant 48 : index
// CHECK:     %c49 = arith.constant 49 : index
// CHECK-DAG: call @func({{.*}}, {{.*}}, %c48, %c49) : (memref<256xi32>, memref<256xi32>, index, index) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }

module @test {
 aie.device(npu2) {
  %tile13 = aie.tile(1, 3)
  %buf_in  = aie.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
  %buf_out = aie.buffer(%tile13) { sym_name = "b" } : memref<256xi32>
  %prod_lock = aie.lock(%tile13, 0) { sym_name = "prod_lock", init = 1 : i32 }
  %cons_lock = aie.lock(%tile13, 1) { sym_name = "cons_lock", init = 0 : i32 }

  // External function declaration (no body) — linked with precompiled C kernel
  func.func private @func(%A: memref<256xi32>, %B: memref<256xi32>,
                           %acq_lock: index, %rel_lock: index) -> ()

  %core13 = aie.core(%tile13) {
    func.call @func(%buf_in, %buf_out, %prod_lock, %cons_lock)
      : (memref<256xi32>, memref<256xi32>, index, index) -> ()
    aie.end
  } { link_with = "kernel.o" }
 }
}
