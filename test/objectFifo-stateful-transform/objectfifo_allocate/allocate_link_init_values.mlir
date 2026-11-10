//===- allocate_link_init_values.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s

// in0 owns the link's pool and shares memory with the link tile through its
// allocate, so its core fills the pool directly: the pool starts with in0's
// initial objects, full, as in0's own pool would without the allocate.
// Before, the pool started empty and the initial values were lost.

// CHECK-LABEL: module @input_owner
// CHECK:     %[[L:.*]] = aie.tile(0, 3)
// CHECK-DAG: aie.buffer(%[[L]]) {{.*}}sym_name = "in0__cons__buf0"} : memref<16xi32> = dense<5>
// CHECK-DAG: aie.buffer(%[[L]]) {{.*}}sym_name = "in0__cons__buf1"} : memref<16xi32> = dense<6>
// CHECK-DAG: aie.lock(%[[L]]) {init = 0 : i32, sym_name = "in0__cons__prodlock0"}
// CHECK-DAG: aie.lock(%[[L]]) {init = 2 : i32, sym_name = "in0__cons__conslock0"}
module @input_owner {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %l = aie.tile(0, 3)
    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<16xi32>> = [dense<5> : memref<16xi32>, dense<6> : memref<16xi32>]
    aie.objectfifo @out(%l, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @in0 (%l)
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
  }
}

// -----

// The output's initial values make it the pool's owner, so the pool starts
// with them; in0's own initial objects have no place in it.

module @both_initial {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %l = aie.tile(0, 3)
    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<16xi32>> = [dense<5> : memref<16xi32>, dense<6> : memref<16xi32>]
    aie.objectfifo @out(%l, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>> = [dense<7> : memref<16xi32>, dense<8> : memref<16xi32>]
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    // expected-error@+1 {{places @in0's objects in the link's pool, which starts with @out's initial objects, but @in0 has initial objects of its own}}
    aie.objectfifo.allocate @in0 (%l)
  }
}
