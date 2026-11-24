//===- forever_loop_counter.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-scf-to-control-flow --split-input-file %s | FileCheck %s

// An INT64_MAX loop with an unused counter keeps its body and loops back
// unconditionally.
// CHECK-LABEL: aie.core
// CHECK-NOT:   arith.cmpi
// CHECK:       cf.br ^[[HEAD:bb[0-9]+]]
// CHECK:       ^[[HEAD]]:
// CHECK:       %[[TRUE:.*]] = arith.constant true
// CHECK:       cf.cond_br %[[TRUE]], ^[[BODY:bb[0-9]+]], ^[[EXIT:bb[0-9]+]]
// CHECK:       ^[[BODY]]:
// CHECK-NEXT:  aie.use_lock(%{{.*}}, AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT:  func.call @kernel()
// CHECK-NEXT:  aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK-NEXT:  cf.br ^[[HEAD]]
// CHECK:       ^[[EXIT]]:
// CHECK-NEXT:  aie.end
// CHECK-NOT:   arith.cmpi
// CHECK-NOT:   arith.addi
aie.device(npu1) {
  func.func private @kernel()
  %t = aie.tile(0, 2)
  %lock = aie.lock(%t, 0) {init = 1 : i32}
  %core = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %max = arith.constant 9223372036854775807 : index
    %one = arith.constant 1 : i32
    scf.for %i = %c0 to %max step %c1 {
      aie.use_lock(%lock, AcquireGreaterEqual, %one)
      func.call @kernel() : () -> ()
      aie.use_lock(%lock, Release, %one)
    }
    aie.end
  }
}

// -----

// A 2^32-1 bound is finite and reachable, so the loop stays counted.
// CHECK-LABEL: aie.core
// CHECK:       arith.cmpi
aie.device(npu1) {
  %t = aie.tile(0, 2)
  %core = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %bound = arith.constant 4294967295 : index
    scf.for %i = %c0 to %bound step %c1 {
    }
    aie.end
  }
}

// -----

// A nonzero lower bound stays counted.
// CHECK-LABEL: aie.core
// CHECK:       arith.cmpi
aie.device(npu1) {
  %t = aie.tile(0, 2)
  %core = aie.core(%t) {
    %c1 = arith.constant 1 : index
    %max = arith.constant 9223372036854775807 : index
    scf.for %i = %c1 to %max step %c1 {
    }
    aie.end
  }
}

// -----

// A non-unit step stays counted.
// CHECK-LABEL: aie.core
// CHECK:       arith.cmpi
aie.device(npu1) {
  %t = aie.tile(0, 2)
  %core = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %max = arith.constant 9223372036854775807 : index
    scf.for %i = %c0 to %max step %c2 {
    }
    aie.end
  }
}

// -----

// The same loop outside an aie.core is not rewritten.
// CHECK-LABEL: func.func @outside_core
// CHECK:       arith.cmpi
func.func @outside_core() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %max = arith.constant 9223372036854775807 : index
  scf.for %i = %c0 to %max step %c1 {
  }
  return
}

// -----

// A used induction variable prevents the rewrite.
// CHECK-LABEL: aie.core
// CHECK:       arith.cmpi
aie.device(npu1) {
  %t = aie.tile(0, 2)
  %core = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %max = arith.constant 9223372036854775807 : index
    scf.for %i = %c0 to %max step %c1 {
      %next = arith.addi %i, %c1 : index
    }
    aie.end
  }
}

// -----

// A loop with iter_args also stays counted, even when its IV is unused.
// CHECK-LABEL: aie.core
// CHECK:       arith.cmpi
aie.device(npu1) {
  %t = aie.tile(0, 2)
  %core = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %max = arith.constant 9223372036854775807 : index
    %zero = arith.constant 0 : i32
    %result = scf.for %i = %c0 to %max step %c1 iter_args(%arg = %zero) -> (i32) {
      scf.yield %arg : i32
    }
    aie.end
  }
}
