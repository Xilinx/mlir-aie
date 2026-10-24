//===- forever_loop_counter.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-scf-to-control-flow --split-input-file %s | FileCheck %s

// Unused counters with effectively infinite bounds become counter-free loops.
// CHECK-LABEL: aie.core
// CHECK-NOT:   arith.cmpi
// CHECK-NOT:   arith.addi
// CHECK:       cf.cond_br
// CHECK-NOT:   arith.cmpi
// CHECK-NOT:   arith.addi
aie.device(npu1) {
  %t = aie.tile(0, 2)
  %core = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %u32max = arith.constant 4294967295 : index
    %i64max = arith.constant 9223372036854775807 : index
    scf.for %i = %c0 to %u32max step %c1 {
    }
    scf.for %i = %c0 to %i64max step %c1 {
    }
    aie.end
  }
}

// -----

// A bound below the threshold remains a counted loop.
// CHECK-LABEL: aie.core
// CHECK:       arith.cmpi
aie.device(npu1) {
  %t = aie.tile(0, 2)
  %core = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %bound = arith.constant 4294967294 : index
    scf.for %i = %c0 to %bound step %c1 {
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
  %bound = arith.constant 4294967295 : index
  scf.for %i = %c0 to %bound step %c1 {
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
    %bound = arith.constant 4294967295 : index
    scf.for %i = %c0 to %bound step %c1 {
      %next = arith.addi %i, %c1 : index
    }
    aie.end
  }
}

// -----

// A loop with iter_args also remains counted, even when its IV is unused.
// CHECK-LABEL: aie.core
// CHECK:       arith.cmpi
aie.device(npu1) {
  %t = aie.tile(0, 2)
  %core = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %bound = arith.constant 4294967295 : index
    %zero = arith.constant 0 : i32
    %result = scf.for %i = %c0 to %bound step %c1 iter_args(%arg = %zero) -> (i32) {
      scf.yield %arg : i32
    }
    aie.end
  }
}
