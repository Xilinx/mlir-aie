//===- scf_to_control_flow.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-scf-to-control-flow --split-input-file %s | FileCheck %s

// scf in a core body is lowered to cf, exactly like the generic
// convert-scf-to-cf.

// CHECK-LABEL: aie.core
// CHECK-NOT:   scf.for
// CHECK:       cf.br
// CHECK:       cf.cond_br
aie.device(npu1) {
  %t = aie.tile(0, 2)
  %core = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c8 step %c1 {
    }
    aie.end
  }
}


// -----

// scf inside a runtime sequence is left untouched: the runtime sequence is
// NoTerminator and its loops are lowered by the dedicated runtime-sequence
// path, not by scf->cf.

// CHECK-LABEL: aie.runtime_sequence @runtime_seq_scf_preserved
// CHECK:       scf.for
// CHECK-NOT:   cf.br
aie.device(npu1) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence @runtime_seq_scf_preserved(%arg0: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c8 step %c1 {
    }
  }
}


// -----

// Both in one device: the core loop is lowered to cf while the runtime-sequence
// loop is preserved.

// CHECK-LABEL: aie.core
// CHECK:         cf.cond_br
// CHECK:       aie.runtime_sequence @mixed
// CHECK:         scf.for
aie.device(npu1) {
  %t02 = aie.tile(0, 2)
  %t00 = aie.tile(0, 0)
  %core = aie.core(%t02) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c8 step %c1 {
    }
    aie.end
  }
  aie.runtime_sequence @mixed(%arg0: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c8 step %c1 {
    }
  }
}
