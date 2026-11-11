//===- lowering_failure.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A failed rewrite fails the pass. Previously the op was left in place, the
// outlined core did not legalize, and the core was dropped from the output.

// RUN: aie-opt --aie-standard-lowering --split-input-file --verify-diagnostics %s

// AIE2 has no wide stream intrinsic.
module @wide_stream_aie2 {
  aie.device(npu1) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) {
      %ch = arith.constant 0 : i32
      %v = arith.constant 1 : i128
      // expected-error@+2 {{'aie.put_stream' op Could not find the intrinsic function llvm.aie2.put.wms}}
      // expected-error@+1 {{failed to legalize operation 'aie.put_stream' that was explicitly marked illegal}}
      aie.put_stream(%ch : i32, %v : i128)
      aie.end
    }
  }
}

// -----

module @float_stream_aie2p {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) {
      %ch = arith.constant 0 : i32
      // expected-error@+2 {{'aie.get_stream' op Could not find the intrinsic function llvm.aie2p.get.fss}}
      // expected-error@+1 {{failed to legalize operation 'aie.get_stream' that was explicitly marked illegal}}
      %v = aie.get_stream(%ch : i32) : f32
      aie.end
    }
  }
}

// -----

// The outlined body must legalize; scf is not lowered by this pass.
module @unlowered_scf {
  aie.device(npu1) {
    %t = aie.tile(0, 2)
    // expected-error@+1 {{failed to legalize operation 'aie.core' that was explicitly marked illegal}}
    %c = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
      }
      aie.end
    }
  }
}
