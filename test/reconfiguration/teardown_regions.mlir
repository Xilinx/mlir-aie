//===- teardown_regions.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-expand-load-pdi="teardown=true inline-config=true" %s -verify-diagnostics

module {
  aie.device(npu2_1col) @cfg {
    %compute = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @seq {
    aie.runtime_sequence @loop() {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      aiex.npu.load_pdi {device_ref = @cfg}
      scf.for %i = %c0 to %c2 step %c1 {
        // expected-error @+1 {{with teardown, must be in the runtime sequence's body, not inside scf.for}}
        aiex.npu.load_pdi {device_ref = @cfg}
      }
    }
  }
}

// -----

module {
  aie.device(npu2_1col) @cfg {
    %compute = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @seq {
    aie.runtime_sequence @branch() {
      %true = arith.constant true
      aiex.npu.load_pdi {device_ref = @cfg}
      scf.if %true {
        // expected-error @+1 {{with teardown, must be in the runtime sequence's body, not inside scf.if}}
        aiex.npu.load_pdi {id = 4 : i32}
      }
    }
  }
}
