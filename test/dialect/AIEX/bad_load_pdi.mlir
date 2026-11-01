//===- bad_load_pdi.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A load_pdi's device_ref names the device whose PDI it loads, so it must be
// an aie.device of the module.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      // expected-error @+1 {{@missing is not a device}}
      aiex.npu.load_pdi {device_ref = @missing}
    }
  }
}

// -----

module {
  aie.device(npu2_1col) @main {
    %t = aie.tile(0, 2)
    %buf = aie.buffer(%t) {sym_name = "buf"} : memref<16xi32>
    aie.runtime_sequence() {
      // expected-error @+1 {{@buf is not a device}}
      aiex.npu.load_pdi {device_ref = @buf}
    }
  }
}
