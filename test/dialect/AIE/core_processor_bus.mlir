// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// External kernel code can request processor-bus access explicitly.
// CHECK-LABEL: module @explicit
// CHECK: aie.core
// CHECK: } {processor_bus}
module @explicit {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %core = aie.core(%tile) {
      aie.end
    } {processor_bus}
  }
}

// -----

// The attribute is a presence flag, not a boolean-valued option.
module @invalid_type {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    // expected-error @+3 {{attribute 'processor_bus' failed to satisfy constraint: unit attribute}}
    %core = aie.core(%tile) {
      aie.end
    } {processor_bus = false}
  }
}
