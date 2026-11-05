// RUN: aie-opt --verify-diagnostics --split-input-file %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A buffer's initial value gives every element, in the buffer's shape and
// element type. The generic form skips the parser's shape inference.

module {
  aie.device(npu2) {
    %m = aie.tile(0, 1)
    // expected-error@+1 {{initial value of type 'tensor<2xi16>' does not cover the buffer's 'memref<60xi16>'}}
    %b = "aie.buffer"(%m) <{aligned = true, initial_value = dense<[1, -1]> : tensor<2xi16>}> : (index) -> memref<60xi16>
  }
}

// -----

module {
  aie.device(npu2) {
    %m = aie.tile(0, 1)
    // expected-error@+1 {{initial value of type 'tensor<2xi32>' does not cover the buffer's 'memref<2xi16>'}}
    %b = "aie.buffer"(%m) <{aligned = true, initial_value = dense<[1, -1]> : tensor<2xi32>}> : (index) -> memref<2xi16>
  }
}

// -----

module {
  aie.device(npu2) {
    %m = aie.tile(0, 1)
    %b = aie.buffer(%m) : memref<2x3xi16> = dense<7>
  }
}
