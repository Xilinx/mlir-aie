//===- bad_init_values.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

module @bad_dims_from_stream {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    // expected-error@+1 {{`init_values` unavailable on stream end}}
    aie.objectfifo @of_stream (%tile12, {%tile33}, 1 : i32)
                              {prod_port = #aie.end_port<Core : 0>}
                              : !aie.objectfifo<memref<3xi32>> = [dense<[0, 1, 3]> : memref<3xi32>]
  }
}
