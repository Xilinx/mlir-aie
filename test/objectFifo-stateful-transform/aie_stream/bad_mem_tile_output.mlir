//===- bad_mem_tile_output.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

module @bad_mem_tile_output {
 aie.device(xcve2302) {
    %tile11 = aie.tile(1, 1)
    %tile33 = aie.tile(3, 3)

    // expected-error@+1 {{a Core stream port end is not available for shim and mem tiles}}
    aie.objectfifo @of_stream (%tile11, {%tile33}, 2 : i32) {prod_port = #aie.end_port<Core : 0>} : !aie.objectfifo<memref<16xi32>>
  }
}
