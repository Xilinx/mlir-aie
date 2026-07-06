//===- bad_mem_tile_input.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

module @bad_mem_tile_input {
 aie.device(xcve2302) {
    %tile11 = aie.tile(1, 1) 
    %tile33 = aie.tile(3, 3)

    // expected-error@+1 {{`aie_stream` is not available for shim and mem tiles}}
    aie.objectfifo @of_stream (%tile33, {%tile11}, 2 : i32) {aie_stream = 1 : i32, aie_stream_port = 0 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}
