//===- bad_shim_tile_output.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

module @bad_shim_tile_output {
 aie.device(xcve2302) {
    %tile10 = aie.tile(1, 0) 
    %tile33 = aie.tile(3, 3)

    // expected-error@+1 {{`aie_stream` is not available for shim and mem tiles}}
    aie.objectfifo @of_stream (%tile10, {%tile33}, 2 : i32) {aie_stream = 0 : i32, aie_stream_port = 0 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}
