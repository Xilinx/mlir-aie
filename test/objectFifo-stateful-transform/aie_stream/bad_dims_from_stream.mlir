//===- bad_dims_from_stream.mlir --------------------------------*- MLIR -*-===//
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

    // expected-error@+1 {{`dimensionsFromStreamPerConsumer` data layout transformations are unavailable on stream end}}
    aie.objectfifo @of_stream (%tile12, {%tile33 dimensionsFromStream [<size = 16, stride = 1>]}, 2 : i32)
                               {aie_stream = 1 : i32, aie_stream_port = 0 : i32}
                               : !aie.objectfifo<memref<16xi32>>
  }
}
