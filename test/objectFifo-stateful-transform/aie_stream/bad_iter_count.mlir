//===- bad_iter_count.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// TODO: stream with repeat, iter_count

// RUN: aie-opt --aie-objectFifo-stateful-transform --verify-diagnostics %s

module @bad_dims_from_stream {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2) 
    %tile33 = aie.tile(3, 3)

    // expected-error@+1 {{`iter_count` unavailable on stream end}}
    aie.objectfifo @of_stream (%tile12, {%tile33}, 1 : i32)
                              {aie_stream = 0 : i32, aie_stream_port = 0 : i32, iter_count = 5 : i32}
                              : !aie.objectfifo<memref<3xi32>>
  }
}
