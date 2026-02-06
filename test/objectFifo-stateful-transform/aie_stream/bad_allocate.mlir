//===- bad_allocate.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

module @bad_allocate {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2) 
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of_stream (%tile12, {%tile13}, 1 : i32)
                              {aie_stream = 0 : i32, aie_stream_port = 0 : i32}
                              : !aie.objectfifo<memref<3xi32>>
    // expected-error@+1 {{cannot allocate a shared memory module to objectfifo using stream port}}
    aie.objectfifo.allocate @of_stream (%tile13)
  }
}
