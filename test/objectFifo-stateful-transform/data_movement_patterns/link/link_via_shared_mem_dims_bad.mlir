//===- link_via_shared_dims_bad.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --verify-diagnostics %s

module @link_AIE2 {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile12 = aie.tile(1, 2)
        %tile22 = aie.tile(2, 2)

        aie.objectfifo @of1 (%tile20, {%tile12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        // expected-error@+1 {{'aie.objectfifo' op no access to shared memory module specified by `via_shared_mem`}}
        aie.objectfifo @of2 (%tile12 dimensionsToStream [<size = 1, stride = 2>], {%tile22}, 2 : i32) {via_shared_mem = 1 : i32} : !aie.objectfifo<memref<16xi32>>
        
        aie.objectfifo.link [@of1] -> [@of2] ([] [])
    }
}