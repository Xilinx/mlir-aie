//===- acquire_release_bad.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// CHECK:   error: 'aie.objectfifo' op does not have enough depths specified for producer and for each consumer.

module @same_core {
    aie.device(xcve2302) {
        %tile12 = aie.tile(1, 2)
        %tile23 = aie.tile(2, 3)

        aie.objectfifo @of (%tile12, {%tile23}, [1, 4]) : !aie.objectfifo<memref<16xi32>>

        %core12 = aie.core(%tile12) {
            %subview0 = aie.objectfifo.acquire @of (Produce, 2) : !aie.objectfifosubview<memref<16xi32>>
            aie.objectfifo.release @of (Produce, 3)
            aie.end
        }
    }
}
