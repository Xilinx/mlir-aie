//===- broadcast_error.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: June 6th 2023
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=false" --aie-assign-lock-ids %s 2>&1 | FileCheck %s

// CHECK:   error: 'aie.objectfifo' op does not have enough depths specified for producer and for each consumer.

module @broadcast_error {
    aie.device(xcvc1902) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)
        %tile14 = aie.tile(1, 4)
        %tile32 = aie.tile(3, 2)
        %tile33 = aie.tile(3, 3)

        aie.objectfifo @broadcast_of (%tile13, {%tile12, %tile14, %tile32, %tile33}, [2, 2, 3]) : !aie.objectfifo<memref<16xi32>>
    }
}
