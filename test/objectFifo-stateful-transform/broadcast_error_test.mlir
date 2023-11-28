//===- broadcast_error.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: June 6th 2023
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// CHECK:   error: 'AIE.objectfifo' op does not have enough depths specified for producer and for each consumer.

module @broadcast_error {
    AIE.device(xcvc1902) {
        %tile12 = AIE.tile(1, 2)
        %tile13 = AIE.tile(1, 3)
        %tile14 = AIE.tile(1, 4)
        %tile32 = AIE.tile(3, 2)
        %tile33 = AIE.tile(3, 3)

        AIE.objectfifo @broadcast_of (%tile13, {%tile12, %tile14, %tile32, %tile33}, [2, 2, 3]) : !AIE.objectfifo<memref<16xi32>>
    }
}
