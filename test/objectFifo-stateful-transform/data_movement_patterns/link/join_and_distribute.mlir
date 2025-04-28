//===- join_and_distribute.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s



module @duplicate_link {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile30 = aie.tile(3, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)
        %tile33 = aie.tile(3, 3)

        aie.objectfifo @of_in1 (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of_in2 (%tile30, {%tile21}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of_out1 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of_out2 (%tile21, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.objectfifo.link [@of_in1, @of_in2] -> [@of_out1, @of_out2] ([0, 16][0, 16])
    }
}
