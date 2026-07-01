//===- duplicate_link_test.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=false" %s 2>&1 | FileCheck %s

// CHECK:   error: 'aie.objectfifo' op objectfifo cannot be in more than one ObjectFifoLinkOp

module @duplicate_link {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)
        %tile33 = aie.tile(3, 3)

        aie.objectfifo @of0 (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of1 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of2 (%tile22, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.objectfifo.link [@of0] -> [@of1] ([][])
        aie.objectfifo.link [@of1] -> [@of2] ([][])
    }
}
