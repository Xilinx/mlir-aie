//===- objectfifo_bad_link.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK:   error: ObjectFifoLinkOp does not support 'join' and 'distribute' at the same time

aie.device(xcve2302) {
   %tile20 = aie.tile(2, 0)
   %tile21 = aie.tile(2, 1)
   %tile22 = aie.tile(2, 2)
   %tile23 = aie.tile(2, 3)
   %tile33 = aie.tile(3, 3)

   aie.objectfifo @link1 (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
   aie.objectfifo @link2 (%tile23, {%tile21}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
   aie.objectfifo @link3 (%tile33, {%tile21}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

   aie.objectfifo @link4 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
   aie.objectfifo @link5 (%tile21, {%tile23}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
   aie.objectfifo @link6 (%tile21, {%tile33}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

   aie.objectfifo.link [@link1, @link2, @link3] -> [@link4, @link5, @link6] ([0, 16, 36][0, 16, 36])
}

// -----

// CHECK: ObjectFifoLinkOp must have a link point, i.e., a shared tile between objectFifos

aie.device(xcve2302) {
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of_in (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<12xi32>>
    aie.objectfifo @of_out (%tile33, {%tile21}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

    aie.objectfifo.link [@of_in] -> [@of_out] ([][])
}

// -----

// CHECK: ObjectFifoLinkOp must have a link point, i.e., a shared tile between objectFifos

aie.device(xcve2302) {
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile31 = aie.tile(3, 1)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @link1 (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link2 (%tile23, {%tile21}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link3 (%tile33, {%tile31}, 2 : i32) : !aie.objectfifo<memref<12xi32>>
    aie.objectfifo @link4 (%tile21, {%tile33}, 2 : i32) : !aie.objectfifo<memref<48xi32>>

    aie.objectfifo.link [@link1, @link2, @link3] -> [@link4] ([0, 16, 36][])
}

// -----

// CHECK: ObjectFifoLinkOp must have a link point, i.e., a shared tile between objectFifos

aie.device(xcve2302) {
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile31 = aie.tile(3, 1)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @link1 (%tile33, {%tile21}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
    aie.objectfifo @link2 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link3 (%tile21, {%tile23}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link4 (%tile31, {%tile33}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

    aie.objectfifo.link [@link1] -> [@link2, @link3, @link4] ([][0, 16, 36])
}

// -----

// CHECK: ObjectFifoLinkOp join and distribute are unavailable on compute or shim tiles

aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @link1 (%tile22, {%tile20}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link2 (%tile23, {%tile20}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link3 (%tile33, {%tile20}, 2 : i32) : !aie.objectfifo<memref<12xi32>>
    aie.objectfifo @link4 (%tile20, {%tile33}, 2 : i32) : !aie.objectfifo<memref<48xi32>>

    aie.objectfifo.link [@link1, @link2, @link3] -> [@link4] ([0, 16, 36][])
}

// -----

// CHECK: ObjectFifoLinkOp join and distribute are unavailable on compute or shim tiles

aie.device(xcve2302) {
    %tile32 = aie.tile(3, 2)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @link1 (%tile22, {%tile32}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link2 (%tile23, {%tile32}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link3 (%tile33, {%tile32}, 2 : i32) : !aie.objectfifo<memref<12xi32>>
    aie.objectfifo @link4 (%tile32, {%tile33}, 2 : i32) : !aie.objectfifo<memref<48xi32>>

    aie.objectfifo.link [@link1, @link2, @link3] -> [@link4] ([0, 16, 36][])
}

// -----

// CHECK: ObjectFifoLinkOp join and distribute are unavailable on compute or shim tiles

aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @link1 (%tile33, {%tile20}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
    aie.objectfifo @link2 (%tile20, {%tile22}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link3 (%tile20, {%tile23}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link4 (%tile20, {%tile33}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

    aie.objectfifo.link [@link1] -> [@link2, @link3, @link4] ([][0, 16, 36])
}

// -----

// CHECK: ObjectFifoLinkOp join and distribute are unavailable on compute or shim tiles

aie.device(xcve2302) {
    %tile32 = aie.tile(3, 2)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @link1 (%tile33, {%tile32}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
    aie.objectfifo @link2 (%tile32, {%tile22}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link3 (%tile32, {%tile23}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link4 (%tile32, {%tile33}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

    aie.objectfifo.link [@link1] -> [@link2, @link3, @link4] ([][0, 16, 36])
}

// -----

// CHECK: number of provided src offsets must be equal to the number of input objectFifos

aie.device(xcve2302) {
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @link1 (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link2 (%tile23, {%tile21}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link3 (%tile33, {%tile21}, 2 : i32) : !aie.objectfifo<memref<12xi32>>
    aie.objectfifo @link4 (%tile21, {%tile33}, 2 : i32) : !aie.objectfifo<memref<48xi32>>

    aie.objectfifo.link [@link1, @link2, @link3] -> [@link4] ([0, 16][])
}

// -----

// CHECK: dst offsets should be empty for join

aie.device(xcve2302) {
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @link1 (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link2 (%tile23, {%tile21}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link3 (%tile33, {%tile21}, 2 : i32) : !aie.objectfifo<memref<12xi32>>
    aie.objectfifo @link4 (%tile21, {%tile33}, 2 : i32) : !aie.objectfifo<memref<48xi32>>

    aie.objectfifo.link [@link1, @link2, @link3] -> [@link4] ([0, 16, 36][0, 16, 36])
}

// -----

// CHECK: number of provided dst offsets must be equal to the number of output objectFifos

aie.device(xcve2302) {
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @link1 (%tile33, {%tile21}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
    aie.objectfifo @link2 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link3 (%tile21, {%tile23}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link4 (%tile21, {%tile33}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

    aie.objectfifo.link [@link1] -> [@link2, @link3, @link4] ([][0, 16])
}

// -----

// CHECK: src offsets should be empty for distribute

aie.device(xcve2302) {
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @link1 (%tile33, {%tile21}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
    aie.objectfifo @link2 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link3 (%tile21, {%tile23}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link4 (%tile21, {%tile33}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

    aie.objectfifo.link [@link1] -> [@link2, @link3, @link4] ([0, 16, 36][0, 16, 36])
}

// -----

// CHECK: currently does not support objectFifos with dimensionsFromStreamPerConsumer for distribute input.

aie.device(xcve2302) {
   %tile23 = aie.tile(2, 3)
   %tile21 = aie.tile(2, 1)
   %tile13 = aie.tile(1, 3)
   %tile33 = aie.tile(3, 3)

   aie.objectfifo @of_in (%tile23, {%tile21 dimensionsFromStream [<size = 1, stride = 1>, <size = 1, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
   aie.objectfifo @of_out1 (%tile21, {%tile13}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
   aie.objectfifo @of_out2 (%tile21, {%tile33}, 2 : i32) : !aie.objectfifo<memref<8xi32>>

   aie.objectfifo.link [@of_in] -> [@of_out1, @of_out2] ([][0, 8])
}
