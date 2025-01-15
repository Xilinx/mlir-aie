//===- objectfifo_link_no_shared_tile.mlir ----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s

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
