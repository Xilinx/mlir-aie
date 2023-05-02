//===- memtile.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s
// CHECK: %[[T04:.*]] = AIE.tile(0, 4)
// CHECK: %[[T03:.*]] = AIE.tile(0, 3)
// CHECK: %[[T02:.*]] = AIE.tile(0, 2)
// CHECK: %[[T01:.*]] = AIE.tile(0, 1)
//
// CHECK: AIE.flow(%[[T01]], DMA : 0, %[[T04]], DMA : 0)
// CHECK: AIE.flow(%[[T01]], DMA : 1, %[[T04]], DMA : 1)
// CHECK: AIE.flow(%[[T04]], DMA : 0, %[[T01]], DMA : 0)
// CHECK: AIE.flow(%[[T04]], DMA : 1, %[[T01]], DMA : 1)
// CHECK: AIE.flow(%[[T01]], DMA : 2, %[[T03]], DMA : 0)
// CHECK: AIE.flow(%[[T01]], DMA : 3, %[[T03]], DMA : 1)
// CHECK: AIE.flow(%[[T03]], DMA : 0, %[[T01]], DMA : 2)
// CHECK: AIE.flow(%[[T03]], DMA : 1, %[[T01]], DMA : 3)
// CHECK: AIE.flow(%[[T01]], DMA : 4, %[[T02]], DMA : 0)
// CHECK: AIE.flow(%[[T01]], DMA : 5, %[[T02]], DMA : 1)
// CHECK: AIE.flow(%[[T02]], DMA : 0, %[[T01]], DMA : 4)
// CHECK: AIE.flow(%[[T02]], DMA : 1, %[[T01]], DMA : 5)

module {
    AIE.device(xcve2302) {
        %t04 = AIE.tile(0, 4)
        %t03 = AIE.tile(0, 3)
        %t02 = AIE.tile(0, 2)
        %t01 = AIE.tile(0, 1)

        AIE.flow(%t01, DMA : 0, %t04, DMA : 0)
        AIE.flow(%t01, DMA : 1, %t04, DMA : 1)
        AIE.flow(%t04, DMA : 0, %t01, DMA : 0)
        AIE.flow(%t04, DMA : 1, %t01, DMA : 1)

        AIE.flow(%t01, DMA : 2, %t03, DMA : 0)
        AIE.flow(%t01, DMA : 3, %t03, DMA : 1)
        AIE.flow(%t03, DMA : 0, %t01, DMA : 2)
        AIE.flow(%t03, DMA : 1, %t01, DMA : 3)

        AIE.flow(%t01, DMA : 4, %t02, DMA : 0)
        AIE.flow(%t01, DMA : 5, %t02, DMA : 1)
        AIE.flow(%t02, DMA : 0, %t01, DMA : 4)
        AIE.flow(%t02, DMA : 1, %t01, DMA : 5)
    }
}

