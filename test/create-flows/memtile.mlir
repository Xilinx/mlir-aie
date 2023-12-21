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
// CHECK: %[[T04:.*]] = aie.tile(0, 4)
// CHECK: %[[T03:.*]] = aie.tile(0, 3)
// CHECK: %[[T02:.*]] = aie.tile(0, 2)
// CHECK: %[[T01:.*]] = aie.tile(0, 1)
//
// CHECK: aie.flow(%[[T04]], DMA : 0, %[[T02]], DMA : 4)
// CHECK: aie.flow(%[[T04]], DMA : 1, %[[T02]], DMA : 5)
// CHECK: aie.flow(%[[T03]], DMA : 0, %[[T02]], DMA : 2)
// CHECK: aie.flow(%[[T03]], DMA : 1, %[[T02]], DMA : 3)
// CHECK: aie.flow(%[[T02]], DMA : 0, %[[T01]], DMA : 0)
// CHECK: aie.flow(%[[T02]], DMA : 1, %[[T01]], DMA : 1)
// CHECK: aie.flow(%[[T02]], DMA : 2, %[[T03]], DMA : 0)
// CHECK: aie.flow(%[[T02]], DMA : 3, %[[T03]], DMA : 1)
// CHECK: aie.flow(%[[T02]], DMA : 4, %[[T04]], DMA : 0)
// CHECK: aie.flow(%[[T02]], DMA : 5, %[[T04]], DMA : 1)
// CHECK: aie.flow(%[[T01]], DMA : 0, %[[T02]], DMA : 0)
// CHECK: aie.flow(%[[T01]], DMA : 1, %[[T02]], DMA : 1)

module {
    aie.device(xcve2802) {
        %t04 = aie.tile(0, 4)
        %t03 = aie.tile(0, 3)
        %t02 = aie.tile(0, 2)
        %t01 = aie.tile(0, 1)

        aie.flow(%t01, DMA : 0, %t02, DMA : 0)
        aie.flow(%t01, DMA : 1, %t02, DMA : 1)
        aie.flow(%t02, DMA : 0, %t01, DMA : 0)
        aie.flow(%t02, DMA : 1, %t01, DMA : 1)

        aie.flow(%t02, DMA : 2, %t03, DMA : 0)
        aie.flow(%t02, DMA : 3, %t03, DMA : 1)
        aie.flow(%t03, DMA : 0, %t02, DMA : 2)
        aie.flow(%t03, DMA : 1, %t02, DMA : 3)

        aie.flow(%t02, DMA : 4, %t04, DMA : 0)
        aie.flow(%t02, DMA : 5, %t04, DMA : 1)
        aie.flow(%t04, DMA : 0, %t02, DMA : 4)
        aie.flow(%t04, DMA : 1, %t02, DMA : 5)
    }
}

