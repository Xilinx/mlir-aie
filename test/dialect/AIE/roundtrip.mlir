//===- roundtrip.mlir ------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file %s | FileCheck %s

// aie.objectfifo.link with multiple consumers with toStream
// CHECK: aie.device
// CHECK: %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK: %[[TILE_0_3:.+]] = aie.tile(0, 3)
// CHECK: %[[TILE_1_2:.+]] = aie.tile(1, 2)
// CHECK: %[[TILE_1_3:.+]] = aie.tile(1, 3)
// CHECK: %[[TILE_0_0:.+]] = aie.tile(0, 0)
// CHECK: %[[TILE_0_1:.+]] = aie.tile(0, 1)
// CHECK: aie.objectfifo @obj1(%[[TILE_0_0]], {%[[TILE_0_1]]}, 4 : i32) : !aie.objectfifo<memref<2048xi32, 1>>
// CHECK: aie.objectfifo @obj2(%[[TILE_0_1]] toStream [<size = 8, stride = 4>, <size = 32, stride = 32>, <size = 4, stride = 1>], {%[[TILE_0_2]], %[[TILE_0_3]]}, 4 : i32) : !aie.objectfifo<memref<1024xi32, 1>>
// CHECK: aie.objectfifo @obj3(%[[TILE_0_1]] toStream [<size = 8, stride = 4>, <size = 32, stride = 32>, <size = 4, stride = 1>], {%[[TILE_1_2]], %[[TILE_1_3]]}, 4 : i32) : !aie.objectfifo<memref<1024xi32, 1>>
// CHECK: aie.objectfifo.link [@obj1] -> [@obj2, @obj3]()
aie.device(npu1_4col) {
  memref.global "public" @out0 : memref<16xi32>
  %tile_0_2 = aie.tile(0, 2)
  %tile_0_3 = aie.tile(0, 3)
  %tile_1_2 = aie.tile(1, 2)
  %tile_1_3 = aie.tile(1, 3)
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  aie.objectfifo @obj1(%tile_0_0, {%tile_0_1}, 4 : i32) : !aie.objectfifo<memref<2048xi32, 1>>
  aie.objectfifo @obj2(%tile_0_1 toStream [<size = 8, stride = 4>, <size = 32, stride = 32>, <size = 4, stride = 1>], {%tile_0_2, %tile_0_3}, 4 : i32) : !aie.objectfifo<memref<1024xi32, 1>>
  aie.objectfifo @obj3(%tile_0_1 toStream [<size = 8, stride = 4>, <size = 32, stride = 32>, <size = 4, stride = 1>], {%tile_1_2, %tile_1_3}, 4 : i32) : !aie.objectfifo<memref<1024xi32, 1>>
  aie.objectfifo.link [@obj1] -> [@obj2, @obj3]()
}
