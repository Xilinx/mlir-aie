// REQUIRES: aie_found
// RUN: phy-opt --inline --convert-physical-to-aie --symbol-dce %s | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK: %[[Tile:.*]] = AIE.tile(6, 3)

// CHECK: %[[Lock:.*]] = AIE.lock(%[[Tile]], 0)
// CHECK: AIE.useLock(%[[Lock]], Release, 1)
%L = physical.lock<1>() { aie.tile = "6.3", aie.id = "0" }

// CHECK: %[[BufA:.*]] = AIE.buffer(%[[Tile]])
// CHECK: %[[BufB:.*]] = AIE.buffer(%[[Tile]])
%A  = physical.buffer() { aie.tile = "6.3" }: memref<1024xi32>
%B  = physical.buffer() { aie.tile = "6.3" }: memref<1024xi32>

func.func private @extern_kernel(%0: memref<1024xi32>, %1: memref<1024xi32>) -> ()

// CHECK: %[[BufB:.*]] = AIE.core(%[[Tile]]) {
// CHECK:   cf.br ^[[bb:.*]]
// CHECK: ^[[bb]]:
// CHECK:   func.call @extern_kernel
// CHECK:   cf.br ^[[bb:.*]]
// CHECK: }

func.func private @kernel_middle(%0: memref<1024xi32>, %1: memref<1024xi32>, %2: !physical.lock) {
  func.call @kernel_internal(%0, %1, %2) : (memref<1024xi32>, memref<1024xi32>, !physical.lock) -> ()
  func.return
}

func.func private @kernel_internal(%0: memref<1024xi32>, %1: memref<1024xi32>, %2: !physical.lock) {
  cf.br ^br0
^br0:
  func.call @extern_kernel(%0, %1) : (memref<1024xi32>, memref<1024xi32>) -> ()
  cf.br ^br0
}

func.func private @kernel(%0: memref<1024xi32>, %1: memref<1024xi32>, %2: !physical.lock) {
  cf.br ^br0
^br0:
  func.call @kernel_middle(%0, %1, %2) : (memref<1024xi32>, memref<1024xi32>, !physical.lock) -> ()
  cf.br ^br0
}

physical.core @kernel(%A, %B, %L) { aie.tile = "6.3" }: (memref<1024xi32>, memref<1024xi32>, !physical.lock) -> !physical.core
