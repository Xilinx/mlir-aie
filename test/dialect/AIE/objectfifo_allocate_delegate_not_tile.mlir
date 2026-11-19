//===- objectfifo_allocate_delegate_not_tile.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

aie.device(npu2_1col) {
  %a = aie.tile(0, 2)
  %l = aie.tile(0, 3)
  %c = arith.constant 3 : index
  aie.objectfifo @f(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  // expected-error@+1 {{'aie.objectfifo.allocate' op delegate must be a tile}}
  aie.objectfifo.allocate @f (%c)
  aie.core(%a) {
    %0 = aie.objectfifo.acquire @f(Produce, 1) : memref<16xi32>
    aie.objectfifo.release @f(Produce, 1)
    aie.end
  }
}
