//===- allocate_delegate_unplaced.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --verify-diagnostics %s

// The split keeps an allocated fifo's objects on its delegate, so the delegate
// must be placed too, like the fifo's ends.

aie.device(npu2_1col) {
  %a = aie.tile(0, 2)
  %l = aie.tile(0, 3)
  %d = aie.logical_tile<CoreTile>(?, ?)
  aie.objectfifo @f(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  // expected-error@+1 {{delegate tile is not a placed aie.tile; run --aie-place-tiles before this pass}}
  aie.objectfifo.allocate @f (%d)
  aie.core(%a) {
    %0 = aie.objectfifo.acquire @f(Produce, 1) : memref<16xi32>
    aie.objectfifo.release @f(Produce, 1)
    aie.end
  }
}
