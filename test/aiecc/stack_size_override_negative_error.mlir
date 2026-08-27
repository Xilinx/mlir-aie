//===- stack_size_override_negative_error.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// python/dialects/aie.py's external_func() already rejects a negative
// stack_size_override before it reaches the IR (see its `>= 0` check), but
// that guard is bypassed by hand-written MLIR like this file. A negative
// override would subtract from whatever path reaches it in maxPathFrom,
// undercounting a core's real requirement -- exactly the direction this
// analysis must never be wrong in -- so aiecc must reject it directly
// instead of trusting the attribute.
//
// This is checked unconditionally for every func.func in the module before
// any core-level stack computation runs, so no real kernel object is needed
// to observe the error.

// RUN: not %aiecc %s 2>&1 | FileCheck %s

// CHECK: error: stack_size_override must be >= 0, got -1

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @entry_a(memref<512xi8>) attributes {stack_size_override = -1 : i32}

    %core_0_2 = aie.core(%tile_0_2) {
      %sv = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<512xi8>>
      %e = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<512xi8>> -> memref<512xi8>
      func.call @entry_a(%e) : (memref<512xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    }

    aie.runtime_sequence(%out : memref<512xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c512 = arith.constant 512 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c512][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<512xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
