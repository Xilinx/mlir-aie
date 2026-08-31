//===- stack_size_override_negative_error.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// external_func() rejects a negative stack_size_override before it reaches the
// IR, and hand-written MLIR skips that check. A negative override subtracts
// from a path total in maxPathFrom and undercounts, so aiecc rejects it too.
// aiecc checks every func.func ahead of any core-level computation, so this
// test needs no kernel object. @entry_a has no definition, so `--get` stops the
// build at the check, ahead of the link.

// RUN: not %aiecc --get=measured_stack_sizes.mlir --output-dir=%t.out %s 2>&1 | FileCheck %s

// CHECK: error: stack_size_override must be >= 0, got -1

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @entry_a(memref<512xi8>) attributes {stack_size_override = -1 : i32}

    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<512xi8>
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
