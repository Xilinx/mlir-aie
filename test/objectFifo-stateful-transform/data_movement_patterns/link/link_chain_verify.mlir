//===- link_chain_verify.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A fifo that is one link's output and another's input chains them: a join
// of joins.

module @chain_on_two_tiles {
  aie.device(npu2) {
    %shim = aie.tile(1, 0)
    %mem11 = aie.tile(1, 1)
    %mem21 = aie.tile(2, 1)
    %core12 = aie.tile(1, 2)
    %core13 = aie.tile(1, 3)
    aie.objectfifo @a (%core12, {%mem11}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b (%core13, {%mem11}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @mid (%mem11, {%mem21}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out (%mem21, {%shim}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@a, @b] -> [@mid] ([0, 16] [])
    aie.objectfifo.link [@mid] -> [@out] ([] [])
  }
}

// -----

// One link's output may not also be another's output.

module @output_of_two_links {
  aie.device(npu2) {
    %shim = aie.tile(1, 0)
    %mem11 = aie.tile(1, 1)
    %core12 = aie.tile(1, 2)
    %core13 = aie.tile(1, 3)
    aie.objectfifo @a (%core12, {%mem11}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b (%core13, {%mem11}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{'aie.objectfifo' op objectfifo cannot be filled by more than one ObjectFifoLinkOp}}
    aie.objectfifo @out (%mem11, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a] -> [@out] ([] [])
    aie.objectfifo.link [@b] -> [@out] ([] [])
  }
}
