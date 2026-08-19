//===- link_core_access.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-objectfifo-split %s 2>&1 | FileCheck %s

// CHECK: core on the shared tile cannot access 'in': a link moves objects by DMA

module @core_consumes_link_input {
  aie.device(npu1) {
    %tile00 = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)
    %tile04 = aie.tile(0, 4)

    aie.objectfifo @in (%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out (%tile02, {%tile04}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@out] ([][])

    %core02 = aie.core(%tile02) {
      %e = aie.objectfifo.acquire @in(Consume) : memref<16xi32>
      aie.objectfifo.release @in(Consume) [1]
      aie.end
    }
  }
}

// -----

// CHECK: core on the shared tile cannot access 'out': a link moves objects by DMA

module @core_produces_link_output {
  aie.device(npu1) {
    %tile00 = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)
    %tile04 = aie.tile(0, 4)

    aie.objectfifo @in (%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out (%tile02, {%tile04}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@out] ([][])

    %core02 = aie.core(%tile02) {
      %e = aie.objectfifo.acquire @out(Produce) : memref<16xi32>
      aie.objectfifo.release @out(Produce) [1]
      aie.end
    }
  }
}
