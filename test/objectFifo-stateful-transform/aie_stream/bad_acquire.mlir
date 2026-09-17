//===- bad_acquire.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --verify-diagnostics %s

module @bad_acquire {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of_stream (%tile12, {%tile33}, 2 : i32) {aie_stream = 0 : i32, aie_stream_port = 0 : i32} : !aie.objectfifo<memref<16xi32>>

    %core12 = aie.core(%tile12) {
      // expected-error@+1 {{'aie.objectfifo.acquire' op cannot acquire from objectfifo stream port}}
      %subview0_obj0 = aie.objectfifo.acquire @of_stream (Produce, 1) : memref<16xi32>
      aie.end
    }
  }
}
