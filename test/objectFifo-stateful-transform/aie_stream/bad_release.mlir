//===- bad_release.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=false" --verify-diagnostics %s

module @bad_release {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of_stream (%tile12, {%tile33}, 2 : i32) {aie_stream = 1 : i32, aie_stream_port = 0 : i32} : !aie.objectfifo<memref<16xi32>>

    %core33 = aie.core(%tile33) {
      // expected-error@+1 {{'aie.objectfifo.release' op cannot release from objectfifo stream port}}
      aie.objectfifo.release @of_stream (Consume, 1)
      aie.end
    }
  }
}
