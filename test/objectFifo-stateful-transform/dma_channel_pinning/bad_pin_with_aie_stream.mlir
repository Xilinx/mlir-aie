//===- bad_pin_with_aie_stream.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=false" %s 2>&1 | FileCheck %s

// aie_stream routes through stream ports, which bypass DMA channels, so pinning
// a DMA channel on the same endpoint is contradictory and is rejected.

// CHECK: error: 'aie.objectfifo' op cannot pin a DMA channel on an objectfifo that also uses aie_stream (stream ports bypass DMA channels)

module @bad_pin_with_aie_stream {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of (%tile12, {%tile33}, 2 : i32) {prod_dma_channel = 0 : i32, aie_stream = 0 : i32, aie_stream_port = 0 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}
