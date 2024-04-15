//===- link_error.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// CHECK:   error: 'aie.objectfifo.acquire' op currently cannot access objectFifo used in ObjectFifoLinkOp

module {
  aie.device(ipu) {
    
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)

    aie.objectfifo @a(%tile_2_2, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @b(%tile_1_2 toStream [<size = 32, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>>

    aie.objectfifo.link [@a] -> [@b]()

    %core12 = aie.core(%tile_1_2) {
      %subview0 = aie.objectfifo.acquire @a (Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
      aie.end
    }

  }
}
