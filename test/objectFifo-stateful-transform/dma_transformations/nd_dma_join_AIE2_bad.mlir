//===- nd_dma_join_AIE2_bad.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --verify-diagnostics %s

module @ndDMAObjFifoAIE2 {
 aie.device(xcve2302) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile22, {%tile11}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

    aie.objectfifo @of1 (%tile23, {%tile11}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

    aie.objectfifo @of2 (%tile11 dimensionsToStream [<size = 8, stride = 1>,
                                           <size = 3, stride = 8>],
                        {%tile10}, 2 : i32) : !aie.objectfifo<memref<24xi32>>
   // expected-error@+1 {{'aie.objectfifo.link' op specified output stride(s) and size(s) result in out of bounds access in input objectfifo buffer, for index 23 in memref of length 12.}}
   aie.objectfifo.link [ @of0, @of1 ] -> [ @of2 ] ([0, 12] [])
 }
}
