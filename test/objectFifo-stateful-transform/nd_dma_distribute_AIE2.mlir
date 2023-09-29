//===- nd_dma_distribute_AIE2.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --verify-diagnostics %s  
// expected-error@+1 {{Specified stepsize(s) and wrap(s) result in out of bounds access in buffer }}

module @ndDMAObjFifoAIE2 {
 AIE.device(xcve2302) {
    %tile10 = AIE.tile(1, 0)
    %tile11 = AIE.tile(1, 1)
    %tile22 = AIE.tile(2, 2)
    %tile23 = AIE.tile(2, 3)

    AIE.objectFifo @of0 (%tile10, {%tile11 fromStream [<32, 16>,
                                                       < 8,  1>]}, 
                         2 : i32) : !AIE.objectFifo<memref<256xi32>>

    AIE.objectFifo @of1 (%tile11 toStream [< 4,64>,
                                           < 2, 4>, 
                                           < 8, 8>, 
                                           < 4, 1>],
                        {%tile22}, 2 : i32) : !AIE.objectFifo<memref<128xi32>>

    AIE.objectFifo @of2 (%tile11 toStream [< 4,64>,
                                           < 2, 4>, 
                                           < 8, 8>, 
                                           < 4, 1>],
                        {%tile23}, 2 : i32) : !AIE.objectFifo<memref<128xi32>>
 }
}
