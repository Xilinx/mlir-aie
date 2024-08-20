//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_2col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t03 = aie.tile(0, 3)
    %t13 = aie.tile(1, 3)
    %t12 = aie.tile(1, 2)

    aie.cascade_flow(%t03, %t13)
    aie.cascade_flow(%t13, %t12)
  
    aie.objectfifo @objFifo_in0(%t00, {%t01}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @objFifo_in1(%t01, {%t03}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@objFifo_in0] -> [@objFifo_in1] ([] [])

    aie.objectfifo @objFifo_out1(%t12, {%t01}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @objFifo_out0(%t01, {%t00}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@objFifo_out1] -> [@objFifo_out0] ([] [])

    func.func private @extern_kernel1() -> ()
    func.func private @extern_kernel2() -> ()
    func.func private @extern_kernel3(%b: memref<64xi32>, %size: i32) -> ()
  
    %core02 = aie.core(%t03) {
      %subview0 = aie.objectfifo.acquire @objFifo_in1(Consume, 1) : !aie.objectfifosubview<memref<64xi32>>

      func.call @extern_kernel1() : () -> ()

      aie.objectfifo.release @objFifo_in1(Consume, 1)

      aie.end
    } { link_with="kernel1.o" }

    %core13 = aie.core(%t13) {
      func.call @extern_kernel2() : () -> ()

      aie.end
    } { link_with="kernel2.o" }

    %core12 = aie.core(%t12) {
      %size = arith.constant 64 : i32
 
      %subview1 = aie.objectfifo.acquire @objFifo_out1(Produce, 1) : !aie.objectfifosubview<memref<64xi32>>
      %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>

      func.call @extern_kernel3(%elem1, %size) : (memref<64xi32>, i32) -> ()

      aie.objectfifo.release @objFifo_out1(Produce, 1)
      aie.end
    } { link_with="kernel3.o" }

    aiex.runtime_sequence(%in : memref<64xi32>, %buf : memref<32xi32>, %out : memref<64xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd (0, 0, %out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0, %c1]) { metadata = @objFifo_out0, id = 1 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_memcpy_nd (0, 0, %in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0, %c1]) { metadata = @objFifo_in0, id = 0 : i64 } : memref<64xi32>
      aiex.npu.dma_wait { symbol = @objFifo_out0 }
    }
  }
}
