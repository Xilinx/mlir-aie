//===- nd_dma_relay_consumer_dims.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// `dimensionsFromStream` on the MemTile consumer of a relay link. The MemTile
// is the consumer of the incoming fifo and the producer of the outgoing one,
// so the transform has to stay on the S2MM chain that fills it and not follow
// the data out.

// CHECK-LABEL: @relayConsumerDims
// CHECK:         aie.memtile_dma(%{{.*}}mem_tile_0_1)

// The transform stays on the chain that fills the MemTile.
// CHECK:           aie.dma_start(S2MM, 0
// CHECK:           aie.dma_bd(%{{.*}}of_in_cons_buff_0 : memref<32xi32> offset = 0 len = 32 sizes = [4, 8] strides = [8, 1])
// CHECK:           aie.dma_bd(%{{.*}}of_in_cons_buff_1 : memref<32xi32> offset = 0 len = 32 sizes = [4, 8] strides = [8, 1])

// It does not follow the data back out.
// CHECK:           aie.dma_start(MM2S, 0
// CHECK:           aie.dma_bd(%{{.*}}of_in_cons_buff_0 : memref<32xi32> offset = 0 len = 32)
// CHECK:           aie.dma_bd(%{{.*}}of_in_cons_buff_1 : memref<32xi32> offset = 0 len = 32)

module @relayConsumerDims {
 aie.device(npu1_1col) {
    %tile00 = aie.tile(0, 0)
    %tile01 = aie.tile(0, 1)
    %tile02 = aie.tile(0, 2)

    aie.objectfifo @of_in (%tile00,
                           {%tile01 dimensionsFromStream [<size = 4, stride = 8>, <size = 8, stride = 1>]},
                           2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @of_out (%tile01, {%tile02}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@of_in] -> [@of_out] ([] [0])

    %core02 = aie.core(%tile02) {
      %c0 = arith.constant 0 : index
      %e = aie.objectfifo.acquire @of_out (Consume, 1) : memref<32xi32>
      %v = memref.load %e[%c0] : memref<32xi32>
      aie.objectfifo.release @of_out (Consume, 1)
      aie.end
    }
 }
}
