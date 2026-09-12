//===- nd_dma_join_output_dims.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// `dimensionsToStream` on the MemTile output of a join. The transform belongs
// to the fifo leaving the MemTile, so it has to reach the MM2S chain that
// drains both join slices rather than the two S2MM chains that fill them.
// Placement tests lower this shape already, but none of them looks at where
// the transform ends up.

// CHECK-LABEL: @joinOutputDims
// The two ingest chains fill their own slice and carry no transform.
// CHECK:         aie.memtile_dma(%{{.*}}mem_tile_2_1)
// CHECK:           aie.dma_start(S2MM, 0
// CHECK:           aie.dma_bd(%{{.*}}of_out_buff_0 : memref<32xi32> offset = 0 len = 16)
// CHECK:           aie.dma_start(S2MM, 1
// CHECK:           aie.dma_bd(%{{.*}}of_out_buff_0 : memref<32xi32> offset = 16 len = 16)

// The drain chain visits both slices of both objects, and every one of its
// descriptors carries the output transform.
// CHECK:           aie.dma_start(MM2S, 0
// CHECK:           aie.dma_bd(%{{.*}}of_out_buff_0 : memref<32xi32> offset = 0 len = 16 sizes = [2, 4] strides = [4, 1])
// CHECK:           aie.dma_bd(%{{.*}}of_out_buff_0 : memref<32xi32> offset = 16 len = 16 sizes = [2, 4] strides = [4, 1])
// CHECK:           aie.dma_bd(%{{.*}}of_out_buff_1 : memref<32xi32> offset = 0 len = 16 sizes = [2, 4] strides = [4, 1])
// CHECK:           aie.dma_bd(%{{.*}}of_out_buff_1 : memref<32xi32> offset = 16 len = 16 sizes = [2, 4] strides = [4, 1])

module @joinOutputDims {
 aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of_a (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_b (%tile23, {%tile21}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out (%tile21 dimensionsToStream [<size = 2, stride = 4>, <size = 4, stride = 1>],
                            {%tile20}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@of_a, @of_b] -> [@of_out] ([0, 16][])

    %core22 = aie.core(%tile22) {
      %e = aie.objectfifo.acquire @of_a (Produce, 1) : memref<16xi32>
      aie.objectfifo.release @of_a (Produce, 1)
      aie.end
    }

    %core23 = aie.core(%tile23) {
      %e = aie.objectfifo.acquire @of_b (Produce, 1) : memref<16xi32>
      aie.objectfifo.release @of_b (Produce, 1)
      aie.end
    }
 }
}
