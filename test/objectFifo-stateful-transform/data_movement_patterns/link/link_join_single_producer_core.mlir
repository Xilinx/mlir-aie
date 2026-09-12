//===- link_join_single_producer_core.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// Both join sources come from one core, which fills them in turn inside a
// loop. The two fifos leave the same tile and land in adjacent slices of one
// MemTile buffer, so the producer tile needs two MM2S chains rather than the
// one it would need if the sources sat on separate tiles.

// CHECK-LABEL: @joinSingleProducer
// Both sources leave the one producer tile, so it needs two MM2S chains.
// CHECK:         aie.mem(%{{.*}}tile_0_2)
// CHECK:           aie.dma_start(MM2S, 0
// CHECK:           aie.dma_bd(%{{.*}}of_a_buff_0 : memref<64xi8> offset = 0 len = 64)
// CHECK:           aie.dma_bd(%{{.*}}of_a_buff_1 : memref<64xi8> offset = 0 len = 64)
// CHECK:           aie.dma_start(MM2S, 1
// CHECK:           aie.dma_bd(%{{.*}}of_b_buff_0 : memref<64xi8> offset = 0 len = 64)
// CHECK:           aie.dma_bd(%{{.*}}of_b_buff_1 : memref<64xi8> offset = 0 len = 64)

// The MemTile lands them in adjacent slices of one joined object.
// CHECK:         aie.memtile_dma(%{{.*}}mem_tile_0_1)
// CHECK:           aie.dma_start(S2MM, 0
// CHECK:           aie.dma_bd(%{{.*}}of_out_buff_0 : memref<128xi8> offset = 0 len = 64)
// CHECK:           aie.dma_start(S2MM, 1
// CHECK:           aie.dma_bd(%{{.*}}of_out_buff_0 : memref<128xi8> offset = 64 len = 64)

module @joinSingleProducer {
 aie.device(npu2) {
    %tile00 = aie.tile(0, 0)
    %tile01 = aie.tile(0, 1)
    %tile02 = aie.tile(0, 2)

    aie.objectfifo @of_a (%tile02, {%tile01}, 2 : i32) : !aie.objectfifo<memref<64xi8>>
    aie.objectfifo @of_b (%tile02, {%tile01}, 2 : i32) : !aie.objectfifo<memref<64xi8>>
    aie.objectfifo @of_out (%tile01, {%tile00}, 2 : i32) : !aie.objectfifo<memref<128xi8>>
    aie.objectfifo.link [@of_a, @of_b] -> [@of_out] ([0, 64][])

    %core02 = aie.core(%tile02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %v = arith.constant 3 : i8
      scf.for %i = %c0 to %c4 step %c1 {
        %ea = aie.objectfifo.acquire @of_a (Produce, 1) : memref<64xi8>
        memref.store %v, %ea[%c0] : memref<64xi8>
        aie.objectfifo.release @of_a (Produce, 1)
        %eb = aie.objectfifo.acquire @of_b (Produce, 1) : memref<64xi8>
        memref.store %v, %eb[%c0] : memref<64xi8>
        aie.objectfifo.release @of_b (Produce, 1)
      }
      aie.end
    }
 }
}
