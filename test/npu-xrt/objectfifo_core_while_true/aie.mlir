//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A core body written as an scf.while with a constant-true condition. The loop
// has no trip count, so the object fifo locks are the only thing that paces the
// core: it processes exactly the number of tiles the runtime sequence feeds it.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    %shim = aie.tile(0, 0)
    %core_tile = aie.tile(0, 2)

    aie.objectfifo @of_in(%shim, {%core_tile}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out(%core_tile, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core_tile) {
      %true = arith.constant true
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c41_i32 = arith.constant 41 : i32

      scf.while : () -> () {
        scf.condition(%true)
      } do {
        %in = aie.objectfifo.acquire @of_in(Consume, 1) : memref<16xi32>
        %out = aie.objectfifo.acquire @of_out(Produce, 1) : memref<16xi32>
        scf.for %i = %c0 to %c16 step %c1 {
          %v = memref.load %in[%i] : memref<16xi32>
          %w = arith.addi %v, %c41_i32 : i32
          memref.store %w, %out[%i] : memref<16xi32>
        }
        aie.objectfifo.release @of_in(Consume, 1)
        aie.objectfifo.release @of_out(Produce, 1)
        scf.yield
      }

      aie.end
    }

    aie.runtime_sequence(%in : memref<128xi32>, %out : memref<128xi32>) {
      %t_out = aiex.dma_configure_task_for @of_out {
        aie.dma_bd(%out : memref<128xi32> offset = 0 len = 128)
        aie.end
      } {issue_token = true}
      %t_in = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%in : memref<128xi32> offset = 0 len = 128)
        aie.end
      }
      aiex.dma_start_task(%t_out)
      aiex.dma_start_task(%t_in)
      aiex.dma_await_task(%t_out)
    }
  }
}
