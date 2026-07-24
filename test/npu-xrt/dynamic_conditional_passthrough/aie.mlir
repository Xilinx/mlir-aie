//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dynamic CONDITIONAL passthrough: the whole n-tile transfer is guarded by a
// runtime i1 %do_copy. When true, the runtime sequence configures/starts the
// output collect + the rolled input ping-pong (drawing BD ids from the dynamic
// free-list pool) and awaits them, all inside the scf.if branch. When false, no
// DMAs are issued and the output buffer stays as the host left it. The core is
// backpressure-driven (while_true), so it produces nothing when nothing is fed.
// Exercises runtime scf.if nesting an scf.for through the dynamic BD pool path.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu2) {
    %logical_core = aie.logical_tile<CoreTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_0 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @of_in(%logical_shim_noc, {%logical_core}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @of_out(%logical_core, {%logical_shim_noc_0}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    %0 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %cmax step %c1 {
        %1 = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<256xi32>>
        %2 = aie.objectfifo.subview.access %1[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
        %3 = aie.objectfifo.acquire @of_in(Consume, 1) : !aie.objectfifosubview<memref<256xi32>>
        %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
        %c0_1 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c1_2 = arith.constant 1 : index
        scf.for %arg1 = %c0_1 to %c256 step %c1_2 {
          %5 = memref.load %4[%arg1] : memref<256xi32>
          memref.store %5, %2[%arg1] : memref<256xi32>
        }
        aie.objectfifo.release @of_in(Consume, 1)
        aie.objectfifo.release @of_out(Produce, 1)
      }
      aie.end
    }
    // %n = runtime tile count; %do_copy = runtime predicate. The transfer is
    // only issued when %do_copy is true; the ping-pong scf.for nests in the if.
    aie.runtime_sequence(%in: memref<256xi32>, %out: memref<4096xi32>, %n: i64, %do_copy: i1) {
      scf.if %do_copy {
        %c1 = arith.constant 1 : index
        %c256_i32 = arith.constant 256 : i32
        %n_idx = arith.index_cast %n : i64 to index
        %n_i32 = arith.trunci %n : i64 to i32
        %len = arith.muli %n_i32, %c256_i32 : i32
        %out_task = aiex.dma_configure_task_for @of_out {
          aie.dma_bd(%out : memref<4096xi32> offset = 0 len = %len sizes = [1, 1, %n, 256] strides = [0, 0, 256, 1])
          aie.end
        } {issue_token = true}
        %init = aiex.dma_configure_task_for @of_in {
          aie.dma_bd(%in : memref<256xi32> offset = 0 len = 256 sizes = [1, 1, 1, 256] strides = [0, 0, 0, 1])
          aie.end
        } {issue_token = true}
        aiex.dma_start_task(%out_task)
        aiex.dma_start_task(%init)
        %last = scf.for %i = %c1 to %n_idx step %c1 iter_args(%prev = %init) -> (index) {
          %t = aiex.dma_configure_task_for @of_in {
            aie.dma_bd(%in : memref<256xi32> offset = 0 len = 256 sizes = [1, 1, 1, 256] strides = [0, 0, 0, 1])
            aie.end
          } {issue_token = true}
          aiex.dma_start_task(%t)
          aiex.dma_free_task(%prev)
          scf.yield %t : index
        }
        aiex.dma_await_task(%last)
        aiex.dma_await_task(%out_task)
        aiex.dma_free_task(%last)
      }
    }
  }
}
