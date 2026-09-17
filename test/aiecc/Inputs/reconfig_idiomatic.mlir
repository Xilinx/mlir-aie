//===- reconfig_idiomatic.mlir --*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
  // Unplaced logical tiles, as real IRON `as_mlir()` output emits (see
  // tmp/sweep/vector_reduce_add.mlir); AIEPlaceTiles assigns coordinates
  // downstream of the reconfig-union fold this fixture guards.
  aie.device(npu2_1col) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %comp = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @out (%comp, {%shim}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.core(%comp) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      %cmax = arith.constant 1 : index
      %sent = arith.constant 11 : i32
      scf.for %it = %c0 to %cmax step %c1 {
        %e = aie.objectfifo.acquire @out (Produce, 1) : memref<4xi32>
        scf.for %k = %c0 to %cn step %c1 {
          memref.store %sent, %e[%k] : memref<4xi32>
        }
        aie.objectfifo.release @out (Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%argout : memref<4xi32>) {
      %t_out = aiex.dma_configure_task_for @out {
        aie.dma_bd(%argout : memref<4xi32> offset = 0 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
      aiex.dma_free_task(%t_out)
    }
  }
}
