//===- reconfig_idiomatic_twochannel.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
  // Unplaced logical tiles (idiomatic single-device design, as real IRON
  // `as_mlir()` output emits; see reconfig_idiomatic.mlir). Two circuit
  // shim-ingress objectFifos on the same shim tile saturate both of column
  // 0's shim MM2S channels, leaving none free for the resident control
  // overlay -- the auto-packetize-control-ingress default-on/opt-out fixture
  // in reconfig_autopacketize_default.mlir.
  aie.device(npu2_1col) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %comp = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0 (%shim, {%comp}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @in1 (%shim, {%comp}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @out (%comp, {%shim}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.core(%comp) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      %cmax = arith.constant 1 : index
      scf.for %it = %c0 to %cmax step %c1 {
        %e0 = aie.objectfifo.acquire @in0 (Consume, 1) : memref<4xi32>
        %e1 = aie.objectfifo.acquire @in1 (Consume, 1) : memref<4xi32>
        %eo = aie.objectfifo.acquire @out (Produce, 1) : memref<4xi32>
        scf.for %k = %c0 to %cn step %c1 {
          %v0 = memref.load %e0[%k] : memref<4xi32>
          %v1 = memref.load %e1[%k] : memref<4xi32>
          %r = arith.addi %v0, %v1 : i32
          memref.store %r, %eo[%k] : memref<4xi32>
        }
        aie.objectfifo.release @in0 (Consume, 1)
        aie.objectfifo.release @in1 (Consume, 1)
        aie.objectfifo.release @out (Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a0 : memref<4xi32>, %a1 : memref<4xi32>, %argout : memref<4xi32>) {
      %t_in0 = aiex.dma_configure_task_for @in0 {
        aie.dma_bd(%a0 : memref<4xi32> offset = 0 len = 4)
        aie.end
      }
      %t_in1 = aiex.dma_configure_task_for @in1 {
        aie.dma_bd(%a1 : memref<4xi32> offset = 0 len = 4)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @out {
        aie.dma_bd(%argout : memref<4xi32> offset = 0 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_in0)
      aiex.dma_start_task(%t_in1)
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
      aiex.dma_free_task(%t_in0)
      aiex.dma_free_task(%t_in1)
      aiex.dma_free_task(%t_out)
    }
  }
}
