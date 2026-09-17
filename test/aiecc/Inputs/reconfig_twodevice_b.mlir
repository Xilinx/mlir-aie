// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
module {
  aie.device(npu2) @main {
    aie.runtime_sequence @configs_2(%argout_b : memref<4xi32>) {
      aiex.configure @cfg_b {
        aiex.run @cfg_b_run(%argout_b) : (memref<4xi32>)
      }
    }
  }
  aie.device(npu2) @cfg_b {
    %shim_b = aie.tile(0, 0)
    %comp_b = aie.tile(1, 2)
    aie.objectfifo @out_b (%comp_b, {%shim_b}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.core(%comp_b) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      %cmax = arith.constant 1 : index
      %sent = arith.constant 12 : i32
      scf.for %it = %c0 to %cmax step %c1 {
        %e = aie.objectfifo.acquire @out_b (Produce, 1) : memref<4xi32>
        scf.for %k = %c0 to %cn step %c1 {
          memref.store %sent, %e[%k] : memref<4xi32>
        }
        aie.objectfifo.release @out_b (Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @cfg_b_run(%argout_b : memref<4xi32>) {
      %t_out_b = aiex.dma_configure_task_for @out_b {
        aie.dma_bd(%argout_b : memref<4xi32> offset = 0 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out_b)
      aiex.dma_await_task(%t_out_b)
      aiex.dma_free_task(%t_out_b)
    }
  }
}
