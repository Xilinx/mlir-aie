// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A reconfiguration-example-shaped input: the host entry sequence carries the
// default name `sequence` (the examples emit every config from ONE template, so
// all N share this name). Folded with its sibling, the two `sequence` names
// collide -- the union must uniquify them to configs_1/configs_2.
module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%argout_a : memref<4xi32>) {
      aiex.configure @cfg_a {
        aiex.run @cfg_a_run(%argout_a) : (memref<4xi32>)
      }
    }
  }
  aie.device(npu2) @cfg_a {
    %shim_a = aie.tile(0, 0)
    %comp_a = aie.tile(0, 2)
    aie.objectfifo @out_a (%comp_a, {%shim_a}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.core(%comp_a) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      %cmax = arith.constant 1 : index
      %sent = arith.constant 11 : i32
      scf.for %it = %c0 to %cmax step %c1 {
        %e = aie.objectfifo.acquire @out_a (Produce, 1) : memref<4xi32>
        scf.for %k = %c0 to %cn step %c1 {
          memref.store %sent, %e[%k] : memref<4xi32>
        }
        aie.objectfifo.release @out_a (Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @cfg_a_run(%argout_a : memref<4xi32>) {
      %t_out_a = aiex.dma_configure_task_for @out_a {
        aie.dma_bd(%argout_a : memref<4xi32> offset = 0 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out_a)
      aiex.dma_await_task(%t_out_a)
      aiex.dma_free_task(%t_out_a)
    }
  }
}
