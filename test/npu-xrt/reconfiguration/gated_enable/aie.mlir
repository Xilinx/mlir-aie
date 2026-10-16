// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A control packet reconfiguration of a two-column pipeline, host -> shim (0, 0) ->
// memory tile (0, 1) -> core (1, 2), which adds 1 -> shim (1, 0) -> host. The
// core waits for data only the host starts, so aie-check-core-start-gating
// proves it gated and its enable goes with column 1's control packets.

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%in : memref<256xi32>, %out : memref<256xi32>) {
      aiex.configure @cfg {
        aiex.run @run(%in, %out) : (memref<256xi32>, memref<256xi32>)
      }
    }
  }

  aie.device(npu2) @cfg {
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %mem = aie.tile(0, 1)
    %comp = aie.tile(1, 2)
    aie.objectfifo @in(%shim0, {%mem}, 1 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @relay(%mem, {%comp}, 1 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo.link [@in] -> [@relay] ([] [])
    aie.objectfifo @out(%comp, {%shim1}, 1 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.core(%comp) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 256 : index
      %one = arith.constant 1 : i32
      %bin = aie.objectfifo.acquire @relay(Consume, 1) : memref<256xi32>
      %bout = aie.objectfifo.acquire @out(Produce, 1) : memref<256xi32>
      scf.for %k = %c0 to %cn step %c1 {
        %v = memref.load %bin[%k] : memref<256xi32>
        %r = arith.addi %v, %one : i32
        memref.store %r, %bout[%k] : memref<256xi32>
      }
      aie.objectfifo.release @relay(Consume, 1)
      aie.objectfifo.release @out(Produce, 1)
      aie.end
    }
    aie.runtime_sequence @run(%a : memref<256xi32>, %b : memref<256xi32>) {
      %t_in = aiex.dma_configure_task_for @in {
        aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @out {
        aie.dma_bd(%b : memref<256xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_in)
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
      aiex.dma_free_task(%t_in)
      aiex.dma_free_task(%t_out)
    }
  }
}
