// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Two cores in one column run the same kernel, each adding 1, on the path
// host -> shim (0, 0) -> core (0, 2) -> core (0, 4) -> shim (0, 0) -> host, and
// control packets configure them. The overlay gives the column a
// control multicast to both cores, and the writes their configurations share go
// once on it.

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%in : memref<256xi32>, %out : memref<256xi32>) {
      aiex.configure @cfg {
        aiex.run @run(%in, %out) : (memref<256xi32>, memref<256xi32>)
      }
    }
  }

  aie.device(npu2) @cfg {
    %shim = aie.tile(0, 0)
    %c2 = aie.tile(0, 2)
    %c4 = aie.tile(0, 4)
    aie.objectfifo @in(%shim, {%c2}, 1 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @mid(%c2, {%c4}, 1 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @out(%c4, {%shim}, 1 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.core(%c2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 256 : index
      %one = arith.constant 1 : i32
      %bin = aie.objectfifo.acquire @in(Consume, 1) : memref<256xi32>
      %bout = aie.objectfifo.acquire @mid(Produce, 1) : memref<256xi32>
      scf.for %k = %c0 to %cn step %c1 {
        %v = memref.load %bin[%k] : memref<256xi32>
        %r = arith.addi %v, %one : i32
        memref.store %r, %bout[%k] : memref<256xi32>
      }
      aie.objectfifo.release @in(Consume, 1)
      aie.objectfifo.release @mid(Produce, 1)
      aie.end
    }
    aie.core(%c4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 256 : index
      %one = arith.constant 1 : i32
      %bin = aie.objectfifo.acquire @mid(Consume, 1) : memref<256xi32>
      %bout = aie.objectfifo.acquire @out(Produce, 1) : memref<256xi32>
      scf.for %k = %c0 to %cn step %c1 {
        %v = memref.load %bin[%k] : memref<256xi32>
        %r = arith.addi %v, %one : i32
        memref.store %r, %bout[%k] : memref<256xi32>
      }
      aie.objectfifo.release @mid(Consume, 1)
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
