// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Three configurations of one pipeline, host -> memtile (0, 1) -> core (0, 2) ->
// host, differing in the order the memtile hands each 16-element slice to the
// core; configuration i writes in[order_i[m]] + 100 to out[m]. Each core waits
// for up to 8 buffers, so the memtile's relay channel is still busy when the
// next configuration starts. Loading the next ELF resets the core tile, but
// only the teardown resets the memtile's channel.

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%in : memref<48xi32>, %out : memref<48xi32>) {
      aiex.configure @cfg_1 {
        %in_sub = memref.subview %in[0] [16] [1] : memref<48xi32> to memref<16xi32, strided<[1], offset: 0>>
        %in_slice = memref.reinterpret_cast %in_sub to offset: [0], sizes: [16], strides: [1] : memref<16xi32, strided<[1], offset: 0>> to memref<16xi32>
        %out_sub = memref.subview %out[0] [16] [1] : memref<48xi32> to memref<16xi32, strided<[1], offset: 0>>
        %out_slice = memref.reinterpret_cast %out_sub to offset: [0], sizes: [16], strides: [1] : memref<16xi32, strided<[1], offset: 0>> to memref<16xi32>
        aiex.run @run(%in_slice, %out_slice) : (memref<16xi32>, memref<16xi32>)
      }
      aiex.configure @cfg_2 {
        %in_sub = memref.subview %in[16] [16] [1] : memref<48xi32> to memref<16xi32, strided<[1], offset: 16>>
        %in_slice = memref.reinterpret_cast %in_sub to offset: [0], sizes: [16], strides: [1] : memref<16xi32, strided<[1], offset: 16>> to memref<16xi32>
        %out_sub = memref.subview %out[16] [16] [1] : memref<48xi32> to memref<16xi32, strided<[1], offset: 16>>
        %out_slice = memref.reinterpret_cast %out_sub to offset: [0], sizes: [16], strides: [1] : memref<16xi32, strided<[1], offset: 16>> to memref<16xi32>
        aiex.run @run(%in_slice, %out_slice) : (memref<16xi32>, memref<16xi32>)
      }
      aiex.configure @cfg_3 {
        %in_sub = memref.subview %in[32] [16] [1] : memref<48xi32> to memref<16xi32, strided<[1], offset: 32>>
        %in_slice = memref.reinterpret_cast %in_sub to offset: [0], sizes: [16], strides: [1] : memref<16xi32, strided<[1], offset: 32>> to memref<16xi32>
        %out_sub = memref.subview %out[32] [16] [1] : memref<48xi32> to memref<16xi32, strided<[1], offset: 32>>
        %out_slice = memref.reinterpret_cast %out_sub to offset: [0], sizes: [16], strides: [1] : memref<16xi32, strided<[1], offset: 32>> to memref<16xi32>
        aiex.run @run(%in_slice, %out_slice) : (memref<16xi32>, memref<16xi32>)
      }
    }
  }

  // In configuration 1 the memtile gathers in contiguous order.
  aie.device(npu2) @cfg_1 {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %comp = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%mem}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @relay(%mem dimensionsToStream [<size = 4, stride = 4>, <size = 4, stride = 1>], {%comp}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@relay] ([] [])
    aie.objectfifo @out(%comp, {%shim}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%comp) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 16 : index
      %cmax = arith.constant 8 : index
      %c100 = arith.constant 100 : i32
      scf.for %it = %c0 to %cmax step %c1 {
        %ein = aie.objectfifo.acquire @relay(Consume, 1) : memref<16xi32>
        %eout = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        scf.for %k = %c0 to %cn step %c1 {
          %v = memref.load %ein[%k] : memref<16xi32>
          %r = arith.addi %v, %c100 : i32
          memref.store %r, %eout[%k] : memref<16xi32>
        }
        aie.objectfifo.release @relay(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence @run(%a : memref<16xi32>, %b : memref<16xi32>) {
      %t_in = aiex.dma_configure_task_for @in {
        aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @out {
        aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_in)
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
      aiex.dma_free_task(%t_in)
      aiex.dma_free_task(%t_out)
    }
  }

  // In configuration 2 the memtile gathers in 4x4 transpose order.
  aie.device(npu2) @cfg_2 {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %comp = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%mem}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @relay(%mem dimensionsToStream [<size = 4, stride = 1>, <size = 4, stride = 4>], {%comp}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@relay] ([] [])
    aie.objectfifo @out(%comp, {%shim}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%comp) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 16 : index
      %cmax = arith.constant 8 : index
      %c100 = arith.constant 100 : i32
      scf.for %it = %c0 to %cmax step %c1 {
        %ein = aie.objectfifo.acquire @relay(Consume, 1) : memref<16xi32>
        %eout = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        scf.for %k = %c0 to %cn step %c1 {
          %v = memref.load %ein[%k] : memref<16xi32>
          %r = arith.addi %v, %c100 : i32
          memref.store %r, %eout[%k] : memref<16xi32>
        }
        aie.objectfifo.release @relay(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence @run(%a : memref<16xi32>, %b : memref<16xi32>) {
      %t_in = aiex.dma_configure_task_for @in {
        aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @out {
        aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_in)
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
      aiex.dma_free_task(%t_in)
      aiex.dma_free_task(%t_out)
    }
  }

  // In configuration 3 the memtile gathers in 2x8 transpose order.
  aie.device(npu2) @cfg_3 {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %comp = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%mem}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @relay(%mem dimensionsToStream [<size = 8, stride = 1>, <size = 2, stride = 8>], {%comp}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@relay] ([] [])
    aie.objectfifo @out(%comp, {%shim}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%comp) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 16 : index
      %cmax = arith.constant 8 : index
      %c100 = arith.constant 100 : i32
      scf.for %it = %c0 to %cmax step %c1 {
        %ein = aie.objectfifo.acquire @relay(Consume, 1) : memref<16xi32>
        %eout = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        scf.for %k = %c0 to %cn step %c1 {
          %v = memref.load %ein[%k] : memref<16xi32>
          %r = arith.addi %v, %c100 : i32
          memref.store %r, %eout[%k] : memref<16xi32>
        }
        aie.objectfifo.release @relay(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence @run(%a : memref<16xi32>, %b : memref<16xi32>) {
      %t_in = aiex.dma_configure_task_for @in {
        aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @out {
        aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
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
