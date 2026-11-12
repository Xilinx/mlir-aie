// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// @leave's run sends three 16-element objects to a core and drains one result.
// The core goes on to the second object, whose result has no drain left and
// stays in the streams to the shim. A reset discards it. A teardown, which
// disables the ports and resets the DMA channels, does not, and @use's drain,
// on the same route, reads it first.

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%in : memref<64xi32>, %out : memref<32xi32>) {
      aiex.configure @leave {
        %in_sub = memref.subview %in[0] [48] [1] : memref<64xi32> to memref<48xi32, strided<[1], offset: 0>>
        %in_slice = memref.reinterpret_cast %in_sub to offset: [0], sizes: [48], strides: [1] : memref<48xi32, strided<[1], offset: 0>> to memref<48xi32>
        %out_sub = memref.subview %out[0] [16] [1] : memref<32xi32> to memref<16xi32, strided<[1], offset: 0>>
        %out_slice = memref.reinterpret_cast %out_sub to offset: [0], sizes: [16], strides: [1] : memref<16xi32, strided<[1], offset: 0>> to memref<16xi32>
        aiex.run @run(%in_slice, %out_slice) : (memref<48xi32>, memref<16xi32>)
      }
      aiex.configure @use {
        %in_sub = memref.subview %in[48] [16] [1] : memref<64xi32> to memref<16xi32, strided<[1], offset: 48>>
        %in_slice = memref.reinterpret_cast %in_sub to offset: [0], sizes: [16], strides: [1] : memref<16xi32, strided<[1], offset: 48>> to memref<16xi32>
        %out_sub = memref.subview %out[16] [16] [1] : memref<32xi32> to memref<16xi32, strided<[1], offset: 16>>
        %out_slice = memref.reinterpret_cast %out_sub to offset: [0], sizes: [16], strides: [1] : memref<16xi32, strided<[1], offset: 16>> to memref<16xi32>
        aiex.run @run(%in_slice, %out_slice) : (memref<16xi32>, memref<16xi32>)
      }
    }
  }

  aie.device(npu2) @leave {
    %shim = aie.tile(0, 0)
    %comp = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%comp}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%comp, {%shim}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%comp) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 16 : index
      %true = arith.constant true
      %c100 = arith.constant 100 : i32
      scf.while : () -> () {
        scf.condition(%true)
      } do {
        %ein = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
        %eout = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        scf.for %k = %c0 to %cn step %c1 {
          %v = memref.load %ein[%k] : memref<16xi32>
          %r = arith.addi %v, %c100 : i32
          memref.store %r, %eout[%k] : memref<16xi32>
        }
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @run(%a : memref<48xi32>, %b : memref<16xi32>) {
      %t_in = aiex.dma_configure_task_for @in {
        aie.dma_bd(%a : memref<48xi32> offset = 0 len = 48)
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

  aie.device(npu2) @use {
    %shim = aie.tile(0, 0)
    %comp = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%comp}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%comp, {%shim}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%comp) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 16 : index
      %true = arith.constant true
      %c100 = arith.constant 100 : i32
      scf.while : () -> () {
        scf.condition(%true)
      } do {
        %ein = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
        %eout = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        scf.for %k = %c0 to %cn step %c1 {
          %v = memref.load %ein[%k] : memref<16xi32>
          %r = arith.addi %v, %c100 : i32
          memref.store %r, %eout[%k] : memref<16xi32>
        }
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
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
