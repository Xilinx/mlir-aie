// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Two configurations whose two inputs leave column 0's shim on both MM2S
// channels. Configuration 1 writes a + b and configuration 2 writes a - b, each
// to its 16-element slice of c. In configuration 1 both inputs go straight to
// the core; in configuration 2, @a goes through the memtile.

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%a : memref<32xi32>, %b : memref<32xi32>, %c : memref<32xi32>) {
      aiex.configure @cfg_add {
        %a_sub = memref.subview %a[0] [16] [1] : memref<32xi32> to memref<16xi32, strided<[1], offset: 0>>
        %a_slice = memref.reinterpret_cast %a_sub to offset: [0], sizes: [16], strides: [1] : memref<16xi32, strided<[1], offset: 0>> to memref<16xi32>
        %b_sub = memref.subview %b[0] [16] [1] : memref<32xi32> to memref<16xi32, strided<[1], offset: 0>>
        %b_slice = memref.reinterpret_cast %b_sub to offset: [0], sizes: [16], strides: [1] : memref<16xi32, strided<[1], offset: 0>> to memref<16xi32>
        %c_sub = memref.subview %c[0] [16] [1] : memref<32xi32> to memref<16xi32, strided<[1], offset: 0>>
        %c_slice = memref.reinterpret_cast %c_sub to offset: [0], sizes: [16], strides: [1] : memref<16xi32, strided<[1], offset: 0>> to memref<16xi32>
        aiex.run @run(%a_slice, %b_slice, %c_slice) : (memref<16xi32>, memref<16xi32>, memref<16xi32>)
      }
      aiex.configure @cfg_sub {
        %a_sub = memref.subview %a[16] [16] [1] : memref<32xi32> to memref<16xi32, strided<[1], offset: 16>>
        %a_slice = memref.reinterpret_cast %a_sub to offset: [0], sizes: [16], strides: [1] : memref<16xi32, strided<[1], offset: 16>> to memref<16xi32>
        %b_sub = memref.subview %b[16] [16] [1] : memref<32xi32> to memref<16xi32, strided<[1], offset: 16>>
        %b_slice = memref.reinterpret_cast %b_sub to offset: [0], sizes: [16], strides: [1] : memref<16xi32, strided<[1], offset: 16>> to memref<16xi32>
        %c_sub = memref.subview %c[16] [16] [1] : memref<32xi32> to memref<16xi32, strided<[1], offset: 16>>
        %c_slice = memref.reinterpret_cast %c_sub to offset: [0], sizes: [16], strides: [1] : memref<16xi32, strided<[1], offset: 16>> to memref<16xi32>
        aiex.run @run(%a_slice, %b_slice, %c_slice) : (memref<16xi32>, memref<16xi32>, memref<16xi32>)
      }
    }
  }

  aie.device(npu2) @cfg_add {
    %shim = aie.tile(0, 0)
    %comp = aie.tile(0, 2)
    aie.objectfifo @a(%shim, {%comp}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%shim, {%comp}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%comp, {%shim}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%comp) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 16 : index
      %true = arith.constant true
      scf.while : () -> () {
        scf.condition(%true)
      } do {
        %ea = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
        %eb = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        %ec = aie.objectfifo.acquire @c(Produce, 1) : memref<16xi32>
        scf.for %k = %c0 to %cn step %c1 {
          %va = memref.load %ea[%k] : memref<16xi32>
          %vb = memref.load %eb[%k] : memref<16xi32>
          %r = arith.addi %va, %vb : i32
          memref.store %r, %ec[%k] : memref<16xi32>
        }
        aie.objectfifo.release @a(Consume, 1)
        aie.objectfifo.release @b(Consume, 1)
        aie.objectfifo.release @c(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @run(%a : memref<16xi32>, %b : memref<16xi32>, %c : memref<16xi32>) {
      %t_a = aiex.dma_configure_task_for @a {
        aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      %t_b = aiex.dma_configure_task_for @b {
        aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      %t_c = aiex.dma_configure_task_for @c {
        aie.dma_bd(%c : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_a)
      aiex.dma_start_task(%t_b)
      aiex.dma_start_task(%t_c)
      aiex.dma_await_task(%t_c)
      aiex.dma_free_task(%t_a)
      aiex.dma_free_task(%t_b)
      aiex.dma_free_task(%t_c)
    }
  }

  aie.device(npu2) @cfg_sub {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %comp = aie.tile(0, 2)
    aie.objectfifo @a(%shim, {%mem}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a_core(%mem, {%comp}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a] -> [@a_core] ([] [])
    aie.objectfifo @b(%shim, {%comp}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%comp, {%shim}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%comp) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 16 : index
      %true = arith.constant true
      scf.while : () -> () {
        scf.condition(%true)
      } do {
        %ea = aie.objectfifo.acquire @a_core(Consume, 1) : memref<16xi32>
        %eb = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        %ec = aie.objectfifo.acquire @c(Produce, 1) : memref<16xi32>
        scf.for %k = %c0 to %cn step %c1 {
          %va = memref.load %ea[%k] : memref<16xi32>
          %vb = memref.load %eb[%k] : memref<16xi32>
          %r = arith.subi %va, %vb : i32
          memref.store %r, %ec[%k] : memref<16xi32>
        }
        aie.objectfifo.release @a_core(Consume, 1)
        aie.objectfifo.release @b(Consume, 1)
        aie.objectfifo.release @c(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @run(%a : memref<16xi32>, %b : memref<16xi32>, %c : memref<16xi32>) {
      %t_a = aiex.dma_configure_task_for @a {
        aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      %t_b = aiex.dma_configure_task_for @b {
        aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      %t_c = aiex.dma_configure_task_for @c {
        aie.dma_bd(%c : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_a)
      aiex.dma_start_task(%t_b)
      aiex.dma_start_task(%t_c)
      aiex.dma_await_task(%t_c)
      aiex.dma_free_task(%t_a)
      aiex.dma_free_task(%t_b)
      aiex.dma_free_task(%t_c)
    }
  }
}
