// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// @a's core (0, 3) reads what core (0, 2) writes into (0, 2)'s memory, then
// loops forever, so @a ends with (0, 3) waiting on (0, 2)'s lock 1. @b programs
// only (0, 2) and uses its lock 1 for its own input. If the torn-down core's
// wait outlives the teardown, it takes @b's first release of lock 1 and @b
// stalls.

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%in : memref<16xi32>, %out : memref<32xi32>) {
      aiex.configure @a {
        %out_sub = memref.subview %out[0] [16] [1] : memref<32xi32> to memref<16xi32, strided<[1], offset: 0>>
        %out_slice = memref.reinterpret_cast %out_sub to offset: [0], sizes: [16], strides: [1] : memref<16xi32, strided<[1], offset: 0>> to memref<16xi32>
        aiex.run @run(%out_slice) : (memref<16xi32>)
      }
      aiex.configure @b {
        %out_sub = memref.subview %out[16] [16] [1] : memref<32xi32> to memref<16xi32, strided<[1], offset: 16>>
        %out_slice = memref.reinterpret_cast %out_sub to offset: [0], sizes: [16], strides: [1] : memref<16xi32, strided<[1], offset: 16>> to memref<16xi32>
        aiex.run @run(%in, %out_slice) : (memref<16xi32>, memref<16xi32>)
      }
    }
  }

  aie.device(npu2) @a {
    %shim = aie.tile(0, 0)
    %low = aie.tile(0, 2)
    %high = aie.tile(0, 3)
    %shared = aie.buffer(%low) {sym_name = "shared"} : memref<16xi32>
    %shared_prod = aie.lock(%low, 0) {init = 1 : i32, sym_name = "shared_prod"}
    %shared_cons = aie.lock(%low, 1) {init = 0 : i32, sym_name = "shared_cons"}
    %res = aie.buffer(%high) {sym_name = "res"} : memref<16xi32>
    %res_prod = aie.lock(%high, 0) {init = 1 : i32, sym_name = "res_prod"}
    %res_cons = aie.lock(%high, 1) {init = 0 : i32, sym_name = "res_cons"}
    aie.flow(%high, DMA : 0, %shim, DMA : 0)
    aie.shim_dma_allocation @a_out(%shim, S2MM, 0)
    aie.core(%low) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c7 = arith.constant 7 : i32
      aie.use_lock(%shared_prod, AcquireGreaterEqual, %one)
      scf.for %k = %c0 to %c16 step %c1 {
        %ki = arith.index_cast %k : index to i32
        %v = arith.addi %ki, %c7 : i32
        memref.store %v, %shared[%k] : memref<16xi32>
      }
      aie.use_lock(%shared_cons, Release, %one)
      aie.end
    }
    aie.core(%high) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %true = arith.constant true
      scf.while : () -> () {
        scf.condition(%true)
      } do {
        aie.use_lock(%shared_cons, AcquireGreaterEqual, %one)
        aie.use_lock(%res_prod, AcquireGreaterEqual, %one)
        scf.for %k = %c0 to %c16 step %c1 {
          %v = memref.load %shared[%k] : memref<16xi32>
          memref.store %v, %res[%k] : memref<16xi32>
        }
        aie.use_lock(%shared_prod, Release, %one)
        aie.use_lock(%res_cons, Release, %one)
        scf.yield
      }
      aie.end
    }
    aie.mem(%high) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.use_lock(%res_cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%res : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%res_prod, Release, %one)
      aie.next_bd ^bd
    ^end:
      aie.end
    }
    aie.runtime_sequence @run(%o : memref<16xi32>) {
      %t = aiex.dma_configure_task_for @a_out {
        aie.dma_bd(%o : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_free_task(%t)
    }
  }

  aie.device(npu2) @b {
    %shim = aie.tile(0, 0)
    %low = aie.tile(0, 2)
    %in_buf = aie.buffer(%low) {sym_name = "in_buf"} : memref<16xi32>
    %out_buf = aie.buffer(%low) {sym_name = "out_buf"} : memref<16xi32>
    %in_prod = aie.lock(%low, 0) {init = 1 : i32, sym_name = "in_prod"}
    %in_cons = aie.lock(%low, 1) {init = 0 : i32, sym_name = "in_cons"}
    %out_prod = aie.lock(%low, 2) {init = 1 : i32, sym_name = "out_prod"}
    %out_cons = aie.lock(%low, 3) {init = 0 : i32, sym_name = "out_cons"}
    aie.flow(%shim, DMA : 0, %low, DMA : 0)
    aie.flow(%low, DMA : 0, %shim, DMA : 0)
    aie.shim_dma_allocation @b_in(%shim, MM2S, 0)
    aie.shim_dma_allocation @b_out(%shim, S2MM, 0)
    aie.core(%low) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c100 = arith.constant 100 : i32
      %true = arith.constant true
      scf.while : () -> () {
        scf.condition(%true)
      } do {
        aie.use_lock(%in_cons, AcquireGreaterEqual, %one)
        aie.use_lock(%out_prod, AcquireGreaterEqual, %one)
        scf.for %k = %c0 to %c16 step %c1 {
          %v = memref.load %in_buf[%k] : memref<16xi32>
          %r = arith.addi %v, %c100 : i32
          memref.store %r, %out_buf[%k] : memref<16xi32>
        }
        aie.use_lock(%in_prod, Release, %one)
        aie.use_lock(%out_cons, Release, %one)
        scf.yield
      }
      aie.end
    }
    aie.mem(%low) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^in, ^next)
    ^in:
      aie.use_lock(%in_prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%in_buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%in_cons, Release, %one)
      aie.next_bd ^in
    ^next:
      %1 = aie.dma_start(MM2S, 0, ^out, ^end)
    ^out:
      aie.use_lock(%out_cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%out_buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%out_prod, Release, %one)
      aie.next_bd ^out
    ^end:
      aie.end
    }
    aie.runtime_sequence @run(%i : memref<16xi32>, %o : memref<16xi32>) {
      %t_in = aiex.dma_configure_task_for @b_in {
        aie.dma_bd(%i : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @b_out {
        aie.dma_bd(%o : memref<16xi32> offset = 0 len = 16)
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
