//===- shim_sharing.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// Three runtime sequences, one per member of a pack, each fills its own end and
// await nothing. One dispatch runs one sequence and finishes its transfers
// before the next starts, so any two may take turns on a channel. Only one
// pair needs to, so in_c keeps MM2S 1 to itself.
// CHECK-LABEL: module @pack
// CHECK-DAG:   aie.shim_dma_allocation @in_a_shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = {{[0-9]+}}>)
// CHECK-DAG:   aie.shim_dma_allocation @in_b_shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = {{[0-9]+}}>)
// CHECK-DAG:   aie.shim_dma_allocation @in_c_shim_alloc(%{{.*}}, MM2S, 1)
module @pack {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %c = aie.tile(0, 4)
    aie.objectfifo @in_a(%s, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_b(%s, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_c(%s, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %x = aie.objectfifo.acquire @in_a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_a(Consume, 1)
      aie.end
    }
    aie.core(%b) {
      %x = aie.objectfifo.acquire @in_b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_b(Consume, 1)
      aie.end
    }
    aie.core(%c) {
      %x = aie.objectfifo.acquire @in_c(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_c(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run_a(%in : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_a, id = 0 : i64} : memref<16xi32>
    }
    aie.runtime_sequence @run_b(%in : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_b, id = 0 : i64} : memref<16xi32>
    }
    aie.runtime_sequence @run_c(%in : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_c, id = 0 : i64} : memref<16xi32>
    }
  }
}

// -----

// Three shim ends need three S2MM channels on a shim tile that has two. Each
// drain is awaited before the next is issued, and a receiving end that is
// awaited is done, so out_a and out_b take turns on S2MM 0, their routes
// packet-switched to tell them apart.
// CHECK-LABEL: module @awaited
// CHECK-DAG:   aie.shim_dma_allocation @out_a_shim_alloc(%{{.*}}, S2MM, 0)
// CHECK-DAG:   aie.shim_dma_allocation @out_b_shim_alloc(%{{.*}}, S2MM, 0)
// CHECK-DAG:   aie.shim_dma_allocation @out_c_shim_alloc(%{{.*}}, S2MM, 1)
module @awaited {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %c = aie.tile(0, 4)
    aie.objectfifo @out_a(%a, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_b(%b, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_c(%c, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %x = aie.objectfifo.acquire @out_a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @out_a(Produce, 1)
      aie.end
    }
    aie.core(%b) {
      %x = aie.objectfifo.acquire @out_b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @out_b(Produce, 1)
      aie.end
    }
    aie.core(%c) {
      %x = aie.objectfifo.acquire @out_c(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @out_c(Produce, 1)
      aie.end
    }
    aie.runtime_sequence(%in : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @out_a, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out_a}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @out_b, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out_b}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @out_c, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out_c}
    }
  }
}

// -----

// Awaiting a sending end proves nothing: a shim MM2S task reports itself
// complete before its words leave the shim, so each fill may still be in
// flight when the next starts, and none share.
// CHECK: error: 'aie.tile' op number of output DMA channel exceeded!
module @sending_awaited {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %c = aie.tile(0, 4)
    aie.objectfifo @in_a(%s, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_b(%s, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_c(%s, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %x = aie.objectfifo.acquire @in_a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_a(Consume, 1)
      aie.end
    }
    aie.core(%b) {
      %x = aie.objectfifo.acquire @in_b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_b(Consume, 1)
      aie.end
    }
    aie.core(%c) {
      %x = aie.objectfifo.acquire @in_c(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_c(Consume, 1)
      aie.end
    }
    aie.runtime_sequence(%in : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_a, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @in_a}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_b, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @in_b}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_c, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @in_c}
    }
  }
}

// -----

// Runtime tasks freed rather than awaited: freeing lets a task's BD ids be
// reused, which is only safe once its transfers are done.
// CHECK-LABEL: module @freed
// CHECK-DAG:   aie.shim_dma_allocation @out_a_shim_alloc(%{{.*}}, S2MM, 0)
// CHECK-DAG:   aie.shim_dma_allocation @out_b_shim_alloc(%{{.*}}, S2MM, 0)
// CHECK-DAG:   aie.shim_dma_allocation @out_c_shim_alloc(%{{.*}}, S2MM, 1)
module @freed {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %c = aie.tile(0, 4)
    aie.objectfifo @out_a(%a, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_b(%b, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_c(%c, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %x = aie.objectfifo.acquire @out_a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @out_a(Produce, 1)
      aie.end
    }
    aie.core(%b) {
      %x = aie.objectfifo.acquire @out_b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @out_b(Produce, 1)
      aie.end
    }
    aie.core(%c) {
      %x = aie.objectfifo.acquire @out_c(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @out_c(Produce, 1)
      aie.end
    }
    aie.runtime_sequence(%in : memref<16xi32>) {
      %ta = aiex.dma_configure_task_for @out_a {
        aie.dma_bd(%in : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      aiex.dma_start_task(%ta)
      aiex.dma_free_task(%ta)
      %tb = aiex.dma_configure_task_for @out_b {
        aie.dma_bd(%in : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      aiex.dma_start_task(%tb)
      aiex.dma_free_task(%tb)
      %tc = aiex.dma_configure_task_for @out_c {
        aie.dma_bd(%in : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      aiex.dma_start_task(%tc)
      aiex.dma_free_task(%tc)
    }
  }
}

// -----

// Two shim tiles each run out of MM2S channels. The ends sharing one channel
// need ids that differ, but the other tile's sharing ends reuse them: the
// router keeps flows with one id apart wherever their destinations differ.
// CHECK-LABEL: module @reused_ids
// CHECK:       aie.shim_dma_allocation @in_a_shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = [[ID0:[0-9]+]]>)
// CHECK:       aie.shim_dma_allocation @in_b_shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = [[ID1:[0-9]+]]>)
// CHECK:       aie.shim_dma_allocation @in_c_shim_alloc(%{{.*}}, MM2S, 1)
// CHECK:       aie.shim_dma_allocation @in_d_shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = [[ID0]]>)
// CHECK:       aie.shim_dma_allocation @in_e_shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = [[ID1]]>)
// CHECK:       aie.shim_dma_allocation @in_f_shim_alloc(%{{.*}}, MM2S, 1)
module @reused_ids {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %c = aie.tile(0, 4)
    %d = aie.tile(1, 2)
    %e = aie.tile(1, 3)
    %f = aie.tile(1, 4)
    aie.objectfifo @in_a(%s0, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_b(%s0, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_c(%s0, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_d(%s1, {%d}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_e(%s1, {%e}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_f(%s1, {%f}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %x = aie.objectfifo.acquire @in_a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_a(Consume, 1)
      aie.end
    }
    aie.core(%b) {
      %x = aie.objectfifo.acquire @in_b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_b(Consume, 1)
      aie.end
    }
    aie.core(%c) {
      %x = aie.objectfifo.acquire @in_c(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_c(Consume, 1)
      aie.end
    }
    aie.core(%d) {
      %x = aie.objectfifo.acquire @in_d(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_d(Consume, 1)
      aie.end
    }
    aie.core(%e) {
      %x = aie.objectfifo.acquire @in_e(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_e(Consume, 1)
      aie.end
    }
    aie.core(%f) {
      %x = aie.objectfifo.acquire @in_f(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_f(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run_a(%in : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_d, id = 1 : i64} : memref<16xi32>
    }
    aie.runtime_sequence @run_b(%in : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_b, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_e, id = 1 : i64} : memref<16xi32>
    }
    aie.runtime_sequence @run_c(%in : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_c, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_f, id = 1 : i64} : memref<16xi32>
    }
  }
}

// -----

// out_a's transfer is never awaited, so it may still be in flight when the
// others go; out_b and out_c take turns on one channel and out_a has the other.
// CHECK-LABEL: module @partly_awaited
// CHECK-DAG:   aie.shim_dma_allocation @out_a_shim_alloc(%{{.*}}, S2MM, 0)
// CHECK-DAG:   aie.shim_dma_allocation @out_b_shim_alloc(%{{.*}}, S2MM, 1)
// CHECK-DAG:   aie.shim_dma_allocation @out_c_shim_alloc(%{{.*}}, S2MM, 1)
module @partly_awaited {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %c = aie.tile(0, 4)
    aie.objectfifo @out_a(%a, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_b(%b, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_c(%c, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %x = aie.objectfifo.acquire @out_a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @out_a(Produce, 1)
      aie.end
    }
    aie.core(%b) {
      %x = aie.objectfifo.acquire @out_b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @out_b(Produce, 1)
      aie.end
    }
    aie.core(%c) {
      %x = aie.objectfifo.acquire @out_c(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @out_c(Produce, 1)
      aie.end
    }
    aie.runtime_sequence(%in : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @out_a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @out_b, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out_b}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @out_c, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out_c}
    }
  }
}

// -----

// A loop issues each transfer again after the others' waits, so the ends
// inside it are in flight across one another's turns, and none share.
// CHECK: error: 'aie.tile' op number of output DMA channel exceeded!
module @looped {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %c = aie.tile(0, 4)
    aie.objectfifo @in_a(%s, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_b(%s, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_c(%s, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %x = aie.objectfifo.acquire @in_a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_a(Consume, 1)
      aie.end
    }
    aie.core(%b) {
      %x = aie.objectfifo.acquire @in_b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_b(Consume, 1)
      aie.end
    }
    aie.core(%c) {
      %x = aie.objectfifo.acquire @in_c(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_c(Consume, 1)
      aie.end
    }
    aie.runtime_sequence(%in : memref<16xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      scf.for %i = %c0 to %c2 step %c1 {
        aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_a, id = 0 : i64, issue_token = true} : memref<16xi32>
        aiex.npu.dma_wait {symbol = @in_a}
        aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_b, id = 0 : i64, issue_token = true} : memref<16xi32>
        aiex.npu.dma_wait {symbol = @in_b}
        aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_c, id = 0 : i64, issue_token = true} : memref<16xi32>
        aiex.npu.dma_wait {symbol = @in_c}
      }
    }
  }
}

// -----

// Nothing is awaited, so all three may be in flight at once.
// CHECK: error: 'aie.tile' op number of output DMA channel exceeded!
module @overlapping {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %c = aie.tile(0, 4)
    aie.objectfifo @in_a(%s, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_b(%s, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_c(%s, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %x = aie.objectfifo.acquire @in_a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_a(Consume, 1)
      aie.end
    }
    aie.core(%b) {
      %x = aie.objectfifo.acquire @in_b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_b(Consume, 1)
      aie.end
    }
    aie.core(%c) {
      %x = aie.objectfifo.acquire @in_c(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_c(Consume, 1)
      aie.end
    }
    aie.runtime_sequence(%in : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_b, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_c, id = 0 : i64} : memref<16xi32>
    }
  }
}
