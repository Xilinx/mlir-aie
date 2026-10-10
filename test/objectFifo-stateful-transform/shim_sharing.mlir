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

// Three shim ends need three MM2S channels on a shim tile that has two. Each
// transfer is awaited before the next is issued, so in_a and in_b take turns
// on MM2S 0, their routes packet-switched to tell them apart.
// CHECK-LABEL: module @awaited
// CHECK-DAG:   aie.shim_dma_allocation @in_a_shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = {{[0-9]+}}>)
// CHECK-DAG:   aie.shim_dma_allocation @in_b_shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = {{[0-9]+}}>)
// CHECK-DAG:   aie.shim_dma_allocation @in_c_shim_alloc(%{{.*}}, MM2S, 1)
module @awaited {
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

// Runtime tasks freed rather than awaited, as IRON's task groups free a fill
// once the drains it feeds are awaited: freeing lets a task's BD ids be
// reused, which is only safe once its transfers are done.
// CHECK-LABEL: module @freed
// CHECK-DAG:   aie.shim_dma_allocation @in_a_shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = {{[0-9]+}}>)
// CHECK-DAG:   aie.shim_dma_allocation @in_b_shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = {{[0-9]+}}>)
// CHECK-DAG:   aie.shim_dma_allocation @in_c_shim_alloc(%{{.*}}, MM2S, 1)
module @freed {
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
      %ta = aiex.dma_configure_task_for @in_a {
        aie.dma_bd(%in : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      aiex.dma_start_task(%ta)
      aiex.dma_free_task(%ta)
      %tb = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%in : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      aiex.dma_start_task(%tb)
      aiex.dma_free_task(%tb)
      %tc = aiex.dma_configure_task_for @in_c {
        aie.dma_bd(%in : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      aiex.dma_start_task(%tc)
      aiex.dma_free_task(%tc)
    }
  }
}

// -----

// in_a's transfer is never awaited, so it may still be in flight when the
// others go; in_b and in_c take turns on one channel and in_a has the other.
// CHECK-LABEL: module @partly_awaited
// CHECK-DAG:   aie.shim_dma_allocation @in_a_shim_alloc(%{{.*}}, MM2S, 0)
// CHECK-DAG:   aie.shim_dma_allocation @in_b_shim_alloc(%{{.*}}, MM2S, 1, <pkt_id = {{[0-9]+}}>)
// CHECK-DAG:   aie.shim_dma_allocation @in_c_shim_alloc(%{{.*}}, MM2S, 1, <pkt_id = {{[0-9]+}}>)
module @partly_awaited {
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
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_b, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @in_b}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_c, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @in_c}
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
