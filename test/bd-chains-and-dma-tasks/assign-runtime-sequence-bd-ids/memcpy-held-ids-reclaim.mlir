//===- memcpy-held-ids-reclaim.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-substitute-shim-dma-allocations \
// RUN:   --aie-assign-runtime-sequence-bd-ids='reclaim-bds=true' \
// RUN:   --verify-diagnostics --split-input-file %s | FileCheck %s
// RUN: not aie-opt --aie-substitute-shim-dma-allocations \
// RUN:   --aie-assign-runtime-sequence-bd-ids='reclaim-bds=true' \
// RUN:   --split-input-file %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ORDER

// %op is in flight over the copy of BD 0, and %v's 0 is the only id reclaim
// can take back (%f is never started), so the tile has no id for %op. The
// remark for the poll that took 0 back comes first.
// ORDER: remark: tile (0,0) has no free buffer descriptor id, so this poll waits until the channel of an earlier task is idle and takes back its ids 0
// ORDER: error: 'aiex.dma_configure_task' op Too many simultaneously active buffer descriptors
// ORDER-SAME: The ids the compiler took back did not cover it, and it could not take more
module @reclaim_frees_held_id {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.shim_dma_allocation @out(%shim, S2MM, 0)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      // expected-note@+1 {{the task it waits for}}
      %v = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%v)
      %f = aiex.dma_configure_task_for @out {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.next_bd ^bd1
      ^bd1: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd2
      ^bd2: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd3
      ^bd3: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd4
      ^bd4: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd5
      ^bd5: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd6
      ^bd6: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd7
      ^bd7: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd8
      ^bd8: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd9
      ^bd9: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd10
      ^bd10: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd11
      ^bd11: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd12
      ^bd12: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd13
      ^bd13: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd14
      ^bd14:
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      // expected-error@+2 {{Too many simultaneously active buffer descriptors on tile (0,0), which supports up to 16. Of those, aiex.npu.dma_memcpy_nd ops program id 0 while this task may run (see the notes).}}
      // expected-remark@+1 {{tile (0,0) has no free buffer descriptor id, so this poll waits until the channel of an earlier task is idle and takes back its ids 0}}
      %op = aiex.dma_configure_task_for @out {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%op)
      aiex.dma_await_task(%v)
      // expected-note@+1 {{this copy reprograms buffer descriptor ID 0 while the task may run}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.dma_await_task(%op)
      aiex.dma_free_task(%f)
    }
  }
}

// -----

// The first allocation reclaims %v's 0 for %op, which the copy then meets in
// flight. The second keeps %op off 0 and reclaims %v2's 1 too; only its two
// polls remain.
// CHECK-LABEL: @retry_reclaims_again
// CHECK-COUNT-2: aiex.npu.maskpoll
// CHECK-NOT: aiex.npu.maskpoll
// CHECK: aie.dma_bd({{.*}}) {bd_id = 1 : i32}
// CHECK-NOT: aiex.npu.maskpoll
module @retry_reclaims_again {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.shim_dma_allocation @out(%shim, S2MM, 0)
    aie.shim_dma_allocation @out2(%shim, S2MM, 1)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      // expected-note@+1 {{the task it waits for}}
      %v = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%v)
      // expected-note@+1 {{the task it waits for}}
      %v2 = aiex.dma_configure_task_for @out2 {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%v2)
      %f = aiex.dma_configure_task_for @out {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.next_bd ^bd1
      ^bd1: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd2
      ^bd2: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd3
      ^bd3: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd4
      ^bd4: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd5
      ^bd5: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd6
      ^bd6: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd7
      ^bd7: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd8
      ^bd8: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd9
      ^bd9: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd10
      ^bd10: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd11
      ^bd11: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd12
      ^bd12: aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) aie.next_bd ^bd13
      ^bd13:
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      // expected-remark@+2 {{takes back its ids 0}}
      // expected-remark@+1 {{takes back its ids 1}}
      %op = aiex.dma_configure_task_for @out {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%op)
      aiex.dma_await_task(%v)
      aiex.dma_await_task(%v2)
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.dma_await_task(%op)
      aiex.dma_free_task(%f)
    }
  }
}


// -----

// Every id on this tile is held by an unretired copy, none by a task: the
// reclaim-bds advice about tasks does not apply, and each copy gets its own
// retirement advice.
module @too_many_all_copies_reclaim {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.shim_dma_allocation @out(%shim, S2MM, 0)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      // expected-note@+1 {{this copy holds buffer descriptor ID 0 until it is retired: it issues no token: set issue_token = true and wait for it, or await a later token-issuing transfer on its channel.}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, metadata = @in} : memref<?xi32>
      // expected-note@+1 {{this copy holds buffer descriptor ID 1 until it is retired: it issues no token: set issue_token = true and wait for it, or await a later token-issuing transfer on its channel.}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 1][1, 1, 1, 3][0, 0, 0, 1]) {id = 1 : i64, metadata = @in} : memref<?xi32>
      // expected-note@+1 {{this copy holds buffer descriptor ID 2 until it is retired: it issues no token: set issue_token = true and wait for it, or await a later token-issuing transfer on its channel.}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 2][1, 1, 1, 3][0, 0, 0, 1]) {id = 2 : i64, metadata = @in} : memref<?xi32>
      // expected-note@+1 {{this copy holds buffer descriptor ID 3 until it is retired: it issues no token: set issue_token = true and wait for it, or await a later token-issuing transfer on its channel.}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 3][1, 1, 1, 3][0, 0, 0, 1]) {id = 3 : i64, metadata = @in} : memref<?xi32>
      // expected-note@+1 {{this copy holds buffer descriptor ID 4 until it is retired: it issues no token: set issue_token = true and wait for it, or await a later token-issuing transfer on its channel.}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 4][1, 1, 1, 3][0, 0, 0, 1]) {id = 4 : i64, metadata = @in} : memref<?xi32>
      // expected-note@+1 {{this copy holds buffer descriptor ID 5 until it is retired: it issues no token: set issue_token = true and wait for it, or await a later token-issuing transfer on its channel.}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 5][1, 1, 1, 3][0, 0, 0, 1]) {id = 5 : i64, metadata = @in} : memref<?xi32>
      // expected-note@+1 {{this copy holds buffer descriptor ID 6 until it is retired: it issues no token: set issue_token = true and wait for it, or await a later token-issuing transfer on its channel.}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 6][1, 1, 1, 3][0, 0, 0, 1]) {id = 6 : i64, metadata = @in} : memref<?xi32>
      // expected-note@+1 {{this copy holds buffer descriptor ID 7 until it is retired: it issues no token: set issue_token = true and wait for it, or await a later token-issuing transfer on its channel.}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 7][1, 1, 1, 3][0, 0, 0, 1]) {id = 7 : i64, metadata = @in} : memref<?xi32>
      // expected-note@+1 {{this copy holds buffer descriptor ID 8 until it is retired: it issues no token: set issue_token = true and wait for it, or await a later token-issuing transfer on its channel.}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 8][1, 1, 1, 3][0, 0, 0, 1]) {id = 8 : i64, metadata = @in} : memref<?xi32>
      // expected-note@+1 {{this copy holds buffer descriptor ID 9 until it is retired: it issues no token: set issue_token = true and wait for it, or await a later token-issuing transfer on its channel.}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 9][1, 1, 1, 3][0, 0, 0, 1]) {id = 9 : i64, metadata = @in} : memref<?xi32>
      // expected-note@+1 {{this copy holds buffer descriptor ID 10 until it is retired: it issues no token: set issue_token = true and wait for it, or await a later token-issuing transfer on its channel.}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 10][1, 1, 1, 3][0, 0, 0, 1]) {id = 10 : i64, metadata = @in} : memref<?xi32>
      // expected-note@+1 {{this copy holds buffer descriptor ID 11 until it is retired: it issues no token: set issue_token = true and wait for it, or await a later token-issuing transfer on its channel.}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 11][1, 1, 1, 3][0, 0, 0, 1]) {id = 11 : i64, metadata = @in} : memref<?xi32>
      // expected-note@+1 {{this copy holds buffer descriptor ID 12 until it is retired: it issues no token: set issue_token = true and wait for it, or await a later token-issuing transfer on its channel.}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 12][1, 1, 1, 3][0, 0, 0, 1]) {id = 12 : i64, metadata = @in} : memref<?xi32>
      // expected-note@+1 {{this copy holds buffer descriptor ID 13 until it is retired: it issues no token: set issue_token = true and wait for it, or await a later token-issuing transfer on its channel.}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 13][1, 1, 1, 3][0, 0, 0, 1]) {id = 13 : i64, metadata = @in} : memref<?xi32>
      // expected-note@+1 {{this copy holds buffer descriptor ID 14 until it is retired: it issues no token: set issue_token = true and wait for it, or await a later token-issuing transfer on its channel.}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 14][1, 1, 1, 3][0, 0, 0, 1]) {id = 14 : i64, metadata = @in} : memref<?xi32>
      // expected-note@+1 {{this copy holds buffer descriptor ID 15 until it is retired: it issues no token: set issue_token = true and wait for it, or await a later token-issuing transfer on its channel.}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 15][1, 1, 1, 3][0, 0, 0, 1]) {id = 15 : i64, metadata = @in} : memref<?xi32>
      // expected-error@+1 {{Too many simultaneously active buffer descriptors on tile (0,0), which supports up to 16. Of those, aiex.npu.dma_memcpy_nd ops program ids 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 while this task may run (see the notes). Every id in use here is an aiex.npu.dma_memcpy_nd copy (reclaim-bds does not consider those); retire the ones the notes name, or give this task's buffer descriptors other ids.}}
      %t = aiex.dma_configure_task_for @out {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.dma_await_task(%t)
    }
  }
}
