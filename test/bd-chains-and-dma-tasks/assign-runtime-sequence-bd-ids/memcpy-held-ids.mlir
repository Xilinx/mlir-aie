//===- memcpy-held-ids.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-substitute-shim-dma-allocations \
// RUN:   --aie-assign-runtime-sequence-bd-ids --verify-diagnostics \
// RUN:   --split-input-file %s | FileCheck %s

// A task in flight while an aiex.npu.dma_memcpy_nd programs BD 0 on its tile
// takes another id, as for the control packets aie-ctrl-packet-to-dma streams
// in with id 0.
// CHECK-LABEL: @task_in_flight_over_copy
// CHECK: aie.dma_bd({{.*}}) {bd_id = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd({{.*}}id = 0 : i64
module @task_in_flight_over_copy {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// A task awaited before the copy is done with BD 0, so it keeps it.
// CHECK-LABEL: @task_done_before_copy
// CHECK: aie.dma_bd({{.*}}) {bd_id = 0 : i32}
module @task_done_before_copy {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}

// -----

// A task that names the copy's id while both may be in flight is refused.
module @named_id_over_copy {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      // expected-error@+1 {{Specified buffer descriptor ID 0 is the one an aiex.npu.dma_memcpy_nd on this tile programs while this task may be in flight}}
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      // expected-note@+1 {{the copy is here}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// The copy and %t share @in's queue, so awaiting %t retires the copy too and
// %u may name its id.
// CHECK-LABEL: @copy_retired_by_later_await
// CHECK: aie.dma_bd({{.*}}) {bd_id = 0 : i32}
// CHECK: aie.dma_bd({{.*}}) {bd_id = 3 : i32}
module @copy_retired_by_later_await {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 0)
    aie.shim_dma_allocation @in2(%shim, MM2S, 1)
    aie.runtime_sequence @run(%x : memref<16xi32>, %w : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%w[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 3 : i64, metadata = @in} : memref<16xi32>
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      %u = aiex.dma_configure_task_for @in2 {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) {bd_id = 3 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%u)
      aiex.dma_await_task(%u)
    }
  }
}

// -----

// A sync consumes one token, the first copy's, so the second copy still
// programs BD 0 when %t is configured.
// CHECK-LABEL: @two_copies_one_sync
// CHECK: aie.dma_bd({{.*}}) {bd_id = 1 : i32}
module @two_copies_one_sync {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 3][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// The await naming %t2 consumes %t1's token, so at the copy %t1 is done and
// %t2 may be in flight: %t1 keeps BD 0 and %t2 stays off it.
// CHECK-LABEL: @await_consumes_older_token
// CHECK: aie.dma_bd({{.*}}) {bd_id = 0 : i32}
// CHECK: aie.dma_bd({{.*}}) {bd_id = 1 : i32}
module @await_consumes_older_token {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      %t1 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      %t2 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.dma_await_task(%t1)
    }
  }
}

// -----

// %t1 names BD 0 and is done when the copy reprograms it (the await naming
// %t2 consumed its token), so it may keep 0.
// CHECK-LABEL: @finished_named_task_keeps_id
// CHECK: aie.dma_bd({{.*}}) {bd_id = 0 : i32}
// CHECK: aie.dma_bd({{.*}}) {bd_id = 1 : i32}
module @finished_named_task_keeps_id {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      %t1 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      %t2 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.dma_await_task(%t1)
    }
  }
}

// -----

// %t is configured before the copy and started after it, so the copy would
// rewrite its BD before it runs.
// CHECK-LABEL: @task_started_after_copy
// CHECK: aie.dma_bd({{.*}}) {bd_id = 1 : i32}
module @task_started_after_copy {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// A task configured while the copy is in flight may not name its id.
module @named_id_while_copy_in_flight {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      // expected-note@+1 {{the copy is here}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 2 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      // expected-error@+1 {{Specified buffer descriptor ID 2 is the one an aiex.npu.dma_memcpy_nd on this tile programs, and that copy may still be in flight here}}
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) {bd_id = 2 : i32}
        aie.end
      } {issue_token = true}
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// Two copies of BD 0 in flight: the second reprograms the first's BD, which
// warns. The first sync retires the first, and the second still holds 0 for
// %t's configure.
// CHECK-LABEL: @second_copy_same_id
// CHECK: aie.dma_bd({{.*}}) {bd_id = 1 : i32}
module @second_copy_same_id {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      // expected-note@+1 {{the earlier copy is here}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      // expected-warning@+1 {{reuses buffer descriptor ID 0 on tile (0,0) while an earlier aiex.npu.dma_memcpy_nd of that ID may still be in flight, so this reprograms it underneath. Give this copy another ID, or retire the earlier copy first: wait for it (aiex.npu.sync or aiex.npu.dma_wait on its channel, consuming its token).}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 3][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// The copy takes BD 0 after the first free, so the second free of %t is
// still a double free.
module @double_free_after_copy {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      aiex.dma_free_task(%t)
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      // expected-error@+1 {{frees buffer descriptor ID 0, which is not currently in use}}
      aiex.dma_free_task(%t)
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}

// -----

// A PDI load ends every transfer begun before it, a copy's as a task's, so
// %t may name the copy's id after it.
// CHECK-LABEL: @load_pdi_after_copy
// CHECK: aie.dma_bd({{.*}}) {bd_id = 0 : i32}
module @load_pdi_after_copy {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      aiex.npu.load_pdi {device_ref = @other}
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
  aie.device(npu2_1col) @other {
    %shim = aie.tile(0, 0)
  }
}

// -----

// The same with a task where the copy was.
// CHECK-LABEL: @load_pdi_after_task
// CHECK: aie.dma_bd({{.*}}) {bd_id = 0 : i32}
// CHECK: aie.dma_bd({{.*}}) {bd_id = 0 : i32}
module @load_pdi_after_task {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      %a = aiex.dma_configure_task_for @ctrl {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%a)
      aiex.npu.load_pdi {device_ref = @other}
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_await_task(%a)
    }
  }
  aie.device(npu2_1col) @other {
    %shim = aie.tile(0, 0)
  }
}

// -----

// A tokenless copy: a wait on its channel does not retire it, so the advice
// names what does.
module @tokenless_copy_named {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      // expected-note@+1 {{the copy is here}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, metadata = @ctrl} : memref<?xi32>
      aiex.npu.dma_wait {symbol = @ctrl}
      // expected-error@+1 {{Give the task another ID, or retire the copy first: it issues no token: set issue_token = true and wait for it, or await a later token-issuing transfer on its channel.}}
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// %t is configured (its BD written) before the copy, but started for the
// first time after it: not a restart, but the copy still overwrites the BD
// the configure wrote before the task ever launches.
module @first_start_after_copy_named {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      // expected-error@+1 {{Specified buffer descriptor ID 0 is the one an aiex.npu.dma_memcpy_nd on this tile programs after this task's own configure wrote it, so starting the task would launch the copy's descriptor instead of its own. Give the task another ID, or start it before the copy.}}
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      // expected-note@+1 {{the copy is here}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      // expected-note@+1 {{the task starts here, after the copy}}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// %t is done before the copy; its restart after it is the clash.
module @restart_after_copy_named {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      // expected-error@+1 {{Specified buffer descriptor ID 0 is the one an aiex.npu.dma_memcpy_nd on this tile programs before this task is started again, so the restart would run the copy's descriptor. Give the task another ID, or configure a new task after the copy.}}
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      // expected-note@+1 {{the copy is here}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      // expected-note@+1 {{the task is started again here}}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// %big holds 0-14 and %t 15, which a copy then reprograms while %t is in
// flight: the tile has no id left for %t, and the error says which copy took
// it.
module @too_many_copy_note {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.shim_dma_allocation @out(%shim, S2MM, 0)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      %big = aiex.dma_configure_task_for @out {
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
      } {issue_token = true}
      aiex.dma_start_task(%big)
      // expected-error@+1 {{Too many simultaneously active buffer descriptors on tile (0,0), which supports up to 16. Of those, aiex.npu.dma_memcpy_nd ops program id 15 while this task may run (see the notes).}}
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      // expected-note@+1 {{this copy reprograms buffer descriptor ID 15 while the task may run}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 15 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.dma_await_task(%t)
      aiex.dma_await_task(%big)
    }
  }
}

// -----

// A tokenless copy reprogrammed by a later copy of its id: the advice names
// what retires a copy without a token.
// CHECK-LABEL: @tokenless_copy_reused
module @tokenless_copy_reused {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    // expected-warning@+1 {{may exit with 2 DMA transfer(s) neither awaited nor freed}}
    aie.runtime_sequence @run(%x : memref<16xi32>) {
      // expected-note@+2 {{the earlier copy is here}}
      // expected-note@+1 {{first outstanding transfer starts here}}
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 2 : i64, metadata = @in} : memref<16xi32>
      // expected-warning@+1 {{reuses buffer descriptor ID 2 on tile (0,0) while an earlier aiex.npu.dma_memcpy_nd of that ID may still be in flight, so this reprograms it underneath. Give this copy another ID, or retire the earlier copy first: it issues no token: set issue_token = true and wait for it, or await a later token-issuing transfer on its channel.}}
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 2 : i64, metadata = @in} : memref<16xi32>
    }
  }
}

// -----

// A wait that retires the first copy leaves the second free to reuse its id.
// CHECK-LABEL: @copy_reused_after_wait
module @copy_reused_after_wait {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.runtime_sequence @run(%pkts : memref<?xi32>) {
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      aiex.npu.dma_wait {symbol = @ctrl}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 3][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      aiex.npu.dma_wait {symbol = @ctrl}
    }
  }
}
