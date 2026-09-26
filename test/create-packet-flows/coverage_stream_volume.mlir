//===- coverage_stream_volume.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s
// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=ERR

// Flows 4 and 6 must share memtile (0,1)'s one free arbiter. Flow 6's
// receiver takes 64 bytes, then waits on flow 4's sender to free its buffer,
// so sharing is safe only if the host sends flow 6 at most 64 bytes.

// A 64-byte memcpy fits.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK-DAG:     aie.amsel<5> (0)
// CHECK-DAG:     aie.amsel<5> (1)

module {
  aie.device(npu1_1col) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %c2 = aie.tile(0, 2)
    %sb = aie.switchbox(%m) {
      %a00 = aie.amsel<0> (0)
      %a01 = aie.amsel<0> (1)
      %a02 = aie.amsel<0> (2)
      %a03 = aie.amsel<0> (3)
      aie.masterset(North : 0, %a00, %a01, %a02, %a03)
      %a10 = aie.amsel<1> (0)
      %a11 = aie.amsel<1> (1)
      %a12 = aie.amsel<1> (2)
      %a13 = aie.amsel<1> (3)
      aie.masterset(North : 1, %a10, %a11, %a12, %a13)
      %a20 = aie.amsel<2> (0)
      %a21 = aie.amsel<2> (1)
      %a22 = aie.amsel<2> (2)
      %a23 = aie.amsel<2> (3)
      aie.masterset(North : 2, %a20, %a21, %a22, %a23)
      %a30 = aie.amsel<3> (0)
      %a31 = aie.amsel<3> (1)
      %a32 = aie.amsel<3> (2)
      %a33 = aie.amsel<3> (3)
      aie.masterset(North : 3, %a30, %a31, %a32, %a33)
      %a40 = aie.amsel<4> (0)
      %a41 = aie.amsel<4> (1)
      %a42 = aie.amsel<4> (2)
      %a43 = aie.amsel<4> (3)
      aie.masterset(North : 4, %a40, %a41, %a42, %a43)
    }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%c2, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
    %prod = aie.lock(%m, 0) {init = 1 : i32}
    %cons = aie.lock(%m, 1) {init = 0 : i32}
    %buf = aie.buffer(%m) : memref<16xi32>
    aie.memtile_dma(%m) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^in, ^ch1)
    ^in:
      aie.use_lock(%prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %one)
      aie.next_bd ^in
    ^ch1:
      %1 = aie.dma_start(MM2S, 0, ^out, ^end)
    ^out:
      aie.use_lock(%cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%prod, Release, %one)
      aie.next_bd ^out
    ^end:
      aie.end
    }
    aie.shim_dma_allocation @in6(%s0, MM2S, 0)
    aie.runtime_sequence(%a: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) { metadata = @in6, id = 0 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @in6}
    }
  }
}

// -----

// A 128-byte one does not.

// ERR: error: Unable to find a legal routing: at tile (0, 1), {{.*}} (id 6) can fill its receiver, and draining that waits on (0, 1) MM2S 0, which sends {{.*}} (id 4).{{$}}

module {
  aie.device(npu1_1col) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %c2 = aie.tile(0, 2)
    %sb = aie.switchbox(%m) {
      %a00 = aie.amsel<0> (0)
      %a01 = aie.amsel<0> (1)
      %a02 = aie.amsel<0> (2)
      %a03 = aie.amsel<0> (3)
      aie.masterset(North : 0, %a00, %a01, %a02, %a03)
      %a10 = aie.amsel<1> (0)
      %a11 = aie.amsel<1> (1)
      %a12 = aie.amsel<1> (2)
      %a13 = aie.amsel<1> (3)
      aie.masterset(North : 1, %a10, %a11, %a12, %a13)
      %a20 = aie.amsel<2> (0)
      %a21 = aie.amsel<2> (1)
      %a22 = aie.amsel<2> (2)
      %a23 = aie.amsel<2> (3)
      aie.masterset(North : 2, %a20, %a21, %a22, %a23)
      %a30 = aie.amsel<3> (0)
      %a31 = aie.amsel<3> (1)
      %a32 = aie.amsel<3> (2)
      %a33 = aie.amsel<3> (3)
      aie.masterset(North : 3, %a30, %a31, %a32, %a33)
      %a40 = aie.amsel<4> (0)
      %a41 = aie.amsel<4> (1)
      %a42 = aie.amsel<4> (2)
      %a43 = aie.amsel<4> (3)
      aie.masterset(North : 4, %a40, %a41, %a42, %a43)
    }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%c2, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
    %prod = aie.lock(%m, 0) {init = 1 : i32}
    %cons = aie.lock(%m, 1) {init = 0 : i32}
    %buf = aie.buffer(%m) : memref<16xi32>
    aie.memtile_dma(%m) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^in, ^ch1)
    ^in:
      aie.use_lock(%prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %one)
      aie.next_bd ^in
    ^ch1:
      %1 = aie.dma_start(MM2S, 0, ^out, ^end)
    ^out:
      aie.use_lock(%cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%prod, Release, %one)
      aie.next_bd ^out
    ^end:
      aie.end
    }
    aie.shim_dma_allocation @in6(%s0, MM2S, 0)
    aie.runtime_sequence(%a: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 32][0, 0, 0, 1]) { metadata = @in6, id = 0 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @in6}
    }
  }
}

// -----

// Nor does a memcpy in a loop.

// ERR: error: Unable to find a legal routing: at tile (0, 1), {{.*}} (id 4). The volume {{.*}} (id 6) carries is unknown, so it is assumed to overrun its receiver.

module {
  aie.device(npu1_1col) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %c2 = aie.tile(0, 2)
    %sb = aie.switchbox(%m) {
      %a00 = aie.amsel<0> (0)
      %a01 = aie.amsel<0> (1)
      %a02 = aie.amsel<0> (2)
      %a03 = aie.amsel<0> (3)
      aie.masterset(North : 0, %a00, %a01, %a02, %a03)
      %a10 = aie.amsel<1> (0)
      %a11 = aie.amsel<1> (1)
      %a12 = aie.amsel<1> (2)
      %a13 = aie.amsel<1> (3)
      aie.masterset(North : 1, %a10, %a11, %a12, %a13)
      %a20 = aie.amsel<2> (0)
      %a21 = aie.amsel<2> (1)
      %a22 = aie.amsel<2> (2)
      %a23 = aie.amsel<2> (3)
      aie.masterset(North : 2, %a20, %a21, %a22, %a23)
      %a30 = aie.amsel<3> (0)
      %a31 = aie.amsel<3> (1)
      %a32 = aie.amsel<3> (2)
      %a33 = aie.amsel<3> (3)
      aie.masterset(North : 3, %a30, %a31, %a32, %a33)
      %a40 = aie.amsel<4> (0)
      %a41 = aie.amsel<4> (1)
      %a42 = aie.amsel<4> (2)
      %a43 = aie.amsel<4> (3)
      aie.masterset(North : 4, %a40, %a41, %a42, %a43)
    }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%c2, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
    %prod = aie.lock(%m, 0) {init = 1 : i32}
    %cons = aie.lock(%m, 1) {init = 0 : i32}
    %buf = aie.buffer(%m) : memref<16xi32>
    aie.memtile_dma(%m) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^in, ^ch1)
    ^in:
      aie.use_lock(%prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %one)
      aie.next_bd ^in
    ^ch1:
      %1 = aie.dma_start(MM2S, 0, ^out, ^end)
    ^out:
      aie.use_lock(%cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%prod, Release, %one)
      aie.next_bd ^out
    ^end:
      aie.end
    }
    aie.shim_dma_allocation @in6(%s0, MM2S, 0)
    aie.runtime_sequence(%a: memref<64xi32>) {
      %k0 = arith.constant 0 : index
      %k1 = arith.constant 1 : index
      %k2 = arith.constant 2 : index
      scf.for %i = %k0 to %k2 step %k1 {
        aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) { metadata = @in6, id = 0 : i64, issue_token = true } : memref<64xi32>
        aiex.npu.dma_wait {symbol = @in6}
      }
    }
  }
}

// -----

// A 64-byte task fits, configured on the tile or for the allocation.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK-DAG:     aie.amsel<5> (0)
// CHECK-DAG:     aie.amsel<5> (1)

module {
  aie.device(npu1_1col) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %c2 = aie.tile(0, 2)
    %sb = aie.switchbox(%m) {
      %a00 = aie.amsel<0> (0)
      %a01 = aie.amsel<0> (1)
      %a02 = aie.amsel<0> (2)
      %a03 = aie.amsel<0> (3)
      aie.masterset(North : 0, %a00, %a01, %a02, %a03)
      %a10 = aie.amsel<1> (0)
      %a11 = aie.amsel<1> (1)
      %a12 = aie.amsel<1> (2)
      %a13 = aie.amsel<1> (3)
      aie.masterset(North : 1, %a10, %a11, %a12, %a13)
      %a20 = aie.amsel<2> (0)
      %a21 = aie.amsel<2> (1)
      %a22 = aie.amsel<2> (2)
      %a23 = aie.amsel<2> (3)
      aie.masterset(North : 2, %a20, %a21, %a22, %a23)
      %a30 = aie.amsel<3> (0)
      %a31 = aie.amsel<3> (1)
      %a32 = aie.amsel<3> (2)
      %a33 = aie.amsel<3> (3)
      aie.masterset(North : 3, %a30, %a31, %a32, %a33)
      %a40 = aie.amsel<4> (0)
      %a41 = aie.amsel<4> (1)
      %a42 = aie.amsel<4> (2)
      %a43 = aie.amsel<4> (3)
      aie.masterset(North : 4, %a40, %a41, %a42, %a43)
    }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%c2, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
    %prod = aie.lock(%m, 0) {init = 1 : i32}
    %cons = aie.lock(%m, 1) {init = 0 : i32}
    %buf = aie.buffer(%m) : memref<16xi32>
    aie.memtile_dma(%m) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^in, ^ch1)
    ^in:
      aie.use_lock(%prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %one)
      aie.next_bd ^in
    ^ch1:
      %1 = aie.dma_start(MM2S, 0, ^out, ^end)
    ^out:
      aie.use_lock(%cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%prod, Release, %one)
      aie.next_bd ^out
    ^end:
      aie.end
    }
    aie.shim_dma_allocation @in6(%s0, MM2S, 0)
    aie.runtime_sequence(%a: memref<64xi32>) {
      %t = aiex.dma_configure_task(%s0, MM2S, 0) {
        aie.dma_bd(%a : memref<64xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 0 : i32}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK-DAG:     aie.amsel<5> (0)
// CHECK-DAG:     aie.amsel<5> (1)

module {
  aie.device(npu1_1col) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %c2 = aie.tile(0, 2)
    %sb = aie.switchbox(%m) {
      %a00 = aie.amsel<0> (0)
      %a01 = aie.amsel<0> (1)
      %a02 = aie.amsel<0> (2)
      %a03 = aie.amsel<0> (3)
      aie.masterset(North : 0, %a00, %a01, %a02, %a03)
      %a10 = aie.amsel<1> (0)
      %a11 = aie.amsel<1> (1)
      %a12 = aie.amsel<1> (2)
      %a13 = aie.amsel<1> (3)
      aie.masterset(North : 1, %a10, %a11, %a12, %a13)
      %a20 = aie.amsel<2> (0)
      %a21 = aie.amsel<2> (1)
      %a22 = aie.amsel<2> (2)
      %a23 = aie.amsel<2> (3)
      aie.masterset(North : 2, %a20, %a21, %a22, %a23)
      %a30 = aie.amsel<3> (0)
      %a31 = aie.amsel<3> (1)
      %a32 = aie.amsel<3> (2)
      %a33 = aie.amsel<3> (3)
      aie.masterset(North : 3, %a30, %a31, %a32, %a33)
      %a40 = aie.amsel<4> (0)
      %a41 = aie.amsel<4> (1)
      %a42 = aie.amsel<4> (2)
      %a43 = aie.amsel<4> (3)
      aie.masterset(North : 4, %a40, %a41, %a42, %a43)
    }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%c2, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
    %prod = aie.lock(%m, 0) {init = 1 : i32}
    %cons = aie.lock(%m, 1) {init = 0 : i32}
    %buf = aie.buffer(%m) : memref<16xi32>
    aie.memtile_dma(%m) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^in, ^ch1)
    ^in:
      aie.use_lock(%prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %one)
      aie.next_bd ^in
    ^ch1:
      %1 = aie.dma_start(MM2S, 0, ^out, ^end)
    ^out:
      aie.use_lock(%cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%prod, Release, %one)
      aie.next_bd ^out
    ^end:
      aie.end
    }
    aie.shim_dma_allocation @in6(%s0, MM2S, 0)
    aie.runtime_sequence(%a: memref<64xi32>) {
      %t = aiex.dma_configure_task_for @in6 {
        aie.dma_bd(%a : memref<64xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 0 : i32}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// Repeating it once does not.

// ERR: error: Unable to find a legal routing: at tile (0, 1), {{.*}} (id 6) can fill its receiver, and draining that waits on (0, 1) MM2S 0, which sends {{.*}} (id 4).{{$}}

module {
  aie.device(npu1_1col) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %c2 = aie.tile(0, 2)
    %sb = aie.switchbox(%m) {
      %a00 = aie.amsel<0> (0)
      %a01 = aie.amsel<0> (1)
      %a02 = aie.amsel<0> (2)
      %a03 = aie.amsel<0> (3)
      aie.masterset(North : 0, %a00, %a01, %a02, %a03)
      %a10 = aie.amsel<1> (0)
      %a11 = aie.amsel<1> (1)
      %a12 = aie.amsel<1> (2)
      %a13 = aie.amsel<1> (3)
      aie.masterset(North : 1, %a10, %a11, %a12, %a13)
      %a20 = aie.amsel<2> (0)
      %a21 = aie.amsel<2> (1)
      %a22 = aie.amsel<2> (2)
      %a23 = aie.amsel<2> (3)
      aie.masterset(North : 2, %a20, %a21, %a22, %a23)
      %a30 = aie.amsel<3> (0)
      %a31 = aie.amsel<3> (1)
      %a32 = aie.amsel<3> (2)
      %a33 = aie.amsel<3> (3)
      aie.masterset(North : 3, %a30, %a31, %a32, %a33)
      %a40 = aie.amsel<4> (0)
      %a41 = aie.amsel<4> (1)
      %a42 = aie.amsel<4> (2)
      %a43 = aie.amsel<4> (3)
      aie.masterset(North : 4, %a40, %a41, %a42, %a43)
    }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%c2, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
    %prod = aie.lock(%m, 0) {init = 1 : i32}
    %cons = aie.lock(%m, 1) {init = 0 : i32}
    %buf = aie.buffer(%m) : memref<16xi32>
    aie.memtile_dma(%m) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^in, ^ch1)
    ^in:
      aie.use_lock(%prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %one)
      aie.next_bd ^in
    ^ch1:
      %1 = aie.dma_start(MM2S, 0, ^out, ^end)
    ^out:
      aie.use_lock(%cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%prod, Release, %one)
      aie.next_bd ^out
    ^end:
      aie.end
    }
    aie.shim_dma_allocation @in6(%s0, MM2S, 0)
    aie.runtime_sequence(%a: memref<64xi32>) {
      %t = aiex.dma_configure_task(%s0, MM2S, 0) {
        aie.dma_bd(%a : memref<64xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// Nor does starting it in a loop.

// ERR: error: Unable to find a legal routing: at tile (0, 1), {{.*}} (id 4). The volume {{.*}} (id 6) carries is unknown, so it is assumed to overrun its receiver.

module {
  aie.device(npu1_1col) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %c2 = aie.tile(0, 2)
    %sb = aie.switchbox(%m) {
      %a00 = aie.amsel<0> (0)
      %a01 = aie.amsel<0> (1)
      %a02 = aie.amsel<0> (2)
      %a03 = aie.amsel<0> (3)
      aie.masterset(North : 0, %a00, %a01, %a02, %a03)
      %a10 = aie.amsel<1> (0)
      %a11 = aie.amsel<1> (1)
      %a12 = aie.amsel<1> (2)
      %a13 = aie.amsel<1> (3)
      aie.masterset(North : 1, %a10, %a11, %a12, %a13)
      %a20 = aie.amsel<2> (0)
      %a21 = aie.amsel<2> (1)
      %a22 = aie.amsel<2> (2)
      %a23 = aie.amsel<2> (3)
      aie.masterset(North : 2, %a20, %a21, %a22, %a23)
      %a30 = aie.amsel<3> (0)
      %a31 = aie.amsel<3> (1)
      %a32 = aie.amsel<3> (2)
      %a33 = aie.amsel<3> (3)
      aie.masterset(North : 3, %a30, %a31, %a32, %a33)
      %a40 = aie.amsel<4> (0)
      %a41 = aie.amsel<4> (1)
      %a42 = aie.amsel<4> (2)
      %a43 = aie.amsel<4> (3)
      aie.masterset(North : 4, %a40, %a41, %a42, %a43)
    }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%c2, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
    %prod = aie.lock(%m, 0) {init = 1 : i32}
    %cons = aie.lock(%m, 1) {init = 0 : i32}
    %buf = aie.buffer(%m) : memref<16xi32>
    aie.memtile_dma(%m) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^in, ^ch1)
    ^in:
      aie.use_lock(%prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %one)
      aie.next_bd ^in
    ^ch1:
      %1 = aie.dma_start(MM2S, 0, ^out, ^end)
    ^out:
      aie.use_lock(%cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%prod, Release, %one)
      aie.next_bd ^out
    ^end:
      aie.end
    }
    aie.shim_dma_allocation @in6(%s0, MM2S, 0)
    aie.runtime_sequence(%a: memref<64xi32>) {
      %t = aiex.dma_configure_task(%s0, MM2S, 0) {
        aie.dma_bd(%a : memref<64xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 0 : i32}
      %k0 = arith.constant 0 : index
      %k1 = arith.constant 1 : index
      %k2 = arith.constant 2 : index
      scf.for %i = %k0 to %k2 step %k1 {
        aiex.dma_start_task(%t)
        aiex.dma_await_task(%t)
      }
    }
  }
}

// -----

// Nor does a 64-byte packet whose header the receiver keeps.

// ERR: error: Unable to find a legal routing: at tile (0, 1), {{.*}} (id 6) can fill its receiver, and draining that waits on (0, 1) MM2S 0, which sends {{.*}} (id 4).{{$}}

module {
  aie.device(npu1_1col) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %c2 = aie.tile(0, 2)
    %sb = aie.switchbox(%m) {
      %a00 = aie.amsel<0> (0)
      %a01 = aie.amsel<0> (1)
      %a02 = aie.amsel<0> (2)
      %a03 = aie.amsel<0> (3)
      aie.masterset(North : 0, %a00, %a01, %a02, %a03)
      %a10 = aie.amsel<1> (0)
      %a11 = aie.amsel<1> (1)
      %a12 = aie.amsel<1> (2)
      %a13 = aie.amsel<1> (3)
      aie.masterset(North : 1, %a10, %a11, %a12, %a13)
      %a20 = aie.amsel<2> (0)
      %a21 = aie.amsel<2> (1)
      %a22 = aie.amsel<2> (2)
      %a23 = aie.amsel<2> (3)
      aie.masterset(North : 2, %a20, %a21, %a22, %a23)
      %a30 = aie.amsel<3> (0)
      %a31 = aie.amsel<3> (1)
      %a32 = aie.amsel<3> (2)
      %a33 = aie.amsel<3> (3)
      aie.masterset(North : 3, %a30, %a31, %a32, %a33)
      %a40 = aie.amsel<4> (0)
      %a41 = aie.amsel<4> (1)
      %a42 = aie.amsel<4> (2)
      %a43 = aie.amsel<4> (3)
      aie.masterset(North : 4, %a40, %a41, %a42, %a43)
    }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%c2, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> } {keep_pkt_header = true}
    %prod = aie.lock(%m, 0) {init = 1 : i32}
    %cons = aie.lock(%m, 1) {init = 0 : i32}
    %buf = aie.buffer(%m) : memref<16xi32>
    aie.memtile_dma(%m) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^in, ^ch1)
    ^in:
      aie.use_lock(%prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %one)
      aie.next_bd ^in
    ^ch1:
      %1 = aie.dma_start(MM2S, 0, ^out, ^end)
    ^out:
      aie.use_lock(%cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%prod, Release, %one)
      aie.next_bd ^out
    ^end:
      aie.end
    }
    aie.shim_dma_allocation @in6(%s0, MM2S, 0)
    aie.runtime_sequence(%a: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) { metadata = @in6, id = 0 : i64, issue_token = true, packet = #aie.packet_info<pkt_type = 0, pkt_id = 6> } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @in6}
    }
  }
}

// ERR-NOT: error
