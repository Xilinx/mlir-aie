//===- coverage_receiver_program_shape.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s
// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=ERR

// As in coverage_stream_volume.mlir, flows 4 and 6 must share memtile (0,1)'s
// one free arbiter, which is safe only if flow 6's receiver takes in all the
// host sends it before it waits on flow 4's sender.

// The receiver's BD chain runs ^in0 once, then loops over ^in1 and ^in2: it
// takes 192 bytes before ^in1 waits a second time. A 192-byte memcpy fits.

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
      %0 = aie.dma_start(S2MM, 0, ^in0, ^ch1)
    ^in0:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^in1
    ^in1:
      aie.use_lock(%prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %one)
      aie.next_bd ^in2
    ^in2:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^in1
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
    aie.runtime_sequence(%a: memref<16400xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 48][0, 0, 0, 1]) { metadata = @in6, id = 0 : i64, issue_token = true } : memref<16400xi32>
      aiex.npu.dma_wait {symbol = @in6}
    }
  }
}

// -----

// A 256-byte one does not: ^in0 is not run again.

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
      %0 = aie.dma_start(S2MM, 0, ^in0, ^ch1)
    ^in0:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^in1
    ^in1:
      aie.use_lock(%prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %one)
      aie.next_bd ^in2
    ^in2:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^in1
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
    aie.runtime_sequence(%a: memref<16400xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @in6, id = 0 : i64, issue_token = true } : memref<16400xi32>
      aiex.npu.dma_wait {symbol = @in6}
    }
  }
}

// -----

// Flow 6 claims ids 6 and 7. A 64-byte memcpy with id 7 fits.

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
    aie.packet_flow(6, mask = 30) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
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
    aie.runtime_sequence(%a: memref<16400xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) { metadata = @in6, id = 0 : i64, issue_token = true, packet = #aie.packet_info<pkt_type = 0, pkt_id = 7> } : memref<16400xi32>
      aiex.npu.dma_wait {symbol = @in6}
    }
  }
}

// -----

// 64 bytes with each of ids 6 and 7 do not.

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
    aie.packet_flow(6, mask = 30) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
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
    aie.runtime_sequence(%a: memref<16400xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) { metadata = @in6, id = 0 : i64, issue_token = true, packet = #aie.packet_info<pkt_type = 0, pkt_id = 6> } : memref<16400xi32>
      aiex.npu.dma_wait {symbol = @in6}
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) { metadata = @in6, id = 0 : i64, issue_token = true, packet = #aie.packet_info<pkt_type = 0, pkt_id = 7> } : memref<16400xi32>
      aiex.npu.dma_wait {symbol = @in6}
    }
  }
}

// -----

// The receiver runs its chain 256 times, more BDs than the analysis steps
// through, and never waits: each run leaves its locks as it found them or
// fuller. It takes 64 KiB, so a 64 KiB memcpy fits.

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
      %0 = aie.dma_start(S2MM, 0, ^in0, ^ch1, repeat_count = 255)
    ^in0:
      aie.use_lock(%prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%prod, Release, %one)
      aie.next_bd ^in1
    ^in1:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %one)
      aie.next_bd ^in2
    ^in2:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^in3
    ^in3:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^last
    ^last:
      aie.end
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
    aie.runtime_sequence(%a: memref<16400xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 64, 256][0, 0, 256, 1]) { metadata = @in6, id = 0 : i64, issue_token = true } : memref<16400xi32>
      aiex.npu.dma_wait {symbol = @in6}
    }
  }
}

// -----

// 64 bytes more do not.

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
      %0 = aie.dma_start(S2MM, 0, ^in0, ^ch1, repeat_count = 255)
    ^in0:
      aie.use_lock(%prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%prod, Release, %one)
      aie.next_bd ^in1
    ^in1:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %one)
      aie.next_bd ^in2
    ^in2:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^in3
    ^in3:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^last
    ^last:
      aie.end
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
    aie.runtime_sequence(%a: memref<16400xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 41, 400][0, 0, 400, 1]) { metadata = @in6, id = 0 : i64, issue_token = true } : memref<16400xi32>
      aiex.npu.dma_wait {symbol = @in6}
    }
  }
}

// ERR-NOT: error
