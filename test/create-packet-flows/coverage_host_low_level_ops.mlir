//===- coverage_host_low_level_ops.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s
// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=ERR

// The host orders shim channels with low-level ops too. As in
// coverage_host_wait_order.mlir, flows 4 and 6 may share the one free arbiter
// at memtile (0,1) only if draining flow 6 never waits on flow 4's sender.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK-DAG:     aie.amsel<5> (0)
// CHECK-DAG:     aie.amsel<5> (1)

// ERR-COUNT-5: error: Unable to find a legal routing: at tile (0, 1), no two of {{.*}} draining that waits on (0, 0) MM2S 0, {{.*}} is unknown, so it is assumed to overrun its receiver.{{$}}
// ERR:         error: Unable to find a legal routing: {{.*}} draining that waits on (0, 0) MM2S 0, {{.*}} Nothing in the design programs (1, 0) S2MM 0, so it is assumed to wait on anything on its tile or on another shim tile.
// ERR-NOT:     error

// A write before any wait orders nothing.

module {
  aie.device(npu1_1col) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %c5 = aie.tile(0, 5)
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
    aie.packet_flow(4) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%c5, DMA : 0>  aie.packet_dest<%s0, DMA : 0> }
    aie.shim_dma_allocation @in4(%s0, MM2S, 0)
    aie.shim_dma_allocation @out6(%s0, S2MM, 0)
    aie.runtime_sequence(%a: memref<64xi32>, %b: memref<64xi32>) {
      %z = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @out6, id = 1 : i64, issue_token = true } : memref<64xi32>
      %addr = arith.constant 119300 : i32
      aiex.npu.write32(%addr, %one) {column = 0 : i32, row = 0 : i32} : i32, i32
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @in4, id = 0 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @in4}
    }
  }
}

// -----

// Every later case orders flow 6's receiver after flow 4's sender.
//
// npu.sync waits on the channel it names.

module {
  aie.device(npu1_1col) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %c5 = aie.tile(0, 5)
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
    aie.packet_flow(4) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%c5, DMA : 0>  aie.packet_dest<%s0, DMA : 0> }
    aie.shim_dma_allocation @in4(%s0, MM2S, 0)
    aie.shim_dma_allocation @out6(%s0, S2MM, 0)
    aie.runtime_sequence(%a: memref<64xi32>, %b: memref<64xi32>) {
      %z = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @in4, id = 0 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.sync(%z, %z, %one, %z, %one, %one) : i32, i32, i32, i32, i32, i32
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @out6, id = 1 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out6}
    }
  }
}

// -----

// An npu.sync on a channel known only at run time may be on any.

module {
  aie.device(npu1_1col) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %c5 = aie.tile(0, 5)
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
    aie.packet_flow(4) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%c5, DMA : 0>  aie.packet_dest<%s0, DMA : 0> }
    aie.shim_dma_allocation @in4(%s0, MM2S, 0)
    aie.shim_dma_allocation @out6(%s0, S2MM, 0)
    aie.runtime_sequence(%a: memref<64xi32>, %b: memref<64xi32>, %dir: i32) {
      %z = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @in4, id = 0 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.sync(%z, %z, %dir, %z, %one, %one) : i32, i32, i32, i32, i32, i32
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @out6, id = 1 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out6}
    }
  }
}

// -----

// So may an npu.maskpoll.

module {
  aie.device(npu1_1col) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %c5 = aie.tile(0, 5)
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
    aie.packet_flow(4) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%c5, DMA : 0>  aie.packet_dest<%s0, DMA : 0> }
    aie.shim_dma_allocation @in4(%s0, MM2S, 0)
    aie.shim_dma_allocation @out6(%s0, S2MM, 0)
    aie.runtime_sequence(%a: memref<64xi32>, %b: memref<64xi32>) {
      %z = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      %addr = arith.constant 119300 : i32
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @in4, id = 0 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.maskpoll(%addr, %z, %one) {column = 0 : i32, row = 0 : i32} : i32, i32, i32
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @out6, id = 1 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out6}
    }
  }
}

// -----

// npu.push_queue issues the channel it names.

module {
  aie.device(npu1_1col) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %c5 = aie.tile(0, 5)
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
    aie.packet_flow(4) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%c5, DMA : 0>  aie.packet_dest<%s0, DMA : 0> }
    aie.runtime_sequence(%a: memref<64xi32>, %b: memref<64xi32>) {
      %z = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %z repeat %z {issue_token = true} : i32, i32
      aiex.npu.sync(%z, %z, %one, %z, %one, %one) : i32, i32, i32, i32, i32, i32
      aiex.npu.push_queue(0, 0, S2MM : 0) bd_id %one repeat %z {issue_token = true} : i32, i32
      aiex.npu.sync(%z, %z, %z, %z, %one, %one) : i32, i32, i32, i32, i32, i32
    }
  }
}

// -----

// A raw register write after a wait may start any channel.

module {
  aie.device(npu1_1col) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %c5 = aie.tile(0, 5)
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
    aie.packet_flow(4) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%c5, DMA : 0>  aie.packet_dest<%s0, DMA : 0> }
    aie.shim_dma_allocation @in4(%s0, MM2S, 0)
    aie.shim_dma_allocation @out6(%s0, S2MM, 0)
    aie.runtime_sequence(%a: memref<64xi32>, %b: memref<64xi32>) {
      %z = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @out6, id = 1 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @in4, id = 0 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @in4}
      %addr = arith.constant 119300 : i32
      aiex.npu.write32(%addr, %one) {column = 0 : i32, row = 0 : i32} : i32, i32
    }
  }
}

// -----

// With no runtime sequence, whoever drives the shims may order one after
// another, here across two shim tiles.

module {
  aie.device(npu1) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %s2 = aie.tile(2, 0)
    %m11 = aie.tile(1, 1)
    %m  = aie.tile(0, 1)
    %c5 = aie.tile(0, 5)
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
    // No way into shim (1,0) but from shim (0,0).
    %sb11 = aie.switchbox(%m11) {
      aie.connect<DMA : 0, South : 0>
      aie.connect<DMA : 1, South : 1>
      aie.connect<DMA : 2, South : 2>
      aie.connect<DMA : 3, South : 3>
    }
    %sb20 = aie.switchbox(%s2) {
      aie.connect<North : 0, West : 0>
      aie.connect<North : 1, West : 1>
      aie.connect<North : 2, West : 2>
      aie.connect<North : 3, West : 3>
    }
    aie.packet_flow(4) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%c5, DMA : 0>  aie.packet_dest<%s1, DMA : 0> }
  }
}
