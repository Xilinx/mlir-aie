//===- coverage_host_wait_order.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s
// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=ERR

// Flows 4 and 6 both pass memtile (0,1), where existing mastersets leave one
// arbiter free. Sharing it is safe unless the host issues flow 6's receiver
// (shim S2MM 0) only after waiting on flow 4's sender (shim MM2S 0): then
// draining flow 6 waits on flow 4.

// Waiting after the issue adds no dependency.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK-DAG:     aie.amsel<5> (0)
// CHECK-DAG:     aie.amsel<5> (1)

// Every later case waits before the issue.

// ERR-COUNT-5: error: Unable to find a legal routing: at tile (0, 1), no two of {{.*}} draining that waits on (0, 0) MM2S 0,
// ERR-NOT:     error

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
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @in4, id = 0 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @out6, id = 1 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @in4}
      aiex.npu.dma_wait {symbol = @out6}
    }
  }
}

// -----

// Waiting before the issue does.

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
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @in4, id = 0 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @in4}
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @out6, id = 1 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out6}
    }
  }
}

// -----

// Likewise through a task configured on the tile.

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
      %t = aiex.dma_configure_task(%s0, MM2S, 0) {
        aie.dma_bd(%a : memref<64xi32> offset = 0 len = 64) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @out6, id = 1 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out6}
    }
  }
}

// -----

// Likewise through a task configured for the allocation.

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
      %t = aiex.dma_configure_task_for @in4 {
        aie.dma_bd(%a : memref<64xi32> offset = 0 len = 64) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @out6, id = 1 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out6}
    }
  }
}

// -----

// Likewise when the receiver is a bd chain for its allocation.

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
    aie.bd_chain @chain(%x: memref<64xi32>) {
      aie.dma_bd(%x : memref<64xi32> offset = 0 len = 64)
      aie.end
    }
    aie.runtime_sequence(%a: memref<64xi32>, %b: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @in4, id = 0 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @in4}
      %t = aiex.dma_start_bd_chain_for @chain(%b) : (memref<64xi32>) for @out6
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// Likewise when the receiver is a bd chain on its tile.

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
    aie.bd_chain @chain(%x: memref<64xi32>) {
      aie.dma_bd(%x : memref<64xi32> offset = 0 len = 64)
      aie.end
    }
    aie.runtime_sequence(%a: memref<64xi32>, %b: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @in4, id = 0 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @in4}
      %t = aiex.dma_start_bd_chain @chain(%b) : (memref<64xi32>) on (%s0, S2MM, 0)
      aiex.dma_await_task(%t)
    }
  }
}
