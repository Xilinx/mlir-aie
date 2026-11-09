//===- memcpy_allocation_header.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-dma-to-npu %s | FileCheck %s

// A transfer on a packet-switched shim allocation stamps the allocation's
// header: word 2 is enable_packet (bit 30) | pkt_id 5 (bits 23:19).
// CHECK-LABEL: @takes_allocation_header
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<8xi32>
// CHECK-SAME: = dense<[{{.*}}, {{.*}}, 1076363264, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]>
module @takes_allocation_header {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0, <pkt_id = 5>)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @in, id = 0 : i64} : memref<64xi32>
    }
  }
}

// -----

// A header written on the transfer wins over the allocation's: pkt_type 3,
// pkt_id 2.
// CHECK-LABEL: @transfer_header_wins
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<8xi32>
// CHECK-SAME: = dense<[{{.*}}, {{.*}}, 1074987008, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]>
module @transfer_header_wins {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0, <pkt_id = 5>)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1], packet = <pkt_id = 2, pkt_type = 3>) {metadata = @in, id = 0 : i64} : memref<64xi32>
    }
  }
}

// -----

// A circuit-switched allocation leaves the header off.
// CHECK-LABEL: @no_header
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<8xi32>
// CHECK-SAME: = dense<[{{.*}}, {{.*}}, 0, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]>
module @no_header {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @in, id = 0 : i64} : memref<64xi32>
    }
  }
}

// -----

// The runtime-sized path stamps it the same way.
// CHECK-LABEL: @dynamic_takes_allocation_header
// CHECK: aiex.npu.blockwrite_values(%{{.*}} : i32) values %{{.*}}, %{{.*}}, %c1076363264_i32,
module @dynamic_takes_allocation_header {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0, <pkt_id = 5>)
    aie.runtime_sequence @seq(%arg0: memref<4096xi32>, %n: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][2, 4, %n, 32][2048, 256, 64, 1]) {id = 0 : i64, metadata = @in} : memref<4096xi32>
    }
  }
}
