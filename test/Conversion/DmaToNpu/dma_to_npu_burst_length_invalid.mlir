//===- dma_to_npu__burst_length_invalid.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-dma-to-npu --split-input-file %s 2>&1 | FileCheck %s
// CHECK: Only ShimTiles support burst length.
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
      aiex.npu.writebd { bd_id = 6 : i32,
                         buffer_length = 1 : i32,
                         buffer_offset = 2 : i32,
                         enable_packet = 0 : i32,
                         out_of_order_id = 0 : i32,
                         packet_id = 0 : i32,
                         packet_type = 0 : i32,
                         column = 3 : i32,
                         row = 1 : i32,
                         d0_stride = 5 : i32,
                         d0_size = 6 : i32,
                         d0_zero_after = 0 : i32,
                         d0_zero_before = 0 : i32,
                         d1_stride = 7 : i32,
                         d1_size = 8 : i32,
                         d1_zero_after = 0 : i32,
                         d1_zero_before = 0 : i32,
                         d2_size = 1 : i32,
                         d2_stride = 9 : i32,
                         d2_zero_after = 0 : i32,
                         d2_zero_before = 0 : i32,
                         ddr_id = 10 : i32,
                         iteration_current = 11 : i32,
                         iteration_stride = 12 : i32,
                         iteration_size = 13 : i32,
                         lock_acq_enable = 1 : i32,
                         lock_acq_id = 1 : i32,
                         lock_acq_val = 2 : i32,
                         lock_rel_id = 3 : i32,
                         lock_rel_val = 4 : i32,
                         next_bd = 5 : i32,
                         use_next_bd = 1 : i32,
                         valid_bd = 1 : i32,
                         burst_length = 20 : i32}
    }
  }
}

// -----
// CHECK: Requested burst length is not supported by the target. Supported burst lengths: 64 128 256 512

module {
  aie.device(npu2) {
    aie.runtime_sequence(%in : memref<4x2x8xi32>, %buf : memref<32xi32>, %out : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd (%in[0,2,0,0][1,2,2,8][0,16,8,1]) { metadata = @of_fromMem, id = 0 : i64, burst_length = 64} : memref<4x2x8xi32>
      aiex.npu.dma_memcpy_nd (%out[0,0,0,0][1,1,1,32][0,0,0,1]) { metadata = @of_toMem, id = 1 : i64, burst_length = 510 } : memref<64xi32>
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
    aie.shim_dma_allocation @of_toMem (S2MM, 0, 0)
  }
}

// -----
// CHECK: Requested burst length is not supported by the target. Supported burst lengths: 64 128 256
// CHECK-NOT: 512
// CHECK: burst_length = 512

module {
  aie.device(npu1) {
    aie.runtime_sequence(%in : memref<4x2x8xi32>, %buf : memref<32xi32>, %out : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd (%in[0,2,0,0][1,2,2,8][0,16,8,1]) { metadata = @of_fromMem, id = 0 : i64, burst_length = 64} : memref<4x2x8xi32>
      aiex.npu.dma_memcpy_nd (%out[0,0,0,0][1,1,1,32][0,0,0,1]) { metadata = @of_toMem, id = 1 : i64, burst_length = 512 } : memref<64xi32>
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
    aie.shim_dma_allocation @of_toMem (S2MM, 0, 0)
  }
}
