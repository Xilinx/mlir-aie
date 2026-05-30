//===- dma_to_npu_sparse_compression.mlir ----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// IRON SparseFifo discardable contract by flipping the AIE2/AIE2P
// tile DMA ``Enable_Compression`` bit (DMA_BDX_1[31]) when the input
// NpuWriteBdOp carries ``aie.enable_compression = true``, and leaves
// it cleared otherwise. The discardable-attr propagation chain is
// (ObjectFifoCreateOp -> DMABDOp -> NpuWriteBdOp); this lit test
// pins the final hop (NpuWriteBdOp -> block-write words).
//
// Hand-computed reference: same field values as
// ``dma_to_npu_core_tile.mlir`` so the only-differing-bit is bit 31
// of the second word.
//
//   Without compression: words[1] = (1<<30)|(21<<24)|(7<<19)|(5<<16)
//                                 = 0x553D0000 = 1430061056
//   With    compression: words[1] = (1<<31) | (above)
//                                 = 0xD53D0000 = 3577544704 (= -717422592 as signed i32)
//
// The other five words are identical to the core-tile reference and
// should match the dense baseline byte-for-byte.

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s

// CHECK-LABEL: module
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<6xi32> = dense<[1180485, -717422592, 9093684, 266847009, 22249971, 1465218380]>
// CHECK: memref.global "private" constant @blockwrite_data_1 : memref<6xi32> = dense<[1180485, 1430061056, 9093684, 266847009, 22249971, 1465218380]>
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence() {
      // Positive case: ``aie.enable_compression = true`` -> bit 31 set
      // on words[1]. This mirrors what
      // ``AIEDMATasksToNPU::rewriteSingleBD`` will set on the
      // NpuWriteBdOp after picking up the same discardable attr from
      // the source ``aie.dma_bd``, which in turn was tagged by
      // ``AIEObjectFifoStatefulTransform::createBdBlock`` from the
      // originating SparseFifo's ``aie.decompress_s2mm = true``.
      aiex.npu.writebd {
        bd_id = 6 : i32,
        buffer_length = 837 : i32,
        buffer_offset = 291 : i32,
        enable_packet = 1 : i32,
        out_of_order_id = 21 : i32,
        packet_id = 7 : i32,
        packet_type = 5 : i32,
        column = 0 : i32,
        row = 2 : i32,
        d0_stride = 564 : i32,
        d0_size = 62 : i32,
        d0_zero_after = 0 : i32,
        d0_zero_before = 0 : i32,
        d1_stride = 1110 : i32,
        d1_size = 127 : i32,
        d1_zero_after = 0 : i32,
        d1_zero_before = 0 : i32,
        d2_size = 0 : i32,
        d2_stride = 801 : i32,
        d2_zero_after = 0 : i32,
        d2_zero_before = 0 : i32,
        ddr_id = 0 : i32,
        iteration_current = 42 : i32,
        iteration_stride = 499 : i32,
        iteration_size = 28 : i32,
        lock_acq_enable = 1 : i32,
        lock_acq_id = 12 : i32,
        lock_acq_val = 42 : i32,
        lock_rel_id = 11 : i32,
        lock_rel_val = 85 : i32,
        next_bd = 10 : i32,
        use_next_bd = 1 : i32,
        valid_bd = 1 : i32,
        burst_length = 0 : i32,
        aie.enable_compression = true
      }
      // Negative case (regression-protect the dense default): no
      // discardable ``aie.enable_compression`` attr -> bit 31 stays 0.
      aiex.npu.writebd {
        bd_id = 6 : i32,
        buffer_length = 837 : i32,
        buffer_offset = 291 : i32,
        enable_packet = 1 : i32,
        out_of_order_id = 21 : i32,
        packet_id = 7 : i32,
        packet_type = 5 : i32,
        column = 0 : i32,
        row = 2 : i32,
        d0_stride = 564 : i32,
        d0_size = 62 : i32,
        d0_zero_after = 0 : i32,
        d0_zero_before = 0 : i32,
        d1_stride = 1110 : i32,
        d1_size = 127 : i32,
        d1_zero_after = 0 : i32,
        d1_zero_before = 0 : i32,
        d2_size = 0 : i32,
        d2_stride = 801 : i32,
        d2_zero_after = 0 : i32,
        d2_zero_before = 0 : i32,
        ddr_id = 0 : i32,
        iteration_current = 42 : i32,
        iteration_stride = 499 : i32,
        iteration_size = 28 : i32,
        lock_acq_enable = 1 : i32,
        lock_acq_id = 12 : i32,
        lock_acq_val = 42 : i32,
        lock_rel_id = 11 : i32,
        lock_rel_val = 85 : i32,
        next_bd = 10 : i32,
        use_next_bd = 1 : i32,
        valid_bd = 1 : i32,
        burst_length = 0 : i32
      }
    }
  }
}
