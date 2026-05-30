//===- sparse_fifo_split_attr_propagation.mlir -----------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
// SparseFifo discardable-attr propagation through
// AIEObjectFifoStatefulTransform's split-fifo path.
//
// (test/Conversion/DmaToNpu/dma_to_npu_sparse_compression.mlir)
// that verified only the FINAL hop — given a hand-constructed
// ``aiex.npu.writebd {aie.enable_compression = true}``, the
// BD-emit pass flips bit 31 in DMA_BDX_1 correctly. That test
// did NOT drive the upstream propagation chain. This test
// closes the gap by exercising the full pipeline:
//
//   ObjectFifoCreateOp{aie.compress_mm2s, aie.decompress_s2mm}
//     -> AIEObjectFifoStatefulTransform splits into
//        (producerFifo, consumerFifo)
//     -> consumerFifo MUST carry the same SparseFifo attrs
//     -> propagateSparseCompressionAttr emits
//        aie.enable_compression on BOTH the producer-side MM2S
//        DMABDOp AND the consumer-side S2MM DMABDOp.
//
// Cross-column tiles (1,2) and (3,3) force the split-fifo
// path (no shared-memory neighbor optimization). Both endpoints
// are compute tiles so the cross-module footgun guard (compute
// vs memtile vs shim) does not skip emission.
//
// Mirrors the shape of base/non_adjacency_test_AIE2.mlir.

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// Producer-side MM2S BDs on tile (1,2): MUST carry
// aie.enable_compression = true.
// CHECK:         aie.mem(%[[VAL_PROD_TILE:.*]]) {
// CHECK:           aie.dma_start(MM2S, 0
// CHECK:           aie.dma_bd({{.*}}) {aie.enable_compression = true}
// CHECK:           aie.dma_bd({{.*}}) {aie.enable_compression = true}
// Consumer-side S2MM BDs on tile (3,3): MUST also carry
// CHECK:         aie.mem(%[[VAL_CONS_TILE:.*]]) {
// CHECK:           aie.dma_start(S2MM, 0
// CHECK:           aie.dma_bd({{.*}}) {aie.enable_compression = true}
// CHECK:           aie.dma_bd({{.*}}) {aie.enable_compression = true}

module @sparse_fifo_split_attr_propagation {
    aie.device(xcve2302) {
        %tile12 = aie.tile(1, 2)
        %tile33 = aie.tile(3, 3)

        // SparseFifo discardable attrs as IRON's
        // aie.iron.sparse.SparseFifo.resolve() pins them.
        // via_DMA = true forces the split-fifo path even when
        // tiles happen to be neighbors (defensive).
        aie.objectfifo @sparse_of (%tile12, {%tile33}, 2 : i32) {
            aie.compress_mm2s = true,
            aie.decompress_s2mm = true,
            aie.sparsity_pattern = "N:M",
            aie.sparsity_n = 2 : i32,
            aie.sparsity_m = 4 : i32,
            via_DMA = true
        } : !aie.objectfifo<memref<16xi32>>

        func.func @some_work(%lineOut : memref<16xi32>) -> () {
            return
        }

        %core12 = aie.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 12 : index

            scf.for %indexInHeight = %c0 to %height step %c1 {
                %subview = aie.objectfifo.acquire @sparse_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem0) : (memref<16xi32>) -> ()
                aie.objectfifo.release @sparse_of (Produce, 1)
            }

            aie.end
        }

        %core33 = aie.core(%tile33) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 12 : index

            scf.for %indexInHeight = %c0 to %height step %c1 {
                %subview = aie.objectfifo.acquire @sparse_of (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem0) : (memref<16xi32>) -> ()
                aie.objectfifo.release @sparse_of (Consume, 1)
            }

            aie.end
        }
    }
}
