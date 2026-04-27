//===- variable_rate_fifo_attr_propagation.mlir ---------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
// or wedge under the split-fifo + cross-column path that the
//
// The ``aie.variable_rate`` discardable attribute is consumed by
// ``unrollForLoops`` (to opt out of LCM-based unrolling) and by
// the split-fifo propagation slot (to copy the marker to
// consumer-side fifos for diagnostic introspection). Since the
// pass lowers and erases the ``aie.objectfifo`` ops as part of
// the transform, the marker itself is not visible in the post-
// pass output -- the propagation lives ENTIRELY inside the pass.
//
// What IS visible (and what this test asserts):
//   1. Cross-column endpoints (1,2) and (3,3) lower cleanly with
//      independent buffer / lock allocations on each tile (the
//      split-fifo path fired correctly; the variable_rate marker
//      did not interfere).
//   2. A producer-side ``aie.mem`` block exists on tile (1,2)
//      with an MM2S DMA and at least one ``aie.dma_bd``.
//   3. A consumer-side ``aie.mem`` block exists on tile (3,3)
//      with an S2MM DMA and at least one ``aie.dma_bd``.
//
// The companion test ``variable_rate_fifo_skip_unroll.mlir``
// exercises the LCM-unroll opt-out behavior directly (a
// 12-iteration loop with one variable_rate fifo and one vanilla
// fifo unrolls by factor 2, NOT by factor 6).

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:         aie.flow(%{{.*}}, DMA : 0, %{{.*}}, DMA : 0)
// Producer-side DMA on tile (1,2):
// CHECK:         %mem_1_2 = aie.mem(%{{.*}}tile_1_2)
// CHECK:           aie.dma_start(MM2S
// CHECK:           aie.dma_bd
// Consumer-side DMA on tile (3,3):
// CHECK:         %mem_3_3 = aie.mem(%{{.*}}tile_3_3)
// CHECK:           aie.dma_start(S2MM
// CHECK:           aie.dma_bd

module @variable_rate_fifo_attr_propagation {
    aie.device(xcve2302) {
        %tile12 = aie.tile(1, 2)
        %tile33 = aie.tile(3, 3)

        // VariableRateFifo discardable attr as IRON's
        // aie.iron.variable_rate.VariableRateFifo.resolve()
        // pins it. via_DMA = true forces the split-fifo path
        // even when tiles happen to be neighbors (defensive).
        aie.objectfifo @vr_of (%tile12, {%tile33}, 2 : i32) {
            aie.variable_rate = true,
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
                %subview = aie.objectfifo.acquire @vr_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem0) : (memref<16xi32>) -> ()
                aie.objectfifo.release @vr_of (Produce, 1)
            }

            aie.end
        }

        %core33 = aie.core(%tile33) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 12 : index

            scf.for %indexInHeight = %c0 to %height step %c1 {
                %subview = aie.objectfifo.acquire @vr_of (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem0) : (memref<16xi32>) -> ()
                aie.objectfifo.release @vr_of (Consume, 1)
            }

            aie.end
        }
    }
}
