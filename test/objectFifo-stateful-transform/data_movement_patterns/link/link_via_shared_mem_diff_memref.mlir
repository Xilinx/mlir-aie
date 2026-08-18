//===- link_via_shared_mem_diff_memref.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: October 1st 2024
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --verify-diagnostics %s
// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL: aie.device(xcve2302) {
// CHECK:       %[[SHARED0:.*]] = aie.buffer({{.*}}) {sym_name = "of1_cons_buff_0"} : memref<32xi32>
// CHECK:       %[[SHARED1:.*]] = aie.buffer({{.*}}) {sym_name = "of1_cons_buff_1"} : memref<32xi32>
// CHECK:       aie.dma_start(S2MM, 0,
// CHECK:       aie.dma_bd(%[[SHARED0]] : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.dma_bd(%[[SHARED0]] : memref<32xi32> offset = 16 len = 16)
// CHECK:       aie.dma_bd(%[[SHARED1]] : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.dma_bd(%[[SHARED1]] : memref<32xi32> offset = 16 len = 16)
// CHECK:       aie.dma_start(MM2S, 0,
// CHECK:       aie.dma_bd(%[[SHARED0]] : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.dma_bd(%[[SHARED1]] : memref<32xi32> offset = 0 len = 16)

module @link_AIE2 {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile12 = aie.tile(1, 2)
        %tile22 = aie.tile(2, 2)

        aie.objectfifo @of1 (%tile20, {%tile12}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
        aie.objectfifo @of2 (%tile12, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        // expected-error@+1 {{segment 1 has no drainer}}
        aie.objectfifo.link [@of1] -> [@of2] ([] [])
    }
}
