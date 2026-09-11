//===- stream_len_decoupled.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A fifo whose channel (de)compresses reaches the runtime through its shim
// allocation, which is where the transfer extent is checked, so the exemption
// has to travel with it.

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-DAG:  aie.shim_dma_allocation @of_cmp_shim_alloc(%{{.*}}, MM2S, 0) {elem_type = memref<64xi16>, stream_len_decoupled}
// CHECK-DAG:  aie.shim_dma_allocation @of_plain_shim_alloc(%{{.*}}, MM2S, 1) {elem_type = memref<64xi16>}

module @decoupled {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile22 = aie.tile(2, 2)

        aie.objectfifo @of_cmp (%tile20, {%tile22}, 2 : i32) {elem_type = memref<64xi16>, stream_len_decoupled} : !aie.objectfifo<memref<64xi16>>
        aie.objectfifo @of_plain (%tile20, {%tile22}, 2 : i32) {elem_type = memref<64xi16>} : !aie.objectfifo<memref<64xi16>>
    }
}
