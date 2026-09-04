//===- stream_len_decoupled_link.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A link's two sides share one pool, so the pool is built from whichever fifo
// owns it -- the input, here. The exemption belongs to the channel rather than
// to that fifo, so it has to survive from the side that declares it, or the
// shim allocation ends up rejecting the transfer the exemption exists for.

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-DAG:  aie.shim_dma_allocation @out_shim_alloc(%{{.*}}, S2MM, 0) {elem_type = memref<64xi16>, stream_len_decoupled}

module @decoupled_link {
    aie.device(npu2) {
        %tile00 = aie.tile(0, 0)
        %tile01 = aie.tile(0, 1)

        aie.objectfifo @in  (%tile00, {%tile01}, 2 : i32) : !aie.objectfifo<memref<64xi16>>
        aie.objectfifo @out (%tile01, {%tile00}, 2 : i32) {stream_len_decoupled} : !aie.objectfifo<memref<64xi16>>
        aie.objectfifo.link [@in] -> [@out] ([] [])
    }
}
