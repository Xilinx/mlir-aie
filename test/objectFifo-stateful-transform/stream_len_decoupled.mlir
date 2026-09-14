//===- stream_len_decoupled.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectfifo-split %s | FileCheck %s

// The attribute has to reach the pool a fifo lowers to, and only that fifo's.

// CHECK-DAG: aie.objectfifo.pool @of_cmp_cons_pool({{.*}}) {{{.*}}streamLenDecoupled{{.*}}}
// CHECK-DAG: aie.objectfifo.pool @of_plain_cons_pool({{.*}}) {depth = 2 : i32, fifoName = "of_plain"}

module @decoupled {
    aie.device(npu2) {
        %tile00 = aie.tile(0, 0)
        %tile02 = aie.tile(0, 2)
        %tile03 = aie.tile(0, 3)

        aie.objectfifo @of_cmp   (%tile00, {%tile02}, 2 : i32) {stream_len_decoupled} : !aie.objectfifo<memref<64xi16>>
        aie.objectfifo @of_plain (%tile00, {%tile03}, 2 : i32) : !aie.objectfifo<memref<64xi16>>
    }
}

// -----

// A link's pool is shared and is built from the incoming fifo, so an attribute
// declared on the outgoing side still has to reach it.

// CHECK-DAG: aie.objectfifo.pool @in_cons_pool({{.*}}) {{{.*}}streamLenDecoupled{{.*}}}

module @decoupled_link {
    aie.device(npu2) {
        %tile00 = aie.tile(0, 0)
        %tile01 = aie.tile(0, 1)

        aie.objectfifo @in  (%tile00, {%tile01}, 2 : i32) : !aie.objectfifo<memref<64xi16>>
        aie.objectfifo @out (%tile01, {%tile00}, 2 : i32) {stream_len_decoupled} : !aie.objectfifo<memref<64xi16>>
        aie.objectfifo.link [@in] -> [@out] ([] [])
    }
}
