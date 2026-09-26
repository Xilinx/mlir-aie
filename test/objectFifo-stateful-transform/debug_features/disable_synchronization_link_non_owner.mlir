//===- disable_synchronization_link_non_owner.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A link's participants share one pool, so the flag has to reach it from
// whichever of them carries it. Here that is an input rather than the fifo the
// pool is named after.

// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s

// CHECK: aie.objectfifo.pool @link3_pool({{.*}}) {{{.*}}disableSynchronization{{.*}}}

module @link_non_owner {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)
        %tile23 = aie.tile(2, 3)

        aie.objectfifo @link1 (%tile22, {%tile21}, 1 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<4x4xi32>>
        aie.objectfifo @link2 (%tile23, {%tile21}, 1 : i32) : !aie.objectfifo<memref<20xi32>>
        aie.objectfifo @link3 (%tile21, {%tile20}, 1 : i32) : !aie.objectfifo<memref<36xi32>>
        aie.objectfifo.link [@link1, @link2] -> [@link3] ([0, 16][])
    }
}
