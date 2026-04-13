//===- depth_two_scalar_test.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// Verify that a scalar depth-2 objectFifo with a max acquire of 1 lowers to
// two buffers (maxAcquire + 1 = 2), not three (declared depth + 1 = 3). The
// buffer count is driven by the maximum acquire count in the core, not the
// declared objectFifo depth. Since the consumer is a shim tile (which has no
// local memory), buffers are only created on the producer side; the shim tile
// side only gets locks.

// CHECK: %[[BUFF0:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of_buff_0"} : memref<16xi32>
// CHECK: %[[BUFF1:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of_buff_1"} : memref<16xi32>
// CHECK-NOT: sym_name = "of_buff_2"
// CHECK-NOT: sym_name = "of_cons_buff_0"

module @depth_two_scalar {
    aie.device(xcve2302) {
        %tile10 = aie.tile(1, 0)
        %tile12 = aie.tile(1, 2)

        aie.objectfifo @of (%tile12, {%tile10}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        func.func @some_work(%buf : memref<16xi32>) -> () {
            return
        }

        %core12 = aie.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c10 = arith.constant 10 : index
            scf.for %i = %c0 to %c10 step %c1 {
                %subview = aie.objectfifo.acquire @of(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem) : (memref<16xi32>) -> ()
                aie.objectfifo.release @of(Produce, 1)
            }
            aie.end
        }
    }
}
