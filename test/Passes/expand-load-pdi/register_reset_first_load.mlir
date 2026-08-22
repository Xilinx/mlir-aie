//===- register_reset_first_load.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt "--aie-expand-load-pdi=register-reset=true" %s | FileCheck %s

// The first load_pdi in a module has no known outgoing device, so
// register-reset=true must fall back to the same empty-device firmware
// reset as the default mode -- there is nothing to register-reset yet.

module {

    // CHECK: aie.device(npu2_1col) @empty_0 {
    // CHECK: }

    aie.device(npu2_1col) @my_switchbox_device {
        %tile = aie.tile(0, 2)
        aie.switchbox(%tile) {
            aie.connect<South : 0, Core : 0>
        }
    }

    aie.device(npu2_1col) @main {
        aie.runtime_sequence (%arg0: memref<1xi32>) {
            // CHECK: aiex.npu.load_pdi {device_ref = @empty_0, expand_mode = 0 : i32}
            // CHECK-NOT: aiex.core_reset
            // CHECK-NOT: aiex.dma_channel_reset_for
            // CHECK-DAG: %[[V0:.*]] = arith.constant -2147483643 : i32
            // CHECK-DAG: %[[A0:.*]] = arith.constant 2355200 : i32
            // CHECK: aiex.npu.write32(%[[A0]], %[[V0]]) : i32, i32
            aiex.npu.load_pdi { device_ref = @my_switchbox_device }
        }
    }

}
