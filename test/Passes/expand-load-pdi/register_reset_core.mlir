//===- register_reset_core.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt "--aie-expand-load-pdi=register-reset=true" %s | FileCheck %s

// register-reset=true replaces the second load_pdi's empty-device firmware
// reset with an aiex.core_reset over the FIRST (outgoing) device's cores: no
// @empty_1 device, and the tile it resets is re-declared inside @main (an
// SSA value cannot cross into a sibling aie.device region).

module {

    // CHECK: aie.device(npu2_1col) @empty_0 {
    // CHECK: }
    // CHECK-NOT: aie.device(npu2_1col) @empty_1 {

    aie.device(npu2_1col) @dev_a {
        %tile = aie.tile(0, 2)
        aie.core(%tile) {
            aie.end
        }
    }

    aie.device(npu2_1col) @dev_b {
        %tile = aie.tile(0, 2)
        aie.switchbox(%tile) {
            aie.connect<South : 0, Core : 0>
        }
    }

    // The tile aiex.core_reset targets is re-declared at @main's device
    // level (an SSA value cannot cross into the sibling @dev_a region).
    // CHECK: %[[T:.*]] = aie.tile(0, 2)
    aie.device(npu2_1col) @main {
        aie.runtime_sequence (%arg0: memref<1xi32>) {
            // First load_pdi: no known outgoing device, falls back to empty reset.
            // CHECK: aiex.npu.load_pdi {device_ref = @empty_0, expand_mode = 0 : i32}
            aiex.npu.load_pdi { device_ref = @dev_a }
            // Second load_pdi: @dev_a's core is reset in place of a firmware reset.
            // CHECK-NOT: aiex.npu.load_pdi {device_ref = @empty
            // CHECK: aiex.core_reset(%[[T]])
            // CHECK-DAG: %[[V1:.*]] = arith.constant -2147483643 : i32
            // CHECK-DAG: %[[A1:.*]] = arith.constant 2355200 : i32
            // CHECK: aiex.npu.write32(%[[A1]], %[[V1]]) : i32, i32
            aiex.npu.load_pdi { device_ref = @dev_b }
        }
    }

}
