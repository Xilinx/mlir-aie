//===- empty_device.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi %s | FileCheck %s

// This tests how some configuration register writes are expanded from a
// load_pdi instructions. The expand-load-pdi pass for this file should generate
// an empty load_pdi instruction (for resetting the device) followed by 
// register writes that configure the switchbox connections specified.


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
        // CHECK: aie.runtime_sequence(%arg0: memref<1xi32>) {
        aie.runtime_sequence (%arg0: memref<1xi32>) {
            // CHECK: aiex.npu.load_pdi {device_ref = @empty_0, expand_mode = 0 : i32}
            // CHECK-DAG: %[[V0:.*]] = arith.constant -2147483643 : i32
            // CHECK-DAG: %[[A0:.*]] = arith.constant 2355200 : i32
            // CHECK: aiex.npu.write32(%[[A0]], %[[V0]]) : i32, i32
            // CHECK-DAG: %[[V1:.*]] = arith.constant -2147483648 : i32
            // CHECK-DAG: %[[A1:.*]] = arith.constant 2355476 : i32
            // CHECK: aiex.npu.write32(%[[A1]], %[[V1]]) : i32, i32
            aiex.npu.load_pdi { device_ref = @my_switchbox_device }
        }
        // CHECK: }
    }


}