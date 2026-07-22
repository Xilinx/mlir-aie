//===- empty_device.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi %s | FileCheck %s

// This tests how multiple load_pdi instructions are expanded. There should be
// empty load_pdi instructions generated between configurations for the device
// reset. Since the firmware 'optimizes' away multiple load_pdi instructions
// with the same address, these should be different empty devices so they 
// don't get skipped.


module {

    // CHECK: aie.device(npu2_1col) @empty_1 {
    // CHECK: }
    // CHECK: aie.device(npu2_1col) @empty_0 {
    // CHECK: }

    aie.device(npu2_1col) @my_switchbox_device_1 {
        %tile = aie.tile(0, 2)
        aie.switchbox(%tile) {
            aie.connect<South : 0, Core : 0>
        }
    }

    aie.device(npu2_1col) @my_switchbox_device_2 {
        %tile = aie.tile(0, 2)
        aie.switchbox(%tile) {
            aie.connect<North : 0, Core : 0>
        }
    }

    aie.device(npu2_1col) @my_switchbox_device_3 {
        %tile = aie.tile(0, 2)
        aie.switchbox(%tile) {
            aie.connect<South : 0, DMA : 0>
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
            aiex.npu.load_pdi { device_ref = @my_switchbox_device_1 }
            // CHECK: aiex.npu.load_pdi {device_ref = @empty_1, expand_mode = 0 : i32}
            // CHECK-DAG: %[[V2:.*]] = arith.constant -2147483633 : i32
            // CHECK-DAG: %[[A2:.*]] = arith.constant 2355200 : i32
            // CHECK: aiex.npu.write32(%[[A2]], %[[V2]]) : i32, i32
            // CHECK-DAG: %[[V3:.*]] = arith.constant -2147483648 : i32
            // CHECK-DAG: %[[A3:.*]] = arith.constant 2355516 : i32
            // CHECK: aiex.npu.write32(%[[A3]], %[[V3]]) : i32, i32
            aiex.npu.load_pdi { device_ref = @my_switchbox_device_2 }
            // CHECK: aiex.npu.load_pdi {device_ref = @empty_0, expand_mode = 0 : i32}
            // CHECK-DAG: %[[V4:.*]] = arith.constant -2147483643 : i32
            // CHECK-DAG: %[[A4:.*]] = arith.constant 2355204 : i32
            // CHECK: aiex.npu.write32(%[[A4]], %[[V4]]) : i32, i32
            // CHECK-DAG: %[[V5:.*]] = arith.constant -2147483648 : i32
            // CHECK-DAG: %[[A5:.*]] = arith.constant 2355476 : i32
            // CHECK: aiex.npu.write32(%[[A5]], %[[V5]]) : i32, i32
            aiex.npu.load_pdi { device_ref = @my_switchbox_device_3 }
        }
        // CHECK:     }
    }


}