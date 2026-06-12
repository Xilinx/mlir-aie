//===- empty_device.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
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
            // CHECK: aiex.npu.load_pdi {device_ref = @empty_0}
            // CHECK-DAG: %[[WA0:.+]] = arith.constant 2355200 : i32
            // CHECK-DAG: %[[WV0:.+]] = arith.constant -2147483643 : i32
            // CHECK: aiex.npu.write32(%[[WA0]], %[[WV0]])
            // CHECK-DAG: %[[WA1:.+]] = arith.constant 2355476 : i32
            // CHECK-DAG: %[[WV1:.+]] = arith.constant -2147483648 : i32
            // CHECK: aiex.npu.write32(%[[WA1]], %[[WV1]])
            aiex.npu.load_pdi { device_ref = @my_switchbox_device_1 }
            // CHECK: aiex.npu.load_pdi {device_ref = @empty_1}
            // CHECK-DAG: %[[WA2:.+]] = arith.constant 2355200 : i32
            // CHECK-DAG: %[[WV2:.+]] = arith.constant -2147483633 : i32
            // CHECK: aiex.npu.write32(%[[WA2]], %[[WV2]])
            // CHECK-DAG: %[[WA3:.+]] = arith.constant 2355516 : i32
            // CHECK-DAG: %[[WV3:.+]] = arith.constant -2147483648 : i32
            // CHECK: aiex.npu.write32(%[[WA3]], %[[WV3]])
            aiex.npu.load_pdi { device_ref = @my_switchbox_device_2 }
            // CHECK: aiex.npu.load_pdi {device_ref = @empty_0}
            // CHECK-DAG: %[[WA4:.+]] = arith.constant 2355204 : i32
            // CHECK-DAG: %[[WV4:.+]] = arith.constant -2147483643 : i32
            // CHECK: aiex.npu.write32(%[[WA4]], %[[WV4]])
            // CHECK-DAG: %[[WA5:.+]] = arith.constant 2355476 : i32
            // CHECK-DAG: %[[WV5:.+]] = arith.constant -2147483648 : i32
            // CHECK: aiex.npu.write32(%[[WA5]], %[[WV5]])
            aiex.npu.load_pdi { device_ref = @my_switchbox_device_3 }
        }
        // CHECK:     }
    }


}