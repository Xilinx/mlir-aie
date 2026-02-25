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
            // CHECK: aiex.npu.write32 {address = 2355200 : ui32, value = 2147483653 : ui32}
            // CHECK: aiex.npu.write32 {address = 2355476 : ui32, value = 2147483648 : ui32}
            aiex.npu.load_pdi { device_ref = @my_switchbox_device_1 }
            // CHECK: aiex.npu.load_pdi {device_ref = @empty_1}
            // CHECK: aiex.npu.write32 {address = 2355200 : ui32, value = 2147483663 : ui32}
            // CHECK: aiex.npu.write32 {address = 2355516 : ui32, value = 2147483648 : ui32}
            aiex.npu.load_pdi { device_ref = @my_switchbox_device_2 }
            // CHECK: aiex.npu.load_pdi {device_ref = @empty_0}
            // CHECK: aiex.npu.write32 {address = 2355204 : ui32, value = 2147483653 : ui32}
            // CHECK: aiex.npu.write32 {address = 2355476 : ui32, value = 2147483648 : ui32}
            aiex.npu.load_pdi { device_ref = @my_switchbox_device_3 }
        }
        // CHECK:     }
    }


}