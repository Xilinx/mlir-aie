//===- empty_device.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi %s | FileCheck %s

// This tests how the load_pdi instruction for an empty device are expanded.
// All that should be generated is an empty device load_pdi, which is used to
// reset the device. There should not follow any write32s/blockwrites, as there
// is nothing to configure in an empty device.

module {

    // CHECK: aie.device(npu2_1col) @empty_0 {
    // CHECK: }

    aie.device(npu2_1col) @my_empty_device {
        aie.end
    }

    aie.device(npu2_1col) @main {
        // CHECK: aie.runtime_sequence(%arg0: memref<1xi32>) {
        aie.runtime_sequence (%arg0: memref<1xi32>) {
            // CHECK: aiex.npu.load_pdi {device_ref = @empty_0, expand_mode = 0 : i32}
            aiex.npu.load_pdi { device_ref = @my_empty_device }
        }
        // CHECK: }
    }

}