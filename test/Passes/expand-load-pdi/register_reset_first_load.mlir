//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi=register-reset=true %s | FileCheck %s

// A single load_pdi has no previous device whose registers could survive, so
// the firmware reset stays.

// CHECK-LABEL: aie.runtime_sequence
// CHECK:       aiex.npu.load_pdi {device_ref = @empty_0

module {
    aie.device(npu2_1col) @dev1 {
        %tile = aie.tile(0, 2)
        aie.switchbox(%tile) { aie.connect<South : 0, Core : 0> }
    }
    aie.device(npu2_1col) @main {
        aie.runtime_sequence(%arg0: memref<1xi32>) {
            aiex.npu.load_pdi { device_ref = @dev1 }
        }
    }
}
