//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi=register-reset=true %s | FileCheck %s

// dev1 enables slave port South_0 and dev2 enables North_0, so the two
// configurations write disjoint slave-config registers and dev2's config leaves
// dev1's port enabled. The boundary restores it instead of loading an empty
// device: 0x23F114 is Stream_Switch_Slave_Config_South_0, reset value 0.

// CHECK-LABEL: aie.runtime_sequence
// The first boundary has no predecessor to undo, so it keeps its firmware reset.
// CHECK:       aiex.npu.load_pdi {device_ref = @empty_0
// CHECK:       aiex.npu.write32
// CHECK:       aiex.npu.write32
// The second is replaced by a write restoring dev1's slave port.
// CHECK-NOT:   aiex.npu.load_pdi
// CHECK:       %[[RA:.*]] = arith.constant 2355476 : i32
// CHECK:       %[[RV:.*]] = arith.constant 0 : i32
// CHECK:       aiex.npu.write32(%[[RA]], %[[RV]])

module {
    aie.device(npu2_1col) @dev1 {
        %tile = aie.tile(0, 2)
        aie.switchbox(%tile) { aie.connect<South : 0, Core : 0> }
    }
    aie.device(npu2_1col) @dev2 {
        %tile = aie.tile(0, 2)
        aie.switchbox(%tile) { aie.connect<North : 0, Core : 0> }
    }
    aie.device(npu2_1col) @main {
        aie.runtime_sequence(%arg0: memref<1xi32>) {
            aiex.npu.load_pdi { device_ref = @dev1 }
            aiex.npu.load_pdi { device_ref = @dev2 }
        }
    }
}
