//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --pass-pipeline='builtin.module(aie-expand-load-pdi{register-reset=true ctrl-pkt=true})' %s 2>&1 | FileCheck %s

// ctrl-pkt configures through control packets, whose targets this pass does not
// read, so a difference over them would be empty and would drop the overlay
// preload the mode depends on.

// CHECK: register-reset and ctrl-pkt are mutually exclusive

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
