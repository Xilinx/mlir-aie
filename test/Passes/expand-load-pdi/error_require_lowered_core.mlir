//===- error_require_lowered_core.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Cores without elf_file are gracefully skipped during PDI expansion.
// This supports lightweight "reset-only" devices that have CoreOps for
// lock/core initialization but no ELF to load.

// RUN: aie-opt --aie-expand-load-pdi %s | FileCheck %s

module {

    aie.device(npu2_1col) @my_core_device {
        %tile = aie.tile(0, 2)
        %lock = aie.lock(%tile, 0) {init = 1 : i32}
        // CoreOp without elf_file — should be skipped by addAieElfs
        // but still trigger core reset/enable in initLocks/addCoreEnable
        aie.core(%tile) {
            aie.end
        }
    }

    aie.device(npu2_1col) @main {
        // CHECK: aie.runtime_sequence
        aie.runtime_sequence (%arg0: memref<1xi32>) {
            // CHECK: aiex.npu.load_pdi
            // CHECK: aiex.npu.write32
            aiex.npu.load_pdi { device_ref = @my_core_device }
        }
    }

}
