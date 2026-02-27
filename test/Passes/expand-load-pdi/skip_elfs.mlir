//===- skip_elfs.mlir -------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi %s | FileCheck %s

// Tests that skip_elfs=true on npu.load_pdi produces a PDI expansion
// that omits core ELF loading but still generates the init config
// (switches, locks, etc.) and core enables.

module {

    // CHECK: aie.device(npu2_1col) @empty_0 {
    // CHECK: }

    aie.device(npu2_1col) @my_device {
        %tile = aie.tile(0, 2)
        %lock = aie.lock(%tile, 0) {init = 1 : i32}
        aie.switchbox(%tile) {
            aie.connect<South : 0, Core : 0>
        }
    }

    aie.device(npu2_1col) @main {
        // CHECK: aie.runtime_sequence
        aie.runtime_sequence (%arg0: memref<1xi32>) {
            // The load_pdi with skip_elfs should expand to register writes
            // for switchbox and lock config, but no ELF data.
            // CHECK: aiex.npu.load_pdi {device_ref = @empty_0}
            // CHECK: aiex.npu.write32
            aiex.npu.load_pdi { device_ref = @my_device, skip_elfs = true }
        }
    }

}
