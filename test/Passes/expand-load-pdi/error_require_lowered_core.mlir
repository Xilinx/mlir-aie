//===- error_require_lowered_core.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --aie-expand-load-pdi %s

// This test should error as load_pdi operations can only be inlined after the
// code in each core has been lowered to an ELF file and linked to the core
// with the `elf_file` attribute.

module {

    aie.device(npu2_1col) @my_core_device {
        %tile = aie.tile(0, 2)
        // expected-error @+1 {{Expected lowered ELF file}}
        aie.core(%tile) {
            aie.end
        }
    }

    aie.device(npu2_1col) @main {
        aie.runtime_sequence (%arg0: memref<1xi32>) {
            // expected-error @+1 {{Failed to generate configuration operations}}
            aiex.npu.load_pdi { device_ref = @my_core_device }
        }
    }


}