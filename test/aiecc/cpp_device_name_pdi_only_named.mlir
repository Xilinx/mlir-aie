//===- cpp_device_name_pdi_only_named.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Outside a full ELF, --device-name builds only the named device, even when
// its sequence loads another: one PDI, under the one name asked for.

// REQUIRES: peano

// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && %aiecc --get-pdi --pdi-name=design.pdi --device-name=main --tmpdir=%t/prj %s
// RUN: ls %t | FileCheck %s --implicit-check-not=.pdi

// CHECK: design.pdi

module {
  aie.device(npu2_1col) @other {
    %t = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @main {
    %t = aie.tile(0, 2)
    aie.runtime_sequence @seq(%buf : memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @other}
    }
  }
}
