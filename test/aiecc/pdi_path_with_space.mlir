//===- pdi_path_with_space.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A build directory whose path holds a space still assembles its PDI: the BIF
// bootgen reads names the CDOs there.

// RUN: rm -rf "%t dir" && mkdir -p "%t dir"
// RUN: cd "%t dir" && %aiecc --tmpdir="%t dir" --get-pdi --pdi-name=design.pdi %s
// RUN: test -s "%t dir/design.pdi"

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %buf = aie.buffer(%mem) {sym_name = "buf"} : memref<16xi32> = dense<7>
    aie.runtime_sequence(%a: memref<16xi32>) {
    }
  }
}
