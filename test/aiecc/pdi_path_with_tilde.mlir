//===- pdi_path_with_tilde.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bootgen cannot read a BIF file name holding a '~'. A build directory that
// truly holds one is reported as such, rather than as a bootgen syntax error.

// RUN: rm -rf %t && mkdir -p "%t/dir~1"
// RUN: cd "%t/dir~1" && not %aiecc --tmpdir="%t/dir~1" --get-pdi --pdi-name=design.pdi %s 2>&1 | FileCheck %s

// CHECK: aiecc: bootgen cannot read a path holding '~': {{.*}}dir~1{{.*}}; build in a directory whose path has none
// CHECK-NOT: BootGen

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    aie.buffer @buf(%mem) : memref<16xi32> = dense<7>
    aie.runtime_sequence(%a: memref<16xi32>) {
    }
  }
}
