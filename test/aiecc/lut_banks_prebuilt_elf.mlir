// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// No compiled cores must not be mistaken for successful LUT verification.
// RUN: not %aiecc --get-core-elfs --check-lut-banks --tmpdir=%t.prj %s 2>&1 | FileCheck %s
// CHECK: --check-lut-banks cannot verify a prebuilt core elf_file without its compiler IR

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %core = aie.core(%tile) {
      aie.end
    } {elf_file = "prebuilt.elf"}
  }
}
