//===- cpp_full_elf_device_name_keeps_loaded.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// --device-name=main keeps the devices main's sequences load and configure,
// so the full ELF carries their PDIs as well as main's.

// REQUIRES: peano

// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && aiecc --get-full-elf --device-name=main --tmpdir=%t %s 2>&1
// RUN: cat %t/full_elf_config.json | FileCheck %s --implicit-check-not=unrelated.pdi

// CHECK-DAG: main.pdi
// CHECK-DAG: other.pdi
// CHECK-DAG: third.pdi
// CHECK-DAG: fourth.pdi

module {
  aie.device(npu2_1col) @unrelated {
    %t = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @fourth {
    %t = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @third {
    %t = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @other {
    %t = aie.tile(0, 2)
    aie.runtime_sequence @other_seq(%buf : memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @third}
    }
  }
  aie.device(npu2_1col) @main {
    %t = aie.tile(0, 2)
    aie.runtime_sequence @seq(%buf : memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @other}
      aiex.configure @fourth {
      }
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}
