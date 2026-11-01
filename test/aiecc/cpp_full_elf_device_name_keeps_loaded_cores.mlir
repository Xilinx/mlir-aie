//===- cpp_full_elf_device_name_keeps_loaded_cores.mlir --------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cpp_full_elf_device_name_keeps_loaded.mlir with a core on every device, so
// core compilation is the first stage --device-name filters. The devices main
// reaches come from the input, whichever stage asks first, and also with
// --expand-load-pdis, which lowers the load_pdi references away.

// REQUIRES: peano

// RUN: rm -rf %t && mkdir -p %t/plain %t/expand
// RUN: cd %t/plain && %aiecc --get-full-elf --device-name=main --tmpdir=%t/plain %s 2>&1
// RUN: cat %t/plain/full_elf_config.json | FileCheck %s --implicit-check-not=unrelated.pdi
// RUN: cd %t/expand && %aiecc --get-full-elf --device-name=main --expand-load-pdis --tmpdir=%t/expand %s 2>&1
// RUN: cat %t/expand/full_elf_config.json | FileCheck %s --check-prefix=EXPAND --implicit-check-not=unrelated.pdi
// RUN: not ls %t/plain/elfs_unrelated_core_0_2
// RUN: not ls %t/expand/elfs_unrelated_core_0_2

// CHECK-DAG: main.pdi
// CHECK-DAG: other.pdi
// CHECK-DAG: third.pdi
// CHECK-DAG: fourth.pdi

// With the load_pdi ops expanded into writes, main's and other's kernels carry
// their own PDIs and the empty reset ones.
// EXPAND-DAG: main.pdi
// EXPAND-DAG: other.pdi
// EXPAND-DAG: empty_0.pdi

module {
  aie.device(npu2_1col) @unrelated {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) {
      aie.end
    }
  }
  aie.device(npu2_1col) @fourth {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) {
      aie.end
    }
  }
  aie.device(npu2_1col) @third {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) {
      aie.end
    }
  }
  aie.device(npu2_1col) @other {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) {
      aie.end
    }
    aie.runtime_sequence @other_seq(%buf : memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @third}
    }
  }
  aie.device(npu2_1col) @main {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) {
      aie.end
    }
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
