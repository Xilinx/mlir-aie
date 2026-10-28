//===- insts_name_several_sequences.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Each runtime sequence gets its own instruction stream. A fixed
// --npu-insts-name can hold only one, so with two sequences aiecc names the
// two that collide; --sequence-name picks one, and the default name keys each.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: not %aiecc --get-npu-insts --npu-insts-name=insts.bin --output-dir=%t.d/fixed --tmpdir=%t.d/fixed/work %s 2>&1 | FileCheck %s --check-prefix=FIXED
// RUN: %aiecc --get-npu-insts --npu-insts-name=insts.bin --sequence-name=second --output-dir=%t.d/one --tmpdir=%t.d/one/work %s
// RUN: ls %t.d/one | FileCheck %s --check-prefix=ONE
// RUN: %aiecc --get-npu-insts --output-dir=%t.d/each --tmpdir=%t.d/each/work %s
// RUN: ls %t.d/each | FileCheck %s --check-prefix=EACH

// FIXED: produced duplicate output path '{{.*}}insts.bin' for items 'main_{{first|second}}' and 'main_{{first|second}}'; its output name needs '{0}' to tell them apart
// ONE: insts.bin
// EACH-DAG: insts_main_first.bin
// EACH-DAG: insts_main_second.bin

module {
  aie.device(npu2) @main {
    %12 = aie.tile(1, 2)
    aie.runtime_sequence @first(%buf : memref<1xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %ch = arith.constant 0 : i32
      aiex.npu.sync(%c0, %c0, %c1, %ch, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
    aie.runtime_sequence @second(%buf : memref<1xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %ch = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %ch, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}
