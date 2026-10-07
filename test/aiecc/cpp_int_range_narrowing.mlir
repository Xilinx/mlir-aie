//===- cpp_int_range_narrowing.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// -O1 and up run aie-core-int-range-narrowing, so a core loop with constant
// bounds reaches LLVM with an i32 induction variable instead of i64, for both
// per-core and unified lowering. The INT64_MAX loop keeps i64 at every level.
// Only IR outputs are requested, so no core compiler is needed.

// RUN: %aiecc -O0 --get=input_with_addresses.mlir --get='perCoreArches_{0}.txt' --get='llvmIR_{0}.ll' --tmpdir=%t.o0 --output-dir=%t.o0 %s
// RUN: FileCheck %s --check-prefix=O0-IR --input-file=%t.o0/input_with_addresses.mlir
// RUN: FileCheck %s --check-prefix=O0-LL --input-file=%t.o0/llvmIR_main_core_0_2.ll

// RUN: %aiecc --get=input_with_addresses.mlir --get='perCoreArches_{0}.txt' --get='llvmIR_{0}.ll' --tmpdir=%t.default --output-dir=%t.default %s
// RUN: FileCheck %s --check-prefix=NARROW-IR --input-file=%t.default/input_with_addresses.mlir
// RUN: FileCheck %s --check-prefix=NARROW-LL --input-file=%t.default/llvmIR_main_core_0_2.ll

// RUN: %aiecc --unified --get='perCoreArches_{0}.txt' --get='llvmIR_{0}.ll' --tmpdir=%t.unified --output-dir=%t.unified %s
// RUN: FileCheck %s --check-prefix=NARROW-LL --input-file=%t.unified/llvmIR_main_core_0_2.ll

// O0-IR-LABEL: aie.core
// O0-IR-NOT: index_castui
// O0-IR: arith.cmpi slt, %{{.*}}, %c32 : index

// O0-LL: phi i64
// O0-LL: icmp slt i64 %{{.*}}, 9223372036854775807
// O0-LL: phi i64
// O0-LL: icmp slt i64 %{{.*}}, 32

// NARROW-IR-LABEL: aie.core
// NARROW-IR: arith.cmpi slt, %{{.*}}, %c9223372036854775807 : index
// NARROW-IR: arith.cmpi slt, %[[I:.*]], %c32_i32 : i32
// NARROW-IR: arith.index_castui %[[I]] : i32 to index

// NARROW-LL: phi i64
// NARROW-LL: icmp slt i64 %{{.*}}, 9223372036854775807
// NARROW-LL: phi i32
// NARROW-LL: icmp slt i32 %{{.*}}, 32

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %buf = aie.buffer(%tile) {sym_name = "buf"} : memref<32xi32>
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %cmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : i32
      scf.for %iter = %c0 to %cmax step %c1 {
        scf.for %i = %c0 to %c32 step %c1 {
          %x = memref.load %buf[%i] : memref<32xi32>
          %y = arith.addi %x, %one : i32
          memref.store %y, %buf[%i] : memref<32xi32>
        }
      }
      aie.end
    }
  }
}
