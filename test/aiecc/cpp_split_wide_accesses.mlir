//===- cpp_split_wide_accesses.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// An AIE2P core's 1024-bit load and store reach LLVM as 512-bit ones at the
// default -O2; an AIE2 core's are left whole. llc runs with the
// combine_concat_unmerge_phis combine off.

// RUN: %aiecc --get='llvmIR_{0}.ll' --tmpdir=%t.npu2 --output-dir=%t.npu2 %s
// RUN: FileCheck %s --check-prefix=NPU2 --input-file=%t.npu2/llvmIR_main_core_0_2.ll
// RUN: sed 's/npu2/npu1/' %s > %t.npu1.mlir
// RUN: %aiecc --get='llvmIR_{0}.ll' --tmpdir=%t.npu1 --output-dir=%t.npu1 %t.npu1.mlir
// RUN: FileCheck %s --check-prefix=NPU1 --input-file=%t.npu1/llvmIR_main_core_0_2.ll
// RUN: rm -rf %t.d && mkdir -p %t.d && cd %t.d
// RUN: %aiecc --get-core-elfs -n --verbose %s 2>&1 | FileCheck %s --check-prefix=LLC

// NPU2-NOT: <32 x float>
// NPU2: load <16 x float>, ptr @in
// NPU2: load <16 x float>, ptr getelementptr inbounds{{.*}}(i8, ptr @in, i64 64)
// NPU2: store <16 x float> %{{.*}}, ptr @out
// NPU2: store <16 x float> %{{.*}}, ptr getelementptr inbounds{{.*}}(i8, ptr @out, i64 64)
// NPU2-NOT: <32 x float>

// NPU1: load <32 x float>, ptr @in
// NPU1: store <32 x float> %{{.*}}, ptr @out

// LLC: llc{{.*}} --aie2pprelegalizercombiner-disable-rule=combine_concat_unmerge_phis

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %in = aie.buffer(%tile) {sym_name = "in"} : memref<32xf32>
    %out = aie.buffer(%tile) {sym_name = "out"} : memref<32xf32>
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.0 : f32
      %v = vector.transfer_read %in[%c0], %cst {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
      vector.transfer_write %v, %out[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32>
      aie.end
    }
  }
}
