//===- cpp_o3_bf16_matmul.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A bf16 8x8x8 matmul core loop. At -O3 the LHS tile is loaded as two
// 512-bit halves instead of one 1024-bit load; the RHS tile stays one load.
// At both levels the transposed RHS is widened by a bf16 multiply by 1.0.
// Only IR outputs are requested, so no core compiler is needed.
// cpp_o3_bf16_matmul_opt.test checks the same core after Peano's opt.

// RUN: %aiecc -O2 --get='perCoreArches_{0}.txt' --get='llvmIR_{0}.ll' --tmpdir=%t.o2 --output-dir=%t.o2 %s
// RUN: FileCheck %s --check-prefixes=CHECK,O2 --input-file=%t.o2/llvmIR_main_core_0_2.ll
// RUN: %aiecc -O3 --get='perCoreArches_{0}.txt' --get='llvmIR_{0}.ll' --tmpdir=%t.o3 --output-dir=%t.o3 %s
// RUN: FileCheck %s --check-prefixes=CHECK,O3 --input-file=%t.o3/llvmIR_main_core_0_2.ll

// CHECK-LABEL: define {{.*}} @core_0_2(
// O2-NOT: load <32 x bfloat>
// O2-COUNT-2: load <64 x bfloat>
// O2-NOT: load <32 x bfloat>

// O3-COUNT-2: load <32 x bfloat>
// O3: load <64 x bfloat>
// O3-NOT: load <64 x bfloat>

// CHECK: call <64 x float> @llvm.aie2p.I1024.I1024.ACC2048.bf.mul.conf(<64 x bfloat> %{{.*}}, <64 x bfloat> splat (bfloat 1.000000e+00), i32 60)
// CHECK: call <64 x i32> @llvm.aie2p.BFP576.BFP576.ACC2048.mac.conf(

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %a = aie.buffer(%tile) {sym_name = "a"} : memref<4x8x8xbf16>
    %b = aie.buffer(%tile) {sym_name = "b"} : memref<4x8x8xbf16>
    %c = aie.buffer(%tile) {sym_name = "c"} : memref<8x8xf32>
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %pbf = arith.constant 0.0 : bf16
      %pf = arith.constant 0.0 : f32
      %acc0 = vector.transfer_read %c[%c0, %c0], %pf {in_bounds = [true, true]} : memref<8x8xf32>, vector<8x8xf32>
      %acc = scf.for %k = %c0 to %c4 step %c1 iter_args(%acck = %acc0) -> (vector<8x8xf32>) {
        %ta = vector.transfer_read %a[%k, %c0, %c0], %pbf {in_bounds = [true, true]} : memref<4x8x8xbf16>, vector<8x8xbf16>
        %tb = vector.transfer_read %b[%k, %c0, %c0], %pbf {in_bounds = [true, true]} : memref<4x8x8xbf16>, vector<8x8xbf16>
        %r = aievec.matmul_aie2p %ta, %tb, %acck : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
        scf.yield %r : vector<8x8xf32>
      }
      vector.transfer_write %acc, %c[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xf32>, memref<8x8xf32>
      aie.end
    }
  }
}
