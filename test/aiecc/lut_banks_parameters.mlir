// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Unresolved parameter bindings must not silently count as verified placement.
// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ -I%S/../../third_party/aie_api/include -fembed-bitcode -c %S/lut_banks_parameters_kernel.cc -o %t.d/lut_banks_parameters.o
// RUN: cd %t.d && not %aiecc --get-core-elfs --check-lut-banks %s 2>&1 | FileCheck %s
// CHECK: the aie::lut tables in 'classify' (parameter 0 and parameter 1)
// CHECK-SAME: have placement that cannot be verified
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ -I%S/../../third_party/aie_api/include -emit-llvm -c %S/lut_banks_parameters_kernel.cc -o %t.d/lut_banks_parameters.bc
// RUN: sed 's/link_with = "lut_banks_parameters.o"/link_with = "lut_banks_parameters.bc", link_with_mode = "merge"/' %s > %t.d/merge.mlir
// RUN: cd %t.d && not %aiecc --get-core-elfs --check-lut-banks merge.mlir 2>&1 | FileCheck %s --check-prefix=MERGE
// RUN: sed 's/sym_name = "cd", mem_bank = 1/sym_name = "cd", mem_bank = 2/' %t.d/merge.mlir > %t.d/separate.mlir
// RUN: cd %t.d && %aiecc --get-core-elfs --check-lut-banks separate.mlir
// RUN: cd %t.d && %aiecc --get-core-elfs --no-unified --check-lut-banks separate.mlir
// MERGE: the aie::lut tables in
// MERGE-SAME: are both in memory bank B

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %ab = aie.buffer(%tile) {sym_name = "ab", mem_bank = 1 : i32} : memref<512xi16>
    %cd = aie.buffer(%tile) {sym_name = "cd", mem_bank = 1 : i32} : memref<512xi16>
    %out = aie.buffer(%tile) {sym_name = "out"} : memref<64xi8>
    func.func private @classify(memref<512xi16>, memref<512xi16>, memref<64xi8>) attributes {link_with = "lut_banks_parameters.o"}
    %core = aie.core(%tile) {
      func.call @classify(%ab, %cd, %out) : (memref<512xi16>, memref<512xi16>, memref<64xi8>) -> ()
      aie.end
    } {stack_size = 1024 : i32}
  }
}
