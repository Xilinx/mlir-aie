// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Check merge-mode kernels even though link_files is empty. Reuse the same
// design with bad and good table placement to test both sides of the check.
// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: cd %t.d && %aiecc -n --get-core-elfs --check-lut-banks %s
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ -I%S/../../third_party/aie_api/include -emit-llvm -c %S/lut_banks_same_bank_kernel.cc -o %t.d/lut_banks_merge.bc
// RUN: cd %t.d && not %aiecc --get-core-elfs --check-lut-banks %s 2>&1 | FileCheck %s
// RUN: cd %t.d && not %aiecc --get-core-elfs --no-unified --check-lut-banks %s 2>&1 | FileCheck %s
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ -I%S/../../third_party/aie_api/include -emit-llvm -c %S/lut_banks_ok_kernel.cc -o %t.d/lut_banks_merge.bc
// RUN: cd %t.d && %aiecc --get-core-elfs --check-lut-banks %s
// CHECK: the aie::lut tables in
// CHECK-SAME: are both in memory bank

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %out = aie.buffer(%tile) {sym_name = "out"} : memref<64xi8>
    func.func private @classify(memref<64xi8>) attributes {link_with = "lut_banks_merge.bc", link_with_mode = "merge"}
    %core = aie.core(%tile) {
      func.call @classify(%out) : (memref<64xi8>) -> ()
      aie.end
    } {stack_size = 1024 : i32, data_size = 4096 : i32}
  }
}
