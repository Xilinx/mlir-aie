// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A probe starts the small .data at its required 4096-byte alignment. Merely
// carrying its 64-byte size into placement puts it at 1024, where final linking
// needs another 3072 bytes of padding and overflows the reservation.
// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -O2 -c %S/data_alignment_kernel.cc -o %t.d/data_alignment_kernel.o
// RUN: cd %t.d && %aiecc --get-core-elfs --get-input-with-addresses --output-dir=%t.out %s
// RUN: FileCheck %s --input-file=%t.out/input_with_addresses.mlir
// RUN: sed 's/} {stack_size = 1024 : i32}/} {stack_size = 1024 : i32, data_size = 64 : i32}/' %s > %t.d/explicit.mlir
// RUN: cd %t.d && %aiecc --get-core-elfs --get-input-with-addresses --output-dir=%t.explicit.out explicit.mlir
// RUN: FileCheck %s --input-file=%t.explicit.out/input_with_addresses.mlir
// RUN: clang++ --target=aie2p-none-unknown-elf -O2 -DPINNED -c %S/data_alignment_kernel.cc -o %t.d/data_alignment_kernel.o
// RUN: cd %t.d && %aiecc --get-core-elfs --get-input-with-addresses --output-dir=%t.pinned.out %s
// RUN: FileCheck %s --check-prefix=PINNED --input-file=%t.pinned.out/input_with_addresses.mlir
//
// CHECK: aie.buffer({{.*}}) {address = 4096 : i32, core_data
// CHECK-SAME: memref<64xi8>
// CHECK: measured_data_alignment = 4096 : i32
// CHECK-SAME: measured_data_size = 64 : i32
// A bank-pinned section whose size is already an alignment multiple still
// overflows if the reservation starts at a merely vector-aligned address.
// PINNED: aie.buffer({{.*}}) {address = 4096 : i32, bank_reserved
// PINNED-SAME: memref<4096xi8>
// PINNED: measured_bank_alignments = array<i32: 4096, 1, 1, 1>
// PINNED-SAME: measured_bank_sizes = array<i32: 4096, 0, 0, 0>
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %out = aie.buffer(%tile) {sym_name = "out"} : memref<16xi32>
    func.func private @touch(memref<16xi32>) attributes {link_with = "data_alignment_kernel.o"}
    %core = aie.core(%tile) {
      func.call @touch(%out) : (memref<16xi32>) -> ()
      aie.end
    } {stack_size = 1024 : i32}
  }
}
