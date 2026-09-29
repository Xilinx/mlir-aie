// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exercise the emitted script with a host object, without AIE intrinsics.
// REQUIRES: system-linux, peano
// RUN: aie-translate --tilecol=0 --tilerow=2 --aie-generate-ldscript %s > %t.ld
// RUN: %host_clang -DPINNED -c %S/bank_default_data.c -o %t.o
// RUN: ld.lld -T %t.ld %t.o -o %t.elf
// RUN: llvm-nm -n %t.elf | FileCheck %s --check-prefix=PINNED
// RUN: %host_clang -DPINNED -DBSS_ONLY -c %S/bank_default_data.c -o %t.o
// RUN: ld.lld -T %t.ld %t.o -o %t.elf
// RUN: llvm-nm -n %t.elf | FileCheck %s --check-prefix=BSS
// RUN: %host_clang -c %S/bank_default_data.c -o %t.o
// RUN: ld.lld -T %t.ld %t.o -o %t.elf
// RUN: llvm-nm -n %t.elf | FileCheck %s --check-prefix=EMPTY
// RUN: %host_clang -DPINNED -fdata-sections -c %S/bank_default_data.c -o %t.o
// RUN: ld.lld --gc-sections -u ordinary -T %t.ld %t.o -o %t.elf
// RUN: llvm-nm -n %t.elf | FileCheck %s --check-prefix=GC

// PINNED: 0000000000070400 D table_a
// PINNED: 0000000000074000 D table_b
// PINNED: 0000000000074040 D initialized
// PINNED: 0000000000074080 B ordinary
// BSS: 0000000000074040 B ordinary
// EMPTY: 0000000000070400 D initialized
// EMPTY: 0000000000070440 B ordinary
// GC: 0000000000070400 B ordinary

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %core = aie.core(%tile) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}
