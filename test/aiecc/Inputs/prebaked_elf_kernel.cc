//===- prebaked_elf_kernel.cc -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A bank-pinned table and ordinary statics, so the prebaked ELF occupies both
// an .aie.bank<N> region and the ordinary data region.
__attribute__((section(".aie.bank2"))) int pinned_tbl[256] = {1};
int statics[512] = {2};

extern "C" void prebaked_kernel(char *b) {
  for (int i = 0; i < 256; i++)
    b[i] = (char)(pinned_tbl[i] + statics[i]);
}
