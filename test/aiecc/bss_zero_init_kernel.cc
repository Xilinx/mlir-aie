//===- bss_zero_init_kernel.cc ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for bss_zero_init.mlir. Deliberately has BOTH initialised data
// (.data, PROGBITS) and a zero-initialised static (.bss, NOBITS) so the linker
// emits a single PT_LOAD with 0 < p_filesz < p_memsz.

#include <stdint.h>

// forces a PROGBITS .data in the same segment as .bss
__attribute__((used, retain))
volatile uint8_t initialised[4096] = {1};

// C++ guarantees this reads as zero before first use ([basic.start.static])
volatile uint8_t zero_state[512];

extern "C" void bss_probe(uint8_t *out) {
  for (int i = 0; i < 512; i++)
    out[i] = zero_state[i];
  out[0] = (uint8_t)(out[0] + (initialised[0] - initialised[0]));
}
