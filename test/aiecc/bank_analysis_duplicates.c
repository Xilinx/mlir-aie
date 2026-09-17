// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifdef SECOND
static int table __attribute__((section(".aie.bank1"))) = 2;
int *get_second(void) { return &table; }
#else
int bank_analysis_anchor __attribute__((section(".aie.bank0"))) = 1;
static int table __attribute__((section(".aie.bank0"))) = 2;
int *get_first(void) { return &table; }
#endif
