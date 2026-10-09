// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifdef SECOND
#ifdef UNPINNED
static int table = 2;
#else
static int table __attribute__((section(".aie.bank1"))) = 2;
#endif
int *get_second(void) { return &table; }
int *unresolved_gather(void) { return get_second(); }
int *same_table(void) { return get_second(); }
#else
int bank_analysis_anchor __attribute__((section(".aie.bank0"))) = 1;
static int table __attribute__((section(".aie.bank0"))) = 2;
int *get_first(void) { return &table; }
int *gather_with_unrelated_blend(void) { return get_first(); }
#endif
