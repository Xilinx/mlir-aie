// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifdef PINNED
int table_a[16] __attribute__((section(".aie.bank0")));
int table_b[16] __attribute__((section(".aie.bank1")));
#endif
#ifndef BSS_ONLY
int initialized = 1;
#endif
int ordinary[16] __attribute__((aligned(64)));
