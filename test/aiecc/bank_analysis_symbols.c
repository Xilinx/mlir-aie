// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

int bank0[4] __attribute__((section(".aie.bank0")));
int bank1[4] __attribute__((section(".aie.bank1")));
int bank2[4] __attribute__((section(".aie.bank2")));
int bank3[4] __attribute__((section(".aie.bank3")));
int bank3_suffix[4] __attribute__((section(".aie.bank3.extra")));
int not_bank[4] __attribute__((section(".aie.bank30")));
int bank_ab[4] __attribute__((section(".data.DM_bankAB")));
extern int undefined;
int *reference = &undefined;
