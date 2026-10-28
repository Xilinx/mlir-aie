// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifdef PINNED
alignas(4096)
    __attribute__((section(".aie.bank0"))) volatile int aligned_data[1024] = {
        1};
#else
alignas(4096) volatile int aligned_data[16] = {1};
#endif

extern "C" void touch(int *out) { out[0] = aligned_data[0]; }
