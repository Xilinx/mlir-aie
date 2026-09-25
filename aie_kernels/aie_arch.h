// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// What each AIE architecture offers the kernel library, one row per
// __AIE_ARCH__, so kernels test a capability rather than the arch number.
// ARCH_TRAITS in python/iron/kernels/_common.py mirrors it, and
// test_arch_traits.py compiles the two against each other.
//
// AIE_BF16_LANES    bf16 lanes in one vector multiply.
// AIE_HAS_*         an instruction the sources use when present.
// AIE_LUT_16B_RUN   uint16 entries per bank run in an aie::lut table.
// AIE_TUNED_*       selects a kernel's code written for that architecture.

#ifndef AIE_KERNELS_AIE_ARCH_H
#define AIE_KERNELS_AIE_ARCH_H

#if __AIE_ARCH__ == 20
#define AIE_ARCH_AIE2 1
#define AIE_ARCH_AIE2P 0
#define AIE_BF16_LANES 16
#define AIE_HAS_NATIVE_TANH 0
#define AIE_HAS_NATIVE_EXP2 0
#define AIE_HAS_BFP16 0
#define AIE_LUT_16B_RUN 8
#elif __AIE_ARCH__ == 21
#define AIE_ARCH_AIE2 0
#define AIE_ARCH_AIE2P 1
#define AIE_BF16_LANES 32
#define AIE_HAS_NATIVE_TANH 1
#define AIE_HAS_NATIVE_EXP2 1
#define AIE_HAS_BFP16 1
#define AIE_LUT_16B_RUN 16
#else
#error "aie_kernels: no row for this __AIE_ARCH__ in aie_arch.h"
#endif

// A kernel's untuned branch tests only the capabilities above, so a new row
// builds every kernel before anything is tuned for it. AIE_KERNELS_PORTABLE
// selects that branch on these architectures too, to keep it building and to
// check its results on hardware that exists.
#ifdef AIE_KERNELS_PORTABLE
#define AIE_TUNED_AIE2 0
#define AIE_TUNED_AIE2P 0
#else
#define AIE_TUNED_AIE2 AIE_ARCH_AIE2
#define AIE_TUNED_AIE2P AIE_ARCH_AIE2P
#endif

// __restrict on AIE2 only: the AIE2 pipeliner needs it to overlap a streaming
// loop's iterations.
#if AIE_TUNED_AIE2
#define AIE2_RESTRICT __restrict
#else
#define AIE2_RESTRICT
#endif

#endif
