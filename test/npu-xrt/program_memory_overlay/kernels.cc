//===- kernels.cc ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Three real aie_kernels behind the one entry signature the resident calls.
//
// The rest of the suite runs generated workloads, which are branch-free padding
// plus a loop and so exercise very little of what a kernel does. These are
// ordinary library kernels: they branch, they call, they use the vector unit,
// and their .text is whatever it is. If the mechanism only worked for simple
// code, this is what would show it.
//
// One source, compiled once per OVL_ID, each linked on its own at the slot.

#include <cstdint>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef OVL_ID
#error "OVL_ID must be defined; each overlay is a separate compilation"
#endif

// N_ELEMS is baked into silu_bf16 and gelu_bf16 in aie_kernels/aie2p, so the
// design's tile size has to agree with it rather than the other way round.
static constexpr int32_t kElems = 1024;

extern "C" void silu_bf16(bfloat16 *restrict in, bfloat16 *restrict out);
extern "C" void gelu_bf16(bfloat16 *restrict in, bfloat16 *restrict out);
extern "C" void softmax_bf16(bfloat16 *restrict in, bfloat16 *restrict out,
                             int32_t n);

extern "C" void overlay_entry(bfloat16 *in, bfloat16 *out) {
#if OVL_ID == 0
  silu_bf16(in, out);
#elif OVL_ID == 1
  gelu_bf16(in, out);
#elif OVL_ID == 2
  softmax_bf16(in, out, kElems);
#else
#error "unknown OVL_ID"
#endif
}
