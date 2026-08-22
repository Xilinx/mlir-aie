//===- kernels.cc ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The overlay kernels: thin wrappers giving three real aie_kernels a common
// entry signature, since the resident calls one fixed address and which kernel
// is behind it is decided at run time by writing program memory.
//
// One source, compiled once per OVL_ID. Each is linked on its own at the slot
// address (overlay.py), so unlike the interchangeable pairs in
// ../pm_write_while_running these are ordinary library kernels: they branch,
// call, and are whatever size they are.

#include <cstdint>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef OVL_ID
#error "OVL_ID must be defined; each overlay is a separate compilation"
#endif

// From aie_kernels/aie2p. silu_bf16 and gelu_bf16 have the tile size baked in,
// which is why TILE_ELEMS below has to match them.
extern "C" void silu_bf16(bfloat16 *restrict in, bfloat16 *restrict out);
extern "C" void gelu_bf16(bfloat16 *restrict in, bfloat16 *restrict out);
extern "C" void softmax_bf16(bfloat16 *restrict in, bfloat16 *restrict out,
                             int32_t n);

extern "C" void overlay_entry(bfloat16 *in, bfloat16 *out, int32_t n) {
#if OVL_ID == 0
  silu_bf16(in, out);
#elif OVL_ID == 1
  gelu_bf16(in, out);
#elif OVL_ID == 2
  softmax_bf16(in, out, n);
#else
#error "unknown OVL_ID"
#endif
}
