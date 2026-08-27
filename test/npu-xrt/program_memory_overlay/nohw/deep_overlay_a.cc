//===- deep_overlay_a.cc --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Same oversized-frame overlay as ./deep_overlay.cc, entry-named for the
// `aie.iron.overlay` API's stack_budget_iron_api.lit instead of pm.py's
// hand-picked "overlay_entry": `ProgramMemorySlot`'s call site is always
// `overlay_entry_<slot name>`, and that test's slot is named "a".

#include <cstdint>

extern "C" void overlay_entry_a(int32_t *in, int32_t *out) {
  volatile int32_t scratch[512];
  for (int i = 0; i < 512; i++)
    scratch[i] = in[i & 63] + i;
  int32_t acc = 0;
  for (int i = 0; i < 512; i++)
    acc += scratch[i];
  *out = acc;
}
