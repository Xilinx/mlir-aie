//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

// Idle between two events, as the trace unit keeps up with events this far
// apart.
#define EVENT_GAP_CYCLES 64

extern "C" {

void emit_events_slow(int32_t n) {
  for (int32_t i = 0; i < n; i++) {
    event0();
    volatile int32_t spin = 0;
    for (int32_t j = 0; j < EVENT_GAP_CYCLES; j++)
      spin = spin + 1;
  }
}

// Raise two events back to back, faster than the trace unit can drain them.
void emit_events_fast(int32_t n) {
  for (int32_t i = 0; i < n; i++) {
    event0();
    event1();
  }
}

} // extern "C"
