//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

// The trace parser merges events raised on consecutive cycles into one
// interval. Idle between two events so that each one appears on its own.
#define EVENT_GAP_CYCLES 16

static inline void gap() {
  volatile int32_t spin = 0;
  for (int32_t i = 0; i < EVENT_GAP_CYCLES; i++)
    spin = spin + 1;
}

extern "C" {

// The runtime parameter sets n, so the number of trace events a run produces
// identifies which runtime sequence dispatched it.
void emit_events_0(int32_t n) {
  for (int32_t i = 0; i < n; i++) {
    event0();
    gap();
  }
}

void emit_events_1(int32_t n) {
  for (int32_t i = 0; i < n; i++) {
    event1();
    gap();
  }
}

} // extern "C"
