//===- kernel.cc - Passthrough kernel using C ObjectFIFO API ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Demonstrates using aie_objectfifo.h for kernel-managed synchronization.
// Lock IDs and buffer references are passed in from MLIR via
// aie.objectfifo.lock and aie.objectfifo.buffer.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "aie_objectfifo.h"

extern "C" {

void passThroughLine(uint8_t *in, uint8_t *out, int32_t lineWidth,
                     int64_t in_acq, int64_t in_rel, int64_t out_acq,
                     int64_t out_rel) {
  // acq_value = -1 selects AcquireGreaterEqual semantics on AIE2/AIE2P:
  // wait until lock value >= 1, then decrement by 1.
  objectfifo_port_t port_in = {(int32_t)in_acq, (int32_t)in_rel, -1, 1};
  objectfifo_port_t port_out = {(int32_t)out_acq, (int32_t)out_rel, -1, 1};

  objectfifo_acquire(&port_in);
  objectfifo_acquire(&port_out);

  for (int i = 0; i < lineWidth; i++)
    out[i] = in[i];

  objectfifo_release(&port_in);
  objectfifo_release(&port_out);
}

} // extern "C"
