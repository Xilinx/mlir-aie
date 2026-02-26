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

void passThroughLine(int32_t *in_buf0, int32_t *in_buf1, int32_t *out_buf0,
                     int32_t *out_buf1, int64_t in_acq_lock,
                     int64_t in_rel_lock, int64_t out_acq_lock,
                     int64_t out_rel_lock) {
  objectfifo_port_t port_in = {(int32_t)in_acq_lock, (int32_t)in_rel_lock, -1,
                               1};
  objectfifo_port_t port_out = {(int32_t)out_acq_lock, (int32_t)out_rel_lock,
                                -1, 1};

  int32_t *in_bufs[2] = {in_buf0, in_buf1};
  int32_t *out_bufs[2] = {out_buf0, out_buf1};

  for (int iter = 0; iter < 8; iter++) {
    objectfifo_acquire(&port_in);
    objectfifo_acquire(&port_out);

    int32_t *in = in_bufs[iter % 2];
    int32_t *out = out_bufs[iter % 2];

    for (int i = 0; i < 1024; i++) {
      out[i] = in[i];
    }

    objectfifo_release(&port_in);
    objectfifo_release(&port_out);
  }
}

} // extern "C"
