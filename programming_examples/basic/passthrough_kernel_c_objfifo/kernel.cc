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
// aie.objectfifo.lock and aie.objectfifo.buffer. The kernel constructs
// objectfifo_t structs and uses objectfifo_get_buffer() for automatic
// buffer rotation.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "aie_objectfifo.h"

extern "C" {

void passThroughLine(int32_t *in_buf0, int32_t *in_buf1, int32_t *out_buf0,
                     int32_t *out_buf1, int64_t in_acq_lock,
                     int64_t in_rel_lock, int64_t out_acq_lock,
                     int64_t out_rel_lock) {
  objectfifo_t of_in = {(int32_t)in_acq_lock, (int32_t)in_rel_lock,
                        -1,          1,
                        2,           {in_buf0, in_buf1}};
  objectfifo_t of_out = {(int32_t)out_acq_lock, (int32_t)out_rel_lock,
                         -1,           1,
                         2,            {out_buf0, out_buf1}};

  for (int iter = 0; iter < 8; iter++) {
    objectfifo_acquire(&of_in);
    objectfifo_acquire(&of_out);

    int32_t *in = (int32_t *)objectfifo_get_buffer(&of_in, iter);
    int32_t *out = (int32_t *)objectfifo_get_buffer(&of_out, iter);

    for (int i = 0; i < 1024; i++) {
      out[i] = in[i];
    }

    objectfifo_release(&of_in);
    objectfifo_release(&of_out);
  }
}

} // extern "C"
