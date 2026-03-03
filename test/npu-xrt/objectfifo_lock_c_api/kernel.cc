//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Demonstrates ObjectFIFO C API usage with depth-2 ping-pong buffering.
// The kernel receives buffer pointers and lock IDs, constructs objectfifo_t
// structs, and manages acquire/release and buffer rotation using
// aie_objectfifo.h.

#include <stdint.h>

#include "aie_objectfifo.h"

extern "C" {

void scale_kernel(int32_t *in_buf0, int32_t *in_buf1, int32_t *out_buf0,
                  int32_t *out_buf1, int64_t in_acq_lock, int64_t in_rel_lock,
                  int64_t out_acq_lock, int64_t out_rel_lock) {
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
      out[i] = in[i] * 3;
    }

    objectfifo_release(&of_in);
    objectfifo_release(&of_out);
  }
}

} // extern "C"
