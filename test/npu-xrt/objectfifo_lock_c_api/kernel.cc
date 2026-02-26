//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Demonstrates ObjectFIFO C API usage with the aie_objectfifo.h header.
// Lock IDs and values are passed from MLIR and used via the struct-based API.

#include <stdint.h>

#include "aie_objectfifo.h"

extern "C" {

void scale_kernel(int32_t *in, int32_t *out, uint32_t in_acq_lock,
                  uint32_t in_rel_lock, uint32_t out_acq_lock,
                  uint32_t out_rel_lock) {
  // Construct port handles from lock IDs
  // For AIE2 semaphore locks, acq_value and rel_value are both 1
  objectfifo_port_t port_in = {(int32_t)in_acq_lock, (int32_t)in_rel_lock, 1,
                               1};
  objectfifo_port_t port_out = {(int32_t)out_acq_lock, (int32_t)out_rel_lock, 1,
                                1};

  // Acquire both ports
  objectfifo_acquire(&port_in);
  objectfifo_acquire(&port_out);

  // Scale each element by 3
  for (int i = 0; i < 1024; i++) {
    out[i] = in[i] * 3;
  }

  // Release both ports
  objectfifo_release(&port_in);
  objectfifo_release(&port_out);
}

} // extern "C"
