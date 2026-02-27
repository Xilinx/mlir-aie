//===- aie_objectfifo.h - ObjectFIFO C API for AIE2 -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Clean C API for ObjectFIFO acquire/release operations in AIE2 kernels.
// Hides the dual-lock (producer + consumer) semantics of AIE2 semaphore locks.
//
// On AIE2, each ObjectFIFO element has two locks:
//   - Producer lock (acq_lock for producer, rel_lock for consumer)
//   - Consumer lock (rel_lock for producer, acq_lock for consumer)
//
// The MLIR `aie.objectfifo.lock` op resolves the correct lock IDs for each
// port and passes them as function arguments. This header provides a struct
// and inline functions to use them.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_OBJECTFIFO_H
#define AIE_OBJECTFIFO_H

#include <stdint.h>

// Lock intrinsics (acquire_equal, release) are provided by the compiler:
//   - Peano: auto-included via aiev2intrin.h / aie2pintrin.h
//   - Chess: compiler built-ins
#ifndef __AIENGINE__
#error                                                                         \
    "aie_objectfifo.h must be compiled for an AIE target (__AIENGINE__ not defined)"
#endif

// ObjectFIFO port handle for C kernels.
// Encapsulates everything needed to acquire/release from a given port.
// The MLIR compiler fills in the correct lock IDs and values based on
// ObjectFIFO configuration and port direction.
typedef struct {
  int32_t acq_lock; // Lock ID for acquire operation
  int32_t rel_lock; // Lock ID for release operation
  int32_t
      acq_value; // Value for acquire_equal(): use -1 for AcquireGreaterEqual
  int32_t rel_value; // Value for release() call (typically 1)
} objectfifo_port_t;

// Acquire an ObjectFIFO port (blocks until available).
// For producers: waits until the buffer is free to write.
// For consumers: waits until data is ready to read.
static inline void objectfifo_acquire(const objectfifo_port_t *port) {
  acquire_equal(port->acq_lock, port->acq_value);
}

// Release an ObjectFIFO port.
// For producers: signals that data has been written.
// For consumers: signals that the buffer is free.
static inline void objectfifo_release(const objectfifo_port_t *port) {
  release(port->rel_lock, port->rel_value);
}

#endif // AIE_OBJECTFIFO_H
