//===- aie_objectfifo.h - ObjectFIFO C API for AIE2P ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// C API for ObjectFIFO operations in AIE2P kernels.
// Provides a self-contained struct that bundles locks, buffers, and depth,
// hiding the dual-lock (producer + consumer) semantics of AIE2P semaphore
// locks.
//
// On AIE2P, each ObjectFIFO element has two locks:
//   - Producer lock (acq_lock for producer, rel_lock for consumer)
//   - Consumer lock (rel_lock for producer, acq_lock for consumer)
//
// The MLIR `aie.objectfifo.lock` and `aie.objectfifo.buffer` ops resolve
// the correct lock IDs and buffer references for each port, passing them
// as function arguments. This header provides a struct and inline functions
// to use them.
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

// Maximum supported ObjectFIFO depth (number of buffers).
#define OBJECTFIFO_MAX_DEPTH 4

// ObjectFIFO handle for C kernels.
// Encapsulates everything needed to acquire/release and access buffers
// for a given ObjectFIFO port (producer or consumer side).
//
// The MLIR compiler fills in the correct lock IDs, buffer pointers, and
// depth based on ObjectFIFO configuration and port direction.
typedef struct {
  int32_t acq_lock;  // Lock ID for acquire operation
  int32_t rel_lock;  // Lock ID for release operation
  int32_t acq_value; // Value for acquire_equal(): -1 for AcquireGreaterEqual
  int32_t rel_value; // Value for release() call (typically 1)
  int32_t depth;     // Number of buffers (ObjectFIFO depth)
  void *buffers[OBJECTFIFO_MAX_DEPTH]; // Buffer pointers
} objectfifo_t;

// Acquire an ObjectFIFO (blocks until available).
// For producers: waits until a buffer is free to write.
// For consumers: waits until data is ready to read.
static inline void objectfifo_acquire(const objectfifo_t *of) {
  acquire_equal(of->acq_lock, of->acq_value);
}

// Release an ObjectFIFO.
// For producers: signals that data has been written.
// For consumers: signals that the buffer is free.
static inline void objectfifo_release(const objectfifo_t *of) {
  release(of->rel_lock, of->rel_value);
}

// Get the buffer pointer for the current iteration.
// Handles buffer rotation using modular indexing: buffers[iter % depth].
// The caller should cast the returned void* to the appropriate type.
static inline void *objectfifo_get_buffer(const objectfifo_t *of,
                                          int32_t iter) {
  return of->buffers[iter % of->depth];
}

#endif // AIE_OBJECTFIFO_H
