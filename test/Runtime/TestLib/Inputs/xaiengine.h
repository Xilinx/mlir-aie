// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Stand-in for aie-rt's xaiengine.h with just the types target.h and
// memory_allocator.h use, so memory_allocator_sim.test builds without aie-rt.

#ifndef MEMORY_ALLOCATOR_SIM_XAIENGINE_H
#define MEMORY_ALLOCATOR_SIM_XAIENGINE_H

#include <cstdint>

typedef uint64_t u64;
typedef struct {
  int unused;
} XAie_MemInst;
typedef struct XAie_Config XAie_Config;
typedef struct XAie_DevInst XAie_DevInst;

#endif
