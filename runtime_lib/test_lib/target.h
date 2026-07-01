//===- target.h ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_TARGET_H
#define AIE_TARGET_H

#include <list>
#include <vector>
#include <xaiengine.h>

#ifdef HSA_RUNTIME
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#endif

struct ext_mem_model_t {
  void *virtualAddr;
  uint64_t physicalAddr;
  size_t size;
  int fd;               // The file descriptor used during allocation
  XAie_MemInst MemInst; // LibXAIE handle if necessary.  This should go away.
};

struct aie_libxaie_ctx_t {
  XAie_Config *XAieConfig;
  XAie_DevInst *XAieDevInst;
  // Some device memory allocators need this to keep track of VA->PA mappings
  std::list<ext_mem_model_t> allocations;
#ifdef HSA_RUNTIME
  hsa_queue_t *cmd_queue;
  std::vector<hsa_agent_t> agents;
  hsa_amd_memory_pool_t global_mem_pool;
#endif
};

#endif
