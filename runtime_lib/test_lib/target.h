//===- target.h ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_TARGET_H
#define AIE_TARGET_H

#include <xaiengine.h>
#include <list>

struct ext_mem_model_t {
  void *virtualAddr;
  uint64_t physicalAddr;
  size_t size;
  int fd; // The file descriptor used during allocation
  XAie_MemInst MemInst; // LibXAIE handle if necessary.  This should go away.
};

struct aie_libxaie_ctx_t {
  XAie_Config AieConfigPtr;
  XAie_DevInst DevInst;
  // Some device memory allocators need this to keep track of VA->PA mappings
  std::list<ext_mem_model_t> allocations;
};

#endif