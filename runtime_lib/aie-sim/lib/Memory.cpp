//===- Memory.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"

#include <cstring>

using namespace aiesim;

bool Memory::read(uint32_t off, void *dst, uint32_t len) const {
  if (!inRange(off, len))
    return false;
  std::memcpy(dst, bytes.data() + off, len);
  return true;
}

bool Memory::write(uint32_t off, const void *src, uint32_t len) {
  if (!inRange(off, len))
    return false;
  std::memcpy(bytes.data() + off, src, len);
  return true;
}
