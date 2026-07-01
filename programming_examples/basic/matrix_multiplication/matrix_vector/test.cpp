//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#define DATATYPES_USING_DEFINED
using A_DATATYPE = int16_t;
using B_DATATYPE = int16_t;
using C_DATATYPE = int32_t; // float;
using ACC_DATATYPE = int32_t;

#include "../test.cpp"
