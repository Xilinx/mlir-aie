//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdfloat>
#include <stdint.h>

#define DATATYPES_USING_DEFINED
using A_DATATYPE = int16_t; // std::bfloat16_t;
using B_DATATYPE = int16_t; // std::bfloat16_t;
using C_DATATYPE = int32_t; // float;
using ACC_DATATYPE = int32_t;

#include "../test.cpp"
