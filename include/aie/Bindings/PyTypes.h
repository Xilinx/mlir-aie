//===- PyTypes.h ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
#ifndef AIE_BINDINGS_PYTYPES_H
#define AIE_BINDINGS_PYTYPES_H

#include "aie-c/TargetModel.h"

class PyAieTargetModel {
public:
  PyAieTargetModel(AieTargetModel model) : model(model) {}
  operator AieTargetModel() const { return model; }
  AieTargetModel get() const { return model; }

private:
  AieTargetModel model;
};

#endif // AIE_BINDINGS_PYTYPES_H