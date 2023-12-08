//===- PythonPass.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PythonPass.h"

PyObject *mlirPassToPythonCapsule(MlirPass pass) {
  return PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(pass),
                       MLIR_PYTHON_CAPSULE_PASS, nullptr);
}

MlirPass mlirPythonCapsuleToPass(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_PASS);
  MlirPass pass = {ptr};
  return pass;
}
