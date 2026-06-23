//===- PythonPass.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_PYTHONPASS_H
#define AIE_PYTHONPASS_H

#include "mlir/Bindings/Python/PybindAdaptors.h"

#define MLIR_PYTHON_CAPSULE_PASS MAKE_MLIR_PYTHON_QUALNAME("ir.Pass._CAPIPtr")

PyObject *mlirPassToPythonCapsule(MlirPass pass);

MlirPass mlirPythonCapsuleToPass(PyObject *capsule);

#endif // AIE_PYTHONPASS_H
