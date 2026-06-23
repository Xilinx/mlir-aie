//===- PythonPass.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_ROUTERPASS_H
#define AIE_ROUTERPASS_H

#include "mlir/CAPI/Pass.h"

#include <pybind11/pybind11.h>

MlirPass mlircreatePythonRouterPass(pybind11::object router);

#endif // AIE_ROUTERPASS_H
