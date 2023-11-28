//===- AIEMLIRModule.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie-c/Dialects.h"
#include "aie-c/Registration.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_aie, m) {

  ::aieRegisterAllPasses();

  m.doc() = R"pbdoc(
    AIE MLIR Python bindings
    --------------------------

    .. currentmodule:: _aie

    .. autosummary::
        :toctree: _generate
  )pbdoc";

  m.def(
      "register_dialect",
      [](MlirDialectRegistry registry) {
        MlirDialectHandle aie_handle = mlirGetDialectHandle__aie__();
        MlirDialectHandle aiex_handle = mlirGetDialectHandle__aiex__();
        MlirDialectHandle aievec_handle = mlirGetDialectHandle__aievec__();
        mlirDialectHandleInsertDialect(aie_handle, registry);
        mlirDialectHandleInsertDialect(aiex_handle, registry);
        mlirDialectHandleInsertDialect(aievec_handle, registry);
      },
      py::arg("registry"));

  m.attr("__version__") = "dev";
}
