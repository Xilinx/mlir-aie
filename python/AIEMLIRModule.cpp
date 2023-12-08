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

  aieRegisterAllPasses();

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
        MlirDialectHandle aieHandle = mlirGetDialectHandle__aie__();
        MlirDialectHandle aiexHandle = mlirGetDialectHandle__aiex__();
        MlirDialectHandle aievecHandle = mlirGetDialectHandle__aievec__();
        mlirDialectHandleInsertDialect(aieHandle, registry);
        mlirDialectHandleInsertDialect(aiexHandle, registry);
        mlirDialectHandleInsertDialect(aievecHandle, registry);
      },
      py::arg("registry"));

  // AIE types bindings
  mlir_type_subclass(m, "ObjectFifoType", aieTypeIsObjectFifoType)
      .def_classmethod(
          "get",
          [](const py::object &cls, const MlirType type) {
            return cls(aieObjectFifoTypeGet(type));
          },
          "Get an instance of ObjectFifoType with given element type.",
          py::arg("self"), py::arg("type") = py::none());

  mlir_type_subclass(m, "ObjectFifoSubviewType", aieTypeIsObjectFifoSubviewType)
      .def_classmethod(
          "get",
          [](const py::object &cls, const MlirType type) {
            return cls(aieObjectFifoSubviewTypeGet(type));
          },
          "Get an instance of ObjectFifoSubviewType with given element type.",
          py::arg("self"), py::arg("type") = py::none());

  m.attr("__version__") = "dev";
}
