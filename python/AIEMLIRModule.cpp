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

  // AIE types bindings
  mlir_type_subclass(m, "ObjectFifoType", aieTypeIsObjectFifoType)
      .def_classmethod(
          "get",
          [](py::object cls, MlirType type) {
            return cls(aieObjectFifoTypeGet(type));
          },
          "Get an instance of ObjectFifoType with given element type.",
          py::arg("self"), py::arg("type") = py::none());

  mlir_type_subclass(m, "ObjectFifoSubviewType", aieTypeIsObjectFifoSubviewType)
      .def_classmethod(
          "get",
          [](py::object cls, MlirType type) {
            return cls(aieObjectFifoSubviewTypeGet(type));
          },
          "Get an instance of ObjectFifoSubviewType with given element type.",
          py::arg("self"), py::arg("type") = py::none());

  m.attr("__version__") = "dev";
}
