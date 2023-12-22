//===- PythonPass.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PythonPass.h"
#include "PybindTypes.h"
#include "RouterPass.h"

using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::adaptors;
using namespace xilinx::AIE;

PyObject *mlirPassToPythonCapsule(MlirPass pass) {
  return PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(pass),
                       MLIR_PYTHON_CAPSULE_PASS, nullptr);
}

MlirPass mlirPythonCapsuleToPass(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_PASS);
  MlirPass pass = {ptr};
  return pass;
}

PYBIND11_MODULE(_aie_python_passes, m) {

  bindTypes(m);

  m.def("create_python_router_pass", [](const py::object &router) {
    MlirPass pass = mlircreatePythonRouterPass(router);
    auto capsule =
        py::reinterpret_steal<py::object>(mlirPassToPythonCapsule(pass));
    return capsule;
  });

  m.def("pass_manager_add_owned_pass",
        [](MlirPassManager passManager, py::handle passHandle) {
          py::object passCapsule = mlirApiObjectToCapsule(passHandle);
          MlirPass pass = mlirPythonCapsuleToPass(passCapsule.ptr());
          mlirPassManagerAddOwnedPass(passManager, pass);
        });

  m.def("get_connecting_bundle", &getConnectingBundle);
}
