//===- PythonPassDemo.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PythonPass.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace py = pybind11;
using namespace mlir::python::adaptors;
using namespace xilinx::AIE;

struct PythonPassDemo
    : public PassWrapper<PythonPassDemo, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PythonPassDemo)

  PythonPassDemo(py::function func) : func(func) {}
  StringRef getArgument() const final { return "python-pass-demo"; }
  void runOnOperation() override {
    this->getOperation()->walk([this](Operation *op) { func(wrap(op)); });
  }

  py::function func;
};

std::unique_ptr<OperationPass<ModuleOp>>
createPythonPassDemoPassWithFunc(py::function func) {
  return std::make_unique<PythonPassDemo>(func);
}

void registerPythonPassDemoPassWithFunc(py::function func) {
  registerPass([func]() { return createPythonPassDemoPassWithFunc(func); });
}
