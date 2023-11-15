//===- PathfinderFlowsWithPython.cpp ---------------------------------------*-
// C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::python::adaptors;
using namespace xilinx::AIE;

namespace py = pybind11;

struct PathfinderFlowsWithPython : public AIEPathfinderPass {

  PathfinderFlowsWithPython(py::function func) : func(func) {}
  StringRef getArgument() const final {
    return "aie-create-pathfinder-flows-with-python";
  }

  py::function func;
};

std::unique_ptr<OperationPass<DeviceOp>>
createPathfinderFlowsWithPythonPassWithFunc(py::function func) {
  return std::make_unique<PathfinderFlowsWithPython>(func);
}

void registerPathfinderFlowsWithPythonPassWithFunc(py::function func) {
  registerPass(
      [func]() { return createPathfinderFlowsWithPythonPassWithFunc(func); });
}

PYBIND11_MODULE(_aie_python_passes, m) {

  m.def("register_pathfinder_flows_with_python", [](py::function func) {
    registerPathfinderFlowsWithPythonPassWithFunc(std::move(func));
  });
}
