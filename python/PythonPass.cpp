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

class PythonPathFinder : public Pathfinder {
public:
  PythonPathFinder(py::function func) : func(std::move(func)) {}
  void initialize(int maxCol, int maxRow,
                  const AIETargetModel &targetModel) override {
    Pathfinder::initialize(maxCol, maxRow, targetModel);
    func(maxCol, maxRow, isLegal());
  }
  py::function func;
};

struct PathfinderFlowsWithPython : public AIEPathfinderPass {
  using AIEPathfinderPass::AIEPathfinderPass;

  StringRef getArgument() const final {
    return "aie-create-pathfinder-flows-with-python";
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
createPathfinderFlowsWithPythonPassWithFunc(py::function func) {
  return std::make_unique<PathfinderFlowsWithPython>(
      DynamicTileAnalysis(std::make_shared<PythonPathFinder>(std::move(func))));
}

void registerPathfinderFlowsWithPythonPassWithFunc(const py::function &func) {
  registerPass(
      [func]() { return createPathfinderFlowsWithPythonPassWithFunc(func); });
}

PYBIND11_MODULE(_aie_python_passes, m) {

  m.def("register_pathfinder_flows_with_python", [](const py::function &func) {
    registerPathfinderFlowsWithPythonPassWithFunc(func);
  });
}
