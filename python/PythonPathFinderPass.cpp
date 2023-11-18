//===- PathfinderFlowsWithPython.cpp ----------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PybindTypes.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::python::adaptors;
using namespace xilinx::AIE;

namespace py = pybind11;

//    ┌─────┐   ┌─────┐
//    │ 1,0 ├   ┤ 1,1 │
//    │     │ N │     │
//    └─────┘   └─────┘
//       W         E
//    ┌─────┐   ┌─────┐
//    │ 0,0 │ S │ 0,1 │
//    │     │   │     │
//    └─────┘   └─────┘

class PythonPathFinder : public Pathfinder {
public:
  explicit PythonPathFinder(py::function findPathsPythonFunc)
      : targetModel(nullptr), maxCol(0), maxRow(0),
        findPathsPythonFunc(std::move(findPathsPythonFunc)) {}

  void initialize(const int maxCol_, const int maxRow_,
                  const AIETargetModel &targetModel_) override {
    maxCol = maxCol_;
    maxRow = maxRow_;
    targetModel = &targetModel_;
  }

  void addFlow(TileID srcCoords, const Port srcPort, TileID dstCoords,
               const Port dstPort) override {
    flows.emplace_back(PathEndPoint{{srcCoords.col, srcCoords.row}, srcPort},
                       PathEndPoint{{dstCoords.col, dstCoords.row}, dstPort});
  }

  bool addFixedConnection(TileID coords, Port port) override {
    fixedConnections.emplace_back(coords, port);
    // TODO(max): not implemented.
    return false;
  }

  bool isLegal() override {
    // TODO(max): not implemented.
    return true;
  }

  std::map<PathEndPoint, SwitchSettings>
  findPaths(const int maxIterations) override {
    auto routingSolution = findPathsPythonFunc(maxCol, maxRow, targetModel,
                                               flows, fixedConnections)
                               .cast<std::map<PathEndPoint, SwitchSettings>>();
    return routingSolution;
  }

  const AIETargetModel *targetModel;
  int maxCol, maxRow;
  std::vector<std::tuple<PathEndPoint, PathEndPoint>> flows;
  std::vector<std::tuple<TileID, Port>> fixedConnections;
  py::function findPathsPythonFunc;
};

struct PathfinderFlowsWithPython : AIEPathfinderPass {
  using AIEPathfinderPass::AIEPathfinderPass;

  StringRef getArgument() const final {
    return "aie-create-pathfinder-flows-with-python";
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
createPathfinderFlowsWithPythonPassWithFunc(py::function findPathsPythonFunc) {
  return std::make_unique<PathfinderFlowsWithPython>(DynamicTileAnalysis(
      std::make_shared<PythonPathFinder>(std::move(findPathsPythonFunc))));
}

void registerPathfinderFlowsWithPythonPassWithFunc(
    const py::function &findPathsPythonFunc) {
  registerPass([findPathsPythonFunc] {
    return createPathfinderFlowsWithPythonPassWithFunc(findPathsPythonFunc);
  });
}

PYBIND11_MODULE(_aie_python_passes, m) {

  bindTypes(m);

  m.def("register_pathfinder_flows_with_python",
        [](const py::function &findPathsPythonFunc) {
          registerPathfinderFlowsWithPythonPassWithFunc(findPathsPythonFunc);
        });

  m.def("get_connecting_bundle", &getConnectingBundle);
}
