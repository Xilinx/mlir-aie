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
  explicit PythonPathFinder(py::object router) : router(std::move(router)) {}

  void initialize(const int maxCol, const int maxRow,
                  const AIETargetModel &targetModel) override {
    // Here we're copying a pointer to targetModel, which is a static somewhere.
    router.attr("initialize")(maxCol, maxRow, &targetModel);
  }

  void addFlow(TileID srcCoords, const Port srcPort, TileID dstCoords,
               const Port dstPort) override {
    router.attr("add_flow")(
        PathEndPoint{{srcCoords.col, srcCoords.row}, srcPort},
        PathEndPoint{{dstCoords.col, dstCoords.row}, dstPort});
  }

  bool addFixedConnection(TileID coords, Port port) override {
    return router.attr("add_fixed_connection")(coords, port).cast<bool>();
  }

  bool isLegal() override { return router.attr("is_legal")().cast<bool>(); }

  std::map<PathEndPoint, SwitchSettings>
  findPaths(const int maxIterations) override {
    return router.attr("find_paths")()
        .cast<std::map<PathEndPoint, SwitchSettings>>();
  }

  py::object router;
};

struct PathfinderFlowsWithPython : AIEPathfinderPass {
  using AIEPathfinderPass::AIEPathfinderPass;

  StringRef getArgument() const final {
    return "aie-create-pathfinder-flows-with-python";
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
createPathfinderFlowsWithPythonPassWithFunc(py::object router) {
  return std::make_unique<PathfinderFlowsWithPython>(DynamicTileAnalysis(
      std::make_shared<PythonPathFinder>(std::move(router))));
}

void registerPathfinderFlowsWithPythonPassWithFunc(const py::object &router) {
  registerPass(
      [router] { return createPathfinderFlowsWithPythonPassWithFunc(router); });
}

PYBIND11_MODULE(_aie_python_passes, m) {

  bindTypes(m);

  m.def("register_pathfinder_flows_with_python", [](const py::object &router) {
    registerPathfinderFlowsWithPythonPassWithFunc(router);
  });

  m.def("get_connecting_bundle", &getConnectingBundle);
}
