//===- PythonRouter.cpp ----------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PybindTypes.h"
#include "PythonPass.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
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

class PythonRouter : public Router {
public:
  explicit PythonRouter(py::object router) : router(std::move(router)) {}

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

  bool addFixedConnection(ConnectOp connectOp) override {
    auto sb = connectOp->getParentOfType<SwitchboxOp>();
    if (sb.getTileOp().isShimNOCTile())
      return true;

    return router.attr("add_fixed_connection")(wrap(connectOp)).cast<bool>();
  }

  std::optional<std::map<PathEndPoint, SwitchSettings>>
  findPaths(const int maxIterations) override {
    return router.attr("find_paths")()
        .cast<std::map<PathEndPoint, SwitchSettings>>();
  }
  Switchbox *getSwitchbox(TileID coords) override { return nullptr; }

  py::object router;
};

struct PythonRouterPass : AIEPathfinderPass {
  using AIEPathfinderPass::AIEPathfinderPass;

  StringRef getArgument() const final { return "aie-create-python-router"; }
};

std::unique_ptr<OperationPass<DeviceOp>>
createPythonRouterPass(py::object router) {
  return std::make_unique<PythonRouterPass>(
      DynamicTileAnalysis(std::make_shared<PythonRouter>(std::move(router))));
}

MlirPass mlircreatePythonRouterPass(py::object router) {
  return wrap(createPythonRouterPass(std::move(router)).release());
}

void registerPythonRouterPassWithRouter(const py::object &router) {
  registerPass([router] { return createPythonRouterPass(router); });
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
