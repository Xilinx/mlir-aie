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

  py::object router;
};

struct PathfinderFlowsWithPython : AIEPathfinderPass {
  using AIEPathfinderPass::AIEPathfinderPass;

  StringRef getArgument() const final {
    return "aie-create-pathfinder-flows-with-python";
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
createPathfinderFlowsWithPythonPassWithRouter(py::object router) {
  return std::make_unique<PathfinderFlowsWithPython>(DynamicTileAnalysis(
      std::make_shared<PythonPathFinder>(std::move(router))));
}

MlirPass mlirCreatePathfinderFlowsWithPythonPassWithRouter(py::object router) {
  return wrap(createPathfinderFlowsWithPythonPassWithRouter(std::move(router))
                  .release());
}

void registerPathfinderFlowsWithPythonPassWithRouter(const py::object &router) {
  registerPass([router] {
    return createPathfinderFlowsWithPythonPassWithRouter(router);
  });
}

#define MLIR_PYTHON_CAPSULE_PASS MAKE_MLIR_PYTHON_QUALNAME("ir.Pass._CAPIPtr")

static PyObject *mlirPassToPythonCapsule(MlirPass pass) {
  return PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(pass),
                       MLIR_PYTHON_CAPSULE_PASS, nullptr);
}

static inline MlirPass mlirPythonCapsuleToPass(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_PASS);
  MlirPass pass = {ptr};
  return pass;
}

PYBIND11_MODULE(_aie_python_passes, m) {

  bindTypes(m);

  m.def("create_pathfinder_flows_with_python_pass",
        [](const py::object &router) {
          MlirPass pass =
              mlirCreatePathfinderFlowsWithPythonPassWithRouter(router);
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
