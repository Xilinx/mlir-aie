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

class PythonPathFinder : public Pathfinder {
public:
  PythonPathFinder(py::function func) : func(std::move(func)) {}

  void initialize(int maxCol, int maxRow,
                  const AIETargetModel &targetModel) override {
    for (int col = 0; col <= maxCol; col++) {
      for (int row = 0; row <= maxRow; row++) {
        (void)grid.insert({{col, row}, Switchbox{col, row}});
        Switchbox &thisNode = grid.at({col, row});
        if (row > 0) { // if not in row 0 add channel to North/South
          Switchbox &southernNeighbor = grid.at({col, row - 1});
          if (uint32_t maxCapacity =
                  targetModel.getNumSourceSwitchboxConnections(
                      col, row, WireBundle::South)) {
            edges.emplace_back(southernNeighbor, thisNode, WireBundle::North,
                               maxCapacity);
          }
          if (uint32_t maxCapacity = targetModel.getNumDestSwitchboxConnections(
                  col, row, WireBundle::South)) {
            edges.emplace_back(thisNode, southernNeighbor, WireBundle::South,
                               maxCapacity);
          }
        }

        if (col > 0) { // if not in col 0 add channel to East/West
          Switchbox &westernNeighbor = grid.at({col - 1, row});
          if (uint32_t maxCapacity =
                  targetModel.getNumSourceSwitchboxConnections(
                      col, row, WireBundle::West)) {
            edges.emplace_back(westernNeighbor, thisNode, WireBundle::East,
                               maxCapacity);
          }
          if (uint32_t maxCapacity = targetModel.getNumDestSwitchboxConnections(
                  col, row, WireBundle::West)) {
            edges.emplace_back(thisNode, westernNeighbor, WireBundle::West,
                               maxCapacity);
          }
        }
      }
    }
  }

  void addFlow(TileID srcCoords, Port srcPort, TileID dstCoords,
               Port dstPort) override {
    // check if a flow with this source already exists
    for (auto &flow : flows) {
      Switchbox *existingSrc = flow.src.sb;
      assert(existingSrc && "nullptr flow source");
      Port existingPort = flow.src.port;
      if (existingSrc->col == srcCoords.col &&
          existingSrc->row == srcCoords.row && existingPort == srcPort) {
        // find the vertex corresponding to the destination
        auto matchingSb =
            std::find_if(grid.cbegin(), grid.cend(),
                         [&](const std::pair<TileID, Switchbox> &item) {
                           return item.second == dstCoords;
                         });
        assert(matchingSb != grid.cend() && "didn't find flow dest");
        flow.dsts.push_back({&grid.at(matchingSb->first), dstPort});
        return;
      }
    }

    // If no existing flow was found with this source, create a new flow.
    auto matchingSrcSb =
        std::find_if(grid.cbegin(), grid.cend(),
                     [&](const std::pair<TileID, Switchbox> &item) {
                       return item.second == srcCoords;
                     });
    assert(matchingSrcSb != grid.cend() && "didn't find flow source");
    auto matchingDstSb =
        std::find_if(grid.cbegin(), grid.cend(),
                     [&](const std::pair<TileID, Switchbox> &item) {
                       return item.second == dstCoords;
                     });
    assert(matchingDstSb != grid.cend() && "didn't add flow destinations");
    flows.push_back(
        {PathEndPoint{&grid.at(matchingSrcSb->first), srcPort},
         std::vector<PathEndPoint>{{&grid.at(matchingDstSb->first), dstPort}}});
  }

  bool addFixedConnection(TileID coords, Port port) override {
    // find the correct Channel and indicate the fixed direction
    auto matchingCh =
        std::find_if(edges.begin(), edges.end(), [&](const Channel &ch) {
          return ch.src == coords && ch.bundle == port.bundle;
        });
    if (matchingCh == edges.end())
      return false;

    matchingCh->fixedCapacity.insert(port.channel);
    return true;
  }

  bool isLegal() override {}

  std::map<PathEndPoint, SwitchSettings>
  findPaths(const int maxIterations) override {
    func(grid, edges, flows);
  }

  Switchbox *getSwitchbox(TileID coords) override {
    auto matchingItem =
        std::find_if(grid.cbegin(), grid.cend(),
                     [&](const std::pair<TileID, Switchbox> &item) {
                       return item.second == coords;
                     });
    assert(matchingItem != grid.cend() && "couldn't find sb");
    return &grid.at(matchingItem->first);
  }

  std::map<TileID, Switchbox> grid;
  std::list<Channel> edges;
  std::vector<Flow> flows;
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

  bindTypes(m);

  m.def("register_pathfinder_flows_with_python", [](const py::function &func) {
    registerPathfinderFlowsWithPythonPassWithFunc(func);
  });
}
