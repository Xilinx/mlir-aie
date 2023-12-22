//===- AIEFlowsToJSON.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

/*
 * Takes as input the mlir after AIECreateFlows and AIEFindFlows.
 * Converts the flows into a JSON file to be read by other tools.
 */

#include "aie/Targets/AIETargets.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include <queue>
#include <set>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx::AIE {

// returns coordinates in the direction indicated by bundle
TileID getNextCoords(int col, int row, WireBundle bundle) {
  switch (bundle) {
  case WireBundle::North:
    return {col, row + 1};
  case WireBundle::South:
    return {col, row - 1};
  case WireBundle::East:
    return {col + 1, row};
  case WireBundle::West:
    return {col - 1, row};
  default:
    return {col, row};
  }
}

mlir::LogicalResult AIEFlowsToJSON(ModuleOp module, raw_ostream &output) {
  output << "{\n";
  if (module.getOps<DeviceOp>().empty()) {
    module.emitOpError("expected AIE.device operation at toplevel");
  }
  DeviceOp targetOp = *(module.getOps<DeviceOp>().begin());

  // count flow sources and destinations
  std::map<TileID, int> sourceCounts;
  std::map<TileID, int> destinationCounts;
  for (FlowOp flowOp : targetOp.getOps<FlowOp>()) {
    TileOp source = cast<TileOp>(flowOp.getSource().getDefiningOp());
    TileOp dest = cast<TileOp>(flowOp.getDest().getDefiningOp());
    TileID srcID = {source.colIndex(), source.rowIndex()};
    TileID dstID = {dest.colIndex(), dest.rowIndex()};
    sourceCounts[srcID]++;
    destinationCounts[dstID]++;
  }

  // for each switchbox, write name, coordinates, and routing demand info
  for (SwitchboxOp switchboxOp : targetOp.getOps<SwitchboxOp>()) {
    int col = switchboxOp.colIndex();
    int row = switchboxOp.rowIndex();
    std::string switchString = "\"switchbox" + std::to_string(col) +
                               std::to_string(row) + "\": {\n" +
                               "\"col\": " + std::to_string(col) + ",\n" +
                               "\"row\": " + std::to_string(row) + ",\n";

    // write source and destination info
    switchString +=
        "\"source_count\": " + std::to_string(sourceCounts[{col, row}]) + ",\n";
    switchString += "\"destination_count\": " +
                    std::to_string(destinationCounts[{col, row}]) + ",\n";

    // write routing demand info
    uint32_t connectCounts[10];
    for (auto &connectCount : connectCounts)
      connectCount = 0;
    for (ConnectOp connectOp : switchboxOp.getOps<ConnectOp>())
      connectCounts[int(connectOp.getDestBundle())]++;

    switchString += "\"northbound\": " +
                    std::to_string(connectCounts[int(WireBundle::North)]) +
                    ",\n";
    switchString += "\"eastbound\": " +
                    std::to_string(connectCounts[int(WireBundle::East)]) +
                    ",\n";
    switchString += "\"southbound\": " +
                    std::to_string(connectCounts[int(WireBundle::South)]) +
                    ",\n";
    switchString += "\"westbound\": " +
                    std::to_string(connectCounts[int(WireBundle::West)]) + "\n";
    switchString += "},\n";
    output << switchString;
  }

  // for each flow, trace it through switchboxes and write the route to JSON
  int flowCount = 0;
  std::set<std::pair<TileOp, Port>> flowSources;
  for (FlowOp flowOp : targetOp.getOps<FlowOp>()) {
    // objects used to trace through the flow
    Port currPort = {flowOp.getSourceBundle(), flowOp.getSourceChannel()};
    SwitchboxOp currSwitchbox;

    TileOp source = cast<TileOp>(flowOp.getSource().getDefiningOp());
    // TileOp dest = cast<TileOp>(flowOp.dest().getDefiningOp());

    // track flow sources to avoid duplicate routes
    std::pair<TileOp, Port> flowSource = {source, currPort};
    if (flowSources.count(flowSource)) {
      continue;
    }
    flowSources.insert(flowSource);

    std::string routeString =
        "\"route" + std::to_string(flowCount++) + "\": [ ";

    // FIFO to handle fanouts
    std::queue<Port> nextPorts;
    std::queue<SwitchboxOp> nextSwitches;

    // find the starting switchbox
    for (SwitchboxOp switchboxOp : targetOp.getOps<SwitchboxOp>()) {
      if (switchboxOp.colIndex() == source.colIndex() &&
          switchboxOp.rowIndex() == source.rowIndex()) {
        currSwitchbox = switchboxOp;
        break;
      }
    }

    // if the flow starts in a shim, handle seperately
    for (ShimMuxOp shimMuxOp : targetOp.getOps<ShimMuxOp>()) {
      if (shimMuxOp.colIndex() == source.colIndex() &&
          shimMuxOp.rowIndex() == source.rowIndex()) {
        for (ConnectOp connectOp : shimMuxOp.getOps<ConnectOp>()) {
          if (connectOp.getSourceBundle() == currPort.bundle &&
              connectOp.getSourceChannel() == currPort.channel) {
            currPort.bundle = getConnectingBundle(connectOp.getDestBundle());
            currPort.channel = connectOp.getDestChannel();
            break;
          }
        }
        break;
      }
    }

    // trace through the flow and add switchbox coordinates to JSON
    bool done = false;
    do {
      // get the coordinates for the next switchbox in the flow
      for (ConnectOp connectOp : currSwitchbox.getOps<ConnectOp>()) {
        // if this connectOp is the end of a flow, skip
        if ((currSwitchbox.rowIndex() == 0 &&
             connectOp.getDestBundle() == WireBundle::South) ||
            connectOp.getDestBundle() == WireBundle::DMA ||
            connectOp.getDestBundle() == WireBundle::Core)
          continue;

        if (connectOp.getSourceBundle() == currPort.bundle &&
            connectOp.getSourceChannel() == currPort.channel) {
          nextPorts.push({getConnectingBundle(connectOp.getDestBundle()),
                          connectOp.getDestChannel()});

          TileID nextCoords =
              getNextCoords(currSwitchbox.colIndex(), currSwitchbox.rowIndex(),
                            connectOp.getDestBundle());

          // search for next switchbox to connect to
          for (SwitchboxOp switchboxOp : targetOp.getOps<SwitchboxOp>()) {
            if (switchboxOp.colIndex() == nextCoords.col &&
                switchboxOp.rowIndex() == nextCoords.row) {
              nextSwitches.push(switchboxOp);
              break;
            }
          }
        }
      }

      // add switchbox to the routeString
      std::string dirString = std::string("[[") +
                              std::to_string(currSwitchbox.colIndex()) + ", " +
                              std::to_string(currSwitchbox.rowIndex()) + "], [";
      int opCount = 0, dirCount = 0;
      for (ConnectOp connectOp : currSwitchbox.getOps<ConnectOp>()) {
        if (connectOp.getSourceBundle() == currPort.bundle &&
            connectOp.getSourceChannel() == currPort.channel) {
          if (opCount++ > 0)
            dirString += ", ";
          dirCount++;
          dirString +=
              "\"" +
              (std::string)stringifyWireBundle(connectOp.getDestBundle()) +
              "\"";
        }
      }
      dirString += "]], ";
      if (dirCount > 0)
        routeString += dirString;

      if (nextPorts.empty() || nextSwitches.empty()) {
        done = true;
        routeString += "[]";
      } else {
        currPort = nextPorts.front();
        currSwitchbox = nextSwitches.front();
        nextPorts.pop();
        nextSwitches.pop();
      }
    } while (!done);
    // write string to JSON
    routeString += std::string(" ],\n");
    output << routeString;
  }
  output << "\"route_all\": [],\n";
  output << "\n\"end json\": 0\n"; // dummy line to avoid errors from commas
  output << "}";
  return success();
} // end AIETranslateToJSON
} // namespace xilinx::AIE
