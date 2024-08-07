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

void translateSwitchboxes(DeviceOp targetOp, raw_ostream &output) {
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
  for (PacketFlowOp pktFlowOp : targetOp.getOps<PacketFlowOp>()) {
    Region &r = pktFlowOp.getPorts();
    Block &b = r.front();
    for (Operation &Op : b.getOperations()) {
      if (auto pktSource = dyn_cast<PacketSourceOp>(Op)) {
        TileOp source = dyn_cast<TileOp>(pktSource.getTile().getDefiningOp());
        TileID srcID = {source.colIndex(), source.rowIndex()};
        sourceCounts[srcID]++;
      } else if (auto pktDest = dyn_cast<PacketDestOp>(Op)) {
        TileOp dest = dyn_cast<TileOp>(pktDest.getTile().getDefiningOp());
        TileID dstID = {dest.colIndex(), dest.rowIndex()};
        destinationCounts[dstID]++;
      }
    }
  }

  std::map<TileID, ShimMuxOp> shimMuxes;
  for (ShimMuxOp shimMuxOp : targetOp.getOps<ShimMuxOp>()) {
    shimMuxes[{shimMuxOp.colIndex(), shimMuxOp.rowIndex()}] = shimMuxOp;
  }

  int totalPathLength = 0;
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

    std::set<Port> ports;
    for (ConnectOp connectOp : switchboxOp.getOps<ConnectOp>())
      ports.insert(connectOp.destPort());
    for (MasterSetOp masterSetOp : switchboxOp.getOps<MasterSetOp>())
      ports.insert(masterSetOp.destPort());
    for (Port port : ports) {
      connectCounts[int(port.bundle)]++;
    }

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

    // calculate total path length
    totalPathLength += connectCounts[int(WireBundle::North)];
    totalPathLength += connectCounts[int(WireBundle::East)];
    totalPathLength += connectCounts[int(WireBundle::South)];
    totalPathLength += connectCounts[int(WireBundle::West)];

    // deduct shim muxes from total path length
    if (shimMuxes.count({col, row})) {
      auto shimMuxOp = shimMuxes[{col, row}];
      std::set<Port> ports;
      for (ConnectOp connectOp : shimMuxOp.getOps<ConnectOp>()) {
        Port port = connectOp.sourcePort();
        if (port.bundle == WireBundle::North) {
          ports.insert(port);
        }
      }
      for (PacketRulesOp packetRulesOp : shimMuxOp.getOps<PacketRulesOp>()) {
        Port port = packetRulesOp.sourcePort();
        if (port.bundle == WireBundle::North) {
          ports.insert(port);
        }
      }
      totalPathLength -= ports.size();
    }
  }

  // write total path length to JSON
  output << "\"total_path_length\": " << totalPathLength << ",\n";
}

void translateCircuitFlows(DeviceOp targetOp, int &flowCount,
                           raw_ostream &output) {
  // for each flow, trace it through switchboxes and write the route to JSON
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
}

void translatePacketFlows(DeviceOp targetOp, int &flowCount,
                          raw_ostream &output) {
  // for each flow, trace it through switchboxes and write the route to JSON
  std::set<std::pair<TileOp, Port>> flowSources;
  for (PacketFlowOp pktFlowOp : targetOp.getOps<PacketFlowOp>()) {
    Region &r = pktFlowOp.getPorts();
    Block &b = r.front();
    Port sourcePort, destPort;
    TileOp source, dest;
    for (Operation &Op : b.getOperations()) {
      if (auto pktSource = dyn_cast<PacketSourceOp>(Op)) {
        source = dyn_cast<TileOp>(pktSource.getTile().getDefiningOp());
        sourcePort = pktSource.port();
      } else if (auto pktDest = dyn_cast<PacketDestOp>(Op)) {
        dest = dyn_cast<TileOp>(pktDest.getTile().getDefiningOp());
        destPort = pktDest.port();
      }
    }

    // objects used to trace through the flow
    Port currPort = sourcePort;
    SwitchboxOp currSwitchbox;

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
      for (auto packetRulesOp : currSwitchbox.getOps<PacketRulesOp>()) {
        Port destPort;
        for (auto masterSetOp : currSwitchbox.getOps<MasterSetOp>()) {
          for (Value amsel : masterSetOp.getAmsels()) {
            for (auto ruleOp :
                 packetRulesOp.getRules().front().getOps<PacketRuleOp>()) {
              if (ruleOp.getAmsel() == amsel) {
                destPort = masterSetOp.destPort();
              }
            }
          }
        }

        // if this packetRulesOp is the end of a flow, skip
        if ((currSwitchbox.rowIndex() == 0 &&
             destPort.bundle == WireBundle::South) ||
            destPort.bundle == WireBundle::DMA ||
            destPort.bundle == WireBundle::Core)
          continue;

        if (packetRulesOp.sourcePort() == currPort) {
          nextPorts.push(
              {getConnectingBundle(destPort.bundle), destPort.channel});

          TileID nextCoords =
              getNextCoords(currSwitchbox.colIndex(), currSwitchbox.rowIndex(),
                            destPort.bundle);

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
      for (auto packetRulesOp : currSwitchbox.getOps<PacketRulesOp>()) {
        if (packetRulesOp.sourcePort() == currPort) {
          Port destPort;
          for (auto masterSetOp : currSwitchbox.getOps<MasterSetOp>()) {
            for (Value amsel : masterSetOp.getAmsels()) {
              for (auto ruleOp :
                   packetRulesOp.getRules().front().getOps<PacketRuleOp>()) {
                if (ruleOp.getAmsel() == amsel) {
                  destPort = masterSetOp.destPort();
                }
              }
            }
          }
          if (opCount++ > 0)
            dirString += ", ";
          dirCount++;
          dirString +=
              "\"" + (std::string)stringifyWireBundle(destPort.bundle) + "\"";
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
}

mlir::LogicalResult AIEFlowsToJSON(ModuleOp module, raw_ostream &output) {
  output << "{\n";
  if (module.getOps<DeviceOp>().empty()) {
    module.emitOpError("expected AIE.device operation at toplevel");
  }

  DeviceOp targetOp = *(module.getOps<DeviceOp>().begin());
  int flowCount = 0;

  translateSwitchboxes(targetOp, output);
  translateCircuitFlows(targetOp, flowCount, output);
  translatePacketFlows(targetOp, flowCount, output);

  output << "\"route_all\": [],\n";
  output << "\n\"end json\": 0\n"; // dummy line to avoid errors from commas
  output << "}";
  return success();
} // end AIETranslateToJSON
} // namespace xilinx::AIE
