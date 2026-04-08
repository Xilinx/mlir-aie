//===- AIEFlowsToDOT.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIETargets.h"
#include "aie/Targets/AIEVisualShared.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include "llvm/Support/Debug.h"

#include <map>
#include <queue>
#include <set>
#include <vector>

#define DEBUG_TYPE "aie-flows-to-dot"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx::AIE {

//===----------------------------------------------------------------------===//
// Routing Reconstruction
//===----------------------------------------------------------------------===//

struct PathSegment {
  TileID from, to;
  Port fromPort;
  Port toPort;
};

struct FlowPath {
  TileID src, dst;
  Port srcPort, dstPort;
  std::vector<TileID> path;
  std::vector<PathSegment> segments;
};

struct SharedMemConnection {
  TileID src, dst, allocTile;
};

static std::pair<int, int>
getDMAChannelCounts(const AIETargetModel &targetModel, int col, int row) {
  AIETileType type = targetModel.getTileType(col, row);

  if (type == AIETileType::ShimNOCTile || type == AIETileType::ShimPLTile) {
    return {
        targetModel.getNumDestShimMuxConnections(col, row, WireBundle::DMA),
        targetModel.getNumSourceShimMuxConnections(col, row, WireBundle::DMA)};
  }

  return {
      targetModel.getNumDestSwitchboxConnections(col, row, WireBundle::DMA),
      targetModel.getNumSourceSwitchboxConnections(col, row, WireBundle::DMA)};
}

static std::pair<std::vector<TileID>, std::vector<PathSegment>>
traceThroughSwitchboxes(DeviceOp device, TileOp startTile, Port startPort) {
  std::vector<TileID> path;
  std::vector<PathSegment> segments;
  path.push_back({startTile.colIndex(), startTile.rowIndex()});

  SwitchboxOp currSwitchbox = nullptr;
  for (auto sb : device.getOps<SwitchboxOp>()) {
    if (sb.colIndex() == startTile.colIndex() &&
        sb.rowIndex() == startTile.rowIndex()) {
      currSwitchbox = sb;
      break;
    }
  }

  if (!currSwitchbox)
    return {path, segments};

  Port currPort = startPort;

  if (startTile.rowIndex() == 0) {
    for (auto shimMux : device.getOps<ShimMuxOp>()) {
      if (shimMux.colIndex() == startTile.colIndex() &&
          shimMux.rowIndex() == startTile.rowIndex()) {
        for (auto connectOp : shimMux.getOps<ConnectOp>()) {
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
  }

  bool done = false;
  while (!done) {
    bool foundNext = false;

    for (auto connectOp : currSwitchbox.getOps<ConnectOp>()) {
      if (connectOp.getDestBundle() == WireBundle::DMA ||
          connectOp.getDestBundle() == WireBundle::Core ||
          (currSwitchbox.rowIndex() == 0 &&
           connectOp.getDestBundle() == WireBundle::South))
        continue;

      if (connectOp.getSourceBundle() == currPort.bundle &&
          connectOp.getSourceChannel() == currPort.channel) {
        TileID currTile = {currSwitchbox.colIndex(), currSwitchbox.rowIndex()};
        TileID nextCoords =
            getNextCoords(currSwitchbox.colIndex(), currSwitchbox.rowIndex(),
                          connectOp.getDestBundle());
        if (path.empty() || path.back() != nextCoords)
          path.push_back(nextCoords);

        Port outPort = {connectOp.getDestBundle(), connectOp.getDestChannel()};
        Port inPort = {getConnectingBundle(connectOp.getDestBundle()),
                       connectOp.getDestChannel()};
        segments.push_back({currTile, nextCoords, outPort, inPort});

        SwitchboxOp nextSwitchbox = nullptr;
        for (auto sb : device.getOps<SwitchboxOp>()) {
          if (sb.colIndex() == nextCoords.col &&
              sb.rowIndex() == nextCoords.row) {
            nextSwitchbox = sb;
            break;
          }
        }

        if (!nextSwitchbox) {
          done = true;
          break;
        }

        currSwitchbox = nextSwitchbox;
        currPort.bundle = getConnectingBundle(connectOp.getDestBundle());
        currPort.channel = connectOp.getDestChannel();
        foundNext = true;
        break;
      }
    }

    if (foundNext)
      continue;

    for (auto pktRulesOp : currSwitchbox.getOps<PacketRulesOp>()) {
      if (pktRulesOp.sourcePort().bundle != currPort.bundle ||
          pktRulesOp.sourcePort().channel != currPort.channel)
        continue;

      Region &r = pktRulesOp.getRules();
      Block &b = r.front();

      for (auto ruleOp : b.getOps<PacketRuleOp>()) {
        Value amsel = ruleOp.getAmsel();

        for (auto masterSetOp : currSwitchbox.getOps<MasterSetOp>()) {
          for (Value masterAmsel : masterSetOp.getAmsels()) {
            if (masterAmsel == amsel) {
              Port destPort = masterSetOp.destPort();
              if (destPort.bundle == WireBundle::DMA ||
                  destPort.bundle == WireBundle::Core ||
                  (currSwitchbox.rowIndex() == 0 &&
                   destPort.bundle == WireBundle::South)) {
                done = true;
                break;
              }

              TileID currTile = {currSwitchbox.colIndex(),
                                 currSwitchbox.rowIndex()};
              TileID nextCoords =
                  getNextCoords(currSwitchbox.colIndex(),
                                currSwitchbox.rowIndex(), destPort.bundle);

              if (path.empty() || path.back() != nextCoords)
                path.push_back(nextCoords);

              Port outPort = destPort;
              Port inPort = {getConnectingBundle(destPort.bundle),
                             destPort.channel};
              segments.push_back({currTile, nextCoords, outPort, inPort});

              SwitchboxOp nextSwitchbox = nullptr;
              for (auto sb : device.getOps<SwitchboxOp>()) {
                if (sb.colIndex() == nextCoords.col &&
                    sb.rowIndex() == nextCoords.row) {
                  nextSwitchbox = sb;
                  break;
                }
              }

              if (!nextSwitchbox) {
                done = true;
                break;
              }

              currSwitchbox = nextSwitchbox;
              currPort.bundle = getConnectingBundle(destPort.bundle);
              currPort.channel = destPort.channel;
              foundNext = true;
              break;
            }
          }
          if (foundNext)
            break;
        }
        if (foundNext)
          break;
      }
      if (foundNext)
        break;
    }

    if (!foundNext)
      done = true;
  }

  return {path, segments};
}

static std::vector<FlowPath> detectCircuitFlows(DeviceOp device) {
  std::vector<FlowPath> connections;
  std::set<std::pair<TileOp, Port>> processedFlows;

  for (auto flowOp : device.getOps<FlowOp>()) {
    TileOp source = cast<TileOp>(flowOp.getSource().getDefiningOp());
    Port sourcePort = {flowOp.getSourceBundle(), flowOp.getSourceChannel()};
    Port destPort = {flowOp.getDestBundle(), flowOp.getDestChannel()};

    if (processedFlows.count({source, sourcePort}))
      continue;
    processedFlows.insert({source, sourcePort});

    auto [path, segments] = traceThroughSwitchboxes(device, source, sourcePort);

    if (path.size() >= 2) {
      connections.push_back(
          {path.front(), path.back(), sourcePort, destPort, path, segments});
    }
  }

  return connections;
}

static std::vector<FlowPath> detectPacketFlows(DeviceOp device) {
  std::vector<FlowPath> connections;
  std::set<std::pair<TileOp, Port>> processedFlows;

  for (auto pktFlowOp : device.getOps<PacketFlowOp>()) {
    Region &r = pktFlowOp.getPorts();
    Block &b = r.front();

    TileOp source = nullptr;
    TileOp dest = nullptr;
    Port sourcePort;
    Port destPort;

    for (Operation &op : b.getOperations()) {
      if (auto pktSource = dyn_cast<PacketSourceOp>(op)) {
        source = dyn_cast<TileOp>(pktSource.getTile().getDefiningOp());
        sourcePort = pktSource.port();
      } else if (auto pktDest = dyn_cast<PacketDestOp>(op)) {
        dest = dyn_cast<TileOp>(pktDest.getTile().getDefiningOp());
        destPort = pktDest.port();
      }
    }

    if (!source)
      continue;

    if (processedFlows.count({source, sourcePort}))
      continue;
    processedFlows.insert({source, sourcePort});

    auto [path, segments] = traceThroughSwitchboxes(device, source, sourcePort);

    if (path.size() >= 2) {
      TileID dstTile =
          dest ? TileID{dest.getCol(), dest.getRow()} : path.back();
      connections.push_back(
          {path.front(), dstTile, sourcePort, destPort, path, segments});
    }
  }

  return connections;
}

static std::vector<SharedMemConnection>
detectSharedMemoryConnections(DeviceOp device,
                              const AIETargetModel &targetModel) {
  std::vector<SharedMemConnection> connections;
  std::map<std::pair<TileOp, TileOp>, TileOp> sharedMemConnect;

  for (auto coreOp : device.getOps<CoreOp>()) {
    auto coreTileOp = coreOp.getTileOp();

    coreOp.walk([&](UseLockOp useLockOp) {
      auto lock = useLockOp.getLockOp();
      auto lockTileOp = lock.getTileOp();

      if (coreTileOp == lockTileOp)
        return;
      if (coreTileOp.isShimTile() != lockTileOp.isShimTile())
        return;
      if (coreTileOp.isMemTile() != lockTileOp.isMemTile())
        return;
      if (!targetModel.isLegalMemAffinity(
              coreTileOp.getCol(), coreTileOp.getRow(), lockTileOp.getCol(),
              lockTileOp.getRow()))
        return;

      if (!lock.hasName())
        return;

      llvm::StringRef lockName = lock.name().getValue();
      bool isProdLock = lockName.contains("_prod_lock");
      bool isConsLock = lockName.contains("_cons_lock");

      if (!isProdLock && !isConsLock)
        return;

      if (useLockOp.release()) {
        std::pair<TileOp, TileOp> edge =
            isConsLock ? std::make_pair(lockTileOp, coreTileOp)
                       : std::make_pair(coreTileOp, lockTileOp);

        if (sharedMemConnect.count(edge) == 0) {
          sharedMemConnect[edge] = lockTileOp;
        }
      }
    });
  }

  for (auto &[tiles, allocTile] : sharedMemConnect) {
    auto [srcTile, dstTile] = tiles;
    connections.push_back({{srcTile.getCol(), srcTile.getRow()},
                           {dstTile.getCol(), dstTile.getRow()},
                           {allocTile.getCol(), allocTile.getRow()}});
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Detected " << connections.size()
                 << " shared-memory connections\n";
    for (const auto &conn : connections) {
      llvm::dbgs() << "  (" << conn.src.col << "," << conn.src.row << ") -> ("
                   << conn.dst.col << "," << conn.dst.row << ") alloc at ("
                   << conn.allocTile.col << "," << conn.allocTile.row << ")\n";
    }
  });

  return connections;
}

//===----------------------------------------------------------------------===//
// DOT Visualization
//===----------------------------------------------------------------------===//

struct TileCoords {
  double gridX, gridY;
  double centerX, centerY;
  double left, right, top, bottom;
};

static TileCoords getTileCoords(int col, int row) {
  double gridX = col * kGridWidth;
  double gridY = row * kGridHeight;
  double centerX =
      gridX + kInternalGap + kSwitchboxWidth + kInternalGap + kTileWidth / 2.0;
  double centerY = gridY + kInternalGap + kTileHeight / 2.0;
  return {gridX,
          gridY,
          centerX,
          centerY,
          centerX - kTileWidth / 2.0,
          centerX + kTileWidth / 2.0,
          centerY + kTileHeight / 2.0,
          centerY - kTileHeight / 2.0};
}

struct DMAPortLayout {
  double memCenterX;
  double portsCenterY;
  double portGap;
  int numS2MM, numMM2S;
};

static DMAPortLayout getDMAPortLayout(const AIETargetModel &targetModel,
                                      const TileCoords &tile, int col,
                                      int row) {
  auto [numS2MM, numMM2S] = getDMAChannelCounts(targetModel, col, row);
  AIETileType type = targetModel.getTileType(col, row);

  double memCenterX, boxSize;

  if (type == AIETileType::CoreTile) {
    boxSize = 1.5;
    double coreCenterX =
        tile.left + kInternalGap + kPortSize + kInternalGap + boxSize / 2.0;
    memCenterX = coreCenterX + boxSize / 2.0 + kInternalGap + kPortSize +
                 kInternalGap + boxSize / 2.0;
  } else {
    memCenterX = tile.centerX;
    boxSize = kTileWidth - 2 * kInternalGap;
  }

  int totalPorts = numS2MM + numMM2S;
  double portGap = totalPorts > 1
                       ? (boxSize - totalPorts * kPortSize) / (totalPorts - 1)
                       : 0;
  double portsCenterY = tile.top - kInternalGap - kDMAPortHeight / 2.0;

  return {memCenterX, portsCenterY, portGap, numS2MM, numMM2S};
}

static double getDMAPortXPosition(const DMAPortLayout &layout, int portIndex) {
  int totalPorts = layout.numS2MM + layout.numMM2S;
  double totalWidth =
      (totalPorts * kPortSize) + ((totalPorts - 1) * layout.portGap);
  double startX = layout.memCenterX - (totalWidth / 2.0) + (kPortSize / 2.0);
  return startX + portIndex * (kPortSize + layout.portGap);
}

static void emitBox(llvm::raw_ostream &output, const std::string &nodeName,
                    const std::string &label, double x, double y, double w,
                    double h, int fontSize = 28) {
  output << "  " << nodeName << " [";
  output << "label=\"" << label << "\"; ";
  output << llvm::format("pos=\"%.2f,%.2f!\"; ", x, y);
  output << "shape=rectangle; fixedsize=true; ";
  output << llvm::format("width=%.2f; height=%.2f; ", w, h);
  output
      << "fillcolor=\"white\"; color=\"black\"; style=filled; penwidth=1.5; ";
  output << "fontsize=" << fontSize << "; fontname=\"Helvetica\"];\n";
}

static void emitSharedMemPort(llvm::raw_ostream &output, int col, int row,
                              char dir, double x, double y) {
  char dirLower = std::tolower(dir);
  output << "  tile_" << col << "_" << row << "_smport_" << dirLower << " [";
  output << "label=\"" << dir << "\"; shape=rectangle; fixedsize=true; ";
  output << llvm::format("width=%.2f; height=%.2f; ", kPortSize, kPortSize);
  output << llvm::format("pos=\"%.2f,%.2f!\"; ", x, y);
  output << "fillcolor=\"white\"; color=\"black\"; style=filled; penwidth=1.2; "
            "fontsize=12; fontname=\"Helvetica\"];\n";
}

static void emitHelperPoint(llvm::raw_ostream &output, const std::string &name,
                            double x, double y) {
  output << "  " << name << " [";
  output << llvm::format("pos=\"%.2f,%.2f!\"; ", x, y);
  output << "shape=point; width=0.01; height=0.01];\n";
}

static std::string getSwitchboxTileCornerName(int col, int row) {
  return "sb_" + std::to_string(col) + "_" + std::to_string(row) +
         "_tile_corner";
}

static std::string getSwitchboxChannelHelperName(int col, int row,
                                                 WireBundle bundle,
                                                 int channel) {
  std::string bundleName;
  switch (bundle) {
  case WireBundle::North:
    bundleName = "n";
    break;
  case WireBundle::South:
    bundleName = "s";
    break;
  case WireBundle::East:
    bundleName = "e";
    break;
  case WireBundle::West:
    bundleName = "w";
    break;
  case WireBundle::DMA:
    bundleName = "dma";
    break;
  case WireBundle::Core:
    bundleName = "core";
    break;
  default:
    bundleName = "unk";
    break;
  }
  return "sb_" + std::to_string(col) + "_" + std::to_string(row) + "_ch_" +
         bundleName + "_" + std::to_string(channel);
}

static GridPosition getSwitchboxChannelPoint(int col, int row,
                                             WireBundle bundle, int channel) {
  double gridX = col * kGridWidth;
  double gridY = row * kGridHeight;
  double sbCenterX = gridX + kInternalGap + kSwitchboxWidth / 2.0;
  double sbCenterY =
      gridY + kInternalGap + kTileHeight + kInternalGap + kSwitchboxWidth / 2.0;

  switch (bundle) {
  case WireBundle::North: {
    double xOffset =
        -kSwitchboxWidth / 2.0 + kChannelSpacing + channel * kChannelSpacing;
    return {sbCenterX + xOffset, sbCenterY + kSwitchboxWidth / 2.0};
  }
  case WireBundle::South: {
    double xOffset =
        -kSwitchboxWidth / 2.0 + kChannelSpacing + channel * kChannelSpacing;
    return {sbCenterX + xOffset, sbCenterY - kSwitchboxWidth / 2.0};
  }
  case WireBundle::East: {
    double yOffset =
        kSwitchboxWidth / 2.0 - kChannelSpacing - channel * kChannelSpacing;
    return {sbCenterX + kSwitchboxWidth / 2.0, sbCenterY + yOffset};
  }
  case WireBundle::West: {
    double yOffset =
        kSwitchboxWidth / 2.0 - kChannelSpacing - channel * kChannelSpacing;
    return {sbCenterX - kSwitchboxWidth / 2.0, sbCenterY + yOffset};
  }
  case WireBundle::DMA:
  case WireBundle::Core: {
    // Position at bottom edge to avoid confusion with East/West
    double xOffset =
        kSwitchboxWidth / 2.0 - kChannelSpacing - channel * kChannelSpacing;
    return {sbCenterX + xOffset, sbCenterY - kSwitchboxWidth / 2.0};
  }
  default:
    return {sbCenterX, sbCenterY};
  }
}

static void emitTileBackground(llvm::raw_ostream &output, int col, int row,
                               const AIETargetModel &targetModel) {
  auto tile = getTileCoords(col, row);

  output << "  tile_" << col << "_" << row << "_bg [";
  output << "label=\"(" << col << "," << row << ")\"; ";
  output << "labelloc=b; labeljust=l; ";
  output << llvm::format("pos=\"%.2f,%.2f!\"; ", tile.centerX, tile.centerY);
  output << "shape=rectangle; fixedsize=true; ";
  output << llvm::format("width=%.2f; height=%.2f; ", kTileWidth, kTileHeight);
  output << "fillcolor=\"white\"; color=\"black\"; ";
  output
      << "style=filled; penwidth=1.5; fontsize=34; fontname=\"Helvetica\"];\n";
}

static void emitDMAPortRow(llvm::raw_ostream &output, int col, int row,
                           int numS2MM, int numMM2S, double centerX,
                           double centerY, double portSize, double portGap) {
  int totalPorts = numS2MM + numMM2S;
  if (totalPorts == 0)
    return;

  double totalPortsWidth =
      (totalPorts * portSize) + ((totalPorts - 1) * portGap);
  double startX = centerX - (totalPortsWidth / 2.0) + (portSize / 2.0);

  int portIndex = 0;

  for (int i = 0; i < numS2MM; i++) {
    double xPos = startX + (portIndex * (portSize + portGap));
    output << "  " << getDMANodeName(col, row, true, i) << " [";
    output << "label=\"S\\n2\\nM\\nM\\n" << i << "\"; ";
    output << llvm::format("pos=\"%.2f,%.2f!\"; ", xPos, centerY);
    output << llvm::format(
        "shape=rectangle; fixedsize=true; width=%.2f; height=%.2f; ", portSize,
        kDMAPortHeight);
    output << "fillcolor=\"white\"; color=\"black\"; style=filled; "
              "penwidth=1.2; fontsize=13; fontname=\"Helvetica\"];\n";
    portIndex++;
  }

  for (int i = 0; i < numMM2S; i++) {
    double xPos = startX + (portIndex * (portSize + portGap));
    output << "  " << getDMANodeName(col, row, false, i) << " [";
    output << "label=\"M\\nM\\n2\\nS\\n" << i << "\"; ";
    output << llvm::format("pos=\"%.2f,%.2f!\"; ", xPos, centerY);
    output << llvm::format(
        "shape=rectangle; fixedsize=true; width=%.2f; height=%.2f; ", portSize,
        kDMAPortHeight);
    output << "fillcolor=\"white\"; color=\"black\"; style=filled; "
              "penwidth=1.2; fontsize=13; fontname=\"Helvetica\"];\n";
    portIndex++;
  }
}

static std::string getFlowEndpointNode(int col, int row, const Port &port,
                                       bool isSource) {
  if (port.bundle == WireBundle::DMA) {
    return getDMANodeName(col, row, !isSource, port.channel);
  }
  return getCoreNodeName(col, row);
}
static void emitCoreTileComponents(llvm::raw_ostream &output, int col, int row,
                                   const AIETargetModel &targetModel) {
  auto tile = getTileCoords(col, row);
  auto dma = getDMAPortLayout(targetModel, tile, col, row);

  double boxSize = 1.5;
  double coreCenterX =
      tile.left + kInternalGap + kPortSize + kInternalGap + boxSize / 2.0;
  double memCenterX = coreCenterX + boxSize / 2.0 + kInternalGap + kPortSize +
                      kInternalGap + boxSize / 2.0;

  emitBox(output, getCoreNodeName(col, row), "Core", coreCenterX, tile.centerY,
          boxSize, boxSize);
  emitBox(output, getBufferNodeName(col, row), "Memory", memCenterX,
          tile.centerY, boxSize, boxSize);

  emitDMAPortRow(output, col, row, dma.numS2MM, dma.numMM2S, dma.memCenterX,
                 dma.portsCenterY, kPortSize, dma.portGap);

  emitSharedMemPort(output, col, row, 'W',
                    tile.left + kInternalGap + kPortSize / 2.0, tile.centerY);
  emitSharedMemPort(output, col, row, 'E',
                    coreCenterX + boxSize / 2.0 + kInternalGap +
                        kPortSize / 2.0,
                    tile.centerY);
  emitSharedMemPort(output, col, row, 'N', coreCenterX,
                    tile.top - kInternalGap - kPortSize / 2.0);
  emitSharedMemPort(output, col, row, 'S', coreCenterX,
                    tile.bottom + kInternalGap + kPortSize / 2.0);
}

static void emitMemTileComponents(llvm::raw_ostream &output, int col, int row,
                                  const AIETargetModel &targetModel) {
  auto tile = getTileCoords(col, row);
  auto dma = getDMAPortLayout(targetModel, tile, col, row);

  double memWidth = kTileWidth - 2 * kInternalGap;
  double memHeight = 1.0;

  emitBox(output, getBufferNodeName(col, row), "Memory", tile.centerX,
          tile.centerY, memWidth / 2.0, memHeight);

  emitDMAPortRow(output, col, row, dma.numS2MM, dma.numMM2S, dma.memCenterX,
                 dma.portsCenterY, kPortSize, dma.portGap);
}

static void emitShimTileComponents(llvm::raw_ostream &output, int col, int row,
                                   const AIETargetModel &targetModel) {
  auto tile = getTileCoords(col, row);
  auto dma = getDMAPortLayout(targetModel, tile, col, row);

  emitDMAPortRow(output, col, row, dma.numS2MM, dma.numMM2S, dma.memCenterX,
                 dma.portsCenterY, kPortSize, dma.portGap);
}

static std::map<TileID, std::set<std::pair<WireBundle, int>>>
collectUsedChannels(const std::vector<FlowPath> &allFlows) {
  std::map<TileID, std::set<std::pair<WireBundle, int>>> usedChannels;

  for (const auto &flow : allFlows) {
    for (const auto &seg : flow.segments) {
      usedChannels[seg.from].insert(
          {seg.fromPort.bundle, seg.fromPort.channel});
      usedChannels[seg.to].insert({seg.toPort.bundle, seg.toPort.channel});
    }
    usedChannels[flow.src].insert({flow.srcPort.bundle, flow.srcPort.channel});
    usedChannels[flow.dst].insert({flow.dstPort.bundle, flow.dstPort.channel});
  }

  return usedChannels;
}

//===----------------------------------------------------------------------===//
// DOT Visualization - Main Generation
//===----------------------------------------------------------------------===//

static void
emitDetailedDOT(llvm::raw_ostream &output, DeviceOp device,
                const AIETargetModel &targetModel,
                const std::vector<FlowPath> &circuitFlows,
                const std::vector<FlowPath> &packetFlows,
                const std::vector<SharedMemConnection> &sharedMemConns) {

  emitDOTHeader(output, "AIE_Routing", "neato");

  output << "  // Background tiles\n";
  for (int col = 0; col < targetModel.columns(); col++) {
    for (int row = 0; row < targetModel.rows(); row++) {
      emitTileBackground(output, col, row, targetModel);
    }
  }

  output << "\n  // Switchboxes\n";
  ColorScheme sbColor = {"#F5F5F5", "#666666"};
  for (int col = 0; col < targetModel.columns(); col++) {
    for (int row = 0; row < targetModel.rows(); row++) {
      double sbCenterX =
          col * kGridWidth + kInternalGap + kSwitchboxWidth / 2.0;
      double sbCenterY = row * kGridHeight + kInternalGap + kTileHeight +
                         kInternalGap + kSwitchboxWidth / 2.0;

      output << "  " << getSwitchboxNodeName(col, row) << " [";
      output << "label=\"\"; ";
      output << llvm::format("pos=\"%.2f,%.2f!\"; ", sbCenterX, sbCenterY);
      output << "fillcolor=\"" << sbColor.fill << "\"; color=\""
             << sbColor.stroke << "\"; ";
      output << "style=filled; pin=true; shape=square; ";
      output << llvm::format(
          "width=%.2f; fontsize=10; fontname=\"Helvetica\"];\n",
          kSwitchboxWidth);
    }
  }

  output << "\n  // Tile components\n";
  for (int col = 0; col < targetModel.columns(); col++) {
    for (int row = 0; row < targetModel.rows(); row++) {
      AIETileType type = targetModel.getTileType(col, row);

      if (type == AIETileType::CoreTile) {
        emitCoreTileComponents(output, col, row, targetModel);
      } else if (type == AIETileType::MemTile) {
        emitMemTileComponents(output, col, row, targetModel);
      } else if (type == AIETileType::ShimNOCTile ||
                 type == AIETileType::ShimPLTile) {
        emitShimTileComponents(output, col, row, targetModel);
      }
    }
  }

  std::vector<FlowPath> allFlows;
  allFlows.insert(allFlows.end(), circuitFlows.begin(), circuitFlows.end());
  allFlows.insert(allFlows.end(), packetFlows.begin(), packetFlows.end());
  auto usedChannels = collectUsedChannels(allFlows);

  output << "\n  // Helper points for DMA port tops\n";
  for (const auto &flow : allFlows) {
    if (flow.srcPort.bundle == WireBundle::DMA) {
      auto tile = getTileCoords(flow.src.col, flow.src.row);
      auto dma =
          getDMAPortLayout(targetModel, tile, flow.src.col, flow.src.row);
      double helperX =
          getDMAPortXPosition(dma, flow.srcPort.channel + dma.numS2MM);
      double helperY = tile.top - kInternalGap;
      emitHelperPoint(output,
                      getDMANodeName(flow.src.col, flow.src.row, false,
                                     flow.srcPort.channel) +
                          "_top",
                      helperX, helperY);
    }
    if (flow.dstPort.bundle == WireBundle::DMA) {
      auto tile = getTileCoords(flow.dst.col, flow.dst.row);
      auto dma =
          getDMAPortLayout(targetModel, tile, flow.dst.col, flow.dst.row);
      double helperX = getDMAPortXPosition(dma, flow.dstPort.channel);
      double helperY = tile.top - kInternalGap;
      emitHelperPoint(output,
                      getDMANodeName(flow.dst.col, flow.dst.row, true,
                                     flow.dstPort.channel) +
                          "_top",
                      helperX, helperY);
    }
  }

  output << "\n  // Helper points for switchbox tile corners\n";
  std::set<TileID> tilesWithDMAFlows;
  for (const auto &flow : allFlows) {
    if (flow.srcPort.bundle == WireBundle::DMA ||
        flow.srcPort.bundle == WireBundle::Core)
      tilesWithDMAFlows.insert(flow.src);
    if (flow.dstPort.bundle == WireBundle::DMA ||
        flow.dstPort.bundle == WireBundle::Core)
      tilesWithDMAFlows.insert(flow.dst);
  }

  for (const auto &tileID : tilesWithDMAFlows) {
    double sbCenterX =
        tileID.col * kGridWidth + kInternalGap + kSwitchboxWidth / 2.0;
    double sbCenterY = tileID.row * kGridHeight + kInternalGap + kTileHeight +
                       kInternalGap + kSwitchboxWidth / 2.0;
    double cornerX = sbCenterX + kSwitchboxWidth / 2.0;
    double cornerY = sbCenterY - kSwitchboxWidth / 2.0;

    emitHelperPoint(output, getSwitchboxTileCornerName(tileID.col, tileID.row),
                    cornerX, cornerY);
  }

  output << "\n  // Helper points for switchbox channels\n";
  for (const auto &[tile, channels] : usedChannels) {
    for (const auto &[bundle, channel] : channels) {
      if (bundle == WireBundle::DMA || bundle == WireBundle::Core)
        continue;

      GridPosition pt =
          getSwitchboxChannelPoint(tile.col, tile.row, bundle, channel);
      emitHelperPoint(
          output,
          getSwitchboxChannelHelperName(tile.col, tile.row, bundle, channel),
          pt.x, pt.y);
    }
  }

  output << "\n  // Flow edges\n";
  int flowIndex = 0;
  auto emitFlowEdges = [&](const auto &flows) {
    for (const auto &conn : flows) {
      std::string color = getRouteColor(flowIndex);
      std::string label = "route" + std::to_string(flowIndex);

      std::string srcNode =
          getFlowEndpointNode(conn.src.col, conn.src.row, conn.srcPort, true);
      std::string srcSB = getSwitchboxNodeName(conn.src.col, conn.src.row);

      if (conn.srcPort.bundle == WireBundle::DMA) {
        std::string srcHelper = getDMANodeName(conn.src.col, conn.src.row,
                                               false, conn.srcPort.channel) +
                                "_top";
        std::string sbCorner =
            getSwitchboxTileCornerName(conn.src.col, conn.src.row);
        output << "  " << srcNode << " -> " << srcHelper
               << " [style=invis; weight=10];\n";
        output << "  " << srcHelper << " -> " << sbCorner;
        output << " [label=\"" << label << "\"; color=\"" << color
               << "\"; penwidth=4.5];\n";
        output << "  " << sbCorner << " -> " << srcSB
               << " [style=invis; weight=10];\n";
      } else {
        output << "  " << srcNode << " -> " << srcSB;
        output << " [label=\"" << label << "\"; color=\"" << color
               << "\"; penwidth=4.5];\n";
      }

      for (const auto &seg : conn.segments) {
        std::string fromSB = getSwitchboxNodeName(seg.from.col, seg.from.row);
        std::string toSB = getSwitchboxNodeName(seg.to.col, seg.to.row);
        std::string fromHelper = getSwitchboxChannelHelperName(
            seg.from.col, seg.from.row, seg.fromPort.bundle,
            seg.fromPort.channel);
        std::string toHelper = getSwitchboxChannelHelperName(
            seg.to.col, seg.to.row, seg.toPort.bundle, seg.toPort.channel);

        output << "  " << fromSB << " -> " << fromHelper
               << " [style=invis; weight=10];\n";
        output << "  " << fromHelper << " -> " << toHelper;
        output << " [label=\"" << label << "\"; color=\"" << color
               << "\"; penwidth=4.5];\n";
        output << "  " << toHelper << " -> " << toSB
               << " [style=invis; weight=10];\n";
      }

      std::string dstNode =
          getFlowEndpointNode(conn.dst.col, conn.dst.row, conn.dstPort, false);
      std::string dstSB = getSwitchboxNodeName(conn.dst.col, conn.dst.row);

      if (conn.dstPort.bundle == WireBundle::DMA) {
        std::string dstHelper = getDMANodeName(conn.dst.col, conn.dst.row, true,
                                               conn.dstPort.channel) +
                                "_top";
        std::string sbCorner =
            getSwitchboxTileCornerName(conn.dst.col, conn.dst.row);
        output << "  " << dstSB << " -> " << sbCorner
               << " [style=invis; weight=10];\n";
        output << "  " << sbCorner << " -> " << dstHelper;
        output << " [label=\"" << label << "\"; color=\"" << color
               << "\"; penwidth=4.5];\n";
        output << "  " << dstHelper << " -> " << dstNode
               << " [style=invis; weight=10];\n";
      } else {
        output << "  " << dstSB << " -> " << dstNode;
        output << " [label=\"" << label << "\"; color=\"" << color
               << "\"; penwidth=4.5];\n";
      }

      flowIndex++;
    }
  };

  emitFlowEdges(circuitFlows);
  emitFlowEdges(packetFlows);

  output << "\n  // Shared-memory edges\n";

  for (const auto &conn : sharedMemConns) {
    int colDiff = conn.dst.col - conn.allocTile.col;
    int rowDiff = conn.dst.row - conn.allocTile.row;

    std::string smPortSuffix;
    if (colDiff == 1 && rowDiff == 0) {
      smPortSuffix = "_smport_w";
    } else if (colDiff == 0 && rowDiff == -1) {
      smPortSuffix = "_smport_n";
    } else if (colDiff == 0 && rowDiff == 1) {
      smPortSuffix = "_smport_s";
    } else {
      llvm_unreachable("Invalid shared memory connection direction");
    }
    output << "  " << getBufferNodeName(conn.allocTile.col, conn.allocTile.row);
    output << " -> tile_" << conn.dst.col << "_" << conn.dst.row
           << smPortSuffix;
    output << " [label=\"shared_mem\"; ";
    output << "color=\"" << getSharedMemoryColor() << "\"; ";
    output << "style=" << getSharedMemoryStyle() << "; ";
    output << "penwidth=4.5];\n";
  }

  emitDOTFooter(output);
}

LogicalResult AIEFlowsToDOT(ModuleOp module, llvm::raw_ostream &output,
                            llvm::StringRef deviceName) {
  DeviceOp device = AIE::DeviceOp::getForSymbolInModule(module, deviceName);
  if (!device)
    return module.emitOpError("expected AIE.device operation");

  const AIETargetModel &targetModel = device.getTargetModel();

  std::vector<FlowPath> circuitFlows = detectCircuitFlows(device);
  std::vector<FlowPath> packetFlows = detectPacketFlows(device);
  std::vector<SharedMemConnection> sharedMemConns =
      detectSharedMemoryConnections(device, targetModel);
  emitDetailedDOT(output, device, targetModel, circuitFlows, packetFlows,
                  sharedMemConns);

  return success();
}

} // namespace xilinx::AIE
