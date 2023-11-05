//===- AIENetlistAnalysis.cpp -----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/AIENetlistAnalysis.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

void xilinx::AIE::NetlistAnalysis::collectTiles(
    DenseMap<TileID, Operation *> &tiles) {
  for (auto tile : device.getOps<TileOp>()) {
    int colIndex = tile.colIndex();
    int rowIndex = tile.rowIndex();
    tiles[{colIndex, rowIndex}] = tile;
  }
}

void xilinx::AIE::NetlistAnalysis::collectCores(
    DenseMap<Operation *, CoreOp> &cores) {
  for (auto core : device.getOps<CoreOp>()) {
    Operation *tileOp = core.getTile().getDefiningOp();
    assert(cores.count(tileOp) == 0 &&
           "Invalid netlist! Expected 1-1 mapping of tile and core");
    cores[tileOp] = core;
  }
}

void xilinx::AIE::NetlistAnalysis::collectBuffers(
    DenseMap<Operation *, SmallVector<BufferOp, 4>> &buffers) {

  for (auto buffer : device.getOps<BufferOp>()) {
    Operation *tileOp = buffer.getTile().getDefiningOp();
    buffers[tileOp].push_back(buffer);
  }
}

void xilinx::AIE::NetlistAnalysis::collectDMAUsage() {
  for (auto map : mems) {
    MemOp mem = map.second;
    Region &r = mem.getBody();
    Block *endBlock = &r.back();
    for (auto op : r.getOps<cf::CondBranchOp>()) {
      auto dmaSt = dyn_cast<DMAStartOp>(op.getCondition().getDefiningOp());
      xilinx::AIE::DMAChannel dmaChan = {
          dmaSt.getChannelDir(), static_cast<int>(dmaSt.getChannelIndex())};
      dmas[{mem, dmaChan}] = dmaSt;
      Block *firstBd = op.getTrueDest();
      Block *curBd = firstBd;

      while (curBd != endBlock) {
        for (auto bdOp : curBd->getOps<DMABDOp>()) {
          Operation *buf = bdOp.getBuffer().getDefiningOp();
          if (std::find(dma2BufMap[dmaSt].begin(), dma2BufMap[dmaSt].end(),
                        buf) != dma2BufMap[dmaSt].end())
            continue;

          dma2BufMap[dmaSt].push_back(bdOp.getBuffer().getDefiningOp());
        }
        curBd = curBd->getSuccessors()[0];
      }
    }
  }
}

// FIXME: make address assignment for buffers explicit and move this function to
// an interface
uint64_t
xilinx::AIE::NetlistAnalysis::getBufferBaseAddress(Operation *bufOp) const {
  if (auto buf = dyn_cast<BufferOp>(bufOp)) {
    return buf.address();
  } else if (auto buf = dyn_cast<ExternalBufferOp>(bufOp)) {
    assert(false && "External buffer addresses are assigned at runtime.");
  }
  llvm_unreachable("unknown buffer type");
  return 0;
}

SmallVector<Operation *, 4> xilinx::AIE::NetlistAnalysis::getNextConnectOps(
    ConnectOp currentConnect) const {

  SmallVector<Operation *, 4> nextConnectOps;

  Operation *swboxOp = currentConnect->getParentOp();
  SwitchboxOp swbox = dyn_cast<SwitchboxOp>(swboxOp);
  int col = swbox.colIndex();
  int row = swbox.rowIndex();
  int nextCol = -1;
  int nextRow = -1;
  WireBundle nextSrcBundle;
  int nextSrcIndex = currentConnect.destIndex();

  if (currentConnect.getDestBundle() == WireBundle::South) {
    nextCol = col;
    nextRow = row - 1;
    nextSrcBundle = WireBundle::North;
  } else if (currentConnect.getDestBundle() == WireBundle::West) {
    nextCol = col - 1;
    nextRow = row;
    nextSrcBundle = WireBundle::East;
  } else if (currentConnect.getDestBundle() == WireBundle::North) {
    nextCol = col;
    nextRow = row + 1;
    nextSrcBundle = WireBundle::South;
  } else if (currentConnect.getDestBundle() == WireBundle::East) {
    nextCol = col + 1;
    nextRow = row;
    nextSrcBundle = WireBundle::West;
  } else {
    return nextConnectOps;
  }

  assert((nextCol >= 0 && nextRow >= 0) &&
         "Invalid ConnectOp! Could not find next tile!");

  Operation *nextTileOp = tiles[{nextCol, nextRow}];
  Operation *nextSwboxOp = switchboxes[nextTileOp];
  SwitchboxOp nextSwbox = dyn_cast<SwitchboxOp>(nextSwboxOp);

  for (auto connect : nextSwbox.getOps<ConnectOp>()) {
    if (connect.getSourceBundle() == nextSrcBundle &&
        connect.sourceIndex() == nextSrcIndex) {
      nextConnectOps.push_back(connect);
    }
  }

  return nextConnectOps;
}

SmallVector<Operation *, 4>
xilinx::AIE::NetlistAnalysis::findDestConnectOps(ConnectOp source,
                                                 WireBundle destBundle) const {

  SmallVector<Operation *, 4> dests;
  SmallVector<Operation *, 4> workList;
  workList.push_back(source);

  while (!workList.empty()) {
    ConnectOp visitor = dyn_cast<ConnectOp>(workList.pop_back_val());
    auto nextConnectOps(getNextConnectOps(visitor));
    for (auto nextConnectOp : nextConnectOps) {
      ConnectOp nextConnect = dyn_cast<ConnectOp>(nextConnectOp);
      if (nextConnect.getDestBundle() != destBundle)
        workList.push_back(nextConnect);
      else
        dests.push_back(nextConnect);
    }
  }

  return dests;
}

void xilinx::AIE::NetlistAnalysis::dmaAnalysis() {
  // Source(DMAChannel, <Buffer0, Buffer1, ...>) --> Dest(DMAChannel, <Buffer0,
  // Buffer1, ...>)
  collectDMAUsage();

  for (auto map : dma2BufMap) {
    Operation *srcDmaOp = map.first;
    DMAStartOp srcDma = dyn_cast<DMAStartOp>(srcDmaOp);
    if (srcDma.isRecv())
      continue;

    int srcChannelIndex = srcDma.getChannelIndex();

    Operation *srcMemOp = srcDmaOp->getParentOp();
    MemOp srcMem = dyn_cast<MemOp>(srcMemOp);
    SwitchboxOp swbox = switchboxes[srcMem.getTile().getDefiningOp()];
    for (auto connect : swbox.getOps<ConnectOp>()) {
      WireBundle srcBundle = connect.getSourceBundle();
      int srcIndex = connect.sourceIndex();
      if (!(srcBundle == WireBundle::DMA && srcIndex == srcChannelIndex))
        continue;

      dma2ConnectsMap[srcDmaOp].push_back(connect);

      auto destConnectOps(findDestConnectOps(connect, WireBundle::DMA));
      for (auto destConnectOp : destConnectOps) {
        ConnectOp destConnect = dyn_cast<ConnectOp>(destConnectOp);
        SwitchboxOp destSwbox =
            dyn_cast<SwitchboxOp>(destConnect->getParentOp());
        Operation *destMemOp = mems[destSwbox.getTile().getDefiningOp()];
        xilinx::AIE::DMAChannel dmaChan = {DMAChannelDir::S2MM,
                                           destConnect.destIndex()};
        Operation *destDmaOp = dmas[{destMemOp, dmaChan}];
        dmaConnections[srcDma].push_back(destDmaOp);
        dma2ConnectsMap[destDmaOp].push_back(destConnect);
      }
    }
  }
}
