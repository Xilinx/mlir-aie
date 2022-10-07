//===- LoweringPatterns.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Transform/AIE/LoweringPatterns.h"

#include "phy/Connectivity/Serialization/LiteralVector.h"
#include "phy/Transform/AIE/Physical/BufferOp.h"
#include "phy/Transform/AIE/Physical/CoreOp.h"
#include "phy/Transform/AIE/Physical/LockOp.h"
#include "phy/Transform/AIE/Physical/StreamDmaOp.h"
#include "phy/Transform/AIE/Physical/StreamHubOp.h"
#include "phy/Transform/AIE/Physical/StreamOp.h"
#include "phy/Transform/AIE/TargetResources.h"

#include <map>
#include <utility>

#include "aie/AIEDialect.h"

using namespace mlir;
using namespace std;
using namespace xilinx;
using namespace xilinx::phy::connectivity;
using namespace xilinx::phy::transform;
using namespace xilinx::phy::transform::aie;

//===----------------------------------------------------------------------===//
// Lists of lowering pattern sets.  Each list is excuted as a pattern set, and
// all lists are executed sequentially.
//===----------------------------------------------------------------------===//

list<list<unique_ptr<LoweringPatternSet>>>
AIELoweringPatternSets::getPatternSets() {
  list<list<unique_ptr<LoweringPatternSet>>> patterns;

  // Convert code regions that use resources first
  patterns.emplace_back();
  patterns.back().push_back(make_unique<CoreOpLoweringPatternSet>(this));
  patterns.back().push_back(make_unique<StreamDmaOpLoweringPatternSet>(this));
  patterns.back().push_back(make_unique<StreamHubOpLoweringPatternSet>(this));

  // Then convert resources
  patterns.emplace_back();
  patterns.back().push_back(make_unique<BufferOpLoweringPatternSet>(this));
  patterns.back().push_back(make_unique<LockOpLoweringPatternSet>(this));
  patterns.back().push_back(make_unique<StreamOpLoweringPatternSet>(this));

  return patterns;
}

//===----------------------------------------------------------------------===//
// Shared resources constructors and getters.
//===----------------------------------------------------------------------===//

template <typename DMAOp>
static DMAOp getDmaGeneric(pair<int, int> index, mlir::ModuleOp module,
                           AIE::TileOp tile, map<pair<int, int>, DMAOp> &dmas) {

  if (!dmas.count(index)) {
    auto builder = OpBuilder::atBlockEnd(module.getBody());
    auto dma = dmas[index] = builder.create<DMAOp>(
        builder.getUnknownLoc(), builder.getIndexType(), tile);

    builder = OpBuilder::atBlockEnd(&dma.getBody().emplaceBlock());
    builder.create<AIE::EndOp>(builder.getUnknownLoc());
  }

  return dmas[index];
}

AIE::MemOp AIELoweringPatternSets::getDma(pair<int, int> index) {
  assert(!TargetResources().isShimTile(index.first, index.second));
  return getDmaGeneric<AIE::MemOp>(index, module, getTile(index), dmas);
}

xilinx::AIE::LockOp AIELoweringPatternSets::getLock(xilinx::AIE::TileOp tile,
                                                    int id) {
  auto pair = std::make_pair(tile, id);

  if (!locks.count(pair)) {
    auto builder = OpBuilder::atBlockBegin(module.getBody());
    builder.setInsertionPointAfter(tile);
    locks[pair] =
        builder.create<xilinx::AIE::LockOp>(builder.getUnknownLoc(), tile, id);
  }

  return locks[pair];
}

AIE::ShimDMAOp AIELoweringPatternSets::getShimDma(pair<int, int> index) {
  assert(TargetResources().isShimTile(index.first, index.second));
  return getDmaGeneric<AIE::ShimDMAOp>(index, module, getTile(index),
                                       shim_dmas);
}

AIE::TileOp AIELoweringPatternSets::getTile(mlir::OpState &op) {
  return getTile(getTileIndex(op));
}

AIE::TileOp AIELoweringPatternSets::getTile(pair<int, int> index) {
  if (!tiles.count(index)) {
    auto builder = OpBuilder::atBlockBegin(module.getBody());
    tiles[index] = builder.create<AIE::TileOp>(builder.getUnknownLoc(),
                                               index.first, index.second);
  }

  return tiles[index];
}

//===----------------------------------------------------------------------===//
// Common attribute getters.
//===----------------------------------------------------------------------===//

AIE::DMAChannel AIELoweringPatternSets::getChannel(mlir::OpState &op,
                                                   physical::StreamOp stream) {
  map<pair<string, int>, AIE::DMAChannel> engine_channels = {
      {{"S2MM", 0}, {AIE::DMAChannelDir::S2MM, 0}},
      {{"S2MM", 1}, {AIE::DMAChannelDir::S2MM, 1}},
      {{"MM2S", 0}, {AIE::DMAChannelDir::MM2S, 0}},
      {{"MM2S", 1}, {AIE::DMAChannelDir::MM2S, 1}}};

  map<pair<string, int>, AIE::DMAChannel> port_channels = {
      {{"DMA.I", 0}, {AIE::DMAChannelDir::S2MM, 0}},
      {{"DMA.I", 1}, {AIE::DMAChannelDir::S2MM, 1}},
      {{"DMA.O", 0}, {AIE::DMAChannelDir::MM2S, 0}},
      {{"DMA.O", 1}, {AIE::DMAChannelDir::MM2S, 1}}};

  auto engine =
      op.getOperation()->getAttrOfType<StringAttr>("aie.engine").str();
  auto engine_id = getId(op);
  auto engine_pair = make_pair(engine, engine_id);

  assert(engine_channels.count(engine_pair) && "unknown engine");

  auto port =
      stream.getOperation()->getAttrOfType<StringAttr>("aie.port").str();
  auto port_id = getId(stream);
  auto port_pair = make_pair(port, port_id);
  assert(port_channels.count(port_pair) && "unknown engine");

  assert(engine_channels[engine_pair] == port_channels[port_pair] &&
         "the dma engine cannot be connected to the given endpoint");

  return engine_channels[engine_pair];
}

int AIELoweringPatternSets::getId(mlir::OpState &op) {
  if (!op.getOperation()->getAttrOfType<StringAttr>("aie.id")) {
    assert(!op.getOperation()->hasAttr("aie.id") && "aie.id must be a string");
    return 0;
  }
  return lexicalCast<int>(
      op.getOperation()->getAttrOfType<StringAttr>("aie.id").str());
}

std::string AIELoweringPatternSets::getImpl(mlir::OpState &op) {
  if (!op.getOperation()->getAttrOfType<StringAttr>("aie.impl"))
    return "";
  return op.getOperation()->getAttrOfType<StringAttr>("aie.impl").str();
}

pair<int, int> AIELoweringPatternSets::getTileIndex(mlir::OpState &op) {
  assert(op.getOperation()->getAttrOfType<StringAttr>("aie.tile") &&
         "tile index must be specified");
  auto tile = LiteralVector<int>(
      op.getOperation()->getAttrOfType<StringAttr>("aie.tile").str());
  return make_pair(tile.vec()[0], tile.vec()[1]);
}

AIE::WireBundle AIELoweringPatternSets::getWireBundle(physical::StreamOp &op) {
  map<string, AIE::WireBundle> bundles = {
      {"Core.I", AIE::WireBundle::Core},   {"Core.O", AIE::WireBundle::Core},
      {"DMA.I", AIE::WireBundle::DMA},     {"DMA.O", AIE::WireBundle::DMA},
      {"FIFO.I", AIE::WireBundle::FIFO},   {"FIFO.O", AIE::WireBundle::FIFO},
      {"North.I", AIE::WireBundle::North}, {"North.O", AIE::WireBundle::North},
      {"East.I", AIE::WireBundle::East},   {"East.O", AIE::WireBundle::East}};
  // TODO: model south and west

  auto port = op.getOperation()->getAttrOfType<StringAttr>("aie.port").str();
  assert(bundles.count(port) && "unknown port");
  return bundles[port];
}
