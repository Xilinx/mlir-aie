//===- BroadcastPacket.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Transform/AIE/Physical/Implementation/BroadcastPacket.h"

#include "phy/Dialect/Physical/PhysicalDialect.h"

#include <set>

#include "aie/AIEDialect.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::phy::physical;
using namespace xilinx::phy::transform::aie;

static std::set<int> getTags(StreamOp &stream);

LogicalResult BroadcastPacketLowering::matchAndRewrite(
    StreamHubOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  if (lowering->getImpl(op) != "broadcast_packet")
    return failure();
  rewriter.setInsertionPointAfter(op);

  // For each broadcast source
  for (auto src_value : op.getEndpoints()) {

    // skip output streams
    if (!src_value.getType().isa<IStreamType>())
      continue;

    auto src = dyn_cast<StreamOp>(src_value.getDefiningOp());
    auto tile = lowering->getTile(src);
    auto bundle = lowering->getWireBundle(src);
    auto bundle_attr = AIE::WireBundleAttr::get(rewriter.getContext(), bundle);
    auto id = lowering->getId(src);
    auto id_attr = rewriter.getI32IntegerAttr(id);

    // AIE.broadcast_packet(%tile, bundle : id) {
    auto broadcast = rewriter.create<AIE::BroadcastPacketOp>(
        rewriter.getUnknownLoc(), tile, bundle_attr, id_attr);
    auto broadcast_builder =
        OpBuilder::atBlockEnd(&broadcast.getPorts().emplaceBlock());
    buildBroadcastPacket(broadcast_builder, op, src);
    // }
  }

  rewriter.eraseOp(op);
  return success();
}

void BroadcastPacketLowering::buildBroadcastPacket(OpBuilder &builder,
                                                   StreamHubOp op,
                                                   StreamOp &src) const {

  assert(src.getTags().has_value() && "broadcast packet requires tags");

  // For each tag
  for (auto src_tag : getTags(src)) {
    auto bp_id_attr = builder.getI8IntegerAttr(src_tag);

    // AIE.bp_id(src_tag) {
    auto bp_id =
        builder.create<AIE::BPIDOp>(builder.getUnknownLoc(), bp_id_attr);
    auto bp_id_builder =
        OpBuilder::atBlockEnd(&bp_id.getPorts().emplaceBlock());
    buildBpId(bp_id_builder, op, src, src_tag);
    // }
  }

  // AIE.end
  builder.create<AIE::EndOp>(builder.getUnknownLoc());
}

void BroadcastPacketLowering::buildBpId(OpBuilder &builder, StreamHubOp op,
                                        StreamOp &src, int tag) const {

  // For each broadcast destination
  for (auto dest_value : op.getEndpoints()) {

    // skip input streams
    if (!dest_value.getType().isa<OStreamType>())
      continue;

    auto dest = dyn_cast<StreamOp>(dest_value.getDefiningOp());
    auto tile = lowering->getTile(dest);
    auto bundle = lowering->getWireBundle(dest);
    auto bundle_attr = AIE::WireBundleAttr::get(builder.getContext(), bundle);
    auto id = lowering->getId(dest);
    auto id_attr = builder.getI32IntegerAttr(id);

    assert(dest.getTags().has_value() && "broadcast packet requires tags");

    // skip destinations that do not accept the tag
    if (!getTags(dest).count(tag))
      continue;

    // AIE.bp_dest<%tile, bundle : id>
    builder.create<AIE::BPDestOp>(builder.getUnknownLoc(), tile, bundle_attr,
                                  id_attr);
  }

  // AIE.end
  builder.create<AIE::EndOp>(builder.getUnknownLoc());
}

static std::set<int> getTags(StreamOp &stream) {
  std::set<int> result;

  for (auto tag_attr : stream.getTags().value()) {
    result.insert(tag_attr.dyn_cast<IntegerAttr>().getValue().getSExtValue());
  }

  return result;
}