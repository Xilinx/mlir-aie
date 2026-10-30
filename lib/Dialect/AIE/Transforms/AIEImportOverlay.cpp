//===- AIEImportOverlay.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"

using namespace mlir;
using namespace xilinx::AIE;

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEIMPORTOVERLAY
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

namespace {
struct AIEImportOverlayPass
    : xilinx::AIE::impl::AIEImportOverlayBase<AIEImportOverlayPass> {
  LogicalResult importDevice(DeviceOp device, DeviceOp overlay) {
    if (overlay == device || overlay.getOverlayAttr())
      return device.emitOpError(
          "requires an overlay without an overlay reference");
    if (device.getDevice() != overlay.getDevice())
      return device.emitOpError("and its overlay must use the same target");
    for (Operation &op : overlay.getBody()->without_terminator())
      if (!isa<TileOp, SwitchboxOp, ShimMuxOp, WireOp, ShimDMAAllocationOp>(op))
        return op.emitOpError("is not a routed overlay resource");

    IRMapping mapping;
    OpBuilder builder(device.getContext());
    for (TileOp tile : overlay.getOps<TileOp>()) {
      builder.setInsertionPointToStart(device.getBody());
      TileOp dest = TileOp::getOrCreate(builder, device, tile.colIndex(),
                                        tile.rowIndex());
      mapping.map(tile.getResult(), dest.getResult());
      if (auto id = tile->getAttr("controller_id")) {
        if (auto own = dest->getAttr("controller_id"); own && own != id)
          return dest.emitOpError(
              "has a different controller ID from its overlay");
        dest->setAttr("controller_id", id);
      }
    }
    auto equivalent = [&](Operation &op, Operation &existing) {
      return OperationEquivalence::isEquivalentTo(
          &op, &existing,
          [&](Value a, Value b) {
            return success(mapping.lookupOrDefault(a) == b);
          },
          nullptr,
          OperationEquivalence::IgnoreLocations |
              OperationEquivalence::IgnoreDiscardableAttrs);
    };
    auto inheritAttrs = [](Operation &source, Operation &dest) {
      for (NamedAttribute attr : source.getDiscardableAttrs())
        if (!dest.hasAttr(attr.getName()))
          dest.setAttr(attr.getName(), attr.getValue());
    };
    auto importOp = [&](Operation &op, Block &block) -> Operation * {
      for (Operation &existing : block.without_terminator()) {
        if (equivalent(op, existing)) {
          mapping.map(op.getResults(), existing.getResults());
          inheritAttrs(op, existing);
          return &existing;
        }
      }
      builder.setInsertionPoint(block.getTerminator());
      return builder.clone(op, mapping);
    };
    for (Operation &op : overlay.getBody()->without_terminator()) {
      if (isa<TileOp, WireOp>(op))
        continue;
      if (auto alloc = dyn_cast<ShimDMAAllocationOp>(op)) {
        auto existing =
            device.lookupSymbol<ShimDMAAllocationOp>(alloc.getSymName());
        if (existing) {
          if (!equivalent(op, *existing))
            return existing.emitOpError(
                "conflicts with its overlay's DMA allocation");
          inheritAttrs(op, *existing);
          continue;
        }
        importOp(op, *device.getBody());
        continue;
      }
      Value tile = mapping.lookup(op.getOperand(0));
      Operation *dest = nullptr;
      for (Operation &existing : device.getBody()->without_terminator())
        if (existing.getName() == op.getName() &&
            existing.getOperand(0) == tile) {
          dest = &existing;
          break;
        }
      if (!dest) {
        builder.setInsertionPoint(device.getBody()->getTerminator());
        dest = builder.cloneWithoutRegions(op, mapping);
        dest->getRegion(0).emplaceBlock();
        builder.setInsertionPointToEnd(&dest->getRegion(0).front());
        EndOp::create(builder, op.getLoc());
      }
      mapping.map(op.getResults(), dest->getResults());
      Block &block = dest->getRegion(0).front();
      for (Operation &setting : op.getRegion(0).front().without_terminator()) {
        auto rules = dyn_cast<PacketRulesOp>(setting);
        if (!rules) {
          importOp(setting, block);
          continue;
        }
        PacketRulesOp destRules;
        for (auto existing : block.getOps<PacketRulesOp>())
          if (existing.sourcePort() == rules.sourcePort())
            destRules = existing;
        if (!destRules) {
          builder.setInsertionPoint(block.getTerminator());
          destRules = PacketRulesOp::create(builder, rules.getLoc(),
                                            rules.getSourceBundle(),
                                            rules.getSourceChannel());
          PacketRulesOp::ensureTerminator(destRules.getRules(), builder,
                                          rules.getLoc());
        }
        if (destRules->hasAttr("aie.is_ctrl_overlay")) {
          for (auto rule : destRules.getRules().front().getOps<PacketRuleOp>())
            rule->setAttr("aie.is_ctrl_overlay", builder.getUnitAttr());
          destRules->removeAttr("aie.is_ctrl_overlay");
        }
        // Rule position is a hardware slot; importing cannot shift fixed rules.
        auto existing = destRules.getRules().front().getOps<PacketRuleOp>();
        auto it = existing.begin();
        for (auto rule : rules.getRules().front().getOps<PacketRuleOp>()) {
          if (it != existing.end()) {
            auto own = *it++;
            if (own.getMask() != rule.getMask() ||
                own.getValue() != rule.getValue() ||
                own.getAmsel() != mapping.lookup(rule.getAmsel()))
              return own.emitOpError(
                  "occupies a packet-rule slot required by its overlay");
            inheritAttrs(*rule, *own);
            if (rules->hasAttr("aie.is_ctrl_overlay"))
              own->setAttr("aie.is_ctrl_overlay", builder.getUnitAttr());
          } else {
            Operation *imported = importOp(*rule, destRules.getRules().front());
            if (rules->hasAttr("aie.is_ctrl_overlay"))
              imported->setAttr("aie.is_ctrl_overlay", builder.getUnitAttr());
          }
        }
      }
    }
    for (auto wire : overlay.getOps<WireOp>())
      importOp(*wire, *device.getBody());
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (DeviceOp device : module.getOps<DeviceOp>()) {
      auto ref = device.getOverlayAttr();
      if (!ref)
        continue;
      auto overlay = module.lookupSymbol<DeviceOp>(ref.getValue());
      if (!overlay) {
        device.emitOpError("references an unknown overlay device ") << ref;
        return signalPassFailure();
      }
      if (failed(importDevice(device, overlay)))
        return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIEImportOverlayPass() {
  return std::make_unique<AIEImportOverlayPass>();
}
