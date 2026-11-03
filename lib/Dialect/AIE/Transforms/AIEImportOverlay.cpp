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
    IRMapping mapping;
    OpBuilder builder(device.getContext());
    auto equivalent = [&](Operation *source, Operation *dest) {
      IRMapping values(mapping);
      return OperationEquivalence::isEquivalentTo(
          source, dest,
          [&](Value a, Value b) {
            return success(values.lookupOrDefault(a) == b);
          },
          [&](Value a, Value b) { values.map(a, b); },
          OperationEquivalence::IgnoreLocations |
              OperationEquivalence::IgnoreDiscardableAttrs);
    };
    auto inherit = [&](Operation *source, Operation *dest) {
      mapping.map(source->getResults(), dest->getResults());
      for (NamedAttribute attr : source->getDiscardableAttrs())
        if (!dest->hasAttr(attr.getName()))
          dest->setAttr(attr.getName(), attr.getValue());
    };
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
    std::function<LogicalResult(Block &, Block &)> importBlock =
        [&](Block &source, Block &dest) -> LogicalResult {
      auto slot = dest.begin();
      for (Operation &op : source.without_terminator()) {
        if (isa<TileOp>(op))
          continue;
        // Rule order fixes both hardware slots and first-match precedence.
        if (isa<PacketRuleOp>(op) && slot != dest.end() && !isa<EndOp>(*slot)) {
          if (!equivalent(&op, &*slot))
            return slot->emitOpError(
                "occupies a packet-rule slot required by its overlay");
          inherit(&op, &*slot++);
          continue;
        }
        bool container = isa<SwitchboxOp, ShimMuxOp, PacketRulesOp>(op);
        Operation *match = nullptr;
        if (container) {
          Operation *shape = op.cloneWithoutRegions();
          for (Operation &candidate : dest.without_terminator()) {
            if (candidate.getName() != op.getName())
              continue;
            Operation *other = candidate.cloneWithoutRegions();
            bool same = equivalent(shape, other);
            other->destroy();
            if (same) {
              match = &candidate;
              break;
            }
          }
          shape->destroy();
        } else {
          for (Operation &candidate : dest.without_terminator()) {
            if (equivalent(&op, &candidate)) {
              match = &candidate;
              break;
            }
            auto name =
                op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
            if (name && name == candidate.getAttrOfType<StringAttr>(
                                    SymbolTable::getSymbolAttrName()))
              return candidate.emitOpError("conflicts with an overlay symbol ")
                     << name;
          }
        }
        if (match) {
          if (container) {
            mapping.map(op.getResults(), match->getResults());
            if (failed(importBlock(op.getRegion(0).front(),
                                   match->getRegion(0).front())))
              return failure();
          } else {
            inherit(&op, match);
          }
        } else {
          builder.setInsertionPoint(dest.getTerminator());
          builder.clone(op, mapping);
        }
      }
      return success();
    };
    if (failed(importBlock(*overlay.getBody(), *device.getBody())))
      return failure();
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
