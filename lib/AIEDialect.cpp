//===- AIEDialect.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/AIEDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;

namespace {

struct AIEInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  // We don't have any special restrictions on what can be inlined into
  // destination regions. Always allow it.
  bool
  isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                  BlockAndValueMapping &valueMapping) const final override {
    return true;
  }
  // Operations in aie dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation *op, Region *, bool wouldBeCloned,
                       BlockAndValueMapping &) const final override {
    return true;
  }
  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the inlined region has more than one block.
  void handleTerminator(Operation *op, Block *newDest) const final override {
    return;
  }
  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the region has only one block.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final override {
    return;
  }
};

struct AIEDialectFoldInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  /// Registered hook to check if the given region, which is attached to an
  /// operation that is *not* isolated from above, should be used when
  /// materializing constants.
  bool shouldMaterializeInto(Region *region) const final {
    // If this is an AIE::CoreOp region, then insert into it.
    return isa<xilinx::AIE::CoreOp>(region->getParentOp());
  }
};

} // end anonymous namespace

namespace xilinx {
namespace AIE {

bool isValidTile(TileID src) {
  // FIXME: what about upper bound?
  return src.first >= 0 && src.second >= 0;
}
// Return the tile ID of the memory to the west of the given tile, if it exists.
Optional<TileID> getMemWest(TileID src) {
  bool isEvenRow = ((src.second % 2) == 0);
  Optional<TileID> ret;
  if (isEvenRow)
    ret = src;
  else
    ret = std::make_pair(src.first - 1, src.second);
  if (!isValidTile(ret.getValue()))
    ret.reset();
  return ret;
}
// Return the tile ID of the memory to the west of the given tile, if it exists.
Optional<TileID> getMemEast(TileID src) {
  bool isEvenRow = ((src.second % 2) == 0);
  Optional<TileID> ret;
  if (isEvenRow)
    ret = std::make_pair(src.first + 1, src.second);
  else
    ret = src;
  if (!isValidTile(ret.getValue()))
    ret.reset();
  return ret;
}
// Return the tile ID of the memory to the west of the given tile, if it exists.
Optional<TileID> getMemNorth(TileID src) {
  Optional<TileID> ret = std::make_pair(src.first, src.second + 1);
  if (!isValidTile(ret.getValue()))
    ret.reset();
  return ret;
}
Optional<TileID> getMemSouth(TileID src) {
  Optional<TileID> ret = std::make_pair(src.first, src.second - 1);
  if (!isValidTile(ret.getValue()))
    ret.reset();
  return ret;
}

bool isInternal(int srcCol, int srcRow, int dstCol, int dstRow) {
  return ((srcCol == dstCol) && (srcRow == dstRow));
}

bool isWest(int srcCol, int srcRow, int dstCol, int dstRow) {
  return ((srcCol == dstCol + 1) && (srcRow == dstRow));
}

bool isMemWest(int srcCol, int srcRow, int dstCol, int dstRow) {
  bool IsEvenRow = ((srcRow % 2) == 0);
  return (IsEvenRow && isInternal(srcCol, srcRow, dstCol, dstRow)) ||
         (!IsEvenRow && isWest(srcCol, srcRow, dstCol, dstRow));
}

bool isEast(int srcCol, int srcRow, int dstCol, int dstRow) {
  return ((srcCol == dstCol - 1) && (srcRow == dstRow));
}

bool isMemEast(int srcCol, int srcRow, int dstCol, int dstRow) {
  bool IsEvenRow = ((srcRow % 2) == 0);
  return (!IsEvenRow && isInternal(srcCol, srcRow, dstCol, dstRow)) ||
         (IsEvenRow && isEast(srcCol, srcRow, dstCol, dstRow));
}

bool isNorth(int srcCol, int srcRow, int dstCol, int dstRow) {
  return ((srcCol == dstCol) && (srcRow == dstRow - 1));
}

bool isMemNorth(int srcCol, int srcRow, int dstCol, int dstRow) {
  return isNorth(srcCol, srcRow, dstCol, dstRow);
}

bool isSouth(int srcCol, int srcRow, int dstCol, int dstRow) {
  return ((srcCol == dstCol) && (srcRow == dstRow + 1));
}

bool isMemSouth(int srcCol, int srcRow, int dstCol, int dstRow) {
  return isSouth(srcCol, srcRow, dstCol, dstRow);
}

bool isLegalMemAffinity(int coreCol, int coreRow, int memCol, int memRow) {
  bool IsEvenRow = ((coreRow % 2) == 0);

  bool IsMemWest = (isWest(coreCol, coreRow, memCol, memRow) && !IsEvenRow) ||
                   (isInternal(coreCol, coreRow, memCol, memRow) && IsEvenRow);

  bool IsMemEast = (isEast(coreCol, coreRow, memCol, memRow) && IsEvenRow) ||
                   (isInternal(coreCol, coreRow, memCol, memRow) && !IsEvenRow);

  bool IsMemNorth = isNorth(coreCol, coreRow, memCol, memRow);
  bool IsMemSouth = isSouth(coreCol, coreRow, memCol, memRow);

  return IsMemSouth || IsMemNorth || IsMemWest || IsMemEast;
}

// FIXME: use Tablegen'd dialect class
AIEDialect::AIEDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect("AIE", ctx, ::mlir::TypeID::get<AIEDialect>()) {
  // addTypes<AIEListType>();
  addOperations<
#define GET_OP_LIST
#include "aie/AIE.cpp.inc"
      >();
  addInterfaces<AIEInlinerInterface, AIEDialectFoldInterface>();
}

} // namespace AIE
} // namespace xilinx

static LogicalResult verify(xilinx::AIE::TileOp op) {
  auto users = op.result().getUsers();
  bool found = false;
  for (auto user : users) {
    if (llvm::isa<xilinx::AIE::SwitchboxOp>(*user)) {
      if (found)
        return op.emitError("Tile can only have one switchbox");
      found = true;
    }
  }

  return success();
}

static LogicalResult verify(xilinx::AIE::SwitchboxOp op) {
  Region &body = op.connections();
  DenseSet<xilinx::AIE::Port> sourceset;
  DenseSet<xilinx::AIE::Port> destset;
  assert(op.getOperation()->getNumRegions());
  assert(!body.empty());
  for (auto &ops : body.front()) {
    if (auto connectOp = dyn_cast<xilinx::AIE::ConnectOp>(ops)) {
      xilinx::AIE::Port source =
          std::make_pair(connectOp.sourceBundle(), connectOp.sourceIndex());
      sourceset.insert(source);

      xilinx::AIE::Port dest =
          std::make_pair(connectOp.destBundle(), connectOp.destIndex());
      if (destset.count(dest)) {
        return connectOp.emitOpError("targets same destination ")
               << stringifyWireBundle(dest.first) << dest.second
               << " as another connect operation";
      } else {
        destset.insert(dest);
      }
      if (connectOp.sourceIndex() < 0) {
        connectOp.emitOpError("source index cannot be less than zero");
      }
      if (connectOp.sourceIndex() >=
          op.getNumSourceConnections(connectOp.sourceBundle())) {
        connectOp.emitOpError("source index for source bundle ")
            << stringifyWireBundle(connectOp.sourceBundle())
            << " must be less than "
            << op.getNumSourceConnections(connectOp.sourceBundle());
      }
      if (connectOp.destIndex() < 0) {
        connectOp.emitOpError("dest index cannot be less than zero");
      }
      if (connectOp.destIndex() >=
          op.getNumDestConnections(connectOp.destBundle())) {
        connectOp.emitOpError("dest index for dest bundle ")
            << stringifyWireBundle(connectOp.destBundle())
            << " must be less than "
            << op.getNumDestConnections(connectOp.destBundle());
      }
    } else if (auto connectOp = dyn_cast<xilinx::AIE::MasterSetOp>(ops)) {
      xilinx::AIE::Port dest =
          std::make_pair(connectOp.destBundle(), connectOp.destIndex());
      if (destset.count(dest)) {
        return connectOp.emitOpError("targets same destination ")
               << stringifyWireBundle(dest.first) << dest.second
               << " as another connect or masterset operation";
      } else {
        destset.insert(dest);
      }
      if (connectOp.destIndex() < 0) {
        connectOp.emitOpError("dest index cannot be less than zero");
      }
      if (connectOp.destIndex() >=
          op.getNumDestConnections(connectOp.destBundle())) {
        connectOp.emitOpError("dest index for dest bundle ")
            << stringifyWireBundle(connectOp.destBundle())
            << " must be less than "
            << op.getNumDestConnections(connectOp.destBundle());
      }

      int arbiter = -1;
      for (auto val : connectOp.amsels()) {
        auto amsel = dyn_cast<xilinx::AIE::AMSelOp>(val.getDefiningOp());
        if ((arbiter != -1) && (arbiter != amsel.arbiterIndex()))
          connectOp.emitOpError(
              "a master port can only be tied to one arbiter");
        arbiter = amsel.arbiterIndex();
      }
    } else if (auto connectOp = dyn_cast<xilinx::AIE::PacketRulesOp>(ops)) {
      xilinx::AIE::Port source =
          std::make_pair(connectOp.sourceBundle(), connectOp.sourceIndex());
      if (sourceset.count(source)) {
        return connectOp.emitOpError("packet switched source ")
               << stringifyWireBundle(source.first) << source.second
               << " cannot match another connect or masterset operation";
      } else {
        sourceset.insert(source);
      }
    } else if (auto amselOp = dyn_cast<xilinx::AIE::AMSelOp>(ops)) {
    } else if (auto endswitchOp = dyn_cast<xilinx::AIE::EndOp>(ops)) {
    } else {
      return ops.emitOpError("cannot be contained in a Switchbox op");
    }
  }

  return success();
}

static LogicalResult verify(xilinx::AIE::ShimSwitchboxOp op) {
  Region &body = op.connections();
  DenseSet<xilinx::AIE::Port> destset;
  assert(op.getOperation()->getNumRegions());
  assert(!body.empty());

  for (auto &ops : body.front()) {
    if (auto connectOp = dyn_cast<xilinx::AIE::ConnectOp>(ops)) {
      xilinx::AIE::Port dest =
          std::make_pair(connectOp.destBundle(), connectOp.destIndex());
      if (destset.count(dest)) {
        return connectOp.emitOpError("targets same destination ")
               << stringifyWireBundle(dest.first) << dest.second
               << " as another connect operation";
      } else {
        destset.insert(dest);
      }
    } else if (auto endswitchOp = dyn_cast<xilinx::AIE::EndOp>(ops)) {
    } else {
      return ops.emitOpError("cannot be contained in a Switchbox op");
    }
  }

  return success();
}

static LogicalResult verify(xilinx::AIE::ShimMuxOp op) {
  Region &body = op.connections();
  DenseSet<xilinx::AIE::Port> destset;
  assert(op.getOperation()->getNumRegions());
  assert(!body.empty());

  auto tileOp = op.getTileOp();
  if (!tileOp.isShimNOCTile())
    return op.emitOpError("must be in a ShimTile with a NOC connection");

  for (auto &ops : body.front()) {
    if (auto connectOp = dyn_cast<xilinx::AIE::ConnectOp>(ops)) {
      xilinx::AIE::Port dest =
          std::make_pair(connectOp.destBundle(), connectOp.destIndex());
      if (destset.count(dest)) {
        return connectOp.emitOpError("targets same destination ")
               << stringifyWireBundle(dest.first) << dest.second
               << " as another connect operation";
      } else {
        destset.insert(dest);
      }
    } else if (auto endswitchOp = dyn_cast<xilinx::AIE::EndOp>(ops)) {
    } else {
      return ops.emitOpError("cannot be contained in a Switchbox op");
    }
  }
  if (!op.getTileOp().isShimTile()) {
    return op.emitOpError("must be in a shim tile, i.e. row == 0.");
  }
  return success();
}

static LogicalResult verify(xilinx::AIE::UseTokenOp op) {
  auto parentOp = op->getParentOp();
  if (isa<FuncOp>(parentOp) || isa<xilinx::AIE::CoreOp>(parentOp) ||
      isa<xilinx::AIE::MemOp>(parentOp) ||
      isa<xilinx::AIE::ShimDMAOp>(parentOp))
    return success();
  return failure();
}

int xilinx::AIE::ShimMuxOp::getNumSourceConnections(WireBundle bundle) {
  switch (bundle) {
  case WireBundle::DMA:
    return 2;
  case WireBundle::NOC:
    return 4;
  case WireBundle::PLIO:
    return 6;
  case WireBundle::South:
    return 6;
  default:
    return 0;
  }
}
int xilinx::AIE::ShimMuxOp::getNumDestConnections(WireBundle bundle) {
  switch (bundle) {
  case WireBundle::DMA:
    return 2;
  case WireBundle::NOC:
    return 4;
  case WireBundle::PLIO:
    return 6;
  case WireBundle::South:
    return 8;
  default:
    return 0;
  }
}
xilinx::AIE::TileOp xilinx::AIE::ShimMuxOp::getTileOp() {
  return cast<xilinx::AIE::TileOp>(tile().getDefiningOp());
}
int xilinx::AIE::ShimMuxOp::colIndex() { return getTileOp().colIndex(); }
int xilinx::AIE::ShimMuxOp::rowIndex() { return getTileOp().rowIndex(); }

// ShimDMAOp
static LogicalResult verify(xilinx::AIE::ShimDMAOp op) {
  assert(op.getOperation()->getNumRegions() == 1 &&
         "ShimDMAOp has zero region!");
  assert(!op.body().empty() && "ShimDMAOp should have non-empty body");
  auto tileOp = op.getTileOp();
  if (!tileOp.isShimNOCTile())
    return op.emitOpError("must be in a ShimTile with a NOC connection");

  return success();
}
xilinx::AIE::TileOp xilinx::AIE::ShimDMAOp::getTileOp() {
  return cast<TileOp>(tile().getDefiningOp());
}
int xilinx::AIE::ShimDMAOp::colIndex() { return getTileOp().colIndex(); }
int xilinx::AIE::ShimDMAOp::rowIndex() { return getTileOp().rowIndex(); }

static LogicalResult verify(xilinx::AIE::PacketFlowOp op) {
  Region &body = op.ports();
  // DenseSet<xilinx::AIE::Port> destset;
  assert(op.getOperation()->getNumRegions());
  assert(!body.empty());
  for (auto &ops : body.front()) {
    if (auto Op = dyn_cast<xilinx::AIE::PacketSourceOp>(ops)) {
    } else if (auto Op = dyn_cast<xilinx::AIE::PacketDestOp>(ops)) {
    } else if (auto endswitchOp = dyn_cast<xilinx::AIE::EndOp>(ops)) {
    } else {
      return ops.emitOpError("cannot be contained in a PacketFlow op");
    }
  }

  return success();
}

// CoreOp
static LogicalResult verify(xilinx::AIE::CoreOp op) {
  assert(op.getOperation()->getNumRegions() == 1 && "CoreOp has zero region!");
  assert(!op.body().empty() && "CoreOp should have non-empty body");

  return success();
}

int xilinx::AIE::CoreOp::colIndex() { return getTileOp().colIndex(); }

int xilinx::AIE::CoreOp::rowIndex() { return getTileOp().rowIndex(); }
xilinx::AIE::TileOp xilinx::AIE::CoreOp::getTileOp() {
  return cast<xilinx::AIE::TileOp>(tile().getDefiningOp());
}

// BufferOp
int64_t xilinx::AIE::BufferOp::getAllocationSize() {
  MemRefType type = getType().cast<MemRefType>();
  return type.getNumElements() * type.getElementTypeBitWidth() / 8;
}
xilinx::AIE::TileOp xilinx::AIE::BufferOp::getTileOp() {
  return cast<xilinx::AIE::TileOp>(tile().getDefiningOp());
}

// MemOp
static LogicalResult verify(xilinx::AIE::MemOp op) {
  Region &body = op.body();
  llvm::SmallSet<xilinx::AIE::DMAChan, 4> used_channels;

  assert(op.getOperation()->getNumRegions() == 1 && "MemOp has zero region!");
  assert(!body.empty() && "MemOp should have non-empty body");

  for (auto &bodyOp : body.getOps()) {
    // check for duplicate DMA channels within the same MemOp
    if (auto DMA_start = dyn_cast<xilinx::AIE::DMAStartOp>(bodyOp)) {
      auto DMA_chan = DMA_start.dmaChan();
      if (used_channels.contains(DMA_chan))
        DMA_start.emitOpError()
            << "Duplicate DMA channel " << stringifyDMAChan(DMA_chan)
            << " detected in MemOp!";
      used_channels.insert(DMA_chan);
    }

    if (auto allocOp = dyn_cast<memref::AllocOp>(bodyOp)) {
      if (!allocOp->getAttr("id"))
        op.emitOpError()
            << "allocOp in MemOp region should have an id attribute\n";
    }
  }

  return success();
}

int xilinx::AIE::MemOp::colIndex() {
  Operation *Op = tile().getDefiningOp();
  xilinx::AIE::TileOp tile = dyn_cast<xilinx::AIE::TileOp>(Op);

  return tile.colIndex();
}

int xilinx::AIE::MemOp::rowIndex() {
  Operation *Op = tile().getDefiningOp();
  xilinx::AIE::TileOp tile = dyn_cast<xilinx::AIE::TileOp>(Op);

  return tile.rowIndex();
}

/// Returns the region on the current operation that is callable. This may
/// return null in the case of an external callable object, e.g. an external
/// function.
Region *xilinx::AIE::MemOp::getCallableRegion() { return &(body()); }

/// Returns the results types that the callable region produces when executed.
ArrayRef<Type> xilinx::AIE::MemOp::getCallableResults() { return getType(); }

xilinx::AIE::TileOp xilinx::AIE::SwitchboxOp::getTileOp() {
  return cast<xilinx::AIE::TileOp>(tile().getDefiningOp());
}
int xilinx::AIE::SwitchboxOp::colIndex() { return getTileOp().colIndex(); }
int xilinx::AIE::SwitchboxOp::rowIndex() { return getTileOp().rowIndex(); }

template <typename... ParentOpTypes> struct HasSomeParent {
  static LogicalResult verifyTrait(Operation *op) {
    Operation *operation = op;
    while (operation) {
      if (llvm::isa<ParentOpTypes...>(operation->getParentOp()))
        return success();
      operation = operation->getParentOp();
    }
    return op->emitOpError()
           << "expects some parent op "
           << (sizeof...(ParentOpTypes) != 1 ? "to be one of '" : "'")
           << llvm::makeArrayRef({ParentOpTypes::getOperationName()...}) << "'";
  }
};

static LogicalResult verify(xilinx::AIE::UseLockOp op) {
  return HasSomeParent<xilinx::AIE::CoreOp, xilinx::AIE::MemOp,
                       xilinx::AIE::ShimDMAOp>::verifyTrait(op);

  //  xilinx::AIE::LockOp lockOp =
  //  dyn_cast_or_null<xilinx::AIE::LockOp>(op.lock().getDefiningOp());
  //   if (!lockOp) {
  //     op.emitOpError() << "Expected LockOp!\n";
  // //    return failure();
  //   }

  return success();
}

#include "aie/AIEEnums.cpp.inc"
#include "aie/AIEInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "aie/AIE.cpp.inc"

namespace xilinx {
namespace AIE {

// void CoreOp::build(Builder *odsBuilder, OperationState &odsState, Type
// resultType0, int col, int row) {
//   odsState.addOperands(colValue);
//   odsState.addOperands(rowValue);
//   odsState.addTypes(resultType0);
// }

//#include "ATenOpInterfaces.cpp.inc"

int SwitchboxOp::getNumSourceConnections(WireBundle bundle) {
  if (getTileOp().isShimTile())
    switch (bundle) {
    // case WireBundle::Core: return 0;
    // case WireBundle::DMA: return 2;
    // case WireBundle::PLIO: return 4;
    case WireBundle::FIFO:
      return 2;
    case WireBundle::North:
      return 4;
    case WireBundle::West:
      return 4;
    case WireBundle::South:
      return 8;
    case WireBundle::East:
      return 4;
    case WireBundle::Trace:
      return 1;
    default:
      return 0;
    }
  else
    switch (bundle) {
    case WireBundle::Core:
      return 2;
    case WireBundle::DMA:
      return 2;
    case WireBundle::FIFO:
      return 2;
    case WireBundle::North:
      return 4;
    case WireBundle::West:
      return 4;
    case WireBundle::South:
      return 6;
    case WireBundle::East:
      return 4;
    case WireBundle::Trace:
      return 2;
    default:
      return 0;
    }
}
int SwitchboxOp::getNumDestConnections(WireBundle bundle) {
  if (getTileOp().isShimTile())
    switch (bundle) {
    // case WireBundle::Core: return 0;
    // case WireBundle::DMA: return 2;
    // case WireBundle::PLIO: return 2;
    case WireBundle::FIFO:
      return 2;
    case WireBundle::North:
      return 6;
    case WireBundle::West:
      return 4;
    case WireBundle::South:
      return 6;
    case WireBundle::East:
      return 4;
    default:
      return 0;
    }
  else
    switch (bundle) {
    case WireBundle::Core:
      return 2;
    case WireBundle::DMA:
      return 2;
    case WireBundle::FIFO:
      return 2;
    case WireBundle::North:
      return 6;
    case WireBundle::West:
      return 4;
    case WireBundle::South:
      return 4;
    case WireBundle::East:
      return 4;
    default:
      return 0;
    }
}
int TileOp::getNumSourceConnections(WireBundle bundle) {
  switch (bundle) {
  case WireBundle::Core:
    return 2;
  case WireBundle::DMA:
    return 2;
  default:
    return 0;
  }
}
int TileOp::getNumDestConnections(WireBundle bundle) {
  switch (bundle) {
  case WireBundle::Core:
    return 2;
  case WireBundle::DMA:
    return 2;
  default:
    return 0;
  }
}
static llvm::SmallDenseSet<unsigned, 16> noc_columns = {
    2, 3, 6, 7, 10, 11, 18, 19, 26, 27, 34, 35, 42, 43, 46, 47};
bool TileOp::isShimNOCTile() {
  return isShimTile() && noc_columns.contains(col());
}
bool TileOp::isShimPLTile() { return isShimNOCorPLTile() && !isShimNOCTile(); }
bool TileOp::isShimNOCorPLTile() { return isShimTile() && (col() > 0); }
} // namespace AIE
} // namespace xilinx
