//===- AIEDialect.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

// Add TableGen'erated dialect definitions (including constructor)
// We implement the initialize() function further below
#include "aie/Dialect/AIE/IR/AIEDialect.cpp.inc"

namespace {

struct AIEInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  // We don't have any special restrictions on what can be inlined into
  // destination regions. Always allow it.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final override {
    return true;
  }
  // Operations in aie dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation *op, Region *, bool wouldBeCloned,
                       IRMapping &) const final override {
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

static xilinx::AIE::VC1902TargetModel VC1902model;
static xilinx::AIE::VE2302TargetModel VE2302model;
static xilinx::AIE::VE2802TargetModel VE2802model;

const xilinx::AIE::AIETargetModel &getTargetModel(Operation *op) {
  if (auto t = dyn_cast<xilinx::AIE::AIETarget>(op))
    return t.getTargetModel();
  if (auto t = op->getParentOfType<xilinx::AIE::AIETarget>())
    return t.getTargetModel();

  // For backward compatibility, return a basic device model compatible with
  // the VCK190
  return VC1902model;
}

// Walk the operation hierarchy until we find a containing TileElement.
// If no parent is a TileElement, then return null.
static xilinx::AIE::TileElement getParentTileElement(Operation *op) {
  auto parent = op->getParentOp();
  while (
      !llvm::isa_and_nonnull<xilinx::AIE::DeviceOp, mlir::ModuleOp>(parent)) {
    if (auto element = llvm::dyn_cast<xilinx::AIE::TileElement>(parent))
      return element;
    parent = parent->getParentOp();
  }
  return llvm::dyn_cast<xilinx::AIE::TileElement>(parent);
}

struct UsesAreAccessable {
  static LogicalResult verifyTrait(Operation *op) {
    auto thisElement = cast<xilinx::AIE::TileElement>(op);
    auto thisID = thisElement.getTileID();
    auto users = op->getResult(0).getUsers();
    const auto &target_model = getTargetModel(op);
    for (auto user : users) {
      // AIE.useLock may be used in a device to set the lock's default value
      // Allow in a toplevel module for backward compatibility
      if (llvm::isa_and_nonnull<xilinx::AIE::DeviceOp, mlir::ModuleOp>(
              user->getParentOp()))
        return success();
      if (auto element = getParentTileElement(user)) {

        auto tileID = element.getTileID();
        if (!target_model.isLegalMemAffinity(tileID.first, tileID.second,
                                             thisID.first, thisID.second))
          return (op->emitOpError("in Column ")
                  << thisID.first << " and Row " << thisID.second
                  << " is accessed from an unreachable tile in Column "
                  << tileID.first << " and Row " << tileID.second)
                     .attachNote(user->getLoc())
                 << "user";
      } else {
        // This should probably be caught elsewhere as well.
        return op->emitOpError("is accessed outside of a tile")
                   .attachNote(user->getLoc())
               << "user";
      }
    }
    return success();
  }
};

namespace detail {
/// This class represents the internal storage of the AIE `ObjectFifoType`.
struct AIEObjectFifoTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage.
  using KeyTy = mlir::Type;

  /// A constructor for the objectFifo type storage instance.
  AIEObjectFifoTypeStorage(mlir::Type elementType) : elementType(elementType) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == KeyTy(elementType); }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`.
  static AIEObjectFifoTypeStorage *
  construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    // Allocate the storage instance and construct it.
    return new (allocator.allocate<AIEObjectFifoTypeStorage>())
        AIEObjectFifoTypeStorage(key);
  }

  mlir::Type elementType;
};
} // namespace detail

AIEObjectFifoType AIEObjectFifoType::get(mlir::Type elementType) {
  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type.
  mlir::MLIRContext *ctx = elementType.getContext();
  return Base::get(ctx, elementType);
}

LogicalResult
AIEObjectFifoType::verify(function_ref<InFlightDiagnostic()> emitError,
                          mlir::Type elementType) {
  // Memref element type expected.
  if (!elementType.isa<MemRefType>())
    return emitError() << "non memref-type passed to 'ObjectFifoType'";
  return success();
}

mlir::Type AIEObjectFifoType::getElementType() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->elementType;
}

namespace detail {
/// This class represents the internal storage of the AIE
/// `ObjectFifoSubviewType`.
struct AIEObjectFifoSubviewTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage.
  using KeyTy = mlir::Type;

  /// A constructor for the subview type storage instance.
  AIEObjectFifoSubviewTypeStorage(mlir::Type elementType)
      : elementType(elementType) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == elementType; }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`.
  static AIEObjectFifoSubviewTypeStorage *
  construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    // Allocate the storage instance and construct it.
    return new (allocator.allocate<AIEObjectFifoSubviewTypeStorage>())
        AIEObjectFifoSubviewTypeStorage(key);
  }

  mlir::Type elementType;
};
} // namespace detail

AIEObjectFifoSubviewType AIEObjectFifoSubviewType::get(mlir::Type elementType) {
  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type.
  mlir::MLIRContext *ctx = elementType.getContext();
  return Base::get(ctx, elementType);
}

/// This method is used to verify the construction invariants.
LogicalResult
AIEObjectFifoSubviewType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 mlir::Type elementType) {
  // Memref element type expected.
  if (!elementType.isa<MemRefType>())
    return emitError() << "non memref-type passed to 'ObjectFifoSubviewType'";
  return success();
}

mlir::Type AIEObjectFifoSubviewType::getElementType() {
  return getImpl()->elementType;
}

/// Parse an instance of a type registered to the AIE dialect.
/// Parse an AIE type in the following forms:
///   AIE-type
///         ::= `objectFifo` `<` type `>`
///         ::= `objectFifoSubview` `<` type `>`
static OptionalParseResult aieTypeParser(MLIRContext *context,
                                         DialectAsmParser &parser,
                                         StringRef name, Type &result) {
  if (name.equals("objectFifo")) {
    mlir::Type elementType;
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    if (parser.parseLess() || parser.parseType(elementType) ||
        parser.parseGreater())
      return failure();

    // Check that the type is a MemRef type.
    if (!elementType.isa<mlir::MemRefType>()) {
      parser.emitError(typeLoc, "element type for an objectFifo must be "
                                "a MemRefType, got: ")
          << elementType;
      return failure();
    }

    return result = AIEObjectFifoType::get(elementType), success();
  }

  if (name.equals("objectFifoSubview")) {
    if (parser.parseLess())
      return failure();

    // Parse the element type of the struct.
    mlir::Type elementType;
    // Parse the current element type.
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    if (parser.parseType(elementType))
      return failure();

    // Check that the type is a MemRefType.
    if (!elementType.isa<mlir::MemRefType>()) {
      parser.emitError(typeLoc, "element type for a subview must be "
                                "a MemRefType, got: ")
          << elementType;
      return failure();
    }

    // Parse: `>`
    if (parser.parseGreater())
      return failure();

    return result = AIEObjectFifoSubviewType::get(elementType), success();
  }

  return {};
}

/// Parse a type defined by this dialect.
/// Emits an error and returns failure if `name` does not
/// refer to a type defined in this dialect.
static ParseResult parse(Type &result, StringRef name,
                         DialectAsmParser &parser) {
  auto *context = parser.getBuilder().getContext();
  OptionalParseResult parseResult;

  parseResult = aieTypeParser(context, parser, name, result);
  if (parseResult.has_value())
    return parseResult.value();

  parser.emitError(parser.getNameLoc(), "unknown AIE dialect type: \"")
      << name << "\"";
  return failure();
}

/// Parse an instance of a type registered to the AIE dialect.
mlir::Type AIEDialect::parseType(mlir::DialectAsmParser &parser) const {
  StringRef name;
  Type result;
  if (parser.parseKeyword(&name) || parse(result, name, parser))
    return Type();
  return result;
}

/// Print an instance of a type registered to the AIE dialect.
void AIEDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  if (type.isa<AIEObjectFifoType>()) {
    AIEObjectFifoType objectFifoType = type.cast<AIEObjectFifoType>();
    printer << "objectFifo<";
    printer << objectFifoType.getElementType();
    printer << '>';

  } else if (type.isa<AIEObjectFifoSubviewType>()) {
    AIEObjectFifoSubviewType subviewType =
        type.cast<AIEObjectFifoSubviewType>();
    printer << "objectFifoSubview<";
    printer << subviewType.getElementType();
    printer << '>';
  }
}

void AIEDialect::initialize() {
  addTypes<
#define GET_TYPE_LIST
#include "aie/Dialect/AIE/IR/AIETypes.cpp.inc"
      >();
  addTypes<AIEObjectFifoType, AIEObjectFifoSubviewType>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aie/Dialect/AIE/IR/AIEAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "aie/Dialect/AIE/IR/AIE.cpp.inc"
      >();
  addInterfaces<AIEInlinerInterface, AIEDialectFoldInterface>();
}

} // namespace AIE
} // namespace xilinx

// Check that the operation only contains terminators in
// TerminatorOpTypes.
template <typename... TerminatorOpTypes> struct HasSomeTerminator {
  static LogicalResult verifyTrait(Operation *op) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        if (!block.empty()) {
          Operation *operation = &block.back();
          if (!llvm::isa_and_nonnull<TerminatorOpTypes...>(operation))
            return operation->emitOpError("is not an allowed terminator")
                .attachNote(op->getLoc())
                .append("in this context: ");
        }
      }
    }
    return success();
  }
};

// Check that the given DMA-like op (e.g. MemOp, ShimDMAOp)
// has valid BDs.
template <typename ConcreteType>
LogicalResult
xilinx::AIE::HasValidBDs<ConcreteType>::verifyTrait(Operation *op) {
  auto element = cast<ConcreteType>(op);
  const auto &target_model = xilinx::AIE::getTargetModel(op);
  int bdMax = target_model.getNumBDs(element.getTileID().first,
                                     element.getTileID().second);

  int bdNum = 0;
  for (auto &block : element.getBody()) {
    if (!block.template getOps<xilinx::AIE::DMABDOp>().empty()) {
      if (bdNum >= bdMax) {
        auto bd = *(block.template getOps<xilinx::AIE::DMABDOp>().begin());
        return (op->emitOpError("has more than ") << bdMax << " blocks")
            .attachNote(bd.getLoc())
            .append("no space for this bd: ");
      }
      bdNum++;
    }
  }
  return success();
}

// Check that the given DMA-like op (e.g. MemOp, ShimDMAOp)
// has valid DMA channels.
template <typename ConcreteType>
LogicalResult
xilinx::AIE::HasValidDMAChannels<ConcreteType>::verifyTrait(Operation *op) {
  auto element = cast<ConcreteType>(op);
  DenseSet<xilinx::AIE::DMAChannel> used_channels;
  for (auto &bodyOp : element.getBody().getOps()) {
    // check for duplicate DMA channels within the same MemTileDMAOp
    if (auto DMA_start = dyn_cast<xilinx::AIE::DMAStartOp>(bodyOp)) {
      xilinx::AIE::DMAChannel dmaChan = std::make_pair(
          DMA_start.getChannelDir(), DMA_start.getChannelIndex());
      if (used_channels.count(dmaChan))
        return DMA_start.emitOpError() << "duplicate DMA channel "
                                       << stringifyDMAChannelDir(dmaChan.first)
                                       << dmaChan.second << " not allowed";
      used_channels.insert(dmaChan);
    }
  }
  return success();
}

// ObjectFifoCreateOp
LogicalResult xilinx::AIE::ObjectFifoCreateOp::verify() {
  if (!hasName())
    return emitOpError("does not have a sym_name.");

  if (isa<ArrayAttr>(getElemNumber())) {
    size_t numDepths = dyn_cast<ArrayAttr>(getElemNumber()).size();
    if (numDepths != (getConsumerTiles().size() + 1)) // +1 for producer depth
      return emitOpError("does not have enough depths specified for producer "
                         "and for each consumer.");
  }

  return success();
}
xilinx::AIE::TileOp xilinx::AIE::ObjectFifoCreateOp::getProducerTileOp() {
  return cast<xilinx::AIE::TileOp>(getProducerTile().getDefiningOp());
}

// ObjectFifoLinkOp
LogicalResult xilinx::AIE::ObjectFifoLinkOp::verify() {
  if (isJoin() && isDistribute())
    return emitError("ObjectFifoLinkOp does not support 'join' and "
                     "'distribute' at the same time");

  auto sharedTile = getOptionalSharedTile();
  if (!sharedTile)
    return emitError("ObjectFifoLinkOp must have a link point, i.e., a "
                     "shared tile between objectFifos");

  if (isJoin()) {
    ObjectFifoCreateOp fifoOut =
        getFifoOuts()[0].getDefiningOp<ObjectFifoCreateOp>();
    AIEObjectFifoType fifoType = fifoOut.getType().cast<AIEObjectFifoType>();
    MemRefType elemType = fifoType.getElementType().cast<MemRefType>();
    int outputSize = (int)elemType.getShape()[0];

    int inputSize = 0;
    for (auto fifoIn : getFifoIns()) {
      auto op = fifoIn.getDefiningOp<ObjectFifoCreateOp>();
      AIEObjectFifoType fifo = op.getType().cast<AIEObjectFifoType>();
      MemRefType elemType = fifo.getElementType().cast<MemRefType>();
      inputSize += (int)elemType.getShape()[0];
    }
    if (inputSize != outputSize)
      return emitError("Total size of input objFifos in ObjectFifoLinkOp must "
                       "be equal to size of output objFifo");

  } else if (isDistribute()) {
    ObjectFifoCreateOp fifoIn =
        getFifoIns()[0].getDefiningOp<ObjectFifoCreateOp>();
    AIEObjectFifoType fifoType = fifoIn.getType().cast<AIEObjectFifoType>();
    MemRefType elemType = fifoType.getElementType().cast<MemRefType>();
    int inputSize = (int)elemType.getShape()[0];

    int outputSize = 0;
    for (auto fifoOut : getFifoOuts()) {
      auto op = fifoOut.getDefiningOp<ObjectFifoCreateOp>();
      AIEObjectFifoType fifo = op.getType().cast<AIEObjectFifoType>();
      MemRefType elemType = fifo.getElementType().cast<MemRefType>();
      outputSize += (int)elemType.getShape()[0];
    }
    if (outputSize != inputSize)
      return emitError("Total size of output objFifos in ObjectFifoLinkOp must "
                       "be equal to size of input objFifo");
  }

  return success();
}
std::optional<Value> xilinx::AIE::ObjectFifoLinkOp::getOptionalSharedTile() {
  if (isJoin()) {
    auto fifoOut = getFifoOuts()[0].getDefiningOp<ObjectFifoCreateOp>();
    for (auto fifoIn : getFifoIns()) {
      ObjectFifoCreateOp fifoInOp = fifoIn.getDefiningOp<ObjectFifoCreateOp>();
      if (fifoOut.getProducerTile() != fifoInOp.getConsumerTiles()[0])
        return {};
    }
    return {fifoOut.getProducerTile()};
  } else {
    auto fifoIn = getFifoIns()[0].getDefiningOp<ObjectFifoCreateOp>();
    for (auto fifoOut : getFifoOuts()) {
      ObjectFifoCreateOp fifoOutOp =
          fifoOut.getDefiningOp<ObjectFifoCreateOp>();
      if (fifoIn.getConsumerTiles()[0] != fifoOutOp.getProducerTile())
        return {};
    }
    return {fifoIn.getConsumerTiles()[0]};
  }
  return {};
}

// ObjectFifoRegisterExternalBuffersOp
LogicalResult xilinx::AIE::ObjectFifoRegisterExternalBuffersOp::verify() {
  if (!getTileOp().isShimTile())
    return emitOpError("tile is not a shim tile");

  return success();
}
xilinx::AIE::TileOp
xilinx::AIE::ObjectFifoRegisterExternalBuffersOp::getTileOp() {
  return cast<xilinx::AIE::TileOp>(getTile().getDefiningOp());
}

// ObjectFifoAcquireOp
LogicalResult xilinx::AIE::ObjectFifoAcquireOp::verify() {
  if (acqNumber() < 1)
    return emitOpError("must acquire at least one element");

  auto parent = getOperation()->getParentOfType<CoreOp>();
  if (parent == nullptr)
    return emitOpError("must be called from inside a CoreOp");

  auto coreTile = parent.getTile();
  auto objFifo = getAIEObjectFifo().getDefiningOp<ObjectFifoCreateOp>();
  if (getPort() == ObjectFifoPort::Produce) {
    if (coreTile != objFifo.getProducerTile())
      return parent.emitOpError(
          "producer port of objectFifo accessed by core running "
          "on non-producer tile");
  } else if (getPort() == ObjectFifoPort::Consume) {
    bool found = false;
    for (auto consumerTile : objFifo.getConsumerTiles()) {
      if (coreTile == consumerTile) {
        found = true;
        break;
      }
    }
    if (!found)
      return parent.emitOpError(
          "consumer port of objectFifo accessed by core running "
          "on non-consumer tile");
  }

  return success();
}

// ObjectFifoReleaseOp
LogicalResult xilinx::AIE::ObjectFifoReleaseOp::verify() {
  if (relNumber() < 1)
    return emitOpError("must release at least one element");

  auto parent = getOperation()->getParentOfType<CoreOp>();
  if (parent == nullptr)
    return emitOpError("must be called from inside a CoreOp");

  auto coreTile = parent.getTile();
  auto objFifo = getAIEObjectFifo().getDefiningOp<ObjectFifoCreateOp>();
  if (getPort() == ObjectFifoPort::Produce) {
    if (coreTile != objFifo.getProducerTile())
      return parent.emitOpError(
          "producer port of objectFifo accessed by core running "
          "on non-producer tile");
  } else if (getPort() == ObjectFifoPort::Consume) {
    bool found = false;
    for (auto consumerTile : objFifo.getConsumerTiles()) {
      if (coreTile == consumerTile) {
        found = true;
        break;
      }
    }
    if (!found)
      return parent.emitOpError(
          "consumer port of objectFifo accessed by core running "
          "on non-consumer tile");
  }

  return success();
}

// ObjectFifoSubviewAccessOp
LogicalResult xilinx::AIE::ObjectFifoSubviewAccessOp::verify() {
  auto parent = getOperation()->getParentOfType<CoreOp>();
  if (parent == nullptr)
    return emitOpError("must be called from inside a CoreOp");

  ObjectFifoAcquireOp acqOp = getSubview().getDefiningOp<ObjectFifoAcquireOp>();
  if ((int)getIndex() >= acqOp.acqNumber())
    return emitOpError("accessed farther than number of acquired elements "
                       "(index out of bounds).");

  return success();
}

// ObjectFifoRegisterProcessOp
LogicalResult xilinx::AIE::ObjectFifoRegisterProcessOp::verify() {
  if (getProcessLength() < 1)
    return emitOpError("process length must be >= 1");

  if (getAcquirePattern().size() != getReleasePattern().size()) {
    // acquire pattern size = process length (i.e., release pattern will be
    // duplicated by process length times) OR the other way around
    if (!(getAcquirePattern().size() == getProcessLength()) &&
        !(getProcessLength() == getReleasePattern().size()))
      return emitOpError(
          "Acquire and Release patterns must be of equal length, or "
          "longest length of one must be equal to process "
          "length of the other");
  }

  return success();
}

const xilinx::AIE::AIETargetModel &xilinx::AIE::DeviceOp::getTargetModel() {
  switch (getDevice()) {
  case AIEDevice::xcvc1902:
    return VC1902model;
  case AIEDevice::xcve2302:
    return VE2302model;
  case AIEDevice::xcve2802:
    return VE2802model;
  }
  return VC1902model;
}

LogicalResult xilinx::AIE::DeviceOp::verify() { return success(); }

LogicalResult xilinx::AIE::TileOp::verify() {
  const auto &target_model = getTargetModel(*this);
  int columns = target_model.columns();
  int rows = target_model.rows();
  if (colIndex() >= columns)
    return emitOpError("column index (")
           << colIndex()
           << ") must be less than the number of columns in the device ("
           << columns << ")";
  if (rowIndex() >= rows)
    return emitOpError("row index (")
           << rowIndex()
           << ") must be less than the number of rows in the device (" << rows
           << ")";

  auto users = getResult().getUsers();
  bool found = false;
  for (auto user : users) {
    if (llvm::isa<xilinx::AIE::SwitchboxOp>(*user)) {
      if (found)
        return emitOpError("can only have one switchbox");
      found = true;
    }
  }

  return success();
}

LogicalResult xilinx::AIE::SwitchboxOp::verify() {
  Region &body = getConnections();
  DenseSet<xilinx::AIE::Port> sourceset;
  DenseSet<xilinx::AIE::Port> destset;
  if (body.empty())
    return emitOpError("should have non-empty body");
  for (auto &ops : body.front()) {
    // Would be simpler if this could be templatized.
    auto checkBound = [&ops](StringRef dir, WireBundle bundle, int index,
                             int bound) -> LogicalResult {
      if (index >= bound) {
        if (bound > 0)
          return ops.emitOpError("index ")
                 << index << " for " << dir << " bundle "
                 << stringifyWireBundle(bundle) << " must be less than "
                 << bound;
        else
          return ops.emitOpError()
                 << dir << " bundle " << stringifyWireBundle(bundle)
                 << " not supported";
      }
      return success();
    };
    if (auto connectOp = dyn_cast<xilinx::AIE::ConnectOp>(ops)) {
      xilinx::AIE::Port source =
          std::make_pair(connectOp.getSourceBundle(), connectOp.sourceIndex());
      sourceset.insert(source);

      xilinx::AIE::Port dest =
          std::make_pair(connectOp.getDestBundle(), connectOp.destIndex());
      if (destset.count(dest)) {
        return connectOp.emitOpError("targets same destination ")
               << stringifyWireBundle(dest.first) << dest.second
               << " as another connect operation";
      } else {
        destset.insert(dest);
      }

      if (connectOp.sourceIndex() < 0)
        return connectOp.emitOpError("source index cannot be less than zero");

      {
        auto boundsCheck = checkBound(
            "source", connectOp.getSourceBundle(), connectOp.sourceIndex(),
            getNumSourceConnections(connectOp.getSourceBundle()));
        if (boundsCheck.failed())
          return boundsCheck;
      }

      if (connectOp.destIndex() < 0)
        return connectOp.emitOpError("dest index cannot be less than zero");

      {
        auto boundsCheck =
            checkBound("dest", connectOp.getDestBundle(), connectOp.destIndex(),
                       getNumDestConnections(connectOp.getDestBundle()));
        if (boundsCheck.failed())
          return boundsCheck;
      }
    } else if (auto connectOp = dyn_cast<xilinx::AIE::MasterSetOp>(ops)) {
      xilinx::AIE::Port dest =
          std::make_pair(connectOp.getDestBundle(), connectOp.destIndex());
      if (destset.count(dest)) {
        return connectOp.emitOpError("targets same destination ")
               << stringifyWireBundle(dest.first) << dest.second
               << " as another connect or masterset operation";
      } else {
        destset.insert(dest);
      }
      if (connectOp.destIndex() < 0)
        return connectOp.emitOpError("dest index cannot be less than zero");

      {
        auto boundsCheck =
            checkBound("dest", connectOp.getDestBundle(), connectOp.destIndex(),
                       getNumDestConnections(connectOp.getDestBundle()));
        if (boundsCheck.failed())
          return boundsCheck;
      }

      int arbiter = -1;
      for (auto val : connectOp.getAmsels()) {
        auto amsel = dyn_cast<xilinx::AIE::AMSelOp>(val.getDefiningOp());
        if ((arbiter != -1) && (arbiter != amsel.arbiterIndex()))
          return connectOp.emitOpError(
              "a master port can only be tied to one arbiter");
        arbiter = amsel.arbiterIndex();
      }
    } else if (auto connectOp = dyn_cast<xilinx::AIE::PacketRulesOp>(ops)) {
      xilinx::AIE::Port source =
          std::make_pair(connectOp.getSourceBundle(), connectOp.sourceIndex());
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

LogicalResult xilinx::AIE::ShimSwitchboxOp::verify() {
  Region &body = getConnections();
  DenseSet<xilinx::AIE::Port> destset;
  if (body.empty())
    return emitOpError("should have non-empty body");

  for (auto &ops : body.front()) {
    if (auto connectOp = dyn_cast<xilinx::AIE::ConnectOp>(ops)) {
      xilinx::AIE::Port dest =
          std::make_pair(connectOp.getDestBundle(), connectOp.destIndex());
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

LogicalResult xilinx::AIE::ShimMuxOp::verify() {
  Region &body = getConnections();
  DenseSet<xilinx::AIE::Port> destset;
  if (body.empty())
    return emitOpError("should have non-empty body");

  for (auto &ops : body.front()) {
    if (auto connectOp = dyn_cast<xilinx::AIE::ConnectOp>(ops)) {
      xilinx::AIE::Port dest =
          std::make_pair(connectOp.getDestBundle(), connectOp.destIndex());
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

int xilinx::AIE::ShimMuxOp::getNumSourceConnections(WireBundle bundle) {
  auto tile = getTileOp();
  const auto &target_model = getTargetModel(*this);
  return target_model.getNumSourceShimMuxConnections(tile.getCol(),
                                                     tile.getRow(), bundle);
}
int xilinx::AIE::ShimMuxOp::getNumDestConnections(WireBundle bundle) {
  auto tile = getTileOp();
  const auto &target_model = getTargetModel(*this);
  return target_model.getNumDestShimMuxConnections(tile.getCol(), tile.getRow(),
                                                   bundle);
}
xilinx::AIE::TileOp xilinx::AIE::ShimMuxOp::getTileOp() {
  return cast<xilinx::AIE::TileOp>(getTile().getDefiningOp());
}
int xilinx::AIE::ShimMuxOp::colIndex() { return getTileOp().colIndex(); }
int xilinx::AIE::ShimMuxOp::rowIndex() { return getTileOp().rowIndex(); }

// ShimDMAOp
LogicalResult xilinx::AIE::ShimDMAOp::verify() {
  if (getBody().empty())
    return emitOpError("should have non-empty body");

  if (!getTileOp().isShimNOCTile())
    return emitOpError("must be in a ShimTile with a NOC connection");

  auto result =
      HasSomeTerminator<xilinx::AIE::DMAStartOp, xilinx::AIE::NextBDOp,
                        xilinx::AIE::EndOp>::verifyTrait(*this);
  if (result.failed()) {
    return result;
  }

  return success();
}
xilinx::AIE::TileOp xilinx::AIE::ShimDMAOp::getTileOp() {
  return cast<TileOp>(getTile().getDefiningOp());
}
int xilinx::AIE::ShimDMAOp::colIndex() { return getTileOp().colIndex(); }
int xilinx::AIE::ShimDMAOp::rowIndex() { return getTileOp().rowIndex(); }

LogicalResult xilinx::AIE::PacketRulesOp::verify() {
  Region &body = getRules();
  if (body.empty())
    return emitOpError("should have non-empty body");

  return success();
}

LogicalResult xilinx::AIE::PacketFlowOp::verify() {
  Region &body = getPorts();
  if (body.empty())
    return emitOpError("should have non-empty body");

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
LogicalResult xilinx::AIE::CoreOp::verify() {
  if (getBody().empty())
    return emitOpError("should have non-empty body");
  if (getTileOp().isShimTile())
    return emitOpError("CoreOp cannot be created on shim tile, i.e. row == 0");
  if (getTileOp().isMemTile())
    return emitOpError("CoreOp cannot be created on mem tile");
  return success();
}

int xilinx::AIE::CoreOp::colIndex() { return getTileOp().colIndex(); }

int xilinx::AIE::CoreOp::rowIndex() { return getTileOp().rowIndex(); }
xilinx::AIE::TileOp xilinx::AIE::CoreOp::getTileOp() {
  return cast<xilinx::AIE::TileOp>(getTile().getDefiningOp());
}

// BufferOp
int64_t xilinx::AIE::BufferOp::getAllocationSize() {
  MemRefType type = getType().cast<MemRefType>();
  return type.getNumElements() * type.getElementTypeBitWidth() / 8;
}
xilinx::AIE::TileOp xilinx::AIE::BufferOp::getTileOp() {
  return cast<xilinx::AIE::TileOp>(getTile().getDefiningOp());
}
LogicalResult xilinx::AIE::BufferOp::verify() {
  auto result = UsesAreAccessable::verifyTrait(*this);
  if (result.failed())
    return result;
  return success();
}

// MemOp
LogicalResult xilinx::AIE::MemOp::verify() {
  Region &body = getBody();
  DenseSet<xilinx::AIE::DMAChannel> used_channels;
  if (body.empty())
    return emitOpError("should have non-empty body");

  auto result =
      HasSomeTerminator<xilinx::AIE::DMAStartOp, xilinx::AIE::NextBDOp,
                        xilinx::AIE::EndOp>::verifyTrait(*this);
  if (result.failed()) {
    return result;
  }

  for (auto &bodyOp : body.getOps()) {
    // check for duplicate DMA channels within the same MemOp
    if (auto DMA_start = dyn_cast<xilinx::AIE::DMAStartOp>(bodyOp)) {
      xilinx::AIE::DMAChannel dmaChan = std::make_pair(
          DMA_start.getChannelDir(), DMA_start.getChannelIndex());
      if (used_channels.count(dmaChan))
        return DMA_start.emitOpError() << "duplicate DMA channel "
                                       << stringifyDMAChannelDir(dmaChan.first)
                                       << dmaChan.second << " in MemOp";
      used_channels.insert(dmaChan);
    }

    if (auto allocOp = dyn_cast<memref::AllocOp>(bodyOp)) {
      if (!allocOp->getAttr("id"))
        return allocOp.emitOpError()
               << "allocOp in MemOp region should have an id attribute";
    }
  }
  return success();
}

xilinx::AIE::TileOp xilinx::AIE::MemOp::getTileOp() {
  return cast<xilinx::AIE::TileOp>(getTile().getDefiningOp());
}
int xilinx::AIE::MemOp::colIndex() { return getTileOp().colIndex(); }

int xilinx::AIE::MemOp::rowIndex() { return getTileOp().rowIndex(); }

/// Returns the region on the current operation that is callable. This may
/// return nullptr in the case of an external callable object, e.g. an external
/// function.
Region *xilinx::AIE::MemOp::getCallableRegion() { return &(getBody()); }

/// Returns the results types that the callable region produces when executed.
ArrayRef<Type> xilinx::AIE::MemOp::getCallableResults() { return getType(); }

// MemTileDMAOp
LogicalResult xilinx::AIE::MemTileDMAOp::verify() {
  assert(getOperation()->getNumRegions() == 1 &&
         "MemTileDMAOp has zero region!");
  assert(!getBody().empty() && "MemTileDMAOp should have non-empty body");

  auto result =
      HasSomeTerminator<xilinx::AIE::DMAStartOp, xilinx::AIE::NextBDOp,
                        xilinx::AIE::EndOp>::verifyTrait(*this);
  if (result.failed()) {
    return result;
  }

  for (auto &bodyOp : getBody().getOps()) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(bodyOp)) {
      if (!allocOp->getAttr("id"))
        return allocOp.emitOpError()
               << "allocOp in MemTileDMAOp region should have an id attribute";
    }
  }
  return success();
}

// DMABDOp
LogicalResult xilinx::AIE::DMABDOp::verify() {
  if (getDimensions()) {
    ::mlir::MemRefType buffer = getBuffer().getType();
    if (!buffer.getElementType().isInteger(32)) {
      // The AIE2 specification prescribes that multi-dimensional address
      // generation creates addresses to 32 bit words. Hence, stepSize and wrap
      // refer to 32 bit words. To avoid confusion, we disallow using multi-
      // dimensional BDs with other memrefs.
      return emitOpError() << "Multi-dimensional buffer descriptors are only "
                              "supported for 32 bit integer elements.";
    }
    uint64_t base_addr = getOffset();
    uint64_t memref_size = 1;
    for (int64_t memref_dim : buffer.getShape()) {
      memref_size *= memref_dim;
    }
    memref_size += 4 * base_addr;
    llvm::ArrayRef<xilinx::AIE::DimTupleAttr> dims = *getDimensions();
    if (dims.size() > 4) {
      return emitOpError() << "Cannot give more than four dimensions.";
    }
    for (xilinx::AIE::DimTupleAttr dim : dims) {
      if (0 == dim.getStepsize()) {
        return emitOpError()
               << "Invalid step size; must be a positive integer.";
      }
      if (dim.getStepsize() > memref_size) {
        return emitOpError()
               << "Step size " << std::to_string(dim.getStepsize()) << " "
               << "exceeds memref size " << std::to_string(memref_size);
      }
      // TODO: There are more meaningful checks that could be added here,
      // such as:
      //   - limits on wrap; note that wrap refers to iterations of the previous
      //     dimension
      //   - last dimension wrap should be implicit from memref size and other
      //     dimension;ensure it isset correctly
    }
  }
  return success();
}

xilinx::AIE::TileOp xilinx::AIE::MemTileDMAOp::getTileOp() {
  return cast<xilinx::AIE::TileOp>(getTile().getDefiningOp());
}
int xilinx::AIE::MemTileDMAOp::colIndex() { return getTileOp().colIndex(); }

int xilinx::AIE::MemTileDMAOp::rowIndex() { return getTileOp().rowIndex(); }

/// Returns the region on the current operation that is callable. This may
/// return nullptr in the case of an external callable object, e.g. an external
/// function.
Region *xilinx::AIE::MemTileDMAOp::getCallableRegion() { return &(getBody()); }

/// Returns the results types that the callable region produces when executed.
ArrayRef<Type> xilinx::AIE::MemTileDMAOp::getCallableResults() {
  return getType();
}

// SwitchboxOp
xilinx::AIE::TileOp xilinx::AIE::SwitchboxOp::getTileOp() {
  return cast<xilinx::AIE::TileOp>(getTile().getDefiningOp());
}
int xilinx::AIE::SwitchboxOp::colIndex() { return getTileOp().colIndex(); }
int xilinx::AIE::SwitchboxOp::rowIndex() { return getTileOp().rowIndex(); }

template <typename... ParentOpTypes> struct HasSomeParent {
  static LogicalResult verifyTrait(Operation *op) {
    Operation *operation = op->getParentOp();
    while (operation) {
      if (llvm::isa_and_nonnull<ParentOpTypes...>(operation))
        return success();
      operation = operation->getParentOp();
    }
    return failure();
  }
};

xilinx::AIE::TileOp xilinx::AIE::LockOp::getTileOp() {
  return cast<xilinx::AIE::TileOp>(getTile().getDefiningOp());
}
int xilinx::AIE::LockOp::colIndex() { return getTileOp().colIndex(); }
int xilinx::AIE::LockOp::rowIndex() { return getTileOp().rowIndex(); }

LogicalResult xilinx::AIE::LockOp::verify() {
  auto result = UsesAreAccessable::verifyTrait(*this);
  if (result.failed())
    return result;

  if (getLockID().has_value()) {
    const auto &target_model = xilinx::AIE::getTargetModel(getTileOp());
    auto tileOp = getTileOp();
    unsigned int numLocks =
        target_model.getNumLocks(tileOp.getCol(), tileOp.getRow());
    if (getLockID().value() >= numLocks)
      return emitOpError("lock assigned invalid id (maximum is ")
             << numLocks - 1 << ")";
  }

  return success();
}

struct UsesReachableLock {
  static LogicalResult verifyTrait(Operation *op) {
    auto useLock = dyn_cast<xilinx::AIE::UseLockOp>(op);
    auto lock =
        dyn_cast<xilinx::AIE::LockOp>(useLock.getLock().getDefiningOp());
    auto parent = dyn_cast<xilinx::AIE::TileElement>(useLock->getParentOp());
    auto tileID = parent.getTileID();
    const auto &target_model = xilinx::AIE::getTargetModel(op);
    if (!target_model.isLegalMemAffinity(tileID.first, tileID.second,
                                         lock.colIndex(), lock.rowIndex()))
      return failure();
    return success();
  }
};

struct UsesOneLockInDMABlock {
  static LogicalResult verifyTrait(Operation *op) {
    auto block = op->getBlock();
    int lockID = -1;
    for (auto op : block->getOps<xilinx::AIE::UseLockOp>()) {
      auto lock = dyn_cast<xilinx::AIE::LockOp>(op.getLock().getDefiningOp());
      if (lock.getLockID().has_value()) {
        if (lockID != -1 && lockID != lock.getLockIDValue())
          return failure();
        lockID = lock.getLockIDValue();
      }
    }
    return success();
  }
};

struct AcquireReleaseOneStateInDMABlock {
  static LogicalResult verifyTrait(Operation *op) {
    auto block = op->getBlock();
    int acqValue = -1, relValue = -1;
    for (auto op : block->getOps<xilinx::AIE::UseLockOp>()) {
      if (op.acquire() || op.acquire_ge()) {
        if (acqValue != -1 && acqValue != op.getLockValue()) {
          return failure();
        }
        acqValue = op.getLockValue();
      } else if (op.release()) {
        if (relValue != -1 && relValue != op.getLockValue()) {
          return failure();
        }
        relValue = op.getLockValue();
      }
    }
    return success();
  }
};

LogicalResult xilinx::AIE::UseLockOp::verify() {
  // AIE.useLock cannot be used at the top level
  if (llvm::isa_and_nonnull<xilinx::AIE::DeviceOp, mlir::ModuleOp>(
          (*this)->getParentOp()))
    return (*this)->emitOpError("must be used in a core or memory operation.");

  const auto &target_model = getTargetModel(*this);
  if (target_model.getTargetArch() == xilinx::AIE::AIEArch::AIE1 &&
      acquire_ge())
    return (*this)->emitOpError(
        "AcquireGreaterEqual is not supported in AIE1.");

  // Otherwise, AIE.useLock should be inside MemOp, MemTileDMAOp, or ShimDMAOp,
  if (HasSomeParent<xilinx::AIE::MemOp, xilinx::AIE::MemTileDMAOp,
                    xilinx::AIE::ShimDMAOp>::verifyTrait(*this)
          .succeeded()) {
    if (!(*this)->getBlock())
      return (*this)->emitOpError("is not in a block.");

    if (target_model.getTargetArch() == xilinx::AIE::AIEArch::AIE1 &&
        UsesOneLockInDMABlock::verifyTrait(*this).failed())
      return (*this)->emitOpError(
          "used in a DMA block that have multiple locks.");

    if (AcquireReleaseOneStateInDMABlock::verifyTrait(*this).failed())
      return (*this)->emitOpError(
          "acquires/releases the lock in a DMA block from/to multiple states.");

    return success();

    // Or it can be in a CoreOp, or some FuncOp called from a CoreOp
  } else if (HasSomeParent<xilinx::AIE::CoreOp, func::FuncOp>::verifyTrait(
                 *this)
                 .succeeded()) {
    return success();

  } else {
    return (*this)->emitOpError()
           << "expects some parent op to be one of "
           << "AIE::device, AIE::core, func::func, AIE::mem, or AIE::shimDMA";
  }
}

#include "aie/Dialect/AIE/IR/AIEEnums.cpp.inc"
#include "aie/Dialect/AIE/IR/AIEInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "aie/Dialect/AIE/IR/AIE.cpp.inc"

namespace xilinx {
namespace AIE {

int SwitchboxOp::getNumSourceConnections(WireBundle bundle) {
  auto tile = getTileOp();
  const auto &target_model = getTargetModel(*this);
  return target_model.getNumSourceSwitchboxConnections(tile.getCol(),
                                                       tile.getRow(), bundle);
}
int SwitchboxOp::getNumDestConnections(WireBundle bundle) {
  auto tile = getTileOp();
  const auto &target_model = getTargetModel(*this);
  return target_model.getNumDestSwitchboxConnections(tile.getCol(),
                                                     tile.getRow(), bundle);
}
int TileOp::getNumSourceConnections(WireBundle bundle) {
  const auto &target_model = getTargetModel(*this);
  if (bundle == WireBundle::Core || bundle == WireBundle::DMA)
    // Note dest is correct here, since direction is reversed.
    if (target_model.isShimNOCTile(getCol(), getRow()) ||
        target_model.isShimPLTile(getCol(), getRow()))
      return target_model.getNumDestShimMuxConnections(getCol(), getRow(),
                                                       bundle);
    else
      return target_model.getNumDestSwitchboxConnections(getCol(), getRow(),
                                                         bundle);
  else
    return 0;
}
int TileOp::getNumDestConnections(WireBundle bundle) {
  const auto &target_model = getTargetModel(*this);
  if (bundle == WireBundle::Core || bundle == WireBundle::DMA)
    // Note source is correct here, since direction is reversed.
    if (target_model.isShimNOCTile(getCol(), getRow()) ||
        target_model.isShimPLTile(getCol(), getRow()))
      return target_model.getNumDestShimMuxConnections(getCol(), getRow(),
                                                       bundle);
    else
      return target_model.getNumSourceSwitchboxConnections(getCol(), getRow(),
                                                           bundle);
  else
    return 0;
}
bool TileOp::isMemTile() {
  const auto &target_model = getTargetModel(*this);
  return target_model.isMemTile(getCol(), getRow());
}
bool TileOp::isShimNOCTile() {
  const auto &target_model = getTargetModel(*this);
  return target_model.isShimNOCTile(getCol(), getRow());
}
bool TileOp::isShimPLTile() {
  const auto &target_model = getTargetModel(*this);
  return target_model.isShimPLTile(getCol(), getRow());
}
bool TileOp::isShimNOCorPLTile() { return isShimNOCTile() || isShimPLTile(); }
} // namespace AIE
} // namespace xilinx

// Include implementations for custom attributes
#define GET_ATTRDEF_CLASSES
#include "aie/Dialect/AIE/IR/AIEAttrDefs.cpp.inc"
