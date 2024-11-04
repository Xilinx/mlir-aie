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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace xilinx::AIE;

// Add TableGen'erated dialect definitions (including constructor)
// We implement the initialize() function further below
#include "aie/Dialect/AIE/IR/AIEDialect.cpp.inc"

namespace {

struct AIEInlinerInterface : DialectInlinerInterface {
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
  void handleTerminator(Operation *op, Block *newDest) const final override {}

  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the region has only one block.
  void handleTerminator(Operation *op,
                        ValueRange valuesToRepl) const final override {}
};

struct AIEDialectFoldInterface : DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  /// Registered hook to check if the given region, which is attached to an
  /// operation that is *not* isolated from above, should be used when
  /// materializing constants.
  bool shouldMaterializeInto(Region *region) const final override {
    // If this is an AIE::CoreOp region, then insert into it.
    return isa<CoreOp>(region->getParentOp());
  }
};

} // end anonymous namespace

namespace xilinx::AIE {

LogicalResult myVerifyOffsetSizeAndStrideOp(OffsetSizeAndStrideOpInterface op) {
  std::array<unsigned, 3> maxRanks = op.getArrayAttrMaxRanks();
  if (!(op.getMixedOffsets().size() == 1 && maxRanks[0] == 1) && // NOLINT
      op.getMixedOffsets().size() != op.getMixedSizes().size())
    return op->emitError(
               "expected mixed offsets rank to match mixed sizes rank (")
           << op.getMixedOffsets().size() << " vs " << op.getMixedSizes().size()
           << ") so the rank of the result type is well-formed.";
  if (failed(verifyListOfOperandsOrIntegers(
          op, "offset", maxRanks[0], op.getStaticOffsets(), op.getOffsets())))
    return failure();
  if (failed(verifyListOfOperandsOrIntegers(
          op, "size", maxRanks[1], op.getStaticSizes(), op.getSizes())))
    return failure();
  if (failed(verifyListOfOperandsOrIntegers(
          op, "stride", maxRanks[2], op.getStaticStrides(), op.getStrides())))
    return failure();
  for (int64_t offset : op.getStaticOffsets())
    if (offset < 0 && !ShapedType::isDynamic(offset))
      return op->emitError("expected offsets to be non-negative, but got ")
             << offset;
  for (int64_t size : op.getStaticSizes())
    if (size < 0 && !ShapedType::isDynamic(size))
      return op->emitError("expected sizes to be non-negative, but got ")
             << size;

  return success();
}

static VC1902TargetModel VC1902model;
static VE2302TargetModel VE2302model;
static VE2802TargetModel VE2802model;
static NPUTargetModel NPUmodel;
static VirtualizedNPUTargetModel NPUmodel1col(1);
static VirtualizedNPUTargetModel NPUmodel2col(2);
static VirtualizedNPUTargetModel NPUmodel3col(3);
static VirtualizedNPUTargetModel NPUmodel4col(4);
static NPU2TargetModel NPU2model;

const AIETargetModel &getTargetModel(Operation *op) {
  if (auto t = dyn_cast<AIETarget>(op))
    return t.getTargetModel();
  if (auto t = op->getParentOfType<AIETarget>())
    return t.getTargetModel();

  // For backward compatibility, return a basic device model compatible with
  // the VCK190
  return VC1902model;
}

const AIETargetModel &getTargetModel(AIEDevice device) {
  switch (device) {
  case AIEDevice::xcvc1902:
    return VC1902model;
  case AIEDevice::xcve2302:
    return VE2302model;
  case AIEDevice::xcve2802:
    return VE2802model;
  case AIEDevice::npu1:
    return NPUmodel;
  case AIEDevice::npu1_1col:
    return NPUmodel1col;
  case AIEDevice::npu1_2col:
    return NPUmodel2col;
  case AIEDevice::npu1_3col:
    return NPUmodel3col;
  case AIEDevice::npu1_4col:
    return NPUmodel4col;
  case AIEDevice::npu2:
    return NPU2model;
  }
  return VC1902model;
}

// Walk the operation hierarchy until we find a containing TileElement.
// If no parent is a TileElement, then return null.
static TileElement getParentTileElement(Operation *op) {
  auto *parent = op->getParentOp();
  while (!llvm::isa_and_nonnull<DeviceOp, ModuleOp>(parent)) {
    if (auto element = llvm::dyn_cast<TileElement>(parent))
      return element;
    parent = parent->getParentOp();
  }
  return llvm::dyn_cast<TileElement>(parent);
}

struct UsesAreAccessible {
  static LogicalResult verifyTrait(Operation *op) {
    auto thisElement = cast<TileElement>(op);
    auto thisID = thisElement.getTileID();
    auto users = op->getResult(0).getUsers();
    const auto &targetModel = getTargetModel(op);
    for (auto *user : users) {
      // AIE.useLock may be used in a device to set the lock's default value
      // Allow in a toplevel module for backward compatibility
      if (llvm::isa_and_nonnull<DeviceOp, ModuleOp>(user->getParentOp())) {
        continue;
      }
      // If any parent prescribes that accessibility checks be skipped,
      // skip the check for that user.
      if (user->getParentWithTrait<SkipAccessibilityCheckTrait>()) {
        continue;
      }
      TileElement element = llvm::dyn_cast<TileElement>(user);
      if (!element) {
        element = getParentTileElement(user);
      }
      if (!element) {
        // This should probably be caught elsewhere as well.
        return op->emitOpError("is accessed outside of a tile")
                   .attachNote(user->getLoc())
               << "user";
      }
      auto tileID = element.getTileID();
      if (!targetModel.isLegalMemAffinity(tileID.col, tileID.row, thisID.col,
                                          thisID.row)) {
        return (op->emitOpError("in Column ")
                << thisID.col << " and Row " << thisID.row
                << " is accessed from an unreachable tile in Column "
                << tileID.col << " and Row " << tileID.row)
                   .attachNote(user->getLoc())
               << "user";
      }
    }
    return success();
  }
};

namespace detail {
/// This class represents the internal storage of the AIE `ObjectFifoType`.
struct AIEObjectFifoTypeStorage : TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage.
  using KeyTy = MemRefType;

  /// A constructor for the objectFifo type storage instance.
  AIEObjectFifoTypeStorage(MemRefType elementType) : elementType(elementType) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == KeyTy(elementType); }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`.
  static AIEObjectFifoTypeStorage *construct(TypeStorageAllocator &allocator,
                                             const KeyTy &key) {
    // Allocate the storage instance and construct it.
    return new (allocator.allocate<AIEObjectFifoTypeStorage>())
        AIEObjectFifoTypeStorage(key);
  }

  MemRefType elementType;
};
} // namespace detail

AIEObjectFifoType AIEObjectFifoType::get(MemRefType elementType) {
  // Call into a helper 'get' method in 'TypeBase' to get an uniqued instance
  // of this type.
  MLIRContext *ctx = elementType.getContext();
  return Base::get(ctx, elementType);
}

LogicalResult
AIEObjectFifoType::verify(function_ref<InFlightDiagnostic()> emitError,
                          MemRefType elementType) {
  return success();
}

mlir::MemRefType AIEObjectFifoType::getElementType() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->elementType;
}

namespace detail {
/// This class represents the internal storage of the AIE
/// `ObjectFifoSubviewType`.
struct AIEObjectFifoSubviewTypeStorage : TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage.
  using KeyTy = MemRefType;

  /// A constructor for the subview type storage instance.
  AIEObjectFifoSubviewTypeStorage(MemRefType elementType)
      : elementType(elementType) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == elementType; }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`.
  static AIEObjectFifoSubviewTypeStorage *
  construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    // Allocate the storage instance and construct it.
    return new (allocator.allocate<AIEObjectFifoSubviewTypeStorage>())
        AIEObjectFifoSubviewTypeStorage(key);
  }

  MemRefType elementType;
};
} // namespace detail

AIEObjectFifoSubviewType AIEObjectFifoSubviewType::get(MemRefType elementType) {
  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type.
  MLIRContext *ctx = elementType.getContext();
  return Base::get(ctx, elementType);
}

/// This method is used to verify the construction invariants.
LogicalResult
AIEObjectFifoSubviewType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 MemRefType elementType) {
  return success();
}

MemRefType AIEObjectFifoSubviewType::getElementType() {
  return getImpl()->elementType;
}

/// Parse an instance of a type registered to the AIE dialect.
/// Parse an AIE type in the following forms:
///   AIE-type
///         ::= `objectfifo` `<` type `>`
///         ::= `objectfifosubview` `<` type `>`
static OptionalParseResult aieTypeParser(DialectAsmParser &parser,
                                         StringRef name, Type &result) {
  if (name == "objectfifo") {
    MemRefType elementType;
    SMLoc typeLoc = parser.getCurrentLocation();
    if (parser.parseLess() || parser.parseType(elementType) ||
        parser.parseGreater())
      return failure();

    // Check that the type is a MemRef type.
    if (!llvm::isa<MemRefType>(elementType)) {
      parser.emitError(typeLoc, "element type for an objectFifo must be "
                                "a MemRefType, got: ")
          << elementType;
      return failure();
    }

    return result = AIEObjectFifoType::get(elementType), success();
  }

  if (name == "objectfifosubview") {
    if (parser.parseLess())
      return failure();

    // Parse the element type of the struct.
    MemRefType elementType;
    // Parse the current element type.
    SMLoc typeLoc = parser.getCurrentLocation();
    if (parser.parseType(elementType))
      return failure();

    // Check that the type is a MemRefType.
    if (!llvm::isa<MemRefType>(elementType)) {
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

  if (OptionalParseResult parseResult = aieTypeParser(parser, name, result);
      parseResult.has_value())
    return parseResult.value();

  parser.emitError(parser.getNameLoc(), "unknown AIE dialect type: \"")
      << name << "\"";
  return failure();
}

/// Parse an instance of a type registered to the AIE dialect.
Type AIEDialect::parseType(DialectAsmParser &parser) const {
  StringRef name;
  Type result;
  if (parser.parseKeyword(&name) || parse(result, name, parser))
    return {};
  return result;
}

/// Print an instance of a type registered to the AIE dialect.
void AIEDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (llvm::isa<AIEObjectFifoType>(type)) {
    auto objectFifoType = llvm::cast<AIEObjectFifoType>(type);
    printer << "objectfifo<";
    printer << objectFifoType.getElementType();
    printer << '>';

  } else if (llvm::isa<AIEObjectFifoSubviewType>(type)) {
    auto subviewType = llvm::cast<AIEObjectFifoSubviewType>(type);
    printer << "objectfifosubview<";
    printer << subviewType.getElementType();
    printer << '>';
  }
}

void AIEDialect::initialize() {
  addTypes<AIEObjectFifoType, AIEObjectFifoSubviewType>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aie/Dialect/AIE/IR/AIEAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "aie/Dialect/AIE/IR/AIEOps.cpp.inc"
      >();
  addInterfaces<AIEInlinerInterface, AIEDialectFoldInterface>();
}

} // namespace xilinx::AIE

// Check that the operation only contains terminators in
// TerminatorOpTypes.
template <typename... TerminatorOpTypes>
struct HasSomeTerminator {
  static LogicalResult verifyTrait(Operation *op) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        if (!block.empty()) {
          if (Operation *operation = &block.back();
              !llvm::isa_and_nonnull<TerminatorOpTypes...>(operation))
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
LogicalResult HasValidBDs<ConcreteType>::verifyTrait(Operation *op) {
  auto element = cast<ConcreteType>(op);
  const auto &targetModel = getTargetModel(op);
  int bdMax =
      targetModel.getNumBDs(element.getTileID().col, element.getTileID().row);

  int bdNum = 0;
  for (auto &block : element.getBody()) {
    if (!block.template getOps<DMABDOp>().empty()) {
      if (bdNum >= bdMax) {
        auto bd = *block.template getOps<DMABDOp>().begin();
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
LogicalResult HasValidDMAChannels<ConcreteType>::verifyTrait(Operation *op) {
  auto element = cast<ConcreteType>(op);
  DenseSet<DMAChannel> usedChannels;
  for (auto &bodyOp : element.getBody().getOps()) {
    // check for duplicate DMA channels within the same MemTileDMAOp
    if (auto dmaStart = dyn_cast<DMAStartOp>(bodyOp)) {
      DMAChannel dmaChan = {dmaStart.getChannelDir(),
                            dmaStart.getChannelIndex()};
      if (usedChannels.count(dmaChan))
        return dmaStart.emitOpError()
               << "duplicate DMA channel "
               << stringifyDMAChannelDir(dmaChan.direction) << dmaChan.channel
               << " not allowed";
      usedChannels.insert(dmaChan);
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ObjectFifoCreateOp
//===----------------------------------------------------------------------===//

LogicalResult ObjectFifoCreateOp::verify() {
  if (isa<ArrayAttr>(getElemNumber())) {
    if (size_t numDepths = dyn_cast<ArrayAttr>(getElemNumber()).size();
        numDepths != getConsumerTiles().size() + 1) // +1 for producer depth
      return emitOpError("does not have enough depths specified for producer "
                         "and for each consumer.");
  }

  if (getProducerTileOp().isShimTile() && !getDimensionsToStream().empty()) {
    return emitError(
        "`dimensionsToStream` data layout transformations are not supported "
        "on shim tile producers");
  }

  if (getViaSharedMem().has_value()) {
    if (getConsumerTiles().size() > 1)
      return emitError(
          "`via_shared_mem` can only be used in 1-to-1 object FIFOs");
  }

  if (getMemtileRepeat().has_value()) {
    if (!getProducerTileOp().isMemTile())
      return emitError("`memtile_repeat` can only be used with a mem tile "
                       "producer");
  }

  return success();
}

TileOp ObjectFifoCreateOp::getProducerTileOp() {
  return cast<TileOp>(getProducerTile().getDefiningOp());
}

namespace xilinx::AIE {

ParseResult parseObjectFifoProducerTile(OpAsmParser &parser,
                                        OpAsmParser::UnresolvedOperand &operand,
                                        BDDimLayoutArrayAttr &dimensions) {
  std::vector<BDDimLayoutAttr> emptyDims = {};
  if (parser.parseOperand(operand))
    return failure();
  if (succeeded(parser.parseOptionalKeyword("dimensionsToStream"))) {
    if (parser.parseCustomAttributeWithFallback<BDDimLayoutArrayAttr>(
            dimensions)) {
      return failure();
    }
  } else {
    dimensions =
        BDDimLayoutArrayAttr::get(parser.getContext(), ArrayRef(emptyDims));
  }
  return success();
}

void printObjectFifoProducerTile(OpAsmPrinter &printer, Operation *op,
                                 Value operand,
                                 BDDimLayoutArrayAttr dimensions) {
  printer << operand;
  if (!dimensions.empty()) {
    printer << " dimensionsToStream ";
    printer.printStrippedAttrOrType(dimensions);
  }
}

ParseResult parseObjectFifoConsumerTiles(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &tiles,
    BDDimLayoutArrayArrayAttr &dimensions) {
  // parseCommaSeparatedList doesn't handle the missing case for "none",
  // so we handle it custom here.
  std::vector<BDDimLayoutArrayAttr> tileDims = {};

  auto parseOneOperand = [&]() -> ParseResult {
    if (parser.parseOperand(tiles.emplace_back(), true)) {
      return failure();
    }
    // By default, create empty dimensions array for each consumer; this way,
    // we can be certain to have as many entries in the dimensions array as
    // there are customer
    BDDimLayoutArrayAttr dimAttr =
        BDDimLayoutArrayAttr::get(parser.getContext(), {});

    if (succeeded(parser.parseOptionalKeyword("dimensionsFromStream"))) {
      // If specified, parse actual data layout transform dimensions
      if (parser.parseCustomAttributeWithFallback<BDDimLayoutArrayAttr>(
              dimAttr)) {
        return failure();
      }
    }
    tileDims.emplace_back(dimAttr);
    return success();
  };

  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::None,
                                     parseOneOperand, " in operand list"))
    return failure();

  dimensions = BDDimLayoutArrayArrayAttr::get(parser.getContext(), tileDims);
  return success();
}

void printObjectFifoConsumerTiles(OpAsmPrinter &printer, Operation *op,
                                  OperandRange tiles,
                                  BDDimLayoutArrayArrayAttr dimsPerTileAttr) {
  size_t tileIdx = 0;
  for (auto tile : tiles) {
    printer << tile;
    if (dimsPerTileAttr && tileIdx < dimsPerTileAttr.size() &&
        dimsPerTileAttr[tileIdx] && !dimsPerTileAttr[tileIdx].empty()) {
      printer << " dimensionsFromStream ";
      printer.printStrippedAttrOrType(dimsPerTileAttr[tileIdx]);
    }
    if (tileIdx < tiles.size() - 1) {
      printer << ", ";
    }
    tileIdx++;
  }
}

} // namespace xilinx::AIE

//===----------------------------------------------------------------------===//
// ObjectFifoLinkOp
//===----------------------------------------------------------------------===//

LogicalResult ObjectFifoLinkOp::verify() {
  if (isJoin() && isDistribute())
    return emitError("ObjectFifoLinkOp does not support 'join' and "
                     "'distribute' at the same time");

  if (auto sharedTile = getOptionalSharedTile(); !sharedTile)
    return emitError("ObjectFifoLinkOp must have a link point, i.e., a "
                     "shared tile between objectFifos");

  if (isJoin()) {
    if (getFifoIns().size() != getSrcOffsets().size())
      return emitOpError("number of provided src offsets must be equal "
                         "to the number of input objectFifos");

    if (!getDstOffsets().empty())
      return emitOpError("dst offsets should be empty for join");

  } else if (isDistribute()) {
    if (getFifoOuts().size() != getDstOffsets().size())
      return emitOpError("number of provided dst offsets must be equal "
                         "to the number of output objectFifos");

    if (!getSrcOffsets().empty())
      return emitOpError("src offsets should be empty for distribute");

    ObjectFifoCreateOp fifoIn = getInputObjectFifos()[0];
    if (!fifoIn.getDimensionsToStream().empty()) {
      return emitOpError("currently does not support objectFifos with "
                         "dimensionsToStream.");
    }
    for (auto dims : fifoIn.getDimensionsFromStreamPerConsumer()) {
      if (!dims.empty())
        return emitOpError("currently does not support objectFifos with "
                           "dimensionsFromStreamPerConsumer.");
    }

    for (auto fifoOut : getOutputObjectFifos()) {
      for (auto dims : fifoOut.getDimensionsFromStreamPerConsumer()) {
        if (!dims.empty())
          return emitOpError("currently does not support objectFifos with "
                             "dimensionsFromStreamPerConsumer.");
      }
    }

    std::vector<int> repeat_counts;
    for (auto fifoOut : getOutputObjectFifos()) {
      if (fifoOut.getMemtileRepeat().has_value())
        repeat_counts.push_back(fifoOut.getMemtileRepeat().value());
      else
        repeat_counts.push_back(0);
    }
    for (auto repeat : repeat_counts)
      if (repeat_counts[0] != repeat)
        return emitError("repeat counts of output object FIFOs must be equal");

  } else {
    if (!getSrcOffsets().empty() && !getDstOffsets().empty())
      return emitOpError("all offsets should be empty if there is no "
                         "join or distribute");
  }

  return success();
}

std::optional<Value> ObjectFifoLinkOp::getOptionalSharedTile() {
  if (isJoin()) {
    auto fifoOut = getOutputObjectFifos()[0];
    for (auto fifoIn : getInputObjectFifos())
      if (fifoOut.getProducerTile() != fifoIn.getConsumerTiles()[0])
        return {};
    return {fifoOut.getProducerTile()};
  }

  if (isDistribute()) {
    auto fifoIn = getInputObjectFifos()[0];
    for (auto fifoOut : getOutputObjectFifos())
      if (fifoIn.getConsumerTiles()[0] != fifoOut.getProducerTile())
        return {};
    return {fifoIn.getConsumerTiles()[0]};
  }

  auto fifoIn = getInputObjectFifos();
  if (auto fifoOut = getOutputObjectFifos();
      !fifoIn.empty() && !fifoOut.empty())
    for (auto consumerIn : fifoIn[0].getConsumerTiles())
      if (consumerIn == fifoOut[0].getProducerTile())
        return {fifoOut[0].getProducerTile()};
  return {};
}

std::vector<ObjectFifoCreateOp> ObjectFifoLinkOp::getInputObjectFifos() {
  std::vector<ObjectFifoCreateOp> inputObjFifos;
  Operation *parent = getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      for (auto sym : getFifoIns()) {
        auto name = dyn_cast<FlatSymbolRefAttr>(sym);
        if (auto *st = SymbolTable::lookupSymbolIn(parent, name);
            isa_and_nonnull<ObjectFifoCreateOp>(st))
          inputObjFifos.push_back(dyn_cast<ObjectFifoCreateOp>(st));
      }
    }
  }
  return inputObjFifos;
}

std::vector<ObjectFifoCreateOp> ObjectFifoLinkOp::getOutputObjectFifos() {
  std::vector<ObjectFifoCreateOp> outputObjFifos;
  Operation *parent = getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      for (auto sym : getFifoOuts()) {
        auto name = dyn_cast<FlatSymbolRefAttr>(sym);
        if (auto *st = SymbolTable::lookupSymbolIn(parent, name);
            isa_and_nonnull<ObjectFifoCreateOp>(st))
          outputObjFifos.push_back(dyn_cast<ObjectFifoCreateOp>(st));
      }
    }
  }
  return outputObjFifos;
}

std::vector<int> ObjectFifoLinkOp::getJoinTransferLengths() {
  std::vector<int> lengths;
  if (isJoin()) {
    auto fifoOut =
        llvm::cast<AIEObjectFifoType>(getOutputObjectFifos()[0].getElemType());
    auto elemTypeOut = llvm::cast<MemRefType>(fifoOut.getElementType());
    int lenOut = elemTypeOut.getNumElements();
    for (size_t i = 0; i < getFifoIns().size(); i++) {
      int len = 0;
      int offset = *getConstantIntValue(getSrcOffsets()[i]);
      if (i == getFifoIns().size() - 1)
        len = lenOut - *getConstantIntValue(getSrcOffsets()[i]);
      else
        len = *getConstantIntValue(getSrcOffsets()[i + 1]) - offset;
      lengths.push_back(len);
    }
  }
  return lengths;
}

std::vector<int> ObjectFifoLinkOp::getDistributeTransferLengths() {
  std::vector<int> lengths;
  if (isDistribute()) {
    auto fifoIn =
        llvm::cast<AIEObjectFifoType>(getInputObjectFifos()[0].getElemType());
    auto elemTypeIn = llvm::cast<MemRefType>(fifoIn.getElementType());
    int lenIn = elemTypeIn.getNumElements();
    for (size_t i = 0; i < getFifoOuts().size(); i++) {
      int offset = *getConstantIntValue(getDstOffsets()[i]);
      int len = 0;
      if (i == getFifoOuts().size() - 1)
        len = lenIn - *getConstantIntValue(getDstOffsets()[i]);
      else
        len = *getConstantIntValue(getDstOffsets()[i + 1]) - offset;
      lengths.push_back(len);
    }
  }
  return lengths;
}

std::optional<int> ObjectFifoLinkOp::getRepeatCount() {
  for (auto fifoOut : getOutputObjectFifos())
    if (fifoOut.getMemtileRepeat().has_value())
      return {fifoOut.getMemtileRepeat().value()};
  return {};
}

//===----------------------------------------------------------------------===//
// ObjectFifoRegisterExternalBuffersOp
//===----------------------------------------------------------------------===//

LogicalResult ObjectFifoRegisterExternalBuffersOp::verify() {
  if (!getTileOp().isShimTile())
    return emitOpError("tile is not a shim tile");

  return success();
}

TileOp ObjectFifoRegisterExternalBuffersOp::getTileOp() {
  return cast<TileOp>(getTile().getDefiningOp());
}

ObjectFifoCreateOp ObjectFifoRegisterExternalBuffersOp::getObjectFifo() {
  Operation *parent = getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      if (auto *st = SymbolTable::lookupSymbolIn(parent, getObjFifoName());
          isa_and_nonnull<ObjectFifoCreateOp>(st))
        return dyn_cast<ObjectFifoCreateOp>(st);
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ObjectFifoAcquireOp
//===----------------------------------------------------------------------===//

LogicalResult ObjectFifoAcquireOp::verify() {
  if (acqNumber() < 1)
    return emitOpError("must acquire at least one element");

  auto parent = getOperation()->getParentOfType<CoreOp>();
  if (parent == nullptr)
    return emitOpError("must be called from inside a CoreOp");

  auto coreTile = parent.getTile();
  auto objFifo = getObjectFifo();
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

  auto objFifoElem =
      llvm::cast<AIEObjectFifoType>(getObjectFifo().getElemType())
          .getElementType();
  auto objFifoSubviewElem =
      llvm::cast<AIEObjectFifoSubviewType>(getResult().getType())
          .getElementType();
  if (objFifoElem != objFifoSubviewElem)
    return emitOpError(
        "ObjectFifo element and ObjectFifoSubview element must match.\n");

  return success();
}

ObjectFifoCreateOp ObjectFifoAcquireOp::getObjectFifo() {
  Operation *parent = getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      if (auto *st = SymbolTable::lookupSymbolIn(parent, getObjFifoName());
          isa_and_nonnull<ObjectFifoCreateOp>(st))
        return dyn_cast<ObjectFifoCreateOp>(st);
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ObjectFifoReleaseOp
//===----------------------------------------------------------------------===//

LogicalResult ObjectFifoReleaseOp::verify() {
  if (relNumber() < 1)
    return emitOpError("must release at least one element");

  auto parent = getOperation()->getParentOfType<CoreOp>();
  if (parent == nullptr)
    return emitOpError("must be called from inside a CoreOp");

  auto coreTile = parent.getTile();
  auto objFifo = getObjectFifo();
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

ObjectFifoCreateOp ObjectFifoReleaseOp::getObjectFifo() {
  Operation *parent = getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      if (auto *st = SymbolTable::lookupSymbolIn(parent, getObjFifoName());
          isa_and_nonnull<ObjectFifoCreateOp>(st))
        return dyn_cast<ObjectFifoCreateOp>(st);
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ObjectFifoSubviewAccessOp
//===----------------------------------------------------------------------===//

LogicalResult ObjectFifoSubviewAccessOp::verify() {
  if (auto parent = getOperation()->getParentOfType<CoreOp>();
      parent == nullptr)
    return emitOpError("must be called from inside a CoreOp");

  if (auto acqOp = getSubview().getDefiningOp<ObjectFifoAcquireOp>();
      getIndex() >= acqOp.acqNumber())
    return emitOpError("accessed farther than number of acquired elements "
                       "(index out of bounds).");

  return success();
}

//===----------------------------------------------------------------------===//
// ObjectFifoRegisterProcessOp
//===----------------------------------------------------------------------===//

LogicalResult ObjectFifoRegisterProcessOp::verify() {
  if (getProcessLength() < 1)
    return emitOpError("process length must be >= 1");

  if (getAcquirePattern().size() != getReleasePattern().size()) {
    // acquire pattern size = process length (i.e., release pattern will be
    // duplicated by process length times) OR the other way around
    if (getAcquirePattern().size() != getProcessLength() &&
        getProcessLength() != getReleasePattern().size())
      return emitOpError(
          "Acquire and Release patterns must be of equal length, or "
          "longest length of one must be equal to process "
          "length of the other");
  }

  return success();
}

ObjectFifoCreateOp ObjectFifoRegisterProcessOp::getObjectFifo() {
  Operation *parent = getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      if (auto *st = SymbolTable::lookupSymbolIn(parent, getObjFifoName());
          isa_and_nonnull<ObjectFifoCreateOp>(st))
        return dyn_cast<ObjectFifoCreateOp>(st);
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// CascadeFlowOp
//===----------------------------------------------------------------------===//

LogicalResult CascadeFlowOp::verify() {
  TileOp src = getSourceTileOp();
  TileOp dst = getDestTileOp();
  const auto &t = getTargetModel(src);

  if (src.isShimTile() || dst.isShimTile())
    return emitOpError("shimTile row has no cascade stream interface");
  if (t.isMemTile(src.colIndex(), src.rowIndex()) ||
      t.isMemTile(dst.colIndex(), dst.rowIndex()))
    return emitOpError("memTile row has no cascade stream interface");

  if (!t.isSouth(src.getCol(), src.getRow(), dst.getCol(), dst.getRow()) &&
      !t.isWest(src.getCol(), src.getRow(), dst.getCol(), dst.getRow()) &&
      !t.isNorth(src.getCol(), src.getRow(), dst.getCol(), dst.getRow()) &&
      !t.isEast(src.getCol(), src.getRow(), dst.getCol(), dst.getRow())) {
    return emitOpError("tiles must be adjacent");
  }
  return success();
}

TileOp CascadeFlowOp::getSourceTileOp() {
  return cast<TileOp>(getSourceTile().getDefiningOp());
}

TileOp CascadeFlowOp::getDestTileOp() {
  return cast<TileOp>(getDestTile().getDefiningOp());
}

//===----------------------------------------------------------------------===//
// ConfigureCascadeOp
//===----------------------------------------------------------------------===//

LogicalResult ConfigureCascadeOp::verify() {
  const auto &t = getTargetModel(*this);
  TileOp tile = cast<TileOp>(getTile().getDefiningOp());
  CascadeDir inputDir = getInputDir();
  CascadeDir outputDir = getOutputDir();

  if (tile.isShimTile())
    return emitOpError("shimTile row has no cascade stream interface");
  if (t.isMemTile(tile.colIndex(), tile.rowIndex()))
    return emitOpError("memTile row has no cascade stream interface");

  if ((t.getTargetArch() == AIEArch::AIE2) ||
      (t.getTargetArch() == AIEArch::AIE2p)) {
    if (inputDir == CascadeDir::South || inputDir == CascadeDir::East) {
      return emitOpError("input direction of cascade must be North or West on ")
             << stringifyAIEArch(t.getTargetArch());
    }
    if (outputDir == CascadeDir::North || outputDir == CascadeDir::West) {
      return emitOpError(
                 "output direction of cascade must be South or East on ")
             << stringifyAIEArch(t.getTargetArch());
    }
  } else {
    return emitOpError("cascade not supported in ")
           << stringifyAIEArch(t.getTargetArch());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PutCascadeOp
//===----------------------------------------------------------------------===//

LogicalResult PutCascadeOp::verify() {
  const auto &targetModel = getTargetModel(*this);
  Type type = getCascadeValue().getType();
  DataLayout dataLayout = DataLayout::closest(*this);
  auto bits = dataLayout.getTypeSizeInBits(type);
  auto archbits = targetModel.getAccumulatorCascadeSize();
  if (bits != archbits)
    return emitOpError("type must match architecture cascade width (")
           << archbits << " bits in "
           << stringifyAIEArch(targetModel.getTargetArch()) << ")";
  return success();
}

//===----------------------------------------------------------------------===//
// GetCascadeOp
//===----------------------------------------------------------------------===//

LogicalResult GetCascadeOp::verify() {
  const auto &targetModel = getTargetModel(*this);
  Type type = getCascadeValue().getType();
  DataLayout dataLayout = DataLayout::closest(*this);
  auto bits = dataLayout.getTypeSizeInBits(type);
  if (targetModel.getTargetArch() == AIEArch::AIE1) {
    if (bits != 384)
      return emitOpError("must be a 384-bit type");
  } else if ((targetModel.getTargetArch() == AIEArch::AIE2) ||
             (targetModel.getTargetArch() == AIEArch::AIE2p)) {
    if (bits != 512)
      return emitOpError("must be a 512-bit type");
  } else
    return emitOpError("cascade not supported in ")
           << stringifyAIEArch(targetModel.getTargetArch());
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceOp
//===----------------------------------------------------------------------===//

const AIETargetModel &DeviceOp::getTargetModel() {
  return xilinx::AIE::getTargetModel(getDevice());
}

LogicalResult DeviceOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

LogicalResult TileOp::verify() {
  const auto &targetModel = getTargetModel(*this);
  int columns = targetModel.columns();
  int rows = targetModel.rows();
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
  for (auto *user : users) {
    if (llvm::isa<SwitchboxOp>(*user)) {
      if (found)
        return emitOpError("can only have one switchbox");
      found = true;
    }
  }

  return success();
}

size_t TileOp::getNumSourceConnections(WireBundle bundle) {
  const auto &targetModel = getTargetModel(*this);
  if (bundle == WireBundle::Core || bundle == WireBundle::DMA)
  // Note dest is correct here, since direction is reversed.
  {
    // Note dest is correct here, since direction is reversed.
    if (targetModel.isShimNOCTile(getCol(), getRow()) ||
        targetModel.isShimPLTile(getCol(), getRow()))
      return targetModel.getNumDestShimMuxConnections(getCol(), getRow(),
                                                      bundle);
    return targetModel.getNumDestSwitchboxConnections(getCol(), getRow(),
                                                      bundle);
  }
  return 0;
}

size_t TileOp::getNumDestConnections(WireBundle bundle) {
  const auto &targetModel = getTargetModel(*this);
  if (bundle == WireBundle::Core || bundle == WireBundle::DMA)
  // Note source is correct here, since direction is reversed.
  {
    // Note source is correct here, since direction is reversed.
    if (targetModel.isShimNOCTile(getCol(), getRow()) ||
        targetModel.isShimPLTile(getCol(), getRow()))
      return targetModel.getNumDestShimMuxConnections(getCol(), getRow(),
                                                      bundle);
    return targetModel.getNumSourceSwitchboxConnections(getCol(), getRow(),
                                                        bundle);
  }
  return 0;
}

bool TileOp::isMemTile() {
  const auto &targetModel = getTargetModel(*this);
  return targetModel.isMemTile(getCol(), getRow());
}

bool TileOp::isShimNOCTile() {
  const auto &targetModel = getTargetModel(*this);
  return targetModel.isShimNOCTile(getCol(), getRow());
}

bool TileOp::isShimPLTile() {
  const auto &targetModel = getTargetModel(*this);
  return targetModel.isShimPLTile(getCol(), getRow());
}

bool TileOp::isShimNOCorPLTile() {
  const auto &targetModel = getTargetModel(*this);
  return targetModel.isShimNOCorPLTile(getCol(), getRow());
}

bool isLegalTileConnection(TileOp tile, const AIETargetModel &targetModel,
                           MasterSetOp masterOp, PacketRulesOp slaveOp) {
  auto srcBundle = slaveOp.sourcePort().bundle;
  auto srcChan = slaveOp.sourcePort().channel;
  auto dstBundle = masterOp.destPort().bundle;
  auto dstChan = masterOp.destPort().channel;
  return targetModel.isLegalTileConnection(
      tile.colIndex(), tile.rowIndex(), srcBundle, srcChan, dstBundle, dstChan);
}

bool isLegalTileConnection(TileOp tile, const AIETargetModel &targetModel,
                           ConnectOp connectOp) {
  auto srcBundle = connectOp.getSourceBundle();
  auto srcChan = connectOp.getSourceChannel();
  auto dstBundle = connectOp.getDestBundle();
  auto dstChan = connectOp.getDestChannel();
  return targetModel.isLegalTileConnection(
      tile.colIndex(), tile.rowIndex(), srcBundle, srcChan, dstBundle, dstChan);
}

TileOp TileOp::getOrCreate(mlir::OpBuilder builder, DeviceOp device, int col,
                           int row) {
  TileOp tile = nullptr;
  // Find matching predefined tile at device top level, ...
  for (auto t : device.getOps<AIE::TileOp>()) {
    if (t.getRow() == row && t.getCol() == col) {
      tile = t;
      break;
    }
  }
  // ... or if undefined, create a new tile op
  if (!tile) {
    OpBuilder::InsertionGuard guard(builder);
    mlir::Block &device_start_block = *device.getBodyRegion().begin();
    builder.setInsertionPointToStart(&device_start_block);
    tile = builder.create<TileOp>(builder.getUnknownLoc(),
                                  builder.getIndexType(), col, row);
  }
  return tile;
}

//===----------------------------------------------------------------------===//
// ShimSwitchboxOp
//===----------------------------------------------------------------------===//

LogicalResult ShimSwitchboxOp::verify() {
  Region &body = getConnections();
  DenseSet<Port> destset;
  if (body.empty())
    return emitOpError("should have non-empty body");

  for (auto &ops : body.front()) {
    if (auto connectOp = dyn_cast<ConnectOp>(ops)) {
      Port dest = {connectOp.getDestBundle(), connectOp.destIndex()};
      if (destset.count(dest))
        return connectOp.emitOpError("targets same destination ")
               << stringifyWireBundle(dest.bundle) << ": " << dest.channel
               << " as another connect operation";
      destset.insert(dest);
    } else if (isa<EndOp>(ops)) {
      // continue;
    } else {
      return ops.emitOpError("cannot be contained in a Switchbox op");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ShimMuxOp
//===----------------------------------------------------------------------===//

LogicalResult ShimMuxOp::verify() {
  Region &body = getConnections();
  DenseSet<Port> destset;
  if (body.empty())
    return emitOpError("should have non-empty body");

  for (auto &ops : body.front()) {
    if (auto connectOp = dyn_cast<ConnectOp>(ops)) {
      Port dest = {connectOp.getDestBundle(), connectOp.destIndex()};
      if (destset.count(dest))
        return connectOp.emitOpError("targets same destination ")
               << stringifyWireBundle(dest.bundle) << ": " << dest.channel
               << " as another connect operation";
      destset.insert(dest);
    } else if (isa<EndOp>(ops)) {
      // continue;
    } else {
      return ops.emitOpError("cannot be contained in a Switchbox op");
    }
  }
  return success();
}

size_t ShimMuxOp::getNumSourceConnections(WireBundle bundle) {
  auto tile = getTileOp();
  const auto &targetModel = getTargetModel(*this);
  return targetModel.getNumSourceShimMuxConnections(tile.getCol(),
                                                    tile.getRow(), bundle);
}

size_t ShimMuxOp::getNumDestConnections(WireBundle bundle) {
  auto tile = getTileOp();
  const auto &targetModel = getTargetModel(*this);
  return targetModel.getNumDestShimMuxConnections(tile.getCol(), tile.getRow(),
                                                  bundle);
}

TileOp ShimMuxOp::getTileOp() {
  return cast<TileOp>(getTile().getDefiningOp());
}

int ShimMuxOp::colIndex() { return getTileOp().colIndex(); }

int ShimMuxOp::rowIndex() { return getTileOp().rowIndex(); }

//===----------------------------------------------------------------------===//
// ShimDMAOp
//===----------------------------------------------------------------------===//

LogicalResult ShimDMAOp::verify() {
  Region &body = getBody();
  DenseSet<DMAChannel> usedChannels;
  std::vector<DMAChannel> inputChannels;
  std::vector<DMAChannel> outputChannels;

  if (getBody().empty())
    return emitOpError("should have non-empty body");

  if (!getTileOp().isShimNOCTile())
    return emitOpError("must be in a ShimTile with a NOC connection");

  if (HasSomeTerminator<DMAStartOp, NextBDOp, EndOp>::verifyTrait(*this)
          .failed())
    return failure();

  for (auto &bodyOp : body.getOps()) {
    // check for duplicate DMA channels within the same ShimDMAOp
    if (auto dmaStart = dyn_cast<DMAStartOp>(bodyOp)) {
      DMAChannel dmaChan = {dmaStart.getChannelDir(),
                            dmaStart.getChannelIndex()};
      if (usedChannels.count(dmaChan))
        return dmaStart.emitOpError()
               << "duplicate DMA channel "
               << stringifyDMAChannelDir(dmaChan.direction) << dmaChan.channel
               << " in MemOp";
      usedChannels.insert(dmaChan);
      // check if number of input and output channels is more than available
      // hardware
      if (dmaChan.direction == DMAChannelDir::S2MM)
        inputChannels.push_back(dmaChan);
      else
        outputChannels.push_back(dmaChan);
    }
  }

  if (inputChannels.size() >
      getTileOp().getNumSourceConnections(WireBundle::DMA))
    return emitOpError("uses more input channels than available on this tile");

  if (outputChannels.size() >
      getTileOp().getNumDestConnections(WireBundle::DMA))
    return emitOpError("uses more output channels than available on this tile");

  return success();
}

TileOp ShimDMAOp::getTileOp() {
  return cast<TileOp>(getTile().getDefiningOp());
}

int ShimDMAOp::colIndex() { return getTileOp().colIndex(); }

int ShimDMAOp::rowIndex() { return getTileOp().rowIndex(); }

LogicalResult PacketRulesOp::verify() {
  if (Region &body = getRules(); body.empty())
    return emitOpError("should have non-empty body");
  return success();
}

LogicalResult PacketFlowOp::verify() {
  Region &body = getPorts();
  if (body.empty())
    return emitOpError("should have non-empty body");

  for (auto &ops : body.front()) {
    if (!isa<PacketSourceOp, PacketDestOp, EndOp>(ops))
      return ops.emitOpError("cannot be contained in a PacketFlow op");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CoreOp
//===----------------------------------------------------------------------===//

LogicalResult CoreOp::verify() {
  if (getBody().empty())
    return emitOpError("should have non-empty body");
  if (getTileOp().isShimTile())
    return emitOpError("CoreOp cannot be created on shim tile, i.e. row == 0");
  if (getTileOp().isMemTile())
    return emitOpError("CoreOp cannot be created on mem tile");
  return success();
}

int CoreOp::colIndex() { return getTileOp().colIndex(); }

int CoreOp::rowIndex() { return getTileOp().rowIndex(); }

TileOp CoreOp::getTileOp() { return cast<TileOp>(getTile().getDefiningOp()); }

//===----------------------------------------------------------------------===//
// BufferOp
//===----------------------------------------------------------------------===//

int64_t BufferOp::getAllocationSize() {
  auto type = llvm::cast<MemRefType>(getType());
  return type.getNumElements() * type.getElementTypeBitWidth() / 8;
}

TileOp BufferOp::getTileOp() { return cast<TileOp>(getTile().getDefiningOp()); }

LogicalResult BufferOp::verify() {
  if (UsesAreAccessible::verifyTrait(*this).failed())
    return failure();
  return success();
}

// FIXME: make address assignment for buffers explicit and move this function to
// an interface
int32_t xilinx::AIE::getBufferBaseAddress(Operation *bufOp) {
  if (auto buf = dyn_cast<BufferOp>(bufOp)) {
    assert(buf.getAddress().has_value() && "buffer must have address assigned");
    return buf.getAddress().value();
  }
  if (isa_and_nonnull<ExternalBufferOp>(bufOp))
    llvm::report_fatal_error(
        "External buffer addresses are assigned at runtime.");
  llvm::report_fatal_error("unknown buffer type");
}

void xilinx::AIE::collectTiles(DeviceOp &device,
                               DenseMap<TileID, Operation *> &tiles) {
  for (auto tile : device.getOps<TileOp>()) {
    int colIndex = tile.colIndex();
    int rowIndex = tile.rowIndex();
    tiles[{colIndex, rowIndex}] = tile;
  }
}

void xilinx::AIE::collectBuffers(
    DeviceOp &device,
    DenseMap<Operation *, SmallVector<BufferOp, 4>> &buffers) {
  for (BufferOp buffer : device.getOps<BufferOp>()) {
    Operation *tileOp = buffer.getTile().getDefiningOp();
    buffers[tileOp].push_back(buffer);
  }
}

static void printBufferInitialValue(OpAsmPrinter &p, BufferOp op, Type type,
                                    Attribute initialValue) {
  if (op.getInitialValue()) {
    p << "= ";
    p.printAttributeWithoutType(initialValue);
  }
}

static ParseResult parseBufferInitialValue(OpAsmParser &parser, Type &type,
                                           Attribute &initialValue) {
  auto memrefType = llvm::cast<MemRefType>(type);
  if (!memrefType.hasStaticShape())
    return parser.emitError(parser.getNameLoc())
           << "type should be static shaped memref, but got " << type;

  if (parser.parseOptionalEqual())
    return success();

  Type tensorType = mlir::memref::getTensorTypeFromMemRefType(memrefType);
  if (parser.parseAttribute(initialValue, tensorType))
    return failure();
  if (!llvm::isa<ElementsAttr>(initialValue))
    return parser.emitError(parser.getNameLoc())
           << "initial value should be an elements attribute";
  return success();
}

//===----------------------------------------------------------------------===//
// MemOp
//===----------------------------------------------------------------------===//

LogicalResult MemOp::verify() {
  Region &body = getBody();
  DenseSet<DMAChannel> usedChannels;
  std::vector<DMAChannel> inputChannels;
  std::vector<DMAChannel> outputChannels;
  if (body.empty())
    return emitOpError("should have non-empty body");

  if (HasSomeTerminator<DMAStartOp, NextBDOp, EndOp>::verifyTrait(*this)
          .failed())
    return failure();

  for (auto &bodyOp : body.getOps()) {
    // check for duplicate DMA channels within the same MemOp
    if (auto dmaStart = dyn_cast<DMAStartOp>(bodyOp)) {
      DMAChannel dmaChan = {dmaStart.getChannelDir(),
                            dmaStart.getChannelIndex()};
      if (usedChannels.count(dmaChan))
        return dmaStart.emitOpError()
               << "duplicate DMA channel "
               << stringifyDMAChannelDir(dmaChan.direction) << dmaChan.channel
               << " in MemOp";
      usedChannels.insert(dmaChan);
      // check if number of input and output channels is more than available
      // hardware
      if (dmaChan.direction == DMAChannelDir::S2MM)
        inputChannels.push_back(dmaChan);
      else
        outputChannels.push_back(dmaChan);
    }

    if (auto allocOp = dyn_cast<memref::AllocOp>(bodyOp))
      if (!allocOp->getAttr("id"))
        return allocOp.emitOpError()
               << "allocOp in MemOp region should have an id attribute";
  }

  if (inputChannels.size() >
      getTileOp().getNumSourceConnections(WireBundle::DMA))
    return emitOpError("uses more input channels than available on this tile");

  if (outputChannels.size() >
      getTileOp().getNumDestConnections(WireBundle::DMA))
    return emitOpError("uses more output channels than available on this tile");

  return success();
}

TileOp MemOp::getTileOp() { return cast<TileOp>(getTile().getDefiningOp()); }

int MemOp::colIndex() { return getTileOp().colIndex(); }

int MemOp::rowIndex() { return getTileOp().rowIndex(); }

/// Returns the region on the current operation that is callable. This may
/// return nullptr in the case of an external callable object, e.g. an external
/// function.
Region *MemOp::getCallableRegion() { return &getBody(); }

//===----------------------------------------------------------------------===//
// MemTileDMAOp
//===----------------------------------------------------------------------===//

LogicalResult MemTileDMAOp::verify() {
  std::vector<DMAChannel> inputChannels;
  std::vector<DMAChannel> outputChannels;

  assert(getOperation()->getNumRegions() == 1 &&
         "MemTileDMAOp has zero region!");
  assert(!getBody().empty() && "MemTileDMAOp should have non-empty body");

  if (HasSomeTerminator<DMAStartOp, NextBDOp, EndOp>::verifyTrait(*this)
          .failed())
    return failure();

  for (auto &bodyOp : getBody().getOps()) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(bodyOp)) {
      if (!allocOp->getAttr("id"))
        return allocOp.emitOpError()
               << "allocOp in MemTileDMAOp region should have an id attribute";
    }
    if (auto startOp = dyn_cast<DMAStartOp>(bodyOp)) {
      // check if number of input and output channels is more than available
      // hardware
      DMAChannel dmaChan = {startOp.getChannelDir(), startOp.getChannelIndex()};
      if (dmaChan.direction == DMAChannelDir::S2MM)
        inputChannels.push_back(dmaChan);
      else
        outputChannels.push_back(dmaChan);

      if (startOp.getChannelIndex() > 3) {
        // Channels 4 and 5 in a memtile are restricted to only access local
        // buffers and locks.

        // TODO: Move this code to the dialect
        // Set of blocks found to be reachable within a given region.
        llvm::SmallSet<Block *, 16> reachable;
        SmallVector<Block *, 16> worklist;
        Block *firstBD = startOp.getSuccessor(0);
        reachable.insert(firstBD);
        worklist.push_back(firstBD);
        while (!worklist.empty()) {
          Block *block = worklist.pop_back_val();
          if (block->empty())
            continue;
          auto successors = block->getTerminator()->getSuccessors();
          for (auto *i : successors) {
            if (!reachable.contains(i)) {
              reachable.insert(i);
              worklist.push_back(i);
            }
          }
        }
        for (Block *b : reachable) {
          for (DMABDOp bd : b->getOps<DMABDOp>()) {
            if (auto bufferOp = bd.getBufferOp();
                bufferOp.getTileOp().colIndex() != colIndex() ||
                bufferOp.getTileOp().rowIndex() != rowIndex()) {
              InFlightDiagnostic err =
                  bd.emitOpError()
                  << "is reachable from DMA channel "
                  << startOp.getChannelIndex()
                  << " and attempts to access a non-local buffer\n";
              err.attachNote(startOp->getLoc()) << "channel";
              err.attachNote(bufferOp->getLoc()) << "buffer";
              return err;
            }
          }
          for (auto useLock : b->getOps<UseLockOp>()) {
            if (auto lockOp = useLock.getLockOp();
                lockOp.getTileOp().colIndex() != colIndex() ||
                lockOp.getTileOp().rowIndex() != rowIndex()) {
              InFlightDiagnostic err =
                  useLock.emitOpError()
                  << "is reachable from DMA channel "
                  << startOp.getChannelIndex()
                  << " and attempts to access a non-local lock\n";
              err.attachNote(startOp->getLoc()) << "channel";
              err.attachNote(lockOp->getLoc()) << "lock";
              return err;
            }
          }
        }
      }
    }
  }

  if (inputChannels.size() >
      getTileOp().getNumSourceConnections(WireBundle::DMA))
    return emitOpError("uses more input channels than available on this tile");

  if (outputChannels.size() >
      getTileOp().getNumDestConnections(WireBundle::DMA))
    return emitOpError("uses more output channels than available on this tile");

  return success();
}

//===----------------------------------------------------------------------===//
// DMAOp
//===----------------------------------------------------------------------===//

LogicalResult DMAOp::verify() {
  auto *parentOp = getOperation()->getParentOp();
  if (parentOp->getRegion(0).getBlocks().size() > 1)
    return emitOpError("DMAOp can only appear in single block region");
  if (!parentOp->getRegion(0).getOps<DMAStartOp>().empty())
    return emitOpError("DMAOp is not compatible with DMAStart ops");
  auto bdRegions = getBds();
  for (auto &bdRegion : bdRegions) {
    if (!bdRegion.hasOneBlock())
      return emitOpError("DMAOp regions must have only one block");
    auto bds = llvm::to_vector_of<DMABDOp>(bdRegion.front().getOps<DMABDOp>());
    if (bds.size() != 1)
      return emitOpError("DMAOp regions/blocks must have exactly one DMABDOp");
    auto useLocks =
        llvm::to_vector_of<UseLockOp>(bdRegion.front().getOps<UseLockOp>());
    if (useLocks.size() != 2)
      return emitOpError(
          "DMAOp regions/blocks must have exactly two UseLock ops");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DMABDOp
//===----------------------------------------------------------------------===//

BufferOp DMABDOp::getBufferOp() {
  return cast<BufferOp>(getBuffer().getDefiningOp());
}

// let assemblyFormat = [{
//   `(` $buffer `:` type($buffer) (`,` $offset^)? (`,` $len^)? (`,`
//   $dimensions^)? (`,` $pad_dimensions^)? (`,` `pad_value` `=` $pad_value^)?
//   `)` attr-dict
// }];
ParseResult DMABDOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand bufferRawOperand{};
  ::llvm::ArrayRef<OpAsmParser::UnresolvedOperand> bufferOperands(
      &bufferRawOperand, 1);
  ::llvm::SMLoc bufferOperandsLoc;
  (void)bufferOperandsLoc;
  Type bufferRawType{};
  ::llvm::ArrayRef<Type> bufferTypes(&bufferRawType, 1);
  IntegerAttr offsetAttr;
  IntegerAttr lenAttr;
  ::xilinx::AIE::BDDimLayoutArrayAttr dimensionsAttr;
  ::xilinx::AIE::BDPadLayoutArrayAttr pad_dimensionsAttr;
  IntegerAttr pad_valueAttr;
  if (parser.parseLParen())
    return failure();

  bufferOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bufferRawOperand))
    return failure();
  if (parser.parseColon())
    return failure();
  if (parser.parseCustomTypeWithFallback(bufferRawType))
    return failure();

  // offset
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseCustomAttributeWithFallback(
            offsetAttr, parser.getBuilder().getIntegerType(32))) {
      return failure();
    }
    if (!offsetAttr)
      offsetAttr = parser.getBuilder().getIntegerAttr(
          parser.getBuilder().getIntegerType(32), 0);
    result.getOrAddProperties<DMABDOp::Properties>().offset = offsetAttr;
  }

  // len
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseCustomAttributeWithFallback(
            lenAttr, parser.getBuilder().getIntegerType(32))) {
      return failure();
    }
    if (lenAttr)
      result.getOrAddProperties<DMABDOp::Properties>().len = lenAttr;
  }

  // dimensions
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseCustomAttributeWithFallback(dimensionsAttr, Type{})) {
      return failure();
    }
    if (dimensionsAttr)
      result.getOrAddProperties<DMABDOp::Properties>().dimensions =
          dimensionsAttr;
  }

  // pad_dimensions
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseCustomAttributeWithFallback(pad_dimensionsAttr, Type{})) {
      return failure();
    }
    if (pad_dimensionsAttr)
      result.getOrAddProperties<DMABDOp::Properties>().pad_dimensions =
          pad_dimensionsAttr;
  }

  // pad_value
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword("pad_value"))
      return failure();
    if (parser.parseEqual())
      return failure();

    if (parser.parseCustomAttributeWithFallback(
            pad_valueAttr, parser.getBuilder().getIntegerType(32))) {
      return failure();
    }
    if (pad_valueAttr)
      result.getOrAddProperties<DMABDOp::Properties>().pad_value =
          pad_valueAttr;
  }
  if (parser.parseRParen())
    return failure();

  auto loc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc)
               << "'" << result.name.getStringRef() << "' op ";
      })))
    return failure();

  if (parser.resolveOperands(bufferOperands, bufferTypes, bufferOperandsLoc,
                             result.operands))
    return failure();

  return success();
}

void DMABDOp::print(::mlir::OpAsmPrinter &printer) {
  printer << "(";
  printer << getBuffer();
  printer << ' ' << ":";
  printer << ' ';
  {
    auto type = getBuffer().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::MemRefType>(type))
      printer.printStrippedAttrOrType(validType);
    else
      printer << type;
  }
  if (getLenAttr() ||
      getOffsetAttr() !=
          ::mlir::OpBuilder((*this)->getContext())
              .getIntegerAttr(
                  ::mlir::OpBuilder((*this)->getContext()).getIntegerType(32),
                  0)) {
    printer << ",";
    printer << ' ';
    printer.printAttributeWithoutType(getOffsetAttr());
  }
  if (getLenAttr()) {
    printer << ",";
    printer << ' ';
    printer.printAttributeWithoutType(getLenAttr());
  }
  if (getDimensionsAttr()) {
    printer << ",";
    printer << ' ';
    printer.printStrippedAttrOrType(getDimensionsAttr());
  }
  if (getPadDimensionsAttr()) {
    printer << ",";
    printer << ' ';
    printer.printStrippedAttrOrType(getPadDimensionsAttr());
  }
  if ((getPadValueAttr() &&
       getPadValueAttr() !=
           ::mlir::OpBuilder((*this)->getContext())
               .getIntegerAttr(
                   ::mlir::OpBuilder((*this)->getContext()).getIntegerType(32),
                   0))) {
    printer << ",";
    printer << ' ' << "pad_value";
    printer << ' ' << "=";
    printer << ' ';
    printer.printAttributeWithoutType(getPadValueAttr());
  }
  printer << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("offset");
  elidedAttrs.push_back("len");
  elidedAttrs.push_back("dimensions");
  elidedAttrs.push_back("pad_dimensions");
  elidedAttrs.push_back("pad_value");
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

LogicalResult DMABDOp::verify() {
  // Skip verification of the BDOp outside of mem operations.
  // BDOps may appear elsewhere and subsequent lowerings will place them in the
  // correct mem ops.
  Operation *p = (*this)->getParentOp();
  if (!llvm::isa<MemOp, MemTileDMAOp, ShimDMAOp, DMAOp>(*p)) {
    return success();
  }

  if (!isa<BufferOp, ExternalBufferOp>(getBuffer().getDefiningOp()))
    return emitOpError(
        "BDs only support BufferOp or ExternalBufferOp operands.");

  if (getLenInBytes() % 4)
    return emitOpError("transfer length must be multiple of 4 (i.e., represent "
                       "4 byte aligned address)");

  TileID parentTileId = getParentTileElement(getOperation()).getTileID();

  if (getOperation()->getParentOfType<MemOp>() &&
      (getBufferOp().getTileOp().colIndex() != parentTileId.col ||
       getBufferOp().getTileOp().rowIndex() != parentTileId.row))
    return emitOpError(
        "Core tile DMAs can only access a buffer in the same tile.");

  const AIETargetModel &targetModel = getTargetModel(getOperation());

  uint32_t maxBds = targetModel.getNumBDs(parentTileId.col, parentTileId.row);
  if (std::optional<int32_t> bdId = getBdId();
      bdId.has_value() && static_cast<uint32_t>(*bdId) >= maxBds)
    return emitOpError("bdId attribute exceeds max: ") << maxBds - 1;
  if (std::optional<int32_t> nextBdId = getNextBdId();
      nextBdId.has_value() && static_cast<uint32_t>(*nextBdId) >= maxBds)
    return emitOpError("nextBdId attribute exceeds max: ") << maxBds - 1;
  if (auto dims = getDimensions(); dims.has_value()) {
    size_t maxNDims = 3;
    if (isa_and_nonnull<MemTileDMAOp>(getOperation()->getParentOp()))
      maxNDims = 4;
    if (dims->size() > maxNDims)
      return emitOpError() << "Cannot give more than "
                           << std::to_string(maxNDims)
                           << " dimensions for step sizes and wraps in this "
                              " tile (got "
                           << std::to_string(dims->size()) << " dimensions).";

    MemRefType buffer = getBuffer().getType();
    int64_t maxIdx = 0;
    for (BDDimLayoutAttr dim : *dims) {
      maxIdx += dim.getStride() * (dim.getSize() - 1);
      if (0 == dim.getStride())
        return emitOpError()
               << "Invalid step size; must be a positive integer.";
      if (dim.getStride() > buffer.getNumElements())
        return emitOpError()
               << "Step size " << std::to_string(dim.getStride()) << " "
               << "exceeds memref size "
               << std::to_string(buffer.getNumElements());
      if (dim.getSize() >= (1UL << 9) + 1)
        return emitOpError() << "Size may not exceed 1023.";
      if (dim.getStride() >= (1UL << 19))
        return emitOpError() << "Stride may not exceed " << (1 << 20);
    }

    if (buffer.getNumElements() <= maxIdx)
      return emitOpError() << "Specified stride(s) and size(s) result in out "
                              "of bounds access in buffer, for index "
                           << std::to_string(maxIdx) << " in memref of length "
                           << std::to_string(buffer.getNumElements()) << ".";

    // Since streams read 32b words, there's no way to read eg 16b with stride
    // of 2 (ie lower halfs of each 32b). So force it to be 1 (and then in
    // CDODirect/XAIEV2 scale the size by 4/getBufferElementTypeWidthInBytes).
    if (getBufferElementTypeWidthInBytes() < 4 && dims->back().getStride() != 1)
      return emitOpError(
          "For <32b width datatypes, inner-most dim stride must be 1");
  }
  if (auto paddims = getPadDimensions(); paddims.has_value()) {
    auto dims = getDimensions();
    if (!dims.has_value())
      return emitOpError() << "Padding requires n-d data layouts expressed as"
                           << " wrap(s) and stride(s).";
    if (dims->size() != paddims->size())
      return emitOpError() << "Mismatch number of dimensions between padding(s)"
                           << " and wrap(s) and stride(s).";
    if (!targetModel.isMemTile(parentTileId.col, parentTileId.row))
      return emitOpError() << "Padding is only supported by memtile dma bds.";
    int actuallen = 1;
    for (unsigned i = 0; i < paddims->size(); i++) {
      auto dim = (*dims)[i];
      auto paddim = (*paddims)[i];
      actuallen *= paddim.getConstPadBefore() + paddim.getConstPadAfter() +
                   dim.getSize();
      if (actuallen > getLen())
        return emitOpError() << "Data exceeds len after padding.";
    }
    if ((paddims->back().getConstPadBefore() *
         getBufferElementTypeWidthInBytes()) %
        4)
      return emitOpError() << "Inner-most padding-before count must result in"
                           << " padding in 32-bit words.";
    if ((paddims->back().getConstPadAfter() *
         getBufferElementTypeWidthInBytes()) %
        4)
      return emitOpError() << "Inner-most padding-after count must result in"
                           << " padding in 32-bit words.";
  }
  if (targetModel.isMemTile(parentTileId.col, parentTileId.row) ||
      targetModel.isCoreTile(parentTileId.col, parentTileId.row)) {
    if (auto baseAddr = getBufferOp().getAddress(); baseAddr.has_value()) {
      int offsetInBytes = *baseAddr + getOffsetInBytes();
      if (offsetInBytes % 4)
        return emitOpError("bd address must be 4 byte (32b) aligned; got "
                           "base+offset: ")
               << offsetInBytes << " (bytes)";
    }
  }
  if (auto packetInfo = getPacket()) {
    if (packetInfo->getPktType() > 7)
      return emitOpError("Packet type field can only hold 3 bits.");
    if (packetInfo->getPktId() > 31)
      return emitOpError("Packet ID field can only hold 5 bits.");
  }

  if (!getLen() && !getBuffer().getType().hasStaticShape())
    return emitOpError() << "buffer with dynamic shape requires static length.";

  return success();
}

TileOp MemTileDMAOp::getTileOp() {
  return cast<TileOp>(getTile().getDefiningOp());
}

int MemTileDMAOp::colIndex() { return getTileOp().colIndex(); }

int MemTileDMAOp::rowIndex() { return getTileOp().rowIndex(); }

/// Returns the region on the current operation that is callable. This may
/// return nullptr in the case of an external callable object, e.g. an
/// external function.
Region *MemTileDMAOp::getCallableRegion() { return &getBody(); }

//===----------------------------------------------------------------------===//
// SwitchboxOp
//===----------------------------------------------------------------------===//

LogicalResult SwitchboxOp::verify() {
  Region &body = getConnections();
  DenseSet<Port> sourceset;
  DenseSet<Port> destset;
  auto tile = getTileOp();
  const auto &targetModel = getTargetModel(tile);
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
        return ops.emitOpError()
               << dir << " bundle " << stringifyWireBundle(bundle)
               << " not supported; index: " << index << ", bound: " << bound;
      }
      return success();
    };

    if (auto connectOp = dyn_cast<ConnectOp>(ops)) {
      Port source = {connectOp.getSourceBundle(), connectOp.sourceIndex()};
      sourceset.insert(source);

      Port dest = {connectOp.getDestBundle(), connectOp.destIndex()};
      if (destset.count(dest)) {
        return connectOp.emitOpError()
               << "; connecting " << to_string(source) << " to "
               << to_string(dest) << " on "
               << to_string(this->getTileOp().getTileID())
               << " targets same dst as another connect op; existing "
                  "destinations: "
               << llvm::join(llvm::map_range(
                                 destset, [](auto &p) { return to_string(p); }),
                             ", ");
      }
      destset.insert(dest);

      if (connectOp.sourceIndex() < 0)
        return connectOp.emitOpError("source index cannot be less than zero");

      if (checkBound("source", connectOp.getSourceBundle(),
                     connectOp.sourceIndex(),
                     getNumSourceConnections(connectOp.getSourceBundle()))
              .failed())
        return failure();

      if (connectOp.destIndex() < 0)
        return connectOp.emitOpError("dest index cannot be less than zero");

      if (checkBound("dest", connectOp.getDestBundle(), connectOp.destIndex(),
                     getNumDestConnections(connectOp.getDestBundle()))
              .failed())
        return failure();

      // Stream switch connection constraints
      if (!isLegalTileConnection(tile, targetModel, connectOp))
        return connectOp.emitOpError("illegal stream switch connection");

    } else if (auto connectOp = dyn_cast<MasterSetOp>(ops)) {
      Port dest = {connectOp.getDestBundle(), connectOp.destIndex()};
      if (destset.count(dest))
        return connectOp.emitOpError("targets same destination ")
               << stringifyWireBundle(dest.bundle) << ": " << dest.channel
               << " as another connect or masterset operation";
      destset.insert(dest);

      if (connectOp.destIndex() < 0)
        return connectOp.emitOpError("dest index cannot be less than zero");

      if (checkBound("dest", connectOp.getDestBundle(), connectOp.destIndex(),
                     getNumDestConnections(connectOp.getDestBundle()))
              .failed())
        return failure();

      int arbiter = -1;
      for (auto val : connectOp.getAmsels()) {
        auto amsel = dyn_cast<AMSelOp>(val.getDefiningOp());
        if (arbiter != -1 && arbiter != amsel.arbiterIndex())
          return connectOp.emitOpError(
              "a master port can only be tied to one arbiter");
        arbiter = amsel.arbiterIndex();
      }
    } else if (auto connectOp = dyn_cast<PacketRulesOp>(ops)) {
      Port source = {connectOp.getSourceBundle(), connectOp.sourceIndex()};
      if (sourceset.count(source))
        return connectOp.emitOpError("packet switched source ")
               << stringifyWireBundle(source.bundle) << source.channel
               << " cannot match another connect or masterset operation";
      sourceset.insert(source);

    } else if (auto amselOp = dyn_cast<AMSelOp>(ops)) {
      std::vector<MasterSetOp> mstrs;
      std::vector<PacketRulesOp> slvs;
      for (auto *user : amselOp.getResult().getUsers()) {
        if (auto s = dyn_cast<PacketRuleOp>(user)) {
          auto pktRules = dyn_cast<PacketRulesOp>(s->getParentOp());
          slvs.push_back(pktRules);
        } else if (auto m = dyn_cast<MasterSetOp>(user))
          mstrs.push_back(m);
      }
      for (auto m : mstrs) {
        for (auto s : slvs) {
          // Stream switch connection constraints
          if (!isLegalTileConnection(tile, targetModel, m, s)) {
            return amselOp->emitOpError("illegal stream switch connection");
          }
        }
      }
    } else if (isa<EndOp>(ops)) {
      // continue;
    } else {
      return ops.emitOpError("cannot be contained in a Switchbox op");
    }
  }

  return success();
}

TileOp SwitchboxOp::getTileOp() {
  return cast<TileOp>(getTile().getDefiningOp());
}

int SwitchboxOp::colIndex() { return getTileOp().colIndex(); }

int SwitchboxOp::rowIndex() { return getTileOp().rowIndex(); }

template <typename... ParentOpTypes>
struct HasSomeParent {
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

TileOp LockOp::getTileOp() { return cast<TileOp>(getTile().getDefiningOp()); }

int LockOp::colIndex() { return getTileOp().colIndex(); }

int LockOp::rowIndex() { return getTileOp().rowIndex(); }

LogicalResult LockOp::verify() {
  if (auto result = UsesAreAccessible::verifyTrait(*this); result.failed())
    return result;

  if (getLockID().has_value()) {
    const auto &targetModel = getTargetModel(getTileOp());
    auto tileOp = getTileOp();
    if (int numLocks =
            targetModel.getNumLocks(tileOp.getCol(), tileOp.getRow());
        getLockID().value() >= numLocks)
      return emitOpError("lock assigned invalid id (maximum is ")
             << numLocks - 1 << ")";
  }

  return success();
}

struct UsesOneLockInDMABlock {
  static LogicalResult verifyTrait(Operation *op) {
    auto *block = op->getBlock();
    int lockID = -1;
    for (auto op : block->getOps<UseLockOp>()) {
      if (auto lock = dyn_cast<LockOp>(op.getLock().getDefiningOp());
          lock.getLockID().has_value()) {
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
    auto *block = op->getBlock();
    int acqValue = -1, relValue = -1;
    for (auto op : block->getOps<UseLockOp>()) {
      if (op.acquire() || op.acquireGE()) {
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

struct AccessesLocalLocks {
  static LogicalResult verifyTrait(Operation *op) {
    if (auto memOp = op->getParentOfType<MemOp>()) {
      auto useLock = dyn_cast<UseLockOp>(op);
      if (auto lock = useLock.getLockOp();
          lock.getTileOp().colIndex() != memOp.colIndex() ||
          lock.getTileOp().rowIndex() != memOp.rowIndex())
        return failure();
    }
    return success();
  }
};

LogicalResult UseLockOp::verify() {
  // AIE.useLock cannot be used at the top level
  if (llvm::isa_and_nonnull<DeviceOp, ModuleOp>((*this)->getParentOp()))
    return (*this)->emitOpError("must be used in a core or memory operation.");

  const auto &targetModel = getTargetModel(*this);
  if (targetModel.getTargetArch() == AIEArch::AIE1 && acquireGE())
    return (*this)->emitOpError(
        "AcquireGreaterEqual is not supported in AIE1.");

  // Otherwise, AIE.useLock should be inside MemOp, MemTileDMAOp, or
  // ShimDMAOp,
  if (HasSomeParent<MemOp, MemTileDMAOp, ShimDMAOp>::verifyTrait(*this)
          .succeeded()) {
    if (!(*this)->getBlock())
      return (*this)->emitOpError("is not in a block.");

    if (targetModel.getTargetArch() == AIEArch::AIE1 &&
        UsesOneLockInDMABlock::verifyTrait(*this).failed())
      return (*this)->emitOpError(
          "used in a DMA block that have multiple locks.");

    if (AcquireReleaseOneStateInDMABlock::verifyTrait(*this).failed())
      return (*this)->emitOpError("acquires/releases the lock in a DMA block "
                                  "from/to multiple states.");

    if (HasSomeParent<MemOp>::verifyTrait(*this).succeeded() &&
        AccessesLocalLocks::verifyTrait(*this).failed())
      return (*this)->emitOpError("can only access a lock in the same tile");
    return success();

    // Or it can be in a CoreOp, or some FuncOp called from a CoreOp
  }
  if (HasSomeParent<CoreOp, func::FuncOp>::verifyTrait(*this).succeeded()) {
    return success();
  }
  return (*this)->emitOpError()
         << "expects some parent op to be one of "
         << "AIE::device, AIE::core, func::func, AIE::mem, or AIE::shimDMA";
}

#include "aie/Dialect/AIE/IR/AIEEnums.cpp.inc"
#include "aie/Dialect/AIE/IR/AIEInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "aie/Dialect/AIE/IR/AIEOps.cpp.inc"

namespace xilinx::AIE {

size_t SwitchboxOp::getNumSourceConnections(WireBundle bundle) {
  auto tile = getTileOp();
  const auto &targetModel = getTargetModel(*this);
  return targetModel.getNumSourceSwitchboxConnections(tile.getCol(),
                                                      tile.getRow(), bundle);
}

size_t SwitchboxOp::getNumDestConnections(WireBundle bundle) {
  auto tile = getTileOp();
  const auto &targetModel = getTargetModel(*this);
  return targetModel.getNumDestSwitchboxConnections(tile.getCol(),
                                                    tile.getRow(), bundle);
}

WireBundle getConnectingBundle(WireBundle dir) {
  switch (dir) {
  case WireBundle::North:
    return WireBundle::South;
  case WireBundle::South:
    return WireBundle::North;
  case WireBundle::East:
    return WireBundle::West;
  case WireBundle::West:
    return WireBundle::East;
  default:
    return dir;
  }
}

} // namespace xilinx::AIE

//===----------------------------------------------------------------------===//
// BDChainOp
//===----------------------------------------------------------------------===//

ParseResult BDChainOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> entryArgs;

  // Symbol name, e.g. @my_chain
  StringAttr symNameAttr;
  if (parser.parseSymbolName(symNameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes)) {
    return failure();
  }

  // Entry arguments (placeholders), e.g. (%addr: memref<1xi32>)
  ParseResult argParseResult = parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
        OpAsmParser::Argument argument;
        if (parser.parseArgument(argument, true, true)) {
          return failure();
        }
        entryArgs.push_back(argument);
        return success();
      });
  if (argParseResult) {
    return argParseResult;
  }

  // BD Chain Body
  auto *body = result.addRegion();
  ParseResult bodyParseResult = parser.parseRegion(*body, entryArgs, false);
  if (bodyParseResult) {
    return bodyParseResult;
  }

  return success();
}

void BDChainOp::print(OpAsmPrinter &printer) {
  auto taskName =
      (*this)
          ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  printer << ' ';
  printer.printSymbolName(taskName);

  Region &body = getRegion();
  auto argsIter = body.getArguments();
  printer << '(';
  for (auto it = argsIter.begin(); it != argsIter.end(); ++it) {
    if (it != argsIter.begin()) {
      printer << ", ";
    }
    printer.printRegionArgument(*it);
  }
  printer << ')';

  printer << ' ';
  printer.printRegion(body, false, true);
}

//===----------------------------------------------------------------------===//
// ShimDMAAllocationOp
//===----------------------------------------------------------------------===//

ShimDMAAllocationOp ShimDMAAllocationOp::getForSymbol(DeviceOp device,
                                                      llvm::StringRef symbol) {
  auto alloc_ops = device.getOps<ShimDMAAllocationOp>();
  for (auto it = alloc_ops.begin(); it != alloc_ops.end(); ++it) {
    AIE::ShimDMAAllocationOp a = *it;
    if (a.getSymName() == symbol) {
      return a;
    }
  }
  return nullptr;
}

// Include implementations for custom attributes
#define GET_ATTRDEF_CLASSES
#include "aie/Dialect/AIE/IR/AIEAttrs.cpp.inc"
