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
static xilinx::AIE::IPUTargetModel IPUmodel;

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
  if (isa<ArrayAttr>(getElemNumber())) {
    size_t numDepths = dyn_cast<ArrayAttr>(getElemNumber()).size();
    if (numDepths != (getConsumerTiles().size() + 1)) // +1 for producer depth
      return emitOpError("does not have enough depths specified for producer "
                         "and for each consumer.");
  }

  if (getProducerTileOp().isShimTile() && getDimensionsToStream().size() > 0) {
    return emitError("`toStream` data layout transformations are not supported "
                     "on shim tile producers");
  }

  return success();
}

xilinx::AIE::TileOp xilinx::AIE::ObjectFifoCreateOp::getProducerTileOp() {
  return cast<xilinx::AIE::TileOp>(getProducerTile().getDefiningOp());
}

namespace xilinx {
namespace AIE {

mlir::ParseResult
parseObjectFifoProducerTile(mlir::OpAsmParser &parser,
                            mlir::OpAsmParser::UnresolvedOperand &tile,
                            DimTupleArrayAttr &dimensions) {
  std::vector<DimTupleAttr> emptyDims = {};
  if (parser.parseOperand(tile))
    return mlir::failure();
  if (mlir::succeeded(parser.parseOptionalKeyword("toStream"))) {
    if (parser.parseCustomAttributeWithFallback<DimTupleArrayAttr>(
            dimensions)) {
      return mlir::failure();
    }
  } else {
    dimensions = DimTupleArrayAttr::get(parser.getContext(),
                                        ArrayRef<DimTupleAttr>(emptyDims));
  }
  return mlir::success();
}

void printObjectFifoProducerTile(mlir::OpAsmPrinter &_odsPrinter, Operation *op,
                                 Value operand, DimTupleArrayAttr dimensions) {
  _odsPrinter << operand;
  if (dimensions && dimensions.size() > 0) {
    _odsPrinter << " toStream ";
    _odsPrinter.printStrippedAttrOrType(dimensions);
  }
}

mlir::ParseResult parseObjectFifoConsumerTiles(
    mlir::OpAsmParser &parser,
    SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &tiles,
    DimTupleArrayArrayAttr &dimensions) {
  // parseCommaSeparatedList doesn't handle the missing case for "none",
  // so we handle it custom here.
  std::vector<DimTupleArrayAttr> tileDims = {};

  auto parseOneOperand = [&]() -> ParseResult {
    if (parser.parseOperand(tiles.emplace_back(), true)) {
      return mlir::failure();
    }
    // By default, create empty dimensions array for each consumer; this way,
    // we can be certain to have as many entries in the dimensions array as
    // there are customer
    DimTupleArrayAttr dimAttr = DimTupleArrayAttr::get(parser.getContext(), {});

    if (mlir::succeeded(parser.parseOptionalKeyword("fromStream"))) {
      // If specified, parse actual data layout transform dimensions
      if (parser.parseCustomAttributeWithFallback<DimTupleArrayAttr>(dimAttr)) {
        return mlir::failure();
      }
    }
    tileDims.emplace_back(dimAttr);
    return mlir::success();
  };

  if (parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::None,
                                     parseOneOperand, " in operand list"))
    return mlir::failure();

  dimensions = DimTupleArrayArrayAttr::get(parser.getContext(), tileDims);
  return mlir::success();
}

void printObjectFifoConsumerTiles(mlir::OpAsmPrinter &_odsPrinter,
                                  Operation *op, OperandRange tiles,
                                  DimTupleArrayArrayAttr dimsPerTileAttr) {
  size_t tileIdx = 0;
  for (auto tile : tiles) {
    _odsPrinter << tile;
    if (dimsPerTileAttr && dimsPerTileAttr.size() == tiles.size() &&
        dimsPerTileAttr[tileIdx] && dimsPerTileAttr[tileIdx].size() > 0) {
      _odsPrinter << " fromStream ";
      _odsPrinter.printStrippedAttrOrType(dimsPerTileAttr[tileIdx]);
    }
    if (tileIdx < tiles.size() - 1) {
      _odsPrinter << ", ";
    }
    tileIdx++;
  }
}

} // namespace AIE
} // namespace xilinx

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
    ObjectFifoCreateOp fifoOut = getOutputObjectFifos()[0];
    AIEObjectFifoType fifoType =
        fifoOut.getElemType().cast<AIEObjectFifoType>();
    MemRefType elemType = fifoType.getElementType().cast<MemRefType>();
    int64_t outputSize = 1;
    for (auto dim : elemType.getShape())
      outputSize *= dim;

    int inputSize = 0;
    for (auto fifoIn : getInputObjectFifos()) {
      AIEObjectFifoType fifo = fifoIn.getElemType().cast<AIEObjectFifoType>();
      MemRefType elemType = fifo.getElementType().cast<MemRefType>();
      int64_t nextInputSize = 1;
      for (auto dim : elemType.getShape())
        nextInputSize *= dim;
      inputSize += nextInputSize;
    }
    if (inputSize != outputSize)
      return emitError("Total size of input objFifos in ObjectFifoLinkOp must "
                       "be equal to size of output objFifo");

  } else if (isDistribute()) {
    ObjectFifoCreateOp fifoIn = getInputObjectFifos()[0];
    if (fifoIn.getDimensionsToStream().size() > 0) {
      return emitOpError("currently does not support objectFifos with "
                         "dimensionsToStream.");
    }
    for (auto dims : fifoIn.getDimensionsFromStreamPerConsumer()) {
      if (dims.size() > 0)
        return emitOpError("currently does not support objectFifos with "
                           "dimensionsFromStreamPerConsumer.");
    }

    AIEObjectFifoType fifoType = fifoIn.getElemType().cast<AIEObjectFifoType>();
    MemRefType elemType = fifoType.getElementType().cast<MemRefType>();
    int64_t inputSize = 1;
    for (auto dim : elemType.getShape())
      inputSize *= dim;

    int outputSize = 0;
    for (auto fifoOut : getOutputObjectFifos()) {
      if ((fifoOut.getDimensionsToStream().size() > 0) &&
          (fifoOut.getConsumerTiles().size() > 1)) {
        return emitOpError("currently does not support objectFifos with "
                           "dimensionsToStream and multiple consumers.");
      }
      for (auto dims : fifoOut.getDimensionsFromStreamPerConsumer()) {
        if (dims.size() > 0)
          return emitOpError("currently does not support objectFifos with "
                             "dimensionsFromStreamPerConsumer.");
      }

      AIEObjectFifoType fifo = fifoOut.getElemType().cast<AIEObjectFifoType>();
      MemRefType elemType = fifo.getElementType().cast<MemRefType>();
      int64_t nextOutputSize = 1;
      for (auto dim : elemType.getShape())
        nextOutputSize *= dim;
      outputSize += nextOutputSize;
    }
    if (outputSize != inputSize)
      return emitError("Total size of output objFifos in ObjectFifoLinkOp must "
                       "be equal to size of input objFifo");
  }

  return success();
}

std::optional<Value> xilinx::AIE::ObjectFifoLinkOp::getOptionalSharedTile() {
  if (isJoin()) {
    auto fifoOut = getOutputObjectFifos()[0];
    for (auto fifoIn : getInputObjectFifos())
      if (fifoOut.getProducerTile() != fifoIn.getConsumerTiles()[0])
        return {};
    return {fifoOut.getProducerTile()};

  } else if (isDistribute()) {
    auto fifoIn = getInputObjectFifos()[0];
    for (auto fifoOut : getOutputObjectFifos())
      if (fifoIn.getConsumerTiles()[0] != fifoOut.getProducerTile())
        return {};
    return {fifoIn.getConsumerTiles()[0]};

  } else {
    auto fifoIn = getInputObjectFifos();
    auto fifoOut = getOutputObjectFifos();
    if (!fifoIn.empty() && !fifoOut.empty())
      for (auto consumerIn : fifoIn[0].getConsumerTiles())
        if (consumerIn == fifoOut[0].getProducerTile())
          return {fifoOut[0].getProducerTile()};
    return {};
  }
  return {};
}

std::vector<xilinx::AIE::ObjectFifoCreateOp>
xilinx::AIE::ObjectFifoLinkOp::getInputObjectFifos() {
  std::vector<ObjectFifoCreateOp> inputObjFifos;
  Operation *parent = getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      for (auto sym : getFifoIns()) {
        auto name = dyn_cast<FlatSymbolRefAttr>(sym);
        auto st = mlir::SymbolTable::lookupSymbolIn(parent, name);
        if (st && isa<ObjectFifoCreateOp>(st))
          inputObjFifos.push_back(dyn_cast<ObjectFifoCreateOp>(st));
      }
    }
  }
  return inputObjFifos;
}

std::vector<xilinx::AIE::ObjectFifoCreateOp>
xilinx::AIE::ObjectFifoLinkOp::getOutputObjectFifos() {
  std::vector<ObjectFifoCreateOp> outputObjFifos;
  Operation *parent = getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      for (auto sym : getFifoOuts()) {
        auto name = dyn_cast<FlatSymbolRefAttr>(sym);
        auto st = mlir::SymbolTable::lookupSymbolIn(parent, name);
        if (st && isa<ObjectFifoCreateOp>(st))
          outputObjFifos.push_back(dyn_cast<ObjectFifoCreateOp>(st));
      }
    }
  }
  return outputObjFifos;
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

xilinx::AIE::ObjectFifoCreateOp
xilinx::AIE::ObjectFifoRegisterExternalBuffersOp::getObjectFifo() {
  Operation *parent = getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      auto st = mlir::SymbolTable::lookupSymbolIn(parent, getObjFifoName());
      if (st && isa<ObjectFifoCreateOp>(st))
        return dyn_cast<ObjectFifoCreateOp>(st);
    }
  }
  return ObjectFifoCreateOp();
}

// ObjectFifoAcquireOp
LogicalResult xilinx::AIE::ObjectFifoAcquireOp::verify() {
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

  return success();
}

xilinx::AIE::ObjectFifoCreateOp
xilinx::AIE::ObjectFifoAcquireOp::getObjectFifo() {
  Operation *parent = getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      auto st = mlir::SymbolTable::lookupSymbolIn(parent, getObjFifoName());
      if (st && isa<ObjectFifoCreateOp>(st))
        return dyn_cast<ObjectFifoCreateOp>(st);
    }
  }
  return ObjectFifoCreateOp();
}

// ObjectFifoReleaseOp
LogicalResult xilinx::AIE::ObjectFifoReleaseOp::verify() {
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

xilinx::AIE::ObjectFifoCreateOp
xilinx::AIE::ObjectFifoReleaseOp::getObjectFifo() {
  Operation *parent = getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      auto st = mlir::SymbolTable::lookupSymbolIn(parent, getObjFifoName());
      if (st && isa<ObjectFifoCreateOp>(st))
        return dyn_cast<ObjectFifoCreateOp>(st);
    }
  }
  return ObjectFifoCreateOp();
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
    if (getAcquirePattern().size() != getProcessLength() &&
        getProcessLength() != getReleasePattern().size())
      return emitOpError(
          "Acquire and Release patterns must be of equal length, or "
          "longest length of one must be equal to process "
          "length of the other");
  }

  return success();
}

xilinx::AIE::ObjectFifoCreateOp
xilinx::AIE::ObjectFifoRegisterProcessOp::getObjectFifo() {
  Operation *parent = getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      auto st = mlir::SymbolTable::lookupSymbolIn(parent, getObjFifoName());
      if (st && isa<ObjectFifoCreateOp>(st))
        return dyn_cast<ObjectFifoCreateOp>(st);
    }
  }
  return ObjectFifoCreateOp();
}

const xilinx::AIE::AIETargetModel &xilinx::AIE::DeviceOp::getTargetModel() {
  switch (getDevice()) {
  case AIEDevice::xcvc1902:
    return VC1902model;
  case AIEDevice::xcve2302:
    return VE2302model;
  case AIEDevice::xcve2802:
    return VE2802model;
  case AIEDevice::ipu:
    return IPUmodel;
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

bool isLegalMemtileConnection(const xilinx::AIE::AIETargetModel &target_model,
                              xilinx::AIE::MasterSetOp masterOp,
                              xilinx::AIE::PacketRulesOp slaveOp) {
  auto srcBundle = masterOp.destPort().first;
  auto srcChan = masterOp.destPort().second;
  auto dstBundle = slaveOp.sourcePort().first;
  auto dstChan = slaveOp.sourcePort().second;
  return target_model.isLegalMemtileConnection(srcBundle, srcChan, dstBundle,
                                               dstChan);
}

bool isLegalMemtileConnection(const xilinx::AIE::AIETargetModel &target_model,
                              xilinx::AIE::ConnectOp connectOp) {
  auto srcBundle = connectOp.getSourceBundle();
  auto srcChan = connectOp.getSourceChannel();
  auto dstBundle = connectOp.getDestBundle();
  auto dstChan = connectOp.getDestChannel();
  return target_model.isLegalMemtileConnection(srcBundle, srcChan, dstBundle,
                                               dstChan);
}

LogicalResult xilinx::AIE::SwitchboxOp::verify() {
  Region &body = getConnections();
  DenseSet<xilinx::AIE::Port> sourceset;
  DenseSet<xilinx::AIE::Port> destset;
  auto tile = getTileOp();
  const auto &target_model = getTargetModel(tile);
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

      // Memtile stream switch connection constraints
      if (tile.isMemTile()) {
        if (!isLegalMemtileConnection(target_model, connectOp))
          return connectOp.emitOpError(
              "illegal memtile stream switch connection");
      }

      // Trace stream switch connection constraints
      if (connectOp.getDestBundle() == xilinx::AIE::WireBundle::Trace)
        return connectOp.emitOpError("Trace port cannot be a destination");
      if (connectOp.getSourceBundle() == xilinx::AIE::WireBundle::Trace) {
        if (!target_model.isValidTraceMaster(tile.getCol(), tile.getRow(),
                                             connectOp.getDestBundle(),
                                             connectOp.getDestChannel()))
          return connectOp.emitOpError("illegal Trace destination");
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
      std::vector<xilinx::AIE::MasterSetOp> mstrs;
      std::vector<xilinx::AIE::PacketRulesOp> slvs;
      for (auto user : amselOp.getResult().getUsers()) {
        if (auto s = dyn_cast<xilinx::AIE::PacketRuleOp>(user)) {
          auto pkt_rules =
              dyn_cast<xilinx::AIE::PacketRulesOp>(s->getParentOp());
          slvs.push_back(pkt_rules);
        } else if (auto m = dyn_cast<xilinx::AIE::MasterSetOp>(user))
          mstrs.push_back(m);
      }
      for (auto m : mstrs) {
        // Trace stream switch connection constraints
        if (m.destPort().first == xilinx::AIE::WireBundle::Trace)
          return connectOp.emitOpError("Trace port cannot be a destination");
        for (auto s : slvs) {
          if (s.sourcePort().first == xilinx::AIE::WireBundle::Trace) {
            if (!target_model.isValidTraceMaster(tile.getCol(), tile.getRow(),
                                                 m.destPort().first,
                                                 m.destPort().second))
              return amselOp.emitOpError("illegal Trace destination");
          }

          // Memtile stream switch connection constraints
          if (tile.isMemTile() && !isLegalMemtileConnection(target_model, m, s))
            return amselOp->emitOpError(
                "illegal memtile stream switch connection");
        }
      }
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
    if (auto startOp = dyn_cast<DMAStartOp>(bodyOp)) {
      if (startOp.getChannelIndex() > 3) {
        // Channels 4 and 5 in a memtile are restricted to only access local
        // buffers and locks.

        // Move this code to the dialect
        // Set of blocks found to be reachable within a given region.
        llvm::SmallSet<Block *, 16> reachable;
        llvm::SmallVector<Block *, 16> worklist;
        Block *firstBD = startOp.getSuccessor(0);
        reachable.insert(firstBD);
        worklist.push_back(firstBD);
        while (!worklist.empty()) {
          Block *block = worklist.pop_back_val();
          if (block->empty())
            continue;
          auto successors = block->getTerminator()->getSuccessors();
          for (auto i : successors) {
            if (!reachable.contains(i)) {
              reachable.insert(i);
              worklist.push_back(i);
            }
          }
        }
        for (auto b : reachable) {
          for (auto bd : b->getOps<xilinx::AIE::DMABDOp>()) {
            auto bufferOp = bd.getBufferOp();
            if (bufferOp.getTileOp().colIndex() != colIndex() ||
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
          for (auto useLock : b->getOps<xilinx::AIE::UseLockOp>()) {
            auto lockOp = useLock.getLockOp();
            if (lockOp.getTileOp().colIndex() != colIndex() ||
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
  return success();
}

// DMABDOp
xilinx::AIE::BufferOp xilinx::AIE::DMABDOp::getBufferOp() {
  return cast<xilinx::AIE::BufferOp>(getBuffer().getDefiningOp());
}

LogicalResult xilinx::AIE::DMABDOp::verify() {
  if (auto memOp = getOperation()->getParentOfType<xilinx::AIE::MemOp>()) {
    auto bufferOp = getBufferOp();
    if (bufferOp.getTileOp().colIndex() != memOp.colIndex() ||
        bufferOp.getTileOp().rowIndex() != memOp.rowIndex())
      return emitOpError("can only access a buffer in the same tile.");
  }

  // The following checks only apply if non-default strides/wraps are defined.
  if (getDimensions()) {
    mlir::MemRefType buffer = getBuffer().getType();
    // We are not restrictive about the type of the memref used as the input
    // to the DMABD when used with multi-dimensional strides/wraps. Since the
    // BD will use the memref as a base address and copy from it in 32 bit
    // chunks, while assuming the layout of the memref is contiguous. We
    // assume the user/compiler understands and accounts for this.
    uint64_t memref_size = 1; // in bytes
    uint64_t max_idx = 0;
    for (int64_t memref_dim : buffer.getShape()) {
      memref_size *= 4 * memref_dim;
    }
    llvm::ArrayRef<xilinx::AIE::DimTupleAttr> dims = *getDimensions();
    size_t max_n_dims = 3;
    if (isa_and_nonnull<xilinx::AIE::MemTileDMAOp>((*this)->getParentOp())) {
      max_n_dims = 4;
    }
    if (dims.size() > max_n_dims) {
      return emitOpError() << "Cannot give more than "
                           << std::to_string(max_n_dims)
                           << " dimensions for step sizes and wraps in this "
                              " tile (got "
                           << std::to_string(dims.size()) << " dimensions).";
    }
    for (xilinx::AIE::DimTupleAttr dim : dims) {
      max_idx += dim.getStepsize() * (dim.getWrap() - 1);
      if (0 == dim.getStepsize()) {
        return emitOpError()
               << "Invalid step size; must be a positive integer.";
      }
      if (dim.getStepsize() > memref_size) {
        return emitOpError()
               << "Step size " << std::to_string(dim.getStepsize() * 4) << " "
               << "bytes exceeds memref size " << std::to_string(memref_size);
      }
      if (dim.getWrap() >= (1UL << 9) + 1) {
        return emitOpError() << "Wrap may not exceed 1023.";
      }
      if (dim.getStepsize() >= (1UL << 19)) {
        return emitOpError() << "Stepsize may not exceed " << (1 << 20);
      }
    }
    if (memref_size <= 4 * max_idx) {
      return emitOpError() << "Specified stepsize(s) and wrap(s) result in out "
                              "of bounds access in buffer, for index "
                           << std::to_string(max_idx) << ", accessing at "
                           << std::to_string(4 * max_idx)
                           << " byte offset in memref of length "
                           << std::to_string(memref_size) << ".";
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

struct AccessesLocalLocks {
  static LogicalResult verifyTrait(Operation *op) {
    if (auto memOp = op->getParentOfType<xilinx::AIE::MemOp>()) {
      auto useLock = dyn_cast<xilinx::AIE::UseLockOp>(op);
      auto lock = useLock.getLockOp();
      if (lock.getTileOp().colIndex() != memOp.colIndex() ||
          lock.getTileOp().rowIndex() != memOp.rowIndex())
        return failure();
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

    if (HasSomeParent<xilinx::AIE::MemOp>::verifyTrait(*this).succeeded()) {
      if (AccessesLocalLocks::verifyTrait(*this).failed())
        return (*this)->emitOpError("can only access a lock in the same tile");
    }
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

bool TileOp::isShimNOCorPLTile() {
  const auto &target_model = getTargetModel(*this);
  return target_model.isShimNOCorPLTile(getCol(), getRow());
}
} // namespace AIE
} // namespace xilinx

// Include implementations for custom attributes
#define GET_ATTRDEF_CLASSES
#include "aie/Dialect/AIE/IR/AIEAttrDefs.cpp.inc"
