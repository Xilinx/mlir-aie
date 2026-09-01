//===- AIEDialect.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace xilinx::AIE;

// Add TableGen'erated dialect definitions (including constructor)
// We implement the initialize() function further below
#include "aie/Dialect/AIE/IR/AIEDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "aie/Dialect/AIE/IR/AIETypes.cpp.inc"

namespace {

struct AIEInlinerInterface : DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  // We don't have any special restrictions on what can be inlined into
  // destination regions. Always allow it.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }

  // Operations in aie dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation *op, Region *, bool wouldBeCloned,
                       IRMapping &) const final {
    return true;
  }

  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the inlined region has more than one block.
  void handleTerminator(Operation *op, Block *newDest) const final {}

  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the region has only one block.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {}
};

struct AIEDialectFoldInterface : DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  /// Registered hook to check if the given region, which is attached to an
  /// operation that is *not* isolated from above, should be used when
  /// materializing constants.
  bool shouldMaterializeInto(Region *region) const final {
    // Materialize constants into the op that "owns" them rather than letting
    // them hoist up to the enclosing IsolatedFromAbove aie.device:
    //  - aie.core bodies are outlined into standalone funcs, so their
    //    constants must stay local for the func to be self-contained.
    //  - aie.runtime_sequence bodies carry constant operands (e.g. the scalar
    //    fields of npu.* ops). Hoisting them to the device body lets CSE merge
    //    them with a core's identical constants, which would leave the core
    //    referencing a device-level value that is erased when the core is
    //    outlined.
    //  - Make sure SSA values for aie.use_lock operands in
    //    aie.mem/aie.memtile_dma/aie.shim_dma bodies do not get
    //    hoisted.
    return isa<CoreOp, RuntimeSequenceOp, MemOp, MemTileDMAOp, ShimDMAOp>(
        region->getParentOp());
  }
};

} // end anonymous namespace

void AIEDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "aie/Dialect/AIE/IR/AIETypes.cpp.inc"
      >();
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

// Helper methods to retrieve the encoding associated to a burst length,
// or to find the highest available burst length if the requested one is 0
// (default value).

static std::pair<uint32_t, uint32_t>
getShimBurstLength(const xilinx::AIE::AIETargetModel &tm,
                   uint32_t burstLength) {

  std::vector<std::pair<uint32_t, uint32_t>> bel =
      tm.getShimBurstEncodingsAndLengths();

  // If we have the default burst length (no burst length was specified),
  // use the highest one available on our target model
  if (burstLength == 0) {
    return *llvm::max_element(bel, [](auto pair1, auto pair2) {
      return pair1.second < pair2.second;
    });
  }

  // Note that if we are given a burst size, we are checking its existence in
  // the pass verification already, so we can safely assume it exists.
  return *llvm::find_if(bel, [=](auto p) { return p.second == burstLength; });
}

uint32_t xilinx::AIE::getShimBurstLengthBytes(const AIE::AIETargetModel &tm,
                                              uint32_t burstLength) {

  return getShimBurstLength(tm, burstLength).second;
}

uint32_t xilinx::AIE::getShimBurstLengthEncoding(const AIE::AIETargetModel &tm,
                                                 uint32_t burstLength) {

  return getShimBurstLength(tm, burstLength).first;
}

std::string xilinx::AIE::generateUniqueSymbolName(
    mlir::Operation *symbolTableOp, llvm::StringRef prefix, unsigned &counter) {
  std::string name;
  do {
    name = (prefix + llvm::Twine(counter++)).str();
  } while (mlir::SymbolTable::lookupSymbolIn(symbolTableOp, name));
  return name;
}

LogicalResult
xilinx::AIE::myVerifyOffsetSizeAndStrideOp(OffsetSizeAndStrideOpInterface op) {
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
static VE3858TargetModel VE3858model;
static VirtualizedNPU1TargetModel NPUmodel1col(1);
static VirtualizedNPU1TargetModel NPUmodel2col(2);
static VirtualizedNPU1TargetModel NPUmodel3col(3);
static VirtualizedNPU1TargetModel NPUmodel4col(4);
static NPU2TargetModel NPU2model;
static VirtualizedNPU2TargetModel NPU2model1col(1);
static VirtualizedNPU2TargetModel NPU2model2col(2);
static VirtualizedNPU2TargetModel NPU2model3col(3);
static VirtualizedNPU2TargetModel NPU2model4col(4);
static VirtualizedNPU2TargetModel NPU2model5col(5);
static VirtualizedNPU2TargetModel NPU2model6col(6);
static VirtualizedNPU2TargetModel NPU2model7col(7);

const AIETargetModel &xilinx::AIE::getTargetModel(Operation *op) {
  if (auto t = dyn_cast<AIETarget>(op))
    return t.getTargetModel();
  if (auto t = op->getParentOfType<AIETarget>())
    return t.getTargetModel();

  // For backward compatibility, return a basic device model compatible with
  // the VCK190
  return VC1902model;
}

AIETargetModel::SharedMemory xilinx::AIE::sharedMemory(TileOp a, TileOp b) {
  return getTargetModel(a.getOperation())
      .getSharedMemory({a.getCol(), a.getRow()}, {b.getCol(), b.getRow()});
}

const AIETargetModel &xilinx::AIE::getTargetModel(AIEDevice device) {
  switch (device) {
  case AIEDevice::xcvc1902:
    return VC1902model;
  case AIEDevice::xcve2302:
    return VE2302model;
  case AIEDevice::xcve2802:
    return VE2802model;
  case AIEDevice::npu1:
    return NPUmodel4col;
  case AIEDevice::npu1_1col:
    return NPUmodel1col;
  case AIEDevice::npu1_2col:
    return NPUmodel2col;
  case AIEDevice::npu1_3col:
    return NPUmodel3col;
  case AIEDevice::npu2:
    return NPU2model;
  case AIEDevice::npu2_1col:
    return NPU2model1col;
  case AIEDevice::npu2_2col:
    return NPU2model2col;
  case AIEDevice::npu2_3col:
    return NPU2model3col;
  case AIEDevice::npu2_4col:
    return NPU2model4col;
  case AIEDevice::npu2_5col:
    return NPU2model5col;
  case AIEDevice::npu2_6col:
    return NPU2model6col;
  case AIEDevice::npu2_7col:
    return NPU2model7col;
  case AIEDevice::xcve3858:
    return VE3858model;
  }
  // No default: label above, so -Wswitch still reports a newly added device.
  // This handles values that are not enumerators at all, which
  // aieGetTargetModel admits by casting an unchecked uint32_t.
  llvm::report_fatal_error("getTargetModel: unknown AIEDevice value " +
                           llvm::Twine(static_cast<uint32_t>(device)));
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

// Returns the maximum index described by the input dimensions.
static int64_t getDimsMaxIdx(ArrayRef<BDDimLayoutAttr> dims) {
  int64_t maxIdx = 0;
  for (BDDimLayoutAttr dim : dims) {
    maxIdx += dim.getStride() * (dim.getSize() - 1);
  }
  return maxIdx;
}

namespace {

struct UsesAreAccessible {
  static LogicalResult verifyTrait(Operation *op) {
    auto thisElement = cast<TileElement>(op);

    // Skip accessibility checks for logical tiles as we cannot tell until tile
    // is placed
    if (!isa<TileOp>(thisElement.getTile().getDefiningOp()))
      return success();

    auto thisID = thisElement.getTileID();
    auto users = op->getResult(0).getUsers();
    const auto &targetModel = getTargetModel(op);
    for (auto *user : users) {
      // AIE.useLock may be used in a device to set the lock's default value
      // Allow in a toplevel module for backward compatibility
      if (llvm::isa_and_nonnull<DeviceOp, ModuleOp>(user->getParentOp())) {
        continue;
      }
      // If any parent or the user itself prescribe that accessibility checks be
      // skipped, skip the check for that user.
      if (user->getParentWithTrait<SkipAccessibilityCheckTrait>() ||
          user->hasTrait<SkipAccessibilityCheckTrait>()) {
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

} // namespace

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

  TileLike tile = element.getTileLike();
  if (!tile)
    return op->emitOpError("tile must implement TileLike interface");

  int bdMax = targetModel.getNumBDs(tile.getTileType());

  int bdNum = 0;
  for (auto &block : element.getBody()) {
    auto bdOps = llvm::to_vector_of<DMABDOp>(block.template getOps<DMABDOp>());

    // Skip entry/end block
    if (bdOps.empty())
      continue;

    // Check BD count limit
    if (bdNum >= bdMax) {
      return (op->emitOpError("has more than ") << bdMax << " blocks")
          .attachNote(bdOps.front().getLoc())
          .append("no space for this BD");
    }
    bdNum++;

    // Check exactly 1 DMABDOp per BD block
    if (bdOps.size() != 1) {
      return (op->emitOpError("BD block must have exactly one DMABDOp, found ")
              << bdOps.size())
          .attachNote(block.front().getLoc())
          .append("in this BD block");
    }

    // Check at most 2 UseLockOps per BD block (1 acquire, 1 release)
    auto useLockOps =
        llvm::to_vector_of<UseLockOp>(block.template getOps<UseLockOp>());
    int acquireCount = 0;
    int releaseCount = 0;
    for (auto useLock : useLockOps) {
      if (useLock.acquire() || useLock.acquireGE())
        acquireCount++;
      else if (useLock.release())
        releaseCount++;
    }

    if (acquireCount > 1) {
      return (op->emitOpError(
                  "BD block must have at most one acquire UseLockOp, found ")
              << acquireCount)
          .attachNote(block.front().getLoc())
          .append("in this BD block");
    }
    if (releaseCount > 1) {
      return (op->emitOpError(
                  "BD block must have at most one release UseLockOp, found ")
              << releaseCount)
          .attachNote(block.front().getLoc())
          .append("in this BD block");
    }
  }
  return success();
}

// Check that the given DMA-like op (e.g. MemOp, ShimDMAOp)
// has valid DMA channels.
template <typename ConcreteType>
LogicalResult HasValidDMAChannels<ConcreteType>::verifyTrait(Operation *op) {
  DenseSet<DMAChannel> inputChannels;
  DenseSet<DMAChannel> outputChannels;
  auto element = cast<ConcreteType>(op);
  Region &body = element.getBody();
  if (body.empty())
    return op->emitOpError("should have non-empty body");
  for (auto &bodyOp : body.getOps()) {
    // check for duplicate DMA channels within the same ShimDMAOp
    if (auto dmaStart = dyn_cast<DMAStartOp>(bodyOp)) {
      DMAChannel dmaChan = {dmaStart.getChannelDir(),
                            dmaStart.getChannelIndex()};
      // check if number of input and output channels is more than available
      // hardware
      if (dmaChan.direction == DMAChannelDir::S2MM)
        inputChannels.insert(dmaChan);
      else
        outputChannels.insert(dmaChan);
    }
  }

  TileLike tile = element.getTileLike();
  if (!tile)
    return op->emitOpError("tile must implement TileLike interface");

  if (inputChannels.size() > tile.getNumSourceConnections(WireBundle::DMA))
    return op->emitOpError(
        "uses more input channels than available on this tile");

  if (outputChannels.size() > tile.getNumDestConnections(WireBundle::DMA))
    return op->emitOpError(
        "uses more output channels than available on this tile");
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

  if (getAieStream() && (getProdDmaChannel() || getConsDmaChannels())) {
    return emitOpError(
        "cannot pin a DMA channel on an objectfifo that also uses aie_stream "
        "(stream ports bypass DMA channels)");
  }

  if (getPacketId() && !getPacket()) {
    return emitOpError("packet_id is only meaningful on a packet objectfifo");
  }

  // Helper to get tile interface from Value
  auto getTileLikeFromValue = [](Value v) -> TileLike {
    return llvm::dyn_cast<TileLike>(v.getDefiningOp());
  };

  TileLike producerTile = getTileLikeFromValue(getProducerTile());
  if (!producerTile)
    return emitError("producer tile must implement TileLike interface");

  // data layout transformations on shim tiles are handled by runtime operations
  if (producerTile.isShimTile() && !getDimensionsToStream().empty()) {
    return emitError(
        "`dimensionsToStream` data layout transformations are not supported "
        "on shim tile producers");
  }
  for (auto consTileVal : getConsumerTiles()) {
    TileLike consTile = getTileLikeFromValue(consTileVal);
    if (!consTile)
      return emitError("consumer tile must implement TileLike interface");
    if (consTile.isShimTile() &&
        !getDimensionsFromStream(consTileVal).empty()) {
      return emitError(
          "`dimensionsFromStreamPerConsumer` data layout transformations are "
          "not supported on shim tile consumers");
    }
  }

  if (getRepeatCount().has_value()) {
    if (producerTile.isShimTile())
      return emitError("`repeat_count` unavailable for shim tiles");
  }

  if (getPadValue() != 0) {
    if (!getPadDimensions().has_value())
      return emitError("`padValue` requires `padDimensions`");
    if (!getTargetModel(getOperation()).isMemTilePadValueSupported())
      return emitError("`padValue` requires the CONSTANT_PAD_VALUE register, "
                       "unavailable on this target");
  }

  if (getAieStreamPort().has_value()) {
    if (!getAieStream().has_value())
      return emitError("`aie_stream` must be defined");
  }

  if (auto aieStream = getAieStream()) {
    int aieStreamVal = *aieStream;
    if (getConsumerTiles().size() > 1)
      return emitError("`aie_stream` can only be used in 1-to-1 object FIFOs");

    if (!getAieStreamPort().has_value())
      return emitError("`aie_stream_port` must be defined");

    if (aieStreamVal == 0 || aieStreamVal == 2) {
      if (producerTile.isShimTile() || producerTile.isMemTile())
        return emitError(
            "`aie_stream` is not available for shim and mem tiles");

      if (getRepeatCount().has_value())
        return emitError("`repeat_count` unavailable on stream end");

      if (getInitValues().has_value())
        return emitError("`init_values` unavailable on stream end");

      if (getIterCount().has_value())
        return emitError("`iter_count` unavailable on stream end");

      if (!getDimensionsToStream().empty())
        return emitError("`dimensionsToStream` data layout transformations are "
                         "unavailable on stream end");
    }

    if (aieStreamVal == 1 || aieStreamVal == 2) {
      TileLike consTile = getTileLikeFromValue(getConsumerTiles()[0]);
      if (consTile && (consTile.isShimTile() || consTile.isMemTile()))
        return emitError(
            "`aie_stream` is not available for shim and mem tiles");
    }

    if (!getDimensionsFromStreamPerConsumer()[0].empty())
      return emitError("`dimensionsFromStreamPerConsumer` data layout "
                       "transformations are unavailable on stream end");
  }

  if (getInitValues().has_value()) {
    if (producerTile.isShimTile())
      return emitError("`init_values` unavailable for shim tiles");
  }

  if (auto initValues = getInitValues()) {
    if ((int)initValues->size() != size())
      return emitError("`init_values` does not initialize all objects");
  }

  if (auto iterCountAttr = getIterCount()) {
    int iterCount = *iterCountAttr;
    if (iterCount < 1 || iterCount > 256)
      return emitError("`iter_count` must be between 1 and 256");
  }

  if (auto consumerElemType = getConsumerElemType()) {
    auto consType = llvm::dyn_cast<AIEObjectFifoType>(*consumerElemType);
    if (!consType)
      return emitError("consumer element type must be an "
                       "!aie.objectfifo<memref<...>> type");
    auto prodType = llvm::cast<AIEObjectFifoType>(getElemType());
    auto prodMemref = prodType.getElementType();
    auto consMemref = consType.getElementType();
    if (prodMemref.getElementType() != consMemref.getElementType())
      return emitError("producer and consumer must have the same scalar "
                       "element type, but got ")
             << prodMemref.getElementType() << " vs "
             << consMemref.getElementType();
    int64_t prodSize = prodMemref.getNumElements();
    int64_t consSize = consMemref.getNumElements();
    if (consSize <= 0)
      return emitError("consumer element count must be positive");
    if (prodSize % consSize != 0)
      return emitError("producer element size (")
             << prodSize << ") must be an integer multiple of consumer "
             << "element size (" << consSize << ")";
  }

  return success();
}

TileOp ObjectFifoCreateOp::getProducerTileOp() {
  return cast<TileOp>(getProducerTile().getDefiningOp());
}

//===----------------------------------------------------------------------===//
// ObjectFifoPoolOp
//===----------------------------------------------------------------------===//

namespace {
/// Resolve a list of symbol names to the ops they name, skipping any that are
/// not yet declared.
template <typename OpTy>
SmallVector<OpTy> lookupAll(Operation *from, std::optional<ArrayAttr> names) {
  SmallVector<OpTy> ops;
  if (names) {
    for (auto name : names->getAsRange<FlatSymbolRefAttr>()) {
      if (auto op = dyn_cast_or_null<OpTy>(
              SymbolTable::lookupNearestSymbolFrom(from, name.getAttr()))) {
        ops.push_back(op);
      }
    }
  }
  return ops;
}

} // namespace

TileLike ObjectFifoPoolOp::getTileLike() {
  return dyn_cast<TileLike>(getTile().getDefiningOp());
}

int64_t ObjectFifoPoolOp::getObjectSize() {
  return llvm::cast<MemRefType>(getElemType()).getNumElements();
}

LogicalResult ObjectFifoPoolOp::verify() {
  auto &target = (*this)->getParentOfType<DeviceOp>().getTargetModel();
  bool semaphoreLocks = target.hasProperty(AIETargetModel::UsesSemaphoreLocks);

  // A binary lock travels with the buffer it guards; a semaphore lock counts
  // objects for a whole segment.
  if (auto locks = getLocks()) {
    if (semaphoreLocks) {
      return emitOpError("'locks' names binary locks, which this device has "
                         "no use for");
    }
    if (static_cast<int64_t>(locks->size()) != getDepth()) {
      return emitOpError("expects one lock per buffer");
    }
  }

  if (auto buffers = getBuffers()) {
    if (static_cast<int64_t>(buffers->size()) != getDepth()) {
      return emitOpError("expects 'depth' buffers");
    }
  }

  std::vector<ObjectFifoSegmentOp> segments = getSegmentOps();
  if (segments.empty()) {
    return emitOpError("expects at least one segment");
  }

  int64_t previous = -1;
  int64_t covered = 0;
  for (ObjectFifoSegmentOp segment : segments) {
    if (static_cast<int64_t>(segment.getOffset()) <= previous) {
      return emitOpError("segments must be ordered by increasing offset");
    }
    if (static_cast<int64_t>(segment.getOffset()) != covered) {
      return segment.emitOpError("leaves a gap or overlaps the segment before "
                                 "it; segments must tile the object");
    }
    previous = segment.getOffset();
    covered = segment.getOffset() + segment.getSize();
    if (!semaphoreLocks &&
        (segment.getProduceLock() || segment.getConsumeLock())) {
      return emitOpError("segment locks are counting locks, which this "
                         "device has no use for");
    }
  }

  if (covered != getObjectSize()) {
    return emitOpError("segments cover ")
           << covered << " of " << getObjectSize() << " elements";
  }

  if (segments.size() > 1 && !semaphoreLocks) {
    return emitOpError(
        "multi-segment pools are unsupported on binary lock architectures");
  }

  return success();
}

int64_t ObjectFifoPoolOp::getObjectSizeInBytes() {
  MemRefType elemType = getElemType();
  DataLayout layout = DataLayout::closest(*this);
  return elemType.getNumElements() *
         layout.getTypeSizeInBits(elemType.getElementType()) / 8;
}

std::vector<ObjectFifoSegmentOp> ObjectFifoPoolOp::getSegmentOps() {
  auto ops = getSegments().getOps<ObjectFifoSegmentOp>();
  return {ops.begin(), ops.end()};
}

Value BufferOp::getBufferTile() { return getTile(); }

SmallVector<BufferLike> ObjectFifoPoolOp::getBufferOps() {
  // Buffers and locks live beside the pool, not in the symbol table it opens
  // for its segments.
  return lookupAll<BufferLike>((*this)->getParentOfType<DeviceOp>(),
                               getBuffers());
}

SmallVector<LockOp> ObjectFifoPoolOp::getLockOps() {
  auto device = (*this)->getParentOfType<DeviceOp>();
  SmallVector<LockOp> locks = lookupAll<LockOp>(device, getLocks());
  for (ObjectFifoSegmentOp segment : getSegmentOps()) {
    for (std::optional<FlatSymbolRefAttr> name :
         {segment.getProduceLockAttr(), segment.getConsumeLockAttr()}) {
      if (name && *name) {
        if (auto lock = mlir::SymbolTable::lookupNearestSymbolFrom<LockOp>(
                device, *name)) {
          locks.push_back(lock);
        }
      }
    }
  }
  return locks;
}

StringRef ObjectFifoPoolOp::getBaseName() {
  StringRef name = getSymName();
  return name.consume_back("_pool") ? name : getSymName();
}

//===----------------------------------------------------------------------===//
// ObjectFifo endpoints
//===----------------------------------------------------------------------===//

namespace {

LogicalResult verifyEndpoint(Operation *op, ObjectFifoPoolOp pool,
                             std::optional<ArrayAttr> segments) {
  std::vector<ObjectFifoSegmentOp> all = pool.getSegmentOps();
  if (!segments) {
    if (all.size() > 1) {
      return op->emitOpError(
          "must list segments explicitly for a multi-segment pool");
    }
    return success();
  }
  if (segments->empty()) {
    return op->emitOpError("expects at least one segment");
  }

  int64_t previous = -1;
  for (auto name : segments->getAsRange<FlatSymbolRefAttr>()) {
    auto segment = SymbolTable::lookupNearestSymbolFrom<ObjectFifoSegmentOp>(
        pool, name.getAttr());
    if (!segment) {
      return op->emitOpError("references undefined segment '")
             << name.getValue() << "' in pool '" << pool.getSymName() << "'";
    }
    // The endpoint's memref is one run of the object, and `dimensions` pairs
    // with this list positionally.
    if (static_cast<int64_t>(segment.getOffset()) <= previous) {
      return op->emitOpError("segments must be named in increasing offset "
                             "order");
    }
    previous = segment.getOffset();
  }
  return success();
}

ObjectFifoPoolOp lookupPool(Operation *op, StringRef name) {
  auto device = op->getParentOfType<DeviceOp>();
  return SymbolTable::lookupNearestSymbolFrom<ObjectFifoPoolOp>(
      device, StringAttr::get(op->getContext(), name));
}

} // namespace

/// The subset of `pool`'s segments `selected` names. Omission selects the
/// pool's only segment.
static std::vector<ObjectFifoSegmentOp>
selectSegments(ObjectFifoPoolOp pool, std::optional<ArrayAttr> selected) {
  if (!pool) {
    return {};
  }
  std::vector<ObjectFifoSegmentOp> all = pool.getSegmentOps();
  if (!selected) {
    return all.empty() ? std::vector<ObjectFifoSegmentOp>{}
                       : std::vector<ObjectFifoSegmentOp>{all.front()};
  }
  std::vector<ObjectFifoSegmentOp> chosen;
  for (auto name : selected->getAsRange<FlatSymbolRefAttr>()) {
    if (auto segment =
            SymbolTable::lookupNearestSymbolFrom<ObjectFifoSegmentOp>(
                pool, name.getAttr())) {
      chosen.push_back(segment);
    }
  }
  return chosen;
}

TileLike ObjectFifoCoreEndpointOp::getTileLike() {
  return dyn_cast<TileLike>(getTile().getDefiningOp());
}

ObjectFifoPoolOp ObjectFifoCoreEndpointOp::getPoolOp() {
  return lookupPool(*this, getPool());
}

std::vector<ObjectFifoSegmentOp>
ObjectFifoCoreEndpointOp::getSelectedSegments() {
  return selectSegments(getPoolOp(), getSegments());
}

LogicalResult ObjectFifoCoreEndpointOp::verify() {
  ObjectFifoPoolOp pool = getPoolOp();
  if (!pool) {
    return emitOpError("references undefined pool '") << getPool() << "'";
  }
  if (failed(verifyEndpoint(*this, pool, getSegments()))) {
    return failure();
  }

  // The core sees one memref, so the segments it selects have to be a single
  // run of the object.
  int64_t next = -1;
  for (ObjectFifoSegmentOp segment : getSelectedSegments()) {
    if (next >= 0 && segment.getOffset() != next) {
      return emitOpError("a core endpoint's segments must be contiguous");
    }
    next = segment.getOffset() + segment.getSize();
  }
  return success();
}

std::pair<int64_t, int64_t> ObjectFifoCoreEndpointOp::getExtent() {
  std::vector<ObjectFifoSegmentOp> selected = getSelectedSegments();
  if (selected.empty()) {
    return {0, 0};
  }
  int64_t offset = selected.front().getOffset();
  int64_t end = selected.back().getOffset() + selected.back().getSize();
  return {offset, end - offset};
}

MemRefType ObjectFifoCoreEndpointOp::getAccessType() {
  MemRefType elemType = getPoolOp().getElemType();
  auto [offset, size] = getExtent();
  if (offset == 0 && size == elemType.getNumElements()) {
    return elemType;
  }
  return cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
      {size}, elemType, {offset}, {size}, {1}));
}

TileLike ObjectFifoDmaEndpointOp::getTileLike() {
  return dyn_cast<TileLike>(getTile().getDefiningOp());
}

ObjectFifoPoolOp ObjectFifoDmaEndpointOp::getPoolOp() {
  return lookupPool(*this, getPool());
}

std::vector<ObjectFifoSegmentOp>
ObjectFifoDmaEndpointOp::getSelectedSegments() {
  return selectSegments(getPoolOp(), getSegments());
}

DMAChannelDir ObjectFifoDmaEndpointOp::getRouteDirection() {
  return drains() ? DMAChannelDir::MM2S : DMAChannelDir::S2MM;
}

WireBundle ObjectFifoDmaEndpointOp::getRouteBundle() { return WireBundle::DMA; }

std::optional<int> ObjectFifoDmaEndpointOp::getRouteChannel() {
  return getChannelIndex();
}

void ObjectFifoDmaEndpointOp::setRouteChannel(int channel) {
  setChannelIndex(channel);
}

LogicalResult ObjectFifoDmaEndpointOp::verify() {
  ObjectFifoPoolOp pool = getPoolOp();
  if (!pool) {
    return emitOpError("references undefined pool '") << getPool() << "'";
  }

  if (failed(verifyEndpoint(*this, pool, getSegments()))) {
    return failure();
  }

  std::optional<ArrayAttr> segmentNames = getSegments();
  size_t selectedCount = segmentNames ? segmentNames->size() : 1;
  auto dimensions = getDimensions();
  if (dimensions && dimensions->size() != selectedCount) {
    return emitOpError("dimensions has ")
           << dimensions->size() << " entries for " << selectedCount
           << " selected segments";
  }
  auto padding = getPadDimensions();
  if (padding && padding->size() != selectedCount) {
    return emitOpError("padDimensions has ")
           << padding->size() << " entries for " << selectedCount
           << " selected segments";
  }
  if (padding && !drains()) {
    return emitOpError("padDimensions is only valid on a draining endpoint");
  }
  if (getPadValue() != 0 &&
      (!padding || llvm::none_of(*padding, [](BDPadLayoutArrayAttr entry) {
        return !entry.empty();
      }))) {
    return emitOpError("nonzero padValue requires a non-empty padDimensions "
                       "entry");
  }

  std::vector<ObjectFifoSegmentOp> selected = getSelectedSegments();
  for (size_t position = 0; position < selectedCount; ++position) {
    BDDimLayoutArrayAttr dims =
        dimensions ? (*dimensions)[position] : BDDimLayoutArrayAttr();
    BDPadLayoutArrayAttr pads =
        padding ? (*padding)[position] : BDPadLayoutArrayAttr();
    if (pads && !pads.empty() && (!dims || dims.empty())) {
      return emitOpError("padDimensions entry ")
             << position << " requires dimensions for the same segment";
    }
    if (dims && pads && !pads.empty() && dims.size() != pads.size()) {
      return emitOpError("dimensions and padDimensions entry ")
             << position << " have different ranks";
    }
    if (dims && !dims.empty() &&
        getDimsMaxIdx(dims) >= selected[position].getSize()) {
      return emitOpError("dimensions entry ")
             << position << " exceeds selected segment '"
             << selected[position].getSymName() << "' of size "
             << selected[position].getSize();
    }
  }
  return success();
}

TileLike RouteEndpointOp::getTileLike() {
  return dyn_cast<TileLike>(getTile().getDefiningOp());
}

DMAChannelDir RouteEndpointOp::getRouteDirection() {
  // A flow runs from its source, so an end named there sends and every other
  // end receives.
  auto device = getOperation()->getParentOfType<DeviceOp>();
  StringRef name = getSymName();
  bool named = false;
  for (auto flow : device.getOps<RouteOp>()) {
    if (flow.getSource() == name) {
      return DMAChannelDir::MM2S;
    }
    named |= llvm::any_of(
        flow.getDestinations().getAsRange<FlatSymbolRefAttr>(),
        [&](FlatSymbolRefAttr dest) { return dest.getValue() == name; });
  }
  assert(named && "endpoint is named by no flow, so it has no direction to "
                  "read; --aie-objectfifo-verify rejects this");
  (void)named;
  return DMAChannelDir::S2MM;
}

WireBundle RouteEndpointOp::getRouteBundle() { return getBundle(); }

std::optional<int> RouteEndpointOp::getRouteChannel() {
  return getChannelIndex();
}

void RouteEndpointOp::setRouteChannel(int channel) { setChannelIndex(channel); }

LogicalResult RouteEndpointOp::verify() {
  TileLike tile = getTileLike();
  if (!tile) {
    return emitOpError("tile operand is not an aie.tile or aie.logical_tile");
  }
  // getRouteDirection() reads this end's direction off the flow naming it, so a
  // second mention would leave it ambiguous, including both ends of one flow.
  StringRef name = getSymName();
  int flows = 0;
  for (auto flow : (*this)->getParentOfType<DeviceOp>().getOps<RouteOp>()) {
    if (flow.getSource() == name) {
      flows++;
    }
    flows += llvm::count_if(
        flow.getDestinations().getAsRange<FlatSymbolRefAttr>(),
        [&](FlatSymbolRefAttr dest) { return dest.getValue() == name; });
  }
  if (flows > 1) {
    return emitOpError("drives one channel, so at most one flow may name it, "
                       "but it is named ")
           << flows << " times";
  }
  switch (getBundle()) {
  case WireBundle::DMA:
  case WireBundle::PLIO:
    if (!tile.isShimTile()) {
      return emitOpError("a DMA or PLIO end the runtime drives is on a shim");
    }
    return success();
  case WireBundle::Core:
    return success();
  default:
    return emitOpError("bundle must be DMA, PLIO or Core");
  }
}

//===----------------------------------------------------------------------===//
// RouteOp
//===----------------------------------------------------------------------===//

LogicalResult RouteOp::verify() {
  if (getDestinations().empty()) {
    return emitOpError("expects at least one destination");
  }

  if (getPacketId() && !getPacket()) {
    return emitOpError("packet_id is only meaningful on a packet flow");
  }

  auto device = (*this)->getParentOfType<DeviceOp>();
  auto lookup = [&](StringRef name) {
    return dyn_cast_or_null<RouteEndpoint>(SymbolTable::lookupNearestSymbolFrom(
        device, StringAttr::get(getContext(), name)));
  };

  if (!lookup(getSource())) {
    return emitOpError("source '") << getSource() << "' is not an endpoint";
  }

  for (auto destination : getDestinations().getAsRange<FlatSymbolRefAttr>()) {
    if (!lookup(destination.getValue())) {
      return emitOpError("destination '")
             << destination.getValue() << "' is not an endpoint";
    }
  }

  return success();
}

BDDimLayoutArrayAttr
ObjectFifoCreateOp::getDimensionsFromStream(Value consumerTile) {
  int dimsIndex = 0;
  for (auto cons : getConsumerTiles()) {
    if (cons == consumerTile)
      break;
    dimsIndex++;
  }
  return getDimensionsFromStreamPerConsumer()[dimsIndex];
}

ParseResult xilinx::AIE::parseObjectFifoProducerTile(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &operand,
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

void xilinx::AIE::printObjectFifoProducerTile(OpAsmPrinter &printer,
                                              Operation *op, Value operand,
                                              BDDimLayoutArrayAttr dimensions) {
  printer << operand;
  if (!dimensions.empty()) {
    printer << " dimensionsToStream ";
    printer.printStrippedAttrOrType(dimensions);
  }
}

ParseResult
xilinx::AIE::parseObjectFifoAcquireObjects(OpAsmParser &parser,
                                           ObjectFifoPortAttr &port,
                                           SmallVectorImpl<Type> &objects) {
  if (parser.parseLParen())
    return failure();

  StringRef portName;
  SMLoc portLoc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalKeyword(&portName))) {
    std::optional<ObjectFifoPort> parsed = symbolizeObjectFifoPort(portName);
    if (!parsed)
      return parser.emitError(portLoc, "invalid objectFifo port: ") << portName;
    port = ObjectFifoPortAttr::get(parser.getContext(), *parsed);
    if (parser.parseComma())
      return failure();
  }

  int64_t count;
  SMLoc countLoc = parser.getCurrentLocation();
  if (parser.parseInteger(count) || parser.parseRParen() ||
      parser.parseColon() || parser.parseTypeList(objects))
    return failure();

  if (count != static_cast<int64_t>(objects.size()))
    return parser.emitError(countLoc, "acquires ")
           << count << " objects but names " << objects.size() << " of them";
  return success();
}

void xilinx::AIE::printObjectFifoAcquireObjects(OpAsmPrinter &printer,
                                                Operation *op,
                                                ObjectFifoPortAttr port,
                                                TypeRange objects) {
  printer << "(";
  if (port)
    printer << stringifyObjectFifoPort(port.getValue()) << ", ";
  printer << objects.size() << ") : ";
  llvm::interleaveComma(objects, printer);
}

ParseResult xilinx::AIE::parseObjectFifoReleaseCount(OpAsmParser &parser,
                                                     ObjectFifoPortAttr &port,
                                                     IntegerAttr &size) {
  if (parser.parseLParen())
    return failure();

  StringRef portName;
  SMLoc portLoc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalKeyword(&portName))) {
    std::optional<ObjectFifoPort> parsed = symbolizeObjectFifoPort(portName);
    if (!parsed)
      return parser.emitError(portLoc, "invalid objectFifo port: ") << portName;
    port = ObjectFifoPortAttr::get(parser.getContext(), *parsed);
    if (parser.parseComma())
      return failure();
  }

  int64_t count;
  if (parser.parseInteger(count) || parser.parseRParen())
    return failure();
  size = parser.getBuilder().getI32IntegerAttr(count);
  return success();
}

void xilinx::AIE::printObjectFifoReleaseCount(OpAsmPrinter &printer,
                                              Operation *op,
                                              ObjectFifoPortAttr port,
                                              IntegerAttr size) {
  printer << "(";
  if (port)
    printer << stringifyObjectFifoPort(port.getValue()) << ", ";
  printer << size.getInt() << ")";
}

ParseResult xilinx::AIE::parseObjectFifoConsumerTiles(
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

void xilinx::AIE::printObjectFifoConsumerTiles(
    OpAsmPrinter &printer, Operation *op, OperandRange tiles,
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

static void printObjectFifoConsumerElemType(OpAsmPrinter &p,
                                            ObjectFifoCreateOp op,
                                            TypeAttr consumerElemType) {
  if (consumerElemType)
    p << " -> " << consumerElemType;
}

static ParseResult parseObjectFifoConsumerElemType(OpAsmParser &parser,
                                                   TypeAttr &consumerElemType) {
  if (failed(parser.parseOptionalArrow()))
    return success(); // no consumer type
  Type type;
  if (parser.parseType(type))
    return failure();
  consumerElemType = TypeAttr::get(type);
  return success();
}

static void printObjectFifoInitValues(OpAsmPrinter &p, ObjectFifoCreateOp op,
                                      Attribute numElem, TypeAttr type,
                                      Attribute initValues) {
  if (op.getInitValues()) {
    p << "= [";
    // `numElem` may be an IntegerAttr (scalar depth) or an ArrayAttr of
    // per-endpoint depths; initValues populates the producer side, which
    // is the first entry of the ArrayAttr.
    int depth;
    if (isa<ArrayAttr>(numElem)) {
      depth =
          llvm::cast<mlir::IntegerAttr>(llvm::cast<mlir::ArrayAttr>(numElem)[0])
              .getInt();
    } else {
      depth = llvm::cast<mlir::IntegerAttr>(numElem).getInt();
    }
    for (int i = 0; i < depth; i++) {
      p.printStrippedAttrOrType(llvm::cast<mlir::ArrayAttr>(initValues)[i]);
      if (i < depth - 1) {
        p << ", ";
      }
    }
    p << "]";
  }
}

static ParseResult parseObjectFifoInitValues(OpAsmParser &parser,
                                             Attribute numElem, TypeAttr type,
                                             Attribute &initValues) {
  int depth;
  if (isa<ArrayAttr>(numElem)) {
    depth =
        llvm::cast<mlir::IntegerAttr>(llvm::cast<mlir::ArrayAttr>(numElem)[0])
            .getInt();
  } else {
    depth = llvm::cast<mlir::IntegerAttr>(numElem).getInt();
  }
  auto objfifoType = llvm::cast<AIEObjectFifoType>(type.getValue());
  auto memrefType = llvm::cast<MemRefType>(objfifoType.getElementType());

  if (!memrefType.hasStaticShape())
    return parser.emitError(parser.getNameLoc())
           << "type should be static shaped memref, but got " << memrefType;

  if (parser.parseOptionalEqual())
    return success();

  Type tensorType = mlir::memref::getTensorTypeFromMemRefType(memrefType);
  if (parser.parseAttribute(initValues, tensorType))
    return failure();
  for (int i = 0; i < depth; i++) {
    auto initialValues = llvm::dyn_cast<mlir::ArrayAttr>(initValues);
    if ((int)initialValues.size() != depth)
      return parser.emitError(parser.getNameLoc())
             << "initial values should initialize all objects";
    if (!llvm::isa<ElementsAttr>(initialValues[i]))
      return parser.emitError(parser.getNameLoc())
             << "initial value should be an elements attribute";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ObjectFifoAllocateOp
//===----------------------------------------------------------------------===//

LogicalResult ObjectFifoAllocateOp::verify() {
  ObjectFifoCreateOp objFifo = getObjectFifo();
  if (!objFifo)
    return emitError("cannot retrieve associated object FIFO");
  if (objFifo.getConsumerTiles().size() != 1)
    return emitError("can only be used in 1-to-1 object FIFOs");
  if (objFifo.getVia_DMA())
    return emitError("cannot allocate a shared memory module to objectfifo "
                     "with set `via_DMA` attribute");
  if (objFifo.getRepeatCount().has_value())
    return emitError("cannot allocate a shared memory module to objectfifo "
                     "with set `repeat_count` attribute");
  if (!objFifo.getDimensionsToStream().empty())
    return emitError("cannot allocate a shared memory module to objectfifo "
                     "with set dimensions attribute");
  if (objFifo.getAieStream().has_value())
    return emitError("cannot allocate a shared memory module to objectfifo "
                     "using stream port");
  return success();
}

TileOp ObjectFifoAllocateOp::getDelegateTileOp() {
  return cast<TileOp>(getDelegateTile().getDefiningOp());
}

ObjectFifoCreateOp ObjectFifoAllocateOp::getObjectFifo() {
  Operation *parent = getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      if (auto *st = SymbolTable::lookupSymbolIn(parent, getObjFifoName());
          isa_and_nonnull<ObjectFifoCreateOp>(st))
        return cast<ObjectFifoCreateOp>(st);
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ObjectFifoLinkOp
//===----------------------------------------------------------------------===//

LogicalResult ObjectFifoLinkOp::verify() {
  if (isJoin() && isDistribute())
    return emitError("ObjectFifoLinkOp does not support 'join' and "
                     "'distribute' at the same time");

  auto participants = [](ObjectFifoLinkOp link) {
    std::vector<ObjectFifoCreateOp> all = link.getInputObjectFifos();
    std::vector<ObjectFifoCreateOp> outs = link.getOutputObjectFifos();
    all.insert(all.end(), outs.begin(), outs.end());
    return all;
  };

  // A fifo passes its objects to exactly one place, so it belongs to at most
  // one link.
  for (auto other :
       (*this)->getParentOfType<DeviceOp>().getOps<ObjectFifoLinkOp>()) {
    if (other == *this) {
      break;
    }
    for (ObjectFifoCreateOp mine : participants(*this)) {
      for (ObjectFifoCreateOp theirs : participants(other)) {
        if (mine == theirs) {
          return mine.emitOpError(
              "objectfifo cannot be in more than one ObjectFifoLinkOp");
        }
      }
    }
  }

  auto sharedTile = getOptionalSharedTile();
  if (!sharedTile)
    return emitError("ObjectFifoLinkOp must have a link point, i.e., a "
                     "shared tile between objectFifos");

  TileLike tile = llvm::dyn_cast<TileLike>(sharedTile.value().getDefiningOp());
  if (!tile)
    return emitError("shared tile must implement TileLike interface");
  // Each participant occupies its own slice of the shared object, so the link
  // point needs a memory module to hold that object and one counting lock pair
  // per slice.
  if (isJoin() || isDistribute()) {
    if (tile.isShimTile())
      return emitError("ObjectFifoLinkOp join and distribute are "
                       "unavailable on shim tiles");
    if (!getTargetModel(getOperation())
             .hasProperty(AIETargetModel::UsesSemaphoreLocks))
      return emitError("ObjectFifoLinkOp join and distribute require "
                       "semaphore locks, which this device lacks");
  }

  if (isJoin()) {
    if (getFifoIns().size() != getSrcOffsets().size())
      return emitOpError("number of provided src offsets must be equal "
                         "to the number of input objectFifos");

    if (!getDstOffsets().empty())
      return emitOpError("dst offsets should be empty for join");

    ObjectFifoCreateOp fifoOut = getOutputObjectFifos()[0];
    if (!fifoOut.getDimensionsToStream().empty()) {
      int64_t maxIdx = getDimsMaxIdx(fifoOut.getDimensionsToStream());
      int64_t minInputBufferSize = -1;
      for (auto lenIn : getJoinTransferLengths()) {
        if (lenIn <= minInputBufferSize || minInputBufferSize < 0)
          minInputBufferSize = lenIn;
      }
      if (minInputBufferSize <= maxIdx) {
        return emitOpError()
               << "specified output stride(s) and size(s) result in out "
                  "of bounds access in join input, for index "
               << std::to_string(maxIdx) << " in transfer of length "
               << std::to_string(minInputBufferSize) << ".";
      }
    }

  } else if (isDistribute()) {
    if (getFifoOuts().size() != getDstOffsets().size())
      return emitOpError("number of provided dst offsets must be equal "
                         "to the number of output objectFifos");

    if (!getSrcOffsets().empty())
      return emitOpError("src offsets should be empty for distribute");

    ObjectFifoCreateOp fifoIn = getInputObjectFifos()[0];
    if (!fifoIn.getDimensionsFromStream(sharedTile.value()).empty()) {
      int64_t maxIdx =
          getDimsMaxIdx(fifoIn.getDimensionsFromStream(sharedTile.value()));
      int64_t minOutputBufferSize = -1;
      for (auto lenOut : getDistributeTransferLengths()) {
        if (lenOut <= minOutputBufferSize || minOutputBufferSize < 0)
          minOutputBufferSize = lenOut;
      }
      if (minOutputBufferSize <= maxIdx) {
        return emitOpError()
               << "specified input stride(s) and size(s) result in out "
                  "of bounds access in distribute output, for index "
               << std::to_string(maxIdx) << " in transfer of length "
               << std::to_string(minOutputBufferSize) << ".";
      }
    }

    std::vector<int> repeat_counts;
    for (auto fifoOut : getOutputObjectFifos()) {
      if (auto repeatCount = fifoOut.getRepeatCount()) {
        repeat_counts.push_back(*repeatCount);
      } else {
        repeat_counts.push_back(0);
      }
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
        auto name = cast<FlatSymbolRefAttr>(sym);
        if (auto *st = SymbolTable::lookupSymbolIn(parent, name);
            isa_and_nonnull<ObjectFifoCreateOp>(st))
          inputObjFifos.push_back(cast<ObjectFifoCreateOp>(st));
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
        auto name = cast<FlatSymbolRefAttr>(sym);
        if (auto *st = mlir::SymbolTable::lookupSymbolIn(parent, name);
            isa_and_nonnull<ObjectFifoCreateOp>(st))
          outputObjFifos.push_back(cast<ObjectFifoCreateOp>(st));
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
    // src_offsets is an I64ArrayAttr, so every element is an IntegerAttr and
    // getConstantIntValue always yields a value; assert to keep this fail-fast.
    auto srcOffset = [&](size_t idx) -> int {
      std::optional<int64_t> v = getConstantIntValue(getSrcOffsets()[idx]);
      assert(v && "src_offsets element must be a constant integer");
      return static_cast<int>(*v);
    };
    for (size_t i = 0; i < getFifoIns().size(); i++) {
      int len = 0;
      int offset = srcOffset(i);
      if (i == getFifoIns().size() - 1)
        len = lenOut - offset;
      else
        len = srcOffset(i + 1) - offset;
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
    // dst_offsets is an I64ArrayAttr, so every element is an IntegerAttr and
    // getConstantIntValue always yields a value; assert to keep this fail-fast.
    auto dstOffset = [&](size_t idx) -> int {
      std::optional<int64_t> v = getConstantIntValue(getDstOffsets()[idx]);
      assert(v && "dst_offsets element must be a constant integer");
      return static_cast<int>(*v);
    };
    for (size_t i = 0; i < getFifoOuts().size(); i++) {
      int offset = dstOffset(i);
      int len = 0;
      if (i == getFifoOuts().size() - 1)
        len = lenIn - offset;
      else
        len = dstOffset(i + 1) - offset;
      lengths.push_back(len);
    }
  }
  return lengths;
}

std::optional<int> ObjectFifoLinkOp::getRepeatCount() {
  for (auto fifoOut : getOutputObjectFifos())
    if (fifoOut.getRepeatCount().has_value())
      return {fifoOut.getRepeatCount()};
  return {};
}

//===----------------------------------------------------------------------===//
// ObjectFifoRegisterExternalBuffersOp
//===----------------------------------------------------------------------===//

TileOp ObjectFifoRegisterExternalBuffersOp::getTileOp() {
  return cast<TileOp>(getTile().getDefiningOp());
}

ObjectFifoCreateOp ObjectFifoRegisterExternalBuffersOp::getObjectFifo() {
  Operation *parent = getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      if (auto *st = SymbolTable::lookupSymbolIn(parent, getObjFifoName());
          isa_and_nonnull<ObjectFifoCreateOp>(st))
        return cast<ObjectFifoCreateOp>(st);
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ObjectFifoAcquireOp
//===----------------------------------------------------------------------===//

namespace {

/// Symbol an objectFifo access names: the fifo it belongs to, or the core
/// endpoint it works through.
Operation *lookupAccessTarget(Operation *op, StringRef name) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      if (Operation *target = SymbolTable::lookupSymbolIn(parent, name)) {
        return target;
      }
    }
  }
  return nullptr;
}

/// Element type an access sees, or a null type if the symbol names neither a
/// fifo nor an endpoint.
Type accessElementType(Operation *target) {
  if (auto fifo = dyn_cast_or_null<ObjectFifoCreateOp>(target)) {
    return llvm::cast<AIEObjectFifoType>(fifo.getElemType()).getElementType();
  }
  if (auto endpoint = dyn_cast_or_null<ObjectFifoCoreEndpointOp>(target)) {
    if (endpoint.getPoolOp()) {
      return endpoint.getAccessType();
    }
  }
  return {};
}

/// A core may only reach a fifo end that is placed on its own tile.
LogicalResult verifyAccessPlacement(Operation *op, Operation *target,
                                    std::optional<ObjectFifoPort> port) {
  auto core = op->getParentOfType<CoreOp>();
  Value coreTile = core.getTile();

  if (auto endpoint = dyn_cast_or_null<ObjectFifoCoreEndpointOp>(target)) {
    if (port) {
      return op->emitOpError("port is implied by the endpoint's role");
    }
    if (endpoint.getTile() != coreTile) {
      return core.emitOpError(
          "objectFifo endpoint accessed by core running on another tile");
    }
    return success();
  }

  auto fifo = dyn_cast_or_null<ObjectFifoCreateOp>(target);
  if (!fifo) {
    return op->emitError("cannot retrieve associated object FIFO");
  }
  if (!port) {
    return op->emitOpError("port is required when accessing an objectFifo");
  }

  if (*port == ObjectFifoPort::Produce) {
    if (coreTile != fifo.getProducerTile()) {
      return core.emitOpError(
          "producer port of objectFifo accessed by core running "
          "on non-producer tile");
    }
  } else {
    if (!llvm::is_contained(fifo.getConsumerTiles(), coreTile)) {
      return core.emitOpError(
          "consumer port of objectFifo accessed by core running "
          "on non-consumer tile");
    }
  }
  return success();
}

} // namespace

LogicalResult ObjectFifoAcquireOp::verify() {
  auto parent = getOperation()->getParentOfType<CoreOp>();
  if (parent == nullptr) {
    return emitOpError("must be called from inside a CoreOp");
  }

  Operation *target = lookupAccessTarget(*this, getObjFifoName());
  if (failed(verifyAccessPlacement(*this, target, getPort()))) {
    return failure();
  }

  if (getObjects().empty()) {
    return emitOpError("must acquire at least one object");
  }

  Type elem = accessElementType(target);
  if (!elem) {
    return success();
  }
  auto fifo = dyn_cast<ObjectFifoCreateOp>(target);
  if (fifo) {
    // A fifo may state a depth per endpoint, the producer first.
    int index = 0;
    if (getPort() == ObjectFifoPort::Consume) {
      for (auto [position, consumer] :
           llvm::enumerate(fifo.getConsumerTiles())) {
        if (consumer == parent.getTile()) {
          index = position + 1;
          break;
        }
      }
    }
    if (static_cast<int>(getObjects().size()) > fifo.size(index)) {
      return emitOpError("acquires ")
             << getObjects().size() << " objects from an objectFifo that holds "
             << fifo.size(index);
    }
  }
  // An asymmetric fifo hands the consumer a different element type.
  auto consType = fifo ? llvm::dyn_cast<AIEObjectFifoType>(
                             fifo.getConsumerElemTypeOrDefault())
                       : nullptr;
  if (fifo && !consType) {
    return emitOpError("ObjectFifo consumer element type must be an "
                       "!aie.objectfifo<memref<...>> type");
  }

  for (Type object : getObjects().getTypes()) {
    if (object != elem && (!consType || consType.getElementType() != object)) {
      return emitOpError("acquired object type ")
             << object << " does not match the objectFifo's " << elem;
    }
  }

  return success();
}

ObjectFifoCreateOp ObjectFifoAcquireOp::getObjectFifo() {
  Operation *parent = getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      if (auto *st = SymbolTable::lookupSymbolIn(parent, getObjFifoName());
          isa_and_nonnull<ObjectFifoCreateOp>(st))
        return cast<ObjectFifoCreateOp>(st);
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ObjectFifoReleaseOp
//===----------------------------------------------------------------------===//

LogicalResult ObjectFifoReleaseOp::verify() {
  auto parent = getOperation()->getParentOfType<CoreOp>();
  if (parent == nullptr)
    return emitOpError("must be called from inside a CoreOp");

  Operation *target = lookupAccessTarget(*this, getObjFifoName());
  return verifyAccessPlacement(*this, target, getPort());
}

ObjectFifoCreateOp ObjectFifoReleaseOp::getObjectFifo() {
  Operation *parent = getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      if (auto *st = SymbolTable::lookupSymbolIn(parent, getObjFifoName());
          isa_and_nonnull<ObjectFifoCreateOp>(st))
        return cast<ObjectFifoCreateOp>(st);
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// CascadeFlowOp
//===----------------------------------------------------------------------===//

LogicalResult CascadeFlowOp::verify() {
  TileLike src = getSourceTileLike();
  TileLike dst = getDestTileLike();

  if (!src || !dst)
    return emitOpError("source and dest must be tile-like operations");

  if (src.isShimTile() || dst.isShimTile())
    return emitOpError("shimTile row has no cascade stream interface");
  if (src.isMemTile() || dst.isMemTile())
    return emitOpError("memTile row has no cascade stream interface");

  std::optional<int> srcCol = src.tryGetCol();
  std::optional<int> srcRow = src.tryGetRow();
  std::optional<int> dstCol = dst.tryGetCol();
  std::optional<int> dstRow = dst.tryGetRow();

  if (srcCol && srcRow && dstCol && dstRow) {
    const auto &t = getTargetModel(*this);
    if (!t.isSouth(*srcCol, *srcRow, *dstCol, *dstRow) &&
        !t.isWest(*srcCol, *srcRow, *dstCol, *dstRow) &&
        !t.isNorth(*srcCol, *srcRow, *dstCol, *dstRow) &&
        !t.isEast(*srcCol, *srcRow, *dstCol, *dstRow)) {
      return emitOpError("tiles must be adjacent");
    }
  }

  return success();
}

TileLike CascadeFlowOp::getSourceTileLike() {
  return dyn_cast<TileLike>(getSourceTile().getDefiningOp());
}

TileLike CascadeFlowOp::getDestTileLike() {
  return dyn_cast<TileLike>(getDestTile().getDefiningOp());
}

TileOp CascadeFlowOp::getSourceTileOp() {
  if (auto tileOp = dyn_cast_or_null<TileOp>(getSourceTile().getDefiningOp()))
    return tileOp;
  llvm::report_fatal_error("Calling getSourceTileOp requires TileOp.");
}

TileOp CascadeFlowOp::getDestTileOp() {
  if (auto tileOp = dyn_cast_or_null<TileOp>(getDestTile().getDefiningOp()))
    return tileOp;
  llvm::report_fatal_error("Calling getDestTileOp requires TileOp.");
}

//===----------------------------------------------------------------------===//
// ConfigureCascadeOp
//===----------------------------------------------------------------------===//

LogicalResult ConfigureCascadeOp::verify() {
  if (!isa<TileOp>(getTile().getDefiningOp()))
    return emitOpError("requires a placed tile (aie.tile), not a logical tile");

  const auto &t = getTargetModel(*this);
  TileOp tile = cast<TileOp>(getTile().getDefiningOp());
  CascadeDir inputDir = getInputDir();
  CascadeDir outputDir = getOutputDir();

  if (tile.isShimTile())
    return emitOpError("shimTile row has no cascade stream interface");
  if (tile.isMemTile())
    return emitOpError("memTile row has no cascade stream interface");

  if (isa<AIE2TargetModel>(t)) {
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
  if (isa<AIE1TargetModel>(targetModel)) {
    if (bits != 384)
      return emitOpError("must be a 384-bit type");
  } else if (isa<AIE2TargetModel>(targetModel)) {
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

LogicalResult DeviceOp::verify() {
  // A compute tile has exactly one core in hardware, so at most one aie.core
  // may resolve to any given (col, row). Cores whose tile is a logical_tile
  // with unspecified coordinates are skipped here — they cannot collide until
  // --aie-place-tiles assigns them a position, at which point this same check
  // runs again on the resulting aie.tile coordinates.
  DenseMap<TileID, CoreOp> coreAtTile;
  WalkResult result = walk([&](CoreOp core) {
    auto tile =
        llvm::dyn_cast_or_null<TileLike>(core.getTile().getDefiningOp());
    if (!tile)
      return WalkResult::advance();
    std::optional<int> col = tile.tryGetCol();
    std::optional<int> row = tile.tryGetRow();
    if (!col || !row)
      return WalkResult::advance();
    TileID id{*col, *row};
    auto [it, inserted] = coreAtTile.try_emplace(id, core);
    if (!inserted) {
      InFlightDiagnostic diag =
          core.emitOpError()
          << "tile (" << *col << ", " << *row
          << ") already has a core; each compute tile can host only one core";
      diag.attachNote(it->second.getLoc()) << "the other core is here";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  return success();
}

const AIETargetModel &DeviceOp::getTargetModel() {
  return xilinx::AIE::getTargetModel(getDevice());
}

xilinx::AIE::DeviceOp DeviceOp::getForSymbolInModule(mlir::ModuleOp module,
                                                     llvm::StringRef symbol) {
  DeviceOp deviceOp;
  if (symbol.empty()) {
    // If no device name is given, assume 'main'
    symbol = "main";
  }
  Operation *maybeDeviceOp = mlir::SymbolTable::lookupSymbolIn(module, symbol);
  if (!maybeDeviceOp) {
    return nullptr;
  }
  deviceOp = llvm::dyn_cast<DeviceOp>(maybeDeviceOp);
  return deviceOp;
}

xilinx::AIE::DeviceOp
DeviceOp::getForSymbolInModuleOrError(mlir::ModuleOp module,
                                      llvm::StringRef symbol) {
  DeviceOp deviceOp = getForSymbolInModule(module, symbol);
  if (!deviceOp) {
    if (!symbol.empty()) {
      module.emitError("No such device: ") << symbol;
    } else {
      module.emitError("No 'main' device in module");
    }
  }
  return deviceOp;
}

//===----------------------------------------------------------------------===//
// TileElement
//===----------------------------------------------------------------------===//

TileOp TileElement::tryGetTileOp() {
  auto element = cast<TileElement>(this->getOperation());
  return dyn_cast_or_null<TileOp>(element.getTile().getDefiningOp());
}

TileOp TileElement::getTileOp() {
  if (auto tileOp = tryGetTileOp())
    return tileOp;
  llvm::report_fatal_error("Calling getTileOp requires TileOp.");
}

//===----------------------------------------------------------------------===//
// LogicalTileOp
//===----------------------------------------------------------------------===//

LogicalResult LogicalTileOp::verify() {
  const auto &targetModel = getTargetModel(*this);
  int columns = targetModel.columns();
  int rows = targetModel.rows();

  // Only verify col/row bounds if they are specified
  if (auto col = getCol()) {
    if (*col >= columns)
      return emitOpError("column index (")
             << *col
             << ") must be less than the number of columns in the device ("
             << columns << ")";
  }
  if (auto row = getRow()) {
    if (*row >= rows)
      return emitOpError("row index (")
             << *row << ") must be less than the number of rows in the device ("
             << rows << ")";
  }

  // Check that the specified tile type exists on the target device
  AIETileType tileType = getTileType();
  bool tileTypeExists = false;
  for (int col = 0; col < columns && !tileTypeExists; col++) {
    for (int row = 0; row < rows && !tileTypeExists; row++) {
      if (targetModel.getTileType(col, row) == tileType)
        tileTypeExists = true;
    }
  }
  if (!tileTypeExists) {
    return emitOpError("tile type '")
           << stringifyAIETileType(tileType)
           << "' does not exist on the target device";
  }

  // Check logical tile type matches coordinates on device
  // Only validate when both col and row are specified
  if (auto col = tryGetCol()) {
    if (auto row = tryGetRow()) {
      if (targetModel.getTileType(*col, *row) != tileType) {
        return emitOpError("declared logical tile type does not match "
                           "the tile type at coordinates (")
               << *col << ", " << *row << ")";
      }
    }
  }

  if (isShimNOCorPLTile() && getAllocationScheme())
    return emitOpError("Shim tiles cannot have an allocation scheme");

  return success();
}

TileID LogicalTileOp::getCanonicalTileID() {
  const auto &targetModel = getTargetModel(*this);

  // If col and row are both specified, use them directly
  std::optional<int32_t> col = getCol();
  std::optional<int32_t> row = getRow();
  if (col.has_value() && row.has_value())
    return {*col, *row};

  // Otherwise, find a representative tile of the given type
  AIETileType tileType = getTileType();
  for (int col = 0; col < targetModel.columns(); col++) {
    for (int row = 0; row < targetModel.rows(); row++) {
      if (targetModel.getTileType(col, row) == tileType) {
        return {col, row};
      }
    }
  }
  llvm_unreachable("No tile of matching tile type found in AIE device");
}

size_t LogicalTileOp::getNumSourceConnections(WireBundle bundle) {
  const auto &targetModel = getTargetModel(*this);
  TileID tile = getCanonicalTileID();

  if (bundle == WireBundle::Core || bundle == WireBundle::DMA) {
    // Note dest is correct here, since direction is reversed.
    if (isShimNOCorPLTile())
      return targetModel.getNumDestShimMuxConnections(tile.col, tile.row,
                                                      bundle);
    return targetModel.getNumDestSwitchboxConnections(tile.col, tile.row,
                                                      bundle);
  }
  return 0;
}

size_t LogicalTileOp::getNumDestConnections(WireBundle bundle) {
  const auto &targetModel = getTargetModel(*this);
  TileID tile = getCanonicalTileID();

  if (bundle == WireBundle::Core || bundle == WireBundle::DMA) {
    // Note source is correct here, since direction is reversed.
    if (isShimNOCorPLTile())
      return targetModel.getNumDestShimMuxConnections(tile.col, tile.row,
                                                      bundle);
    return targetModel.getNumSourceSwitchboxConnections(tile.col, tile.row,
                                                        bundle);
  }
  return 0;
}

std::optional<int> LogicalTileOp::tryGetCol() {
  if (auto col = getCol())
    return col;
  return std::nullopt;
}

std::optional<int> LogicalTileOp::tryGetRow() {
  if (auto row = getRow())
    return row;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Custom Printer and Parser for LogicalTileOp
//===----------------------------------------------------------------------===//

ParseResult LogicalTileOp::parse(OpAsmParser &parser, OperationState &result) {
  AIETileType tileType;
  if (parser.parseLess())
    return failure();

  StringRef tileTypeStr;
  if (parser.parseKeyword(&tileTypeStr))
    return failure();

  auto tileTypeOpt = symbolizeAIETileType(tileTypeStr);
  if (!tileTypeOpt)
    return parser.emitError(parser.getCurrentLocation(),
                            "unknown logical tile type: ")
           << tileTypeStr;
  tileType = *tileTypeOpt;

  if (parser.parseGreater())
    return failure();

  if (parser.parseLParen())
    return failure();

  std::optional<int32_t> col;
  if (succeeded(parser.parseOptionalQuestion())) {
    // col is unspecified
  } else {
    int32_t colVal;
    if (parser.parseInteger(colVal))
      return failure();
    col = colVal;
  }

  if (parser.parseComma())
    return failure();

  std::optional<int32_t> row;
  if (succeeded(parser.parseOptionalQuestion())) {
    // row is unspecified
  } else {
    int32_t rowVal;
    if (parser.parseInteger(rowVal))
      return failure();
    row = rowVal;
  }

  if (parser.parseRParen())
    return failure();

  // Parse optional attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Add the parsed attributes to the result
  result.getOrAddProperties<LogicalTileOp::Properties>().tile_type =
      AIETileTypeAttr::get(parser.getContext(), tileType);
  if (col)
    result.getOrAddProperties<LogicalTileOp::Properties>().col =
        parser.getBuilder().getI32IntegerAttr(*col);
  if (row)
    result.getOrAddProperties<LogicalTileOp::Properties>().row =
        parser.getBuilder().getI32IntegerAttr(*row);

  // Add result type (index)
  result.addTypes(parser.getBuilder().getIndexType());

  return success();
}

void LogicalTileOp::print(OpAsmPrinter &printer) {
  printer << "<" << stringifyAIETileType(getTileType()) << ">";

  printer << "(";
  if (auto col = getCol())
    printer << *col;
  else
    printer << "?";
  printer << ", ";
  if (auto row = getRow())
    printer << *row;
  else
    printer << "?";
  printer << ")";

  SmallVector<StringRef, 3> elidedAttrs = {"tile_type", "col", "row"};
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

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

  if (isShimNOCorPLTile() && getAllocationScheme())
    return emitOpError("Shim tiles cannot have an allocation scheme");

  return success();
}

size_t TileOp::getNumSourceConnections(WireBundle bundle) {
  const auto &targetModel = getTargetModel(*this);
  if (bundle == WireBundle::Core || bundle == WireBundle::DMA)
  // Note dest is correct here, since direction is reversed.
  {
    // Note dest is correct here, since direction is reversed.
    if (isShimNOCorPLTile())
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
    if (isShimNOCorPLTile())
      return targetModel.getNumDestShimMuxConnections(getCol(), getRow(),
                                                      bundle);
    return targetModel.getNumSourceSwitchboxConnections(getCol(), getRow(),
                                                        bundle);
  }
  return 0;
}

std::optional<int> TileOp::tryGetCol() { return getCol(); }
std::optional<int> TileOp::tryGetRow() { return getRow(); }

AIETileType TileOp::getTileType() {
  const auto &targetModel = getTargetModel(*this);
  return targetModel.getTileType(getCol(), getRow());
}

static bool isLegalTileConnection(TileOp tile,
                                  const AIETargetModel &targetModel,
                                  MasterSetOp masterOp, PacketRulesOp slaveOp) {
  auto srcBundle = slaveOp.sourcePort().bundle;
  auto srcChan = slaveOp.sourcePort().channel;
  auto dstBundle = masterOp.destPort().bundle;
  auto dstChan = masterOp.destPort().channel;
  return targetModel.isLegalTileConnection(
      tile.colIndex(), tile.rowIndex(), srcBundle, srcChan, dstBundle, dstChan);
}

static bool isLegalTileConnection(TileOp tile,
                                  const AIETargetModel &targetModel,
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
    tile = TileOp::create(builder, device.getLoc(), builder.getIndexType(), col,
                          row);
  }
  return tile;
}

//===----------------------------------------------------------------------===//
// ShimMuxOp
//===----------------------------------------------------------------------===//

LogicalResult ShimMuxOp::verify() {
  // The port/connection checks below are keyed off the target model for a
  // placed tile. Before --aie-place-tiles the tile is still an
  // aie.logical_tile, so defer those checks to the post-placement re-verify --
  // mirroring SwitchboxOp::verify and the UsesAreAccessible trait.
  if (!isa<TileOp>(getTile().getDefiningOp()))
    return success();

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

//===----------------------------------------------------------------------===//
// ShimDMAOp
//===----------------------------------------------------------------------===//

LogicalResult ShimDMAOp::verify() {
  if (HasSomeTerminator<DMAStartOp, NextBDOp, EndOp>::verifyTrait(*this)
          .failed())
    return failure();
  return success();
}

TileOp ShimDMAOp::getTileOp() {
  return cast<TileElement>(this->getOperation()).getTileOp();
}

LogicalResult PacketRulesOp::verify() {
  if (Region &body = getRules(); body.empty())
    return emitOpError("should have non-empty body");
  return success();
}

LogicalResult PacketFlowOp::verify() {
  Region &body = getPorts();
  if (body.empty())
    return emitOpError("should have non-empty body");

  int numSources = 0, numDests = 0;
  for (auto &ops : body.front()) {
    if (!isa<PacketSourceOp, PacketDestOp, EndOp>(ops))
      return ops.emitOpError("cannot be contained in a PacketFlow op");
    if (isa<PacketSourceOp>(ops))
      ++numSources;
    if (isa<PacketDestOp>(ops))
      ++numDests;
  }

  if (numSources < 1)
    return emitOpError("must have at least one aie.packet_source");
  if (numDests < 1)
    return emitOpError("must have at least one aie.packet_dest");

  return success();
}

//===----------------------------------------------------------------------===//
// CoreOp
//===----------------------------------------------------------------------===//

LogicalResult CoreOp::verify() {
  if (getBody().empty())
    return emitOpError("should have non-empty body");
  if (getElfFile()) {
    // If an ELF file is specified, no MLIR body is allowed (to remove
    // ambiguity); the ELF file will fully dictate what runs on the
    // core and any MLIR would be ignored.
    if (!isEmpty()) {
      return emitOpError(
          "When `elf_file` attribute is specified, core body must be empty "
          "(consist of exactly one `aie.end` op).");
    }
  }
  if (getLinkWith() && getLinkFiles())
    return emitOpError(
        "cannot specify both 'link_with' (deprecated) and 'link_files' "
        "on the same core; run aie-assign-core-link-files to migrate");
  if (getLinkWith() && getLinkMergeFiles())
    return emitOpError(
        "cannot specify both 'link_with' (deprecated) and 'link_merge_files' "
        "on the same core; run aie-assign-core-link-files to migrate");
  // An artifact is either merged into the core's LLVM module or handed to the
  // final link, never both: doing both would define its symbols twice.
  if (auto linkFiles = getLinkFiles())
    if (auto mergeFiles = getLinkMergeFiles()) {
      llvm::SmallSet<StringRef, 8> linked;
      for (auto f : linkFiles->getAsRange<StringAttr>())
        linked.insert(f.getValue());
      for (auto f : mergeFiles->getAsRange<StringAttr>())
        if (linked.count(f.getValue()))
          return emitOpError("artifact '")
                 << f.getValue()
                 << "' appears in both 'link_files' and 'link_merge_files'; an "
                    "artifact must be either merged or linked, not both";
    }
  // Checked last so it does not pre-empt the diagnostics above on an op with
  // more than one defect.
  if (uint32_t stackSize = getEffectiveStackSize(),
      localMem = getTargetModel(*this).getLocalMemorySize();
      stackSize >= localMem)
    return emitOpError("stack_size ")
           << stackSize << " leaves no local memory for this tile's buffers ("
           << localMem << " bytes total)";
  return success();
}

bool CoreOp::isEmpty() {
  Region &body = getBody();
  // Return iff. core body contains exactly one block with exactly one AIE.EndOp
  return (body.hasOneBlock() && body.front().getOperations().size() == 1 &&
          llvm::isa<AIE::EndOp>(body.front().front()));
}

TileOp CoreOp::getTileOp() {
  return cast<TileElement>(this->getOperation()).getTileOp();
}

uint32_t CoreOp::getEffectiveStackSize() {
  return getStackSize().value_or(
      getTargetModel(*this).getDefaultCoreStackSize());
}

//===----------------------------------------------------------------------===//
// BufferOp
//===----------------------------------------------------------------------===//

int64_t BufferOp::getAllocationSize() {
  auto type = llvm::cast<MemRefType>(getType());
  DataLayout dataLayout = DataLayout::closest(getOperation());
  return type.getNumElements() * dataLayout.getTypeSize(type.getElementType());
}

LogicalResult BufferOp::verify() {
  if (UsesAreAccessible::verifyTrait(*this).failed())
    return failure();
  return success();
}

// FIXME: make address assignment for buffers explicit and move this function to
// an interface
int32_t xilinx::AIE::getBufferBaseAddress(Operation *bufOp) {
  if (auto buf = dyn_cast<BufferOp>(bufOp)) {
    std::optional<int32_t> address = buf.getAddress();
    assert(address.has_value() && "buffer must have address assigned");
    return *address;
  }
  if (isa_and_nonnull<ExternalBufferOp>(bufOp))
    llvm::report_fatal_error(
        "External buffer addresses are assigned at runtime.");
  llvm::report_fatal_error("unknown buffer type");
}

void xilinx::AIE::collectTiles(DeviceOp &device,
                               DenseMap<TileID, Operation *> &tiles) {
  for (auto tile : device.getOps<TileOp>()) {
    tiles[tile.getTileID()] = tile;
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

TileOp BufferOp::getTileOp() {
  return cast<TileElement>(this->getOperation()).getTileOp();
}

//===----------------------------------------------------------------------===//
// MemOp
//===----------------------------------------------------------------------===//

LogicalResult MemOp::verify() {
  Region &body = getBody();
  if (HasSomeTerminator<DMAStartOp, NextBDOp, EndOp>::verifyTrait(*this)
          .failed())
    return failure();

  for (auto &bodyOp : body.getOps()) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(bodyOp))
      if (!allocOp->getAttr("id"))
        return allocOp.emitOpError()
               << "allocOp in MemOp region should have an id attribute";
  }
  return success();
}

TileOp MemOp::getTileOp() {
  return cast<TileElement>(this->getOperation()).getTileOp();
}

//===----------------------------------------------------------------------===//
// MemTileDMAOp
//===----------------------------------------------------------------------===//

LogicalResult MemTileDMAOp::verify() {
  assert(getOperation()->getNumRegions() == 1 &&
         "MemTileDMAOp has zero region!");

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
                bufferOp.getTile() != getTile()) {
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
                lockOp.getTile() != getTile()) {
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

TileOp MemTileDMAOp::getTileOp() {
  return cast<TileElement>(this->getOperation()).getTileOp();
}

//===----------------------------------------------------------------------===//
// DMAOp
//===----------------------------------------------------------------------===//

static bool isBdPacketEnabled(DMABDOp bd) {
  if (bd.getPacket().has_value())
    return true;
  if (Block *blk = bd->getBlock())
    return !blk->getOps<DMABDPACKETOp>().empty();
  return false;
}

LogicalResult
xilinx::AIE::verifyDMABDOutOfOrderId(DMABDOp bd, bool packetEnabledByContext) {
  std::optional<int32_t> oooId = bd.getOutOfOrderId();
  if (!oooId.has_value())
    return success();
  const AIETargetModel &targetModel = getTargetModel(bd.getOperation());
  if (!targetModel.hasProperty(AIETargetModel::SupportsOutOfOrderDMA))
    return bd.emitOpError("out_of_order_id is not supported on this device");
  if (!packetEnabledByContext && !isBdPacketEnabled(bd))
    return bd.emitOpError("out_of_order_id requires a packet-enabled BD");
  uint32_t maxOooId = targetModel.getMaxOutOfOrderId();
  if (*oooId < 0 || static_cast<uint32_t>(*oooId) > maxOooId)
    return bd.emitOpError("out_of_order_id must be in [0, ") << maxOooId << "]";
  return success();
}

static LogicalResult verifyDMARepeatCount(Operation *op, int32_t repeatCount) {
  uint32_t maxRepeat = getTargetModel(op).getMaxRepeatCount();
  if (maxRepeat == 0) {
    if (repeatCount != 0)
      return op->emitOpError("repeat_count is not supported on this target");
    return success();
  }
  if (repeatCount < 0 || static_cast<uint32_t>(repeatCount) > maxRepeat)
    return op->emitOpError("repeat_count ")
           << repeatCount << " is out of range [0, " << maxRepeat
           << "] for this target";
  return success();
}

LogicalResult
xilinx::AIE::verifyOutOfOrderChannel(Operation *op, DMAChannelDir dir,
                                     bool outOfOrder, ArrayRef<DMABDOp> bds,
                                     bool packetEnabledByContext) {
  if (!outOfOrder)
    return success();
  if (dir != DMAChannelDir::S2MM)
    return op->emitOpError("out_of_order is only valid on an S2MM channel");
  if (!getTargetModel(op).hasProperty(AIETargetModel::SupportsOutOfOrderDMA))
    return op->emitOpError(
        "out-of-order S2MM DMA is not supported on this device");
  if (bds.empty())
    return op->emitOpError("out-of-order S2MM channel must have at least one "
                           "receive buffer descriptor"); // else stall
  for (DMABDOp bd : bds) {
    if (!packetEnabledByContext && !isBdPacketEnabled(bd))
      return bd.emitOpError(
          "out-of-order S2MM receive buffer descriptor must be packet-enabled");
    if (bd.getOutOfOrderId().has_value())
      return bd.emitOpError(
          "out_of_order_id belongs on the sender buffer descriptor, not an "
          "out-of-order S2MM receive buffer descriptor");
  }
  // an inter-BD lock dependency can deadlock because arrival order is unknown
  DenseMap<Operation *, Operation *> releasedByRecvBd;
  for (DMABDOp bd : bds)
    for (auto useLock : bd->getBlock()->getOps<UseLockOp>())
      if (useLock.release())
        if (Operation *lockDef = useLock.getLock().getDefiningOp())
          releasedByRecvBd.try_emplace(lockDef, bd.getOperation());
  for (DMABDOp bd : bds)
    for (auto useLock : bd->getBlock()->getOps<UseLockOp>())
      if (useLock.acquire() || useLock.acquireGE())
        if (Operation *lockDef = useLock.getLock().getDefiningOp()) {
          auto it = releasedByRecvBd.find(lockDef);
          if (it != releasedByRecvBd.end() && it->second != bd.getOperation())
            return bd.emitOpError(
                "out-of-order S2MM prohibits inter-BD lock dependencies; "
                "can deadlock");
        }
  return success();
}

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
  if (getPadValue() != 0) {
    if (!isa<MemTileDMAOp>(parentOp))
      return emitOpError("pad_value is only supported on memtile DMA channels");
    if (getChannelDir() != DMAChannelDir::MM2S)
      return emitOpError("pad_value is only supported on MM2S DMA channels");
    if (!getTargetModel(getOperation()).isMemTilePadValueSupported())
      return emitOpError("pad_value requires the CONSTANT_PAD_VALUE register, "
                         "unavailable on this target");
  }
  if (failed(verifyDMARepeatCount(getOperation(), getRepeatCount())))
    return failure();
  SmallVector<DMABDOp> bds;
  for (auto &bdRegion : getBds())
    llvm::append_range(bds, bdRegion.front().getOps<DMABDOp>());
  return verifyOutOfOrderChannel(getOperation(), getChannelDir(),
                                 getOutOfOrder(), bds);
}

//===----------------------------------------------------------------------===//
// DMABDOp
//===----------------------------------------------------------------------===//

BufferOp DMABDOp::getBufferOp() {
  return cast<BufferOp>(getBuffer().getDefiningOp());
}

// Parse/print hooks for the custom<DynamicScalar>($operand, $static_attr)
// directive: a single scalar that is either an SSA value (runtime, %v) or a
// compile-time integer constant (folded into the attribute), so a constant
// never materializes an operand. The scalar analog of custom<DynamicIndexList>.
static ParseResult
parseDynamicScalar(OpAsmParser &parser,
                   std::optional<OpAsmParser::UnresolvedOperand> &operand,
                   IntegerAttr &staticAttr) {
  int64_t intValue;
  OptionalParseResult intResult = parser.parseOptionalInteger(intValue);
  if (intResult.has_value()) {
    if (failed(*intResult))
      return failure();
    staticAttr =
        parser.getBuilder().getI32IntegerAttr(static_cast<int32_t>(intValue));
    return success();
  }
  // Not a plain integer: parse an SSA operand (resolved to i32 by the caller).
  OpAsmParser::UnresolvedOperand op;
  if (parser.parseOperand(op))
    return failure();
  operand = op;
  return success();
}

static void printDynamicScalar(OpAsmPrinter &printer, Operation *,
                               Value operand, IntegerAttr staticAttr) {
  if (operand)
    printer << operand;
  else
    printer << staticAttr.getInt();
}

// Split a scalar OpFoldResult into an operand (runtime value) or an i32
// attribute (compile-time constant). A null OpFoldResult leaves both unset.
static void splitScalarOfr(mlir::OpBuilder &builder, mlir::OpFoldResult ofr,
                           mlir::Value &operand, mlir::IntegerAttr &attr) {
  operand = nullptr;
  attr = nullptr;
  if (!ofr)
    return;
  if (auto v = llvm::dyn_cast_if_present<mlir::Value>(ofr)) {
    operand = v;
    return;
  }
  auto intAttr =
      llvm::cast<mlir::IntegerAttr>(llvm::cast<mlir::Attribute>(ofr));
  attr = builder.getI32IntegerAttr(static_cast<int32_t>(intAttr.getInt()));
}

void DMABDOp::buildMixed(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::Value buffer, mlir::OpFoldResult offset,
                         mlir::OpFoldResult len,
                         llvm::ArrayRef<mlir::OpFoldResult> sizes,
                         llvm::ArrayRef<mlir::OpFoldResult> strides,
                         BDPadLayoutArrayAttr padDims, PacketInfoAttr packet) {
  // Split each mixed list into its dynamic operands + static array (with the
  // ShapedType::kDynamic sentinel for the runtime entries).
  llvm::SmallVector<int64_t> staticSizes, staticStrides;
  llvm::SmallVector<mlir::Value> dynSizes, dynStrides;
  mlir::dispatchIndexOpFoldResults(sizes, dynSizes, staticSizes);
  mlir::dispatchIndexOpFoldResults(strides, dynStrides, staticStrides);

  // Leave the static arrays unset when there is no ND layout so they elide
  // from the printed form.
  DenseI64ArrayAttr staticSizesAttr =
      staticSizes.empty() ? DenseI64ArrayAttr{}
                          : builder.getDenseI64ArrayAttr(staticSizes);
  DenseI64ArrayAttr staticStridesAttr =
      staticStrides.empty() ? DenseI64ArrayAttr{}
                            : builder.getDenseI64ArrayAttr(staticStrides);

  // A constant offset/len lands in the static_* attribute; a runtime value
  // becomes the operand (same operand-vs-attr split as sizes/strides).
  mlir::Value offsetVal, lenVal;
  mlir::IntegerAttr offsetAttr, lenAttr;
  splitScalarOfr(builder, offset, offsetVal, offsetAttr);
  splitScalarOfr(builder, len, lenVal, lenAttr);

  build(builder, state, buffer, /*offset=*/offsetVal, /*len=*/lenVal,
        /*static_offset=*/offsetAttr, /*static_len=*/lenAttr,
        /*sizes=*/dynSizes, /*strides=*/dynStrides,
        /*static_sizes=*/staticSizesAttr,
        /*static_strides=*/staticStridesAttr,
        /*pad_dimensions=*/padDims,
        /*bd_id_val=*/nullptr,
        /*bd_id=*/nullptr,
        /*packet=*/packet,
        /*out_of_order_id=*/nullptr,
        /*burst_length=*/nullptr,
        /*axcache=*/nullptr,
        /*iteration=*/nullptr,
        /*offset_parameter=*/nullptr,
        /*offset_state_table_idx=*/nullptr,
        /*next_bd_id=*/nullptr);
}

void DMABDOp::buildWithConstants(mlir::OpBuilder &builder,
                                 mlir::OperationState &state,
                                 mlir::Value buffer, int32_t offset,
                                 int32_t len, BDDimLayoutArrayAttr dims,
                                 BDPadLayoutArrayAttr padDims,
                                 PacketInfoAttr packet) {
  // Constant offset/len flow to the static_offset/static_len attributes (no
  // arith.constant materialized). i32 matches AIEI32Attr, keeping the cast
  // sign-exact for negative offsets.
  mlir::OpFoldResult offsetOfr = builder.getI32IntegerAttr(offset);
  mlir::OpFoldResult lenOfr = builder.getI32IntegerAttr(len);

  // Turn the outermost-first BDDimLayoutArrayAttr into all-constant
  // OpFoldResults for buildMixed.
  llvm::SmallVector<mlir::OpFoldResult> sizes, strides;
  if (dims) {
    for (BDDimLayoutAttr d : dims) {
      sizes.push_back(builder.getI64IntegerAttr(d.getSize()));
      strides.push_back(builder.getI64IntegerAttr(d.getStride()));
    }
  }
  buildMixed(builder, state, buffer, offsetOfr, lenOfr, sizes, strides, padDims,
             packet);
}

llvm::SmallVector<mlir::OpFoldResult> DMABDOp::getMixedSizes() {
  return ::mlir::getMixedValues(
      getStaticSizes().value_or(llvm::ArrayRef<int64_t>{}), getSizes(),
      getContext());
}

llvm::SmallVector<mlir::OpFoldResult> DMABDOp::getMixedStrides() {
  return ::mlir::getMixedValues(
      getStaticStrides().value_or(llvm::ArrayRef<int64_t>{}), getStrides(),
      getContext());
}

LogicalResult DMABDOp::verifyMixedSizesAndStrides() {
  llvm::ArrayRef<int64_t> staticSizes =
      getStaticSizes().value_or(llvm::ArrayRef<int64_t>{});
  llvm::ArrayRef<int64_t> staticStrides =
      getStaticStrides().value_or(llvm::ArrayRef<int64_t>{});
  if (failed(mlir::verifyListOfOperandsOrIntegers(
          *this, "sizes", staticSizes.size(), staticSizes, getSizes())))
    return failure();
  if (failed(mlir::verifyListOfOperandsOrIntegers(
          *this, "strides", staticStrides.size(), staticStrides, getStrides())))
    return failure();
  if (staticSizes.size() != staticStrides.size())
    return emitOpError("expected the same number of sizes (")
           << staticSizes.size() << ") and strides (" << staticStrides.size()
           << ")";
  return success();
}

std::optional<llvm::SmallVector<BDDimLayoutAttr>> DMABDOp::getFoldedDimensions(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
  llvm::SmallVector<mlir::OpFoldResult> sizes = getMixedSizes();
  llvm::SmallVector<mlir::OpFoldResult> strides = getMixedStrides();
  if (sizes.size() != strides.size()) {
    emitError() << "expected the same number of sizes (" << sizes.size()
                << ") and strides (" << strides.size() << ")";
    return std::nullopt;
  }
  llvm::SmallVector<BDDimLayoutAttr> dims;
  dims.reserve(sizes.size());
  for (auto [s, t] : llvm::zip(sizes, strides)) {
    std::optional<int64_t> sc = mlir::getConstantIntValue(s);
    std::optional<int64_t> tc = mlir::getConstantIntValue(t);
    if (!sc || !tc) {
      emitError() << "buffer descriptor size/stride is a runtime value; a "
                     "compile-time constant is required on this path";
      return std::nullopt;
    }
    dims.push_back(BDDimLayoutAttr::get(
        getContext(), static_cast<uint32_t>(*sc), static_cast<uint32_t>(*tc)));
  }
  return dims;
}

std::optional<llvm::SmallVector<BDDimLayoutAttr>>
DMABDOp::getConstantDimensions() {
  return getFoldedDimensions([&]() { return emitOpError(); });
}

// A BDDimLayoutAttr array (outermost-first) describes a contiguous row-major
// scan when the innermost stride is 1 and each outer stride equals the product
// of all inner sizes.  Used by both DMABDOp verification and canonicalization.
bool xilinx::AIE::isContiguousBDTransfer(llvm::ArrayRef<BDDimLayoutAttr> dims) {
  if (dims.empty())
    return true; // no ND layout = trivially contiguous
  // Innermost (last) stride must be 1.
  if (dims.back().getStride() != 1)
    return false;
  // Each outer stride must equal the product of all inner sizes.
  // Use uint64_t throughout to match the unsigned getSize()/getStride() types.
  uint64_t product = 1;
  for (int i = static_cast<int>(dims.size()) - 1; i >= 1; --i) {
    product *= dims[i].getSize();
    // A size-1 dimension's stride is irrelevant (it is never stepped).
    if (dims[i - 1].getSize() > 1 && dims[i - 1].getStride() != product)
      return false;
  }
  return true;
}

llvm::SmallVector<uint32_t> xilinx::AIE::getAssignedBdIds(DmaBody program) {
  llvm::SmallVector<uint32_t> ids;
  program.getDmaBody().walk([&](DMABDOp bd) {
    if (auto id = bd.getBdId())
      ids.push_back(*id);
  });
  return ids;
}

LogicalResult DMABDOp::verify() {
  // Skip verification of the BDOp outside of mem operations.
  // BDOps may appear elsewhere and subsequent lowerings will place them in the
  // correct mem ops.
  Operation *p = (*this)->getParentOp();
  if (!llvm::isa<MemOp, MemTileDMAOp, ShimDMAOp, DMAOp>(*p)) {
    return success();
  }

  // Check if buffer is an unranked memref (e.g., from function argument)
  bool isUnrankedMemRef = llvm::isa<UnrankedMemRefType>(getBuffer().getType());

  // For unranked memrefs, we can't verify as strictly since we don't know
  // the buffer's defining op or its full type at compile time
  if (!isUnrankedMemRef) {
    if (!isa<BufferOp, ExternalBufferOp>(getBuffer().getDefiningOp()))
      return emitOpError(
          "BDs only support BufferOp or ExternalBufferOp operands.");
  }

  if (getLenInBytes() % 4)
    return emitOpError("transfer length must be multiple of 4 (i.e., represent "
                       "4 byte aligned address)");

  TileElement parentTileElement = getParentTileElement(getOperation());
  TileLike parentTile = parentTileElement.getTileLike();
  if (!parentTile)
    return emitOpError("parent tile must implement TileLike interface");

  if (!isUnrankedMemRef && getOperation()->getParentOfType<MemOp>() &&
      getBufferOp().getTile() != parentTileElement.getTile())
    return emitOpError(
        "Core tile DMAs can only access a buffer in the same tile.");

  const AIETargetModel &targetModel = getTargetModel(getOperation());

  uint32_t maxBds = targetModel.getNumBDs(parentTile.getTileType());
  if (std::optional<int32_t> bdId = getBdId();
      bdId.has_value() && static_cast<uint32_t>(*bdId) >= maxBds)
    return emitOpError("bdId attribute exceeds max: ") << maxBds - 1;
  if (std::optional<int32_t> nextBdId = getNextBdId();
      nextBdId.has_value() && static_cast<uint32_t>(*nextBdId) >= maxBds)
    return emitOpError("nextBdId attribute exceeds max: ") << maxBds - 1;

  if (getBdIdVal() && getBdId().has_value())
    return emitOpError("bd_id and bd_id_val are mutually exclusive");

  // Issue #1097: the buffer_length field of a DMA buffer descriptor has a
  // tile-type-specific bit width (e.g. on AIE2: 32-bit shim, 17-bit mem tile,
  // 14-bit core tile). The hardware field counts address-generation granules
  // (32-bit words), so convert the byte length accordingly and reject
  // transfers that would overflow the field (they would otherwise be silently
  // truncated during lowering). Only checkable when the length is statically
  // known: either an explicit constant `len`, or (when no `len` is given) a
  // statically-shaped buffer whose full size is used.
  {
    bool lenStaticallyKnown = getConstantLen().has_value();
    if (!lenStaticallyKnown && !hasLen()) {
      if (auto shaped = llvm::dyn_cast<mlir::ShapedType>(getBuffer().getType()))
        lenStaticallyKnown = shaped.hasStaticShape();
    }
    // Skip validation for non-positive constant lengths: negative values
    // indicate a separate issue (e.g. distribute-link offset inference) and
    // would wrap to huge values via getLenInBytes().
    if (lenStaticallyKnown) {
      if (auto constLen = getConstantLen();
          constLen.has_value() && *constLen <= 0) {
        // non-positive explicit length — skip overflow check
      } else {
        uint64_t lenInBytes = getLenInBytes();
        uint32_t granularity = targetModel.getAddressGenGranularity();
        if (granularity != 0) {
          uint64_t lenInWords = lenInBytes * 8 / granularity;
          uint64_t maxLen =
              targetModel.getDmaBdMaxLen(parentTile.getTileType());
          if (lenInWords > maxLen)
            return emitOpError()
                   << "buffer descriptor length (" << lenInWords
                   << " 32-bit words) exceeds the maximum of " << maxLen
                   << " words supported by this tile type";
        }
      }
    }
  }

  if (failed(verifyMixedSizesAndStrides()))
    return failure();

  // Fold the mixed sizes/strides to a constant BDDimLayoutAttr list for
  // verification; a runtime size/stride yields nullopt plus a diagnostic.
  std::optional<llvm::SmallVector<BDDimLayoutAttr>> dims;
  if (!getMixedSizes().empty()) {
    dims = getConstantDimensions();
    if (!dims.has_value())
      return failure();
  }
  if (dims.has_value()) {
    // The per-BD ND access-pattern limit is a hardware property of the tile;
    // query it from the target model by tile type rather than inferring it
    // from the parent op type.
    size_t maxNDims = targetModel.getBDMaxDims(parentTile.getTileType());
    if (dims->size() > maxNDims)
      return emitOpError() << "Cannot give more than "
                           << std::to_string(maxNDims)
                           << " dimensions for step sizes and wraps on this "
                              "tile (got "
                           << std::to_string(dims->size()) << " dimensions).";

    auto buffer = llvm::dyn_cast<MemRefType>(getBuffer().getType());
    if (!buffer)
      return emitOpError() << "dimensions attribute cannot be used with "
                              "unranked memref buffer type.";
    int64_t maxIdx = getDimsMaxIdx(*dims);
    if (buffer.getNumElements() <= maxIdx)
      return emitOpError() << "Specified stride(s) and size(s) result in out "
                              "of bounds access in buffer, for index "
                           << std::to_string(maxIdx) << " in memref of length "
                           << std::to_string(buffer.getNumElements()) << ".";

    // A contiguous row-major access on a shim tile is lowered to linear mode
    // by aie-dma-tasks-to-npu / aie-dma-to-npu, using the wide buffer_length
    // register which is exempt from the 10-bit ND wrap-size limit.
    // Skip the per-dimension size check when the BD is on a shim tile and the
    // access is contiguous, so the natural ND form can be written without
    // triggering a spurious verifier error before lowering.
    //
    // Note: the verifier early-exit above means we only reach this code when
    // the parent op is MemOp, MemTileDMAOp, ShimDMAOp, or DMAOp -- all of
    // which are TileElements, so parentTile is always non-null here.
    bool skipSizeCheck =
        parentTile.isShimTile() && xilinx::AIE::isContiguousBDTransfer(*dims);

    // The wrap (size) and step (stride) field widths are tile-type specific
    // (e.g. on AIE2: core tiles have an 8-bit wrap, mem/shim tiles 10-bit).
    // Wrap fields are unbiased, so a W-bit wrap field admits sizes up to
    // 2^W - 1. Step fields are hardware-encoded as actual-1, so an S-bit
    // step field admits actual (unbiased, as declared here) strides up to
    // 2^S. Deriving the limits from the target model (instead of a single
    // hardcoded shim-sized constant) keeps this check correct for all tile
    // types, and computing the message from the same value it checks
    // against means the two can never disagree again.
    uint32_t wrapBits = targetModel.getDmaBdWrapBits(parentTile.getTileType());
    uint32_t stepBits = targetModel.getDmaBdStepBits(parentTile.getTileType());
    uint64_t maxSize = wrapBits > 0 ? (1ULL << wrapBits) - 1 : 0;
    uint64_t maxStride = stepBits > 0 ? (1ULL << stepBits) : 0;

    for (BDDimLayoutAttr dim : *dims) {
      if (0 == dim.getStride())
        return emitOpError()
               << "Invalid step size; must be a positive integer.";
      if (dim.getStride() > buffer.getNumElements())
        return emitOpError() << "Step size " << std::to_string(dim.getStride())
                             << " exceeds memref size "
                             << std::to_string(buffer.getNumElements());
      if (!skipSizeCheck && dim.getSize() > maxSize)
        return emitOpError() << "Size may not exceed " << maxSize << ".";
      if (dim.getStride() > maxStride)
        return emitOpError() << "Stride may not exceed " << maxStride << ".";
    }

    // Since streams read 32b words, there's no way to read eg 16b with stride
    // of 2 (ie lower halfs of each 32b). So force it to be 1 (and then in
    // CDODirect/XAIEV2 scale the size by 4/getBufferElementTypeWidthInBytes).
    if (getBufferElementTypeWidthInBytes() < 4 && dims->back().getStride() != 1)
      return emitOpError(
          "For <32b width datatypes, inner-most dim stride must be 1");

    if (getBufferElementTypeWidthInBytes() > 4 && dims->back().getStride() != 1)
      return emitOpError(
          "For >32b width datatypes, inner-most dim stride must be 1");

    // The hardware stepsize/wrap registers are denominated in 32-bit words.
    // lib/Targets/AIERT.cpp's static/CDO lowering converts element-
    // granularity strides/sizes to words by scaling by
    // elementWidthInBytes/4.0 and truncating via an unguarded static_cast;
    // when that scale factor isn't an integer, a declared value that isn't
    // itself a whole number of words is silently replaced by a different,
    // hardware-expressible value instead of being rejected. Reject those
    // values here instead, since they are genuinely inexpressible in the
    // hardware's word-granularity registers. The trigger condition is
    // "element width not a multiple of 4 bytes" rather than "< 4 bytes" so
    // that this also covers the `bfp` block-floating-point types, whose
    // widths (9 and 17 bytes) are >4 bytes but still not word multiples.
    // Dimensions with size == 1 are skipped: the loop runs once so stride is
    // never stepped and that dimension's size/stride do not affect hardware
    // address generation; word alignment is not required for them.
    //
    // Which dim is "innermost" is determined by the last dim with size > 1,
    // not by array position: `dims` is outermost-first, and trailing dims
    // with size == 1 (which never step) do not change which dim actually
    // performs the finest-granularity access. If every dim has size == 1,
    // there is no innermost dim to check and the whole block is skipped.
    int32_t elementWidthInBytes = getBufferElementTypeWidthInBytes();
    if (elementWidthInBytes % 4 != 0) {
      std::optional<size_t> effectiveInnermost;
      for (size_t i = 0; i < dims->size(); i++)
        if ((*dims)[i].getSize() > 1)
          effectiveInnermost = i;
      if (effectiveInnermost.has_value()) {
        for (size_t i = 0; i < dims->size(); i++) {
          BDDimLayoutAttr dim = (*dims)[i];
          if (dim.getSize() <= 1)
            continue;
          if (i == *effectiveInnermost) {
            // Effective innermost dim: the size (element count transferred
            // at this level) must itself be a whole number of 32-bit words.
            if ((dim.getSize() * elementWidthInBytes) % 4)
              return emitOpError()
                     << "Innermost dim size (" << dim.getSize() << ") * "
                     << elementWidthInBytes << "-byte element width = "
                     << (dim.getSize() * elementWidthInBytes)
                     << " bytes: innermost dim size must be a multiple of 4 "
                        "bytes for sub-32b element types (the hardware size "
                        "register is 32-bit-word granularity).";
          } else {
            // Dim outside the effective innermost dim: the stride (address
            // step between elements at this level) must itself be a whole
            // number of 32-bit words.
            if ((dim.getStride() * elementWidthInBytes) % 4)
              return emitOpError()
                     << "Dim " << i << " stride (" << dim.getStride() << ") * "
                     << elementWidthInBytes << "-byte element width = "
                     << (dim.getStride() * elementWidthInBytes)
                     << " bytes: non-innermost dim stride must be a multiple "
                        "of 4 bytes for sub-32b element types (the hardware "
                        "stepsize register is 32-bit-word granularity).";
          }
        }
      }
    }
  }
  if (auto paddims = getPadDimensions(); paddims.has_value()) {
    if (!dims.has_value())
      return emitOpError() << "Padding requires n-d data layouts expressed as"
                           << " wrap(s) and stride(s).";
    if (!parentTile.isMemTile())
      return emitOpError() << "Padding is only supported by memtile dma bds.";
    if (dims->size() != paddims->size())
      return emitOpError() << "Mismatch number of dimensions between padding(s)"
                           << " and wrap(s) and stride(s).";
    int actuallen = 1;
    for (unsigned i = 0; i < paddims->size(); i++) {
      auto dim = (*dims)[i];
      auto paddim = (*paddims)[i];
      actuallen *= paddim.getConstPadBefore() + paddim.getConstPadAfter() +
                   dim.getSize();
      if (std::optional<int32_t> len = getConstantLen(); len.has_value()) {
        if (actuallen > *len)
          return emitOpError() << "Data exceeds len after padding.";
      } else if (getLen()) {
        return emitOpError()
               << "Padding with a runtime len operand is not yet supported; "
                  "use a compile-time constant len with padded BDs.";
      }
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
  if (!isUnrankedMemRef &&
      (parentTile.isMemTile() || parentTile.isCoreTile())) {
    if (auto baseAddr = getBufferOp().getAddress(); baseAddr.has_value()) {
      int64_t offsetInBytes = *baseAddr + getOffsetInBytes();
      if (offsetInBytes % 4)
        return emitOpError("bd address must be 4 byte (32b) aligned; got "
                           "base+offset: ")
               << offsetInBytes << " (bytes)";
    }
  }
  if (auto packetInfo = getPacket()) {
    if (packetInfo->getPktType() > 7)
      return emitOpError("Packet type field can only hold 3 bits.");
    if (packetInfo->getPktId() >
        getTargetModel(getOperation()).getMaxPacketId())
      return emitOpError("Packet ID field can only hold 5 bits.");
  }

  if (failed(verifyDMABDOutOfOrderId(*this)))
    return failure();

  // A runtime len operand or the static_len attribute both count as having a
  // length here.
  if (!hasLen() && !getBuffer().getType().hasStaticShape())
    return emitOpError() << "buffer with dynamic shape requires static length.";

  if (getBurstLength() != 0 && !parentTile.isShimNOCTile())
    return emitOpError("Burst length is only supported in Shim NOC tiles that "
                       "are connected to the memory-mapped NOC.");

  if (auto axcache = getAxcache()) {
    if (!parentTile.isShimNOCTile()) {
      return emitOpError("AxCACHE is only supported in Shim NOC tiles "
                         "that are connected to the memory-mapped NOC.");
    }
    if (axcache > 0xF) {
      return emitOpError("AxCache value out of 4-bit range.");
    }
  }

  // BD iteration bounds. Values are true/element (aie-rt encodes value-1);
  // size <= 1 disables iteration (stride ignored). The stride is checked in
  // whole 32-bit words against the tile-specific step field; the wrap is a
  // 6-bit field everywhere. aiex.npu.writebd checks the same tile-correct step
  // limit (getDmaBdStepBits) inline in its own raw-register terms.
  if (auto iter = getIteration()) {
    if (!targetModel.hasProperty(AIETargetModel::UsesBDIteration))
      return emitOpError("BD iteration is not supported on this target");
    uint32_t size = iter->getSize(), current = iter->getCurrent();
    if (size < 1 || size > 64) // 64 = aie-rt IterWrapMax + 1
      return emitOpError("BD iteration size must be in [1, 64]");
    if (size > 1) {
      int64_t strideInBytes = static_cast<int64_t>(iter->getStride()) *
                              getBufferElementTypeWidthInBytes();
      if (strideInBytes % 4)
        return emitOpError(
            "BD iteration stride must be aligned to 32-bit words");
      int64_t stepInWords = strideInBytes / 4;
      int64_t maxStep =
          1LL << targetModel.getDmaBdStepBits(parentTile.getTileType());
      if (stepInWords < 1 || stepInWords > maxStep)
        return emitOpError() << "BD iteration stride must be in [1, " << maxStep
                             << "] 32-bit words";
    }
    if (current >= size)
      return emitOpError("BD iteration current must be in [0, size)");
  }

  return success();
}

uint32_t DMABDOp::getAxcacheOrDefault() {
  return getAxcache().value_or(
      getTargetModel(getOperation()).getDefaultAxCache());
}

//===----------------------------------------------------------------------===//
// DMAStartOp
//===----------------------------------------------------------------------===//

static LogicalResult FoldDMAStartOp(DMAStartOp op, PatternRewriter &rewriter) {

  llvm::SetVector<Block *> reachable;
  SmallVector<Block *, 16> worklist;
  Block *firstBD = op.getSuccessor(0);
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

  // BD chain ends with an EndOp, indicating non-repeating pattern: BD chain
  // folding not applicable.
  if (isa<EndOp>((reachable.back())->getTerminator()))
    return failure();

  // Check for identical bds.
  auto areEquivalentBDs = [](Block *b1, Block *b2) {
    auto b1OpRange = b1->without_terminator();
    auto b2OpRange = b2->without_terminator();
    if (llvm::range_size(b1OpRange) != llvm::range_size(b2OpRange))
      return false;
    if (!llvm::all_of(llvm::zip_equal(b1OpRange, b2OpRange),
                      [](std::tuple<Operation &, Operation &> pair) {
                        return OperationEquivalence::isEquivalentTo(
                            &std::get<0>(pair), &std::get<1>(pair),
                            OperationEquivalence::IgnoreLocations);
                      }))
      return false;
    return true;
  };

  // Get a vector of unique BDs.
  SmallVector<Block *> uniquePattern;
  const auto *patternIt = reachable.begin();
  while (patternIt != reachable.end() &&
         llvm::none_of(uniquePattern, [patternIt, areEquivalentBDs](Block *b1) {
           return areEquivalentBDs(*patternIt, b1);
         })) {
    uniquePattern.push_back(*patternIt);
    patternIt++;
  }

  unsigned idx = 0;
  while (patternIt != reachable.end()) {
    // BD repetition found. Check if repeating pattern.
    if (!areEquivalentBDs(*patternIt, uniquePattern[idx]))
      return failure();
    patternIt++;
    idx = (++idx) % uniquePattern.size();
  }

  // Repeating BD chains detected. Erasing repetitions.
  auto lastBDTerm = cast<NextBDOp>(reachable.back()->getTerminator());
  auto lastUniqueBDTerm = cast<NextBDOp>(uniquePattern.back()->getTerminator());
  lastUniqueBDTerm.setSuccessor(lastBDTerm.getSuccessor());

  return success();
}

// Canonicalization pattern for DMABDOp: on shim tiles, fold a contiguous
// row-major ND access pattern into canonical linear form (no dimensions
// attribute), so that the hardware uses the wide buffer_length register.
namespace {
struct LinearizeContiguousBDTransfer : public mlir::OpRewritePattern<DMABDOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(DMABDOp op, mlir::PatternRewriter &rewriter) const override {
    // Only fire for shim DMA BDs: ExternalBufferOp buffer
    // (dma_configure_task_for stage), parent TileElement is shim
    // (dma_configure_task stage), or inside ShimDMAOp (after full lowering).
    bool bufferIsExternal =
        isa_and_nonnull<ExternalBufferOp>(op.getBuffer().getDefiningOp());
    TileElement parentElem = getParentTileElement(op.getOperation());
    TileLike parentTileLike =
        parentElem ? parentElem.getTileLike() : TileLike{};
    bool parentIsShim = parentTileLike && parentTileLike.isShimTile();
    bool inShimDMA = (bool)op->getParentOfType<ShimDMAOp>();
    if (!bufferIsExternal && !parentIsShim && !inShimDMA)
      return mlir::failure();

    // Only ND dimensions that are present and all-constant can be linearized;
    // decline silently on runtime-valued sizes/strides so valid dynamic IR
    // doesn't get a spurious diagnostic.
    if (op.getMixedSizes().empty())
      return mlir::failure();
    for (mlir::OpFoldResult s : op.getMixedSizes())
      if (!mlir::getConstantIntValue(s))
        return mlir::failure();
    for (mlir::OpFoldResult s : op.getMixedStrides())
      if (!mlir::getConstantIntValue(s))
        return mlir::failure();
    std::optional<llvm::SmallVector<BDDimLayoutAttr>> dims =
        op.getFoldedDimensions([&]() { return op.emitError(); });
    if (!dims.has_value() || dims->empty())
      return mlir::failure();
    if (!xilinx::AIE::isContiguousBDTransfer(*dims))
      return mlir::failure();
    // Already linear (single dimension with stride 1)?
    if (dims->size() == 1 && dims->front().getStride() == 1)
      return mlir::failure();

    int64_t product = 1;
    for (BDDimLayoutAttr dim : *dims)
      product *= dim.getSize();

    // len < product: the outermost dim describes a hardware BD iteration
    // (preserved downstream as iteration_size/stride). Don't fold it away.
    // A runtime len can't be compared, so bail. Also bail if len is absent but
    // the buffer is unranked (can't safely infer product == full transfer).
    if (!op.hasLen() && !op.getBuffer().getType().hasStaticShape())
      return mlir::failure();
    std::optional<int32_t> lenVal = op.getConstantLen();
    if (lenVal.has_value() && static_cast<int64_t>(*lenVal) != product)
      return mlir::failure();
    int32_t len = static_cast<int32_t>(product);

    // Rewrite to a linear BD: len as the static_len attribute, sizes/strides
    // cleared. Other attributes are preserved.
    rewriter.modifyOpInPlace(op, [&]() {
      op.getLenMutable().clear();
      op.setStaticLen(len);
      op.getSizesMutable().clear();
      op.getStridesMutable().clear();
      op.setStaticSizes(std::nullopt);
      op.setStaticStrides(std::nullopt);
    });
    return mlir::success();
  }
};
} // namespace

// Canonicalization pattern for DMABDOp: fold a constant offset/len/size/stride
// operand back into the corresponding static_* attribute, removing the operand.
namespace {
struct FoldConstantBDDimList : public mlir::OpRewritePattern<DMABDOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(DMABDOp op, mlir::PatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::OpFoldResult> sizes = op.getMixedSizes();
    llvm::SmallVector<mlir::OpFoldResult> strides = op.getMixedStrides();
    // foldDynamicIndexList returns success() only if it replaced a dynamic
    // entry with a constant. Bitwise (not logical) OR is deliberate: both
    // calls must run unconditionally, so strides still folds even once
    // sizes already succeeded.
    // NOLINTNEXTLINE(clang-diagnostic-bitwise-instead-of-logical)
    bool changed = succeeded(mlir::foldDynamicIndexList(sizes)) |
                   succeeded(mlir::foldDynamicIndexList(strides));

    // Fold a constant offset/len operand into its static_* attribute.
    std::optional<int32_t> foldOffset, foldLen;
    if (op.getOffset())
      if (auto c = mlir::getConstantIntValue(op.getOffset()))
        foldOffset = static_cast<int32_t>(*c);
    if (op.getLen())
      if (auto c = mlir::getConstantIntValue(op.getLen()))
        foldLen = static_cast<int32_t>(*c);
    changed |= foldOffset.has_value() || foldLen.has_value();

    if (!changed)
      return mlir::failure();

    llvm::SmallVector<int64_t> staticSizes, staticStrides;
    llvm::SmallVector<mlir::Value> dynSizes, dynStrides;
    mlir::dispatchIndexOpFoldResults(sizes, dynSizes, staticSizes);
    mlir::dispatchIndexOpFoldResults(strides, dynStrides, staticStrides);

    rewriter.modifyOpInPlace(op, [&]() {
      op.getSizesMutable().assign(dynSizes);
      op.getStridesMutable().assign(dynStrides);
      // Leave the static arrays unset when there is no ND layout so they elide.
      if (staticSizes.empty())
        op.removeStaticSizesAttr();
      else
        op.setStaticSizes(staticSizes);
      if (staticStrides.empty())
        op.removeStaticStridesAttr();
      else
        op.setStaticStrides(staticStrides);
      if (foldOffset) {
        op.getOffsetMutable().clear();
        op.setStaticOffset(foldOffset);
      }
      if (foldLen) {
        op.getLenMutable().clear();
        op.setStaticLen(foldLen);
      }
    });
    return mlir::success();
  }
};
} // namespace

void DMABDOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<FoldConstantBDDimList, LinearizeContiguousBDTransfer>(context);
}

void DMAStartOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add(FoldDMAStartOp);
}

LogicalResult DMAStartOp::verify() {
  if (getPadValue() != 0) {
    if (!isa<MemTileDMAOp>(getOperation()->getParentOp()))
      return emitOpError("pad_value is only supported on memtile DMA channels");
    if (getChannelDir() != DMAChannelDir::MM2S)
      return emitOpError("pad_value is only supported on MM2S DMA channels");
    if (!getTargetModel(getOperation()).isMemTilePadValueSupported())
      return emitOpError("pad_value requires the CONSTANT_PAD_VALUE register, "
                         "unavailable on this target");
  }
  if (failed(verifyDMARepeatCount(getOperation(), getRepeatCount())))
    return failure();
  SmallVector<DMABDOp> bds;
  if (getOutOfOrder()) {
    llvm::SmallPtrSet<Block *, 8> visited;
    for (Block *b = getDest(); b && visited.insert(b).second;
         b = b->getNumSuccessors() > 0 ? b->getSuccessor(0) : nullptr)
      llvm::append_range(bds, b->getOps<DMABDOp>());
  }
  return verifyOutOfOrderChannel(getOperation(), getChannelDir(),
                                 getOutOfOrder(), bds);
}

//===----------------------------------------------------------------------===//
// SwitchboxOp
//===----------------------------------------------------------------------===//

LogicalResult SwitchboxOp::verify() {
  // The remaining checks (port bounds, legal-connection rules) are all keyed
  // off the target model for a placed tile. Before --aie-place-tiles runs the
  // tile is still an aie.logical_tile, so defer those checks to the
  // post-placement re-verify -- mirroring how UsesAreAccessible and FlowOp
  // skip target-model checks on logical tiles.
  if (!isa<TileOp>(getTile().getDefiningOp()))
    return success();

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
        auto amsel = cast<AMSelOp>(val.getDefiningOp());
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
          auto pktRules = cast<PacketRulesOp>(s->getParentOp());
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

TileOp LockOp::getTileOp() {
  return cast<TileElement>(this->getOperation()).getTileOp();
}

LogicalResult LockOp::verify() {
  if (auto result = UsesAreAccessible::verifyTrait(*this); result.failed())
    return result;

  if (auto lockID = getLockID()) {
    TileLike tileLike = getTileLike();
    if (!tileLike)
      return emitOpError("tile operand must implement TileLike interface");
    const auto &targetModel = getTargetModel(*this);
    auto tileType = tileLike.getTileType();
    if (int numLocks = targetModel.getNumLocks(tileType); *lockID >= numLocks)
      return emitOpError("lock assigned invalid id (maximum is ")
             << numLocks - 1 << ")";
  }

  return success();
}

// Look up for compile-time constant lock values, if any.
// Returns std::nullopt if lock value does not reference an `arith.constant`.
static std::optional<int32_t> getConstantLockValue(UseLockOp op) {
  if (auto constant = op.getValue().getDefiningOp<arith::ConstantOp>())
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(constant.getValue()))
      return (int32_t)intAttr.getInt();
  return std::nullopt;
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
      // Non-constant lock values cannot be compared here; the passes that
      // require a constant enforce that separately via getConstantValue().
      auto value = getConstantLockValue(op);
      if (!value)
        continue;
      if (op.acquire() || op.acquireGE()) {
        if (acqValue != -1 && acqValue != *value) {
          return failure();
        }
        acqValue = *value;
      } else if (op.release()) {
        if (relValue != -1 && relValue != *value) {
          return failure();
        }
        relValue = *value;
      }
    }
    return success();
  }
};

struct AccessesLocalLocks {
  static LogicalResult verifyTrait(Operation *op) {
    if (auto memOp = op->getParentOfType<MemOp>()) {
      auto useLock = cast<UseLockOp>(op);
      if (auto lock = useLock.getLockOp(); lock.getTile() != memOp.getTile())
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

  // Locks used inside a DMA/BD block are configured via static register writes
  // and therefore require a compile-time constant value.
  if (HasSomeParent<MemOp, MemTileDMAOp, ShimDMAOp>::verifyTrait(*this)
          .succeeded() &&
      !getConstantLockValue(*this))
    return (*this)->emitOpError(
        "lock value in a DMA/BD block must be a compile-time constant "
        "(defined by an arith.constant).");

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
  // Or it can be in a DMAConfigureTaskOp (for runtime DMA configuration)
  // Check by operation name to avoid circular dependency with AIEX dialect
  {
    Operation *operation = (*this)->getParentOp();
    while (operation) {
      if (operation->getName().getStringRef() == "aiex.dma_configure_task")
        return success();
      operation = operation->getParentOp();
    }
  }
  return (*this)->emitOpError()
         << "expects some parent op to be one of "
         << "AIE::device, AIE::core, func::func, AIE::mem, AIE::shimDMA, or "
            "AIEX::dma_configure_task";
}

#include "aie/Dialect/AIE/IR/AIEEnums.cpp.inc"
#include "aie/Dialect/AIE/IR/AIEInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// TraceEventAttr
//===----------------------------------------------------------------------===//

// Custom parser for TraceEventAttr value (uses shared helper)
static ParseResult parseTraceEventValue(AsmParser &parser, Attribute &value) {
  return xilinx::AIE::parseTraceEvent(parser, value);
}

// Custom printer for TraceEventAttr value (uses shared helper)
static void printTraceEventValue(AsmPrinter &printer, Attribute value) {
  xilinx::AIE::printTraceEventEnum(printer, value);
}

// Custom parser for TraceEventAttr value (uses shared helper)
static ParseResult parseTraceRegValue(OpAsmParser &parser, Attribute &value) {

  // Try to parse as number
  int64_t intValue;
  OptionalParseResult parseResult = parser.parseOptionalInteger(intValue);
  if (parseResult.has_value() && succeeded(parseResult.value())) {
    MLIRContext *ctx = parser.getContext();
    value = IntegerAttr::get(IntegerType::get(ctx, 32), intValue);
    return success();
  }
  return xilinx::AIE::parseTraceEvent(parser, value);
}

// Custom printer for TraceEventAttr value (uses shared helper)
static void printTraceRegValue(OpAsmPrinter &printer, Operation *op,
                               Attribute value) {
  // if it's an intattr
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(value)) {
    printer << intAttr.getInt();
    return;
  }
  xilinx::AIE::printTraceEventEnum(printer, value);
}

// Helper to parse a LockBlocking enum keyword.
#define GET_OP_CLASSES
#include "aie/Dialect/AIE/IR/AIEOps.cpp.inc"

FailureOr<int32_t> UseLockOp::getConstantValue() {
  if (auto value = getConstantLockValue(*this))
    return *value;
  return emitOpError("expected the lock value to be a compile-time constant "
                     "(defined by an arith.constant).");
}

LogicalResult UseLockOp::canonicalize(UseLockOp op, PatternRewriter &rewriter) {
  // An AcquireGreaterEqual by a compile-time-constant 0 is a no-op: the lock is
  // already >= 0 (semaphore values are non-negative) and it decrements the lock
  // by 0.
  if (op.acquireGE()) {
    if (auto value = getConstantLockValue(op); value && *value == 0) {
      rewriter.eraseOp(op);
      return success();
    }
  }
  return failure();
}

TileOp SwitchboxOp::getTileOp() {
  return cast<TileElement>(this->getOperation()).getTileOp();
}

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

TileOp ShimMuxOp::getTileOp() {
  return cast<TileElement>(this->getOperation()).getTileOp();
}

WireBundle xilinx::AIE::getConnectingBundle(WireBundle dir) {
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
  for (auto *it = argsIter.begin(); it != argsIter.end(); ++it) {
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

LogicalResult ShimDMAAllocationOp::verify() {
  TileLike tileLike = llvm::dyn_cast<TileLike>(getTile().getDefiningOp());
  if (!tileLike) {
    return emitOpError("tile operand must implement TileLike interface");
  }

  if (!tileLike.isShimNOCorPLTile()) {
    // if placed, provide detailed error message
    auto col = tileLike.tryGetCol();
    auto row = tileLike.tryGetRow();
    if (col && row) {
      return emitOpError("tile must be a shim tile, but got tile(")
             << *col << ", " << *row << ")";
    }
    return emitOpError("tile must be a shim tile");
  }

  return success();
}

TileOp ShimDMAAllocationOp::getTileOp() {
  return cast<TileOp>(getTile().getDefiningOp());
}

ShimDMAAllocationOp ShimDMAAllocationOp::getForSymbol(DeviceOp device,
                                                      llvm::StringRef symbol) {
  Operation *maybeOp = device.lookupSymbol(symbol);
  if (maybeOp) {
    if (ShimDMAAllocationOp op = dyn_cast<ShimDMAAllocationOp>(maybeOp)) {
      return op;
    }
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ObjectFifoRearmBindingOp
//===----------------------------------------------------------------------===//

LogicalResult ObjectFifoRearmBindingOp::verify() {
  if (getChannelDirs().size() != getChannelTiles().size())
    return emitOpError("expected one channel_dirs entry per channel tile (")
           << getChannelTiles().size() << " tiles, " << getChannelDirs().size()
           << " dirs)";
  if (getChannelIndices().size() != getChannelTiles().size())
    return emitOpError("expected one channel_indices entry per channel tile (")
           << getChannelTiles().size() << " tiles, "
           << getChannelIndices().size() << " indices)";
  if (getLockInits().size() != getLocks().size())
    return emitOpError("expected one lock_inits entry per lock (")
           << getLocks().size() << " locks, " << getLockInits().size()
           << " inits)";
  // head_bd_ids / repeat_counts are optional (populated by
  // --aie-assign-bd-ids), but if present they are one-per-channel and travel as
  // a pair: the re-push needs both, so a binding carrying one without the other
  // is malformed.
  if (getHeadBdIds().has_value() != getRepeatCounts().has_value())
    return emitOpError("head_bd_ids and repeat_counts must both be set or both "
                       "be absent");
  if (auto headBdIds = getHeadBdIds();
      headBdIds.has_value() && headBdIds->size() != getChannelTiles().size())
    return emitOpError("expected one head_bd_ids entry per channel tile (")
           << getChannelTiles().size() << " tiles, " << headBdIds->size()
           << " ids)";
  if (auto repeatCounts = getRepeatCounts();
      repeatCounts.has_value() &&
      repeatCounts->size() != getChannelTiles().size())
    return emitOpError("expected one repeat_counts entry per channel tile (")
           << getChannelTiles().size() << " tiles, " << repeatCounts->size()
           << " counts)";
  for (int32_t dir : getChannelDirs())
    if (dir != 0 && dir != 1)
      return emitOpError("channel_dirs entries must be 0 (S2MM) or 1 (MM2S), "
                         "got ")
             << dir;
  // The lowering resolves each operand to its aie.tile / aie.lock, so reject a
  // binding whose operands are not those (otherwise the lowering would cast a
  // non-lock/non-tile operand and abort).
  for (Value tile : getChannelTiles()) {
    auto tileOp = tile.getDefiningOp<TileOp>();
    if (!tileOp)
      return emitOpError("channel_tiles operands must be aie.tile values");
    // A shim DMA endpoint is host-managed: the host re-pushes its BD program
    // and re-arms its locks. Re-arming it here would fight the host, so a
    // re-arm binding only ever records non-shim channels -- reject a shim tile
    // even in a hand-authored binding.
    if (tileOp.isShimTile())
      return emitOpError("channel_tiles must be non-shim tiles; a shim DMA "
                         "endpoint is host-managed and is not re-armed here");
  }
  for (Value lock : getLocks())
    if (!lock.getDefiningOp<LockOp>())
      return emitOpError("locks operands must be aie.lock values");
  return success();
}

//===----------------------------------------------------------------------===//
// RuntimeSequenceOp
//===----------------------------------------------------------------------===//

ParseResult RuntimeSequenceOp::parse(OpAsmParser &parser,
                                     OperationState &result) {

  // Name of this runtime sequence
  StringAttr nameAttr;
  (void)parser.parseOptionalSymbolName(
      nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes);

  SmallVector<OpAsmParser::Argument> entryArgs;

  // Entry arguments,  e.g. (%addr: memref<1xi32>)
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

  // Optional `attributes { ... }` clause (before the body so it does not
  // conflict with the `{` that opens the region).
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Body
  auto *body = result.addRegion();
  ParseResult bodyParseResult = parser.parseRegion(*body, entryArgs, false);
  if (bodyParseResult) {
    return bodyParseResult;
  }

  return success();
}

void RuntimeSequenceOp::print(OpAsmPrinter &printer) {
  Region &body = getRegion();

  auto nameAttr = (*this)->getAttrOfType<StringAttr>(
      mlir::SymbolTable::getSymbolAttrName());
  if (nameAttr &&
      nameAttr != ::mlir::OpBuilder((*this)->getContext())
                      .getStringAttr(getDefaultRuntimeSequenceName())) {
    printer << ' ';
    printer.printSymbolName(nameAttr);
  }

  printer << '(';
  for (unsigned i = 0, n = body.getNumArguments(); i < n; i++) {
    if (i > 0) {
      printer << ", ";
    }
    printer.printRegionArgument(body.getArgument(i));
  }
  printer << ')';

  printer.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{mlir::SymbolTable::getSymbolAttrName()});

  printer << ' ';
  printer.printRegion(body, false, true);
}

LogicalResult RuntimeSequenceOp::verify() {
  DeviceOp device = (*this)->getParentOfType<DeviceOp>();
  if (!device) {
    // this check is redudnant with the HasParent trait, but can't hurt
    (*this)->emitOpError() << "must be inside AIE device operation.";
    return failure();
  }
  return success();
}

RuntimeSequenceOp
RuntimeSequenceOp::getForSymbolInDevice(DeviceOp deviceOp,
                                        llvm::StringRef symbol) {
  RuntimeSequenceOp runtimeSequenceOp;
  if (symbol.empty()) {
    auto range = deviceOp.getOps<RuntimeSequenceOp>();
    if (range.begin() == range.end()) {
      // No runtime sequence in the device; let the caller emit a diagnostic
      // rather than dereferencing an end iterator (which crashes).
      return nullptr;
    }
    runtimeSequenceOp = *range.begin();
  } else {
    Operation *maybeRuntimeSequenceOp =
        mlir::SymbolTable::lookupSymbolIn(deviceOp, symbol);
    if (!maybeRuntimeSequenceOp) {
      return nullptr;
    }
    runtimeSequenceOp =
        llvm::dyn_cast<RuntimeSequenceOp>(maybeRuntimeSequenceOp);
  }
  return runtimeSequenceOp;
}

RuntimeSequenceOp
RuntimeSequenceOp::getForSymbolInDeviceOrError(DeviceOp deviceOp,
                                               llvm::StringRef symbol) {
  RuntimeSequenceOp runtimeSequenceOp = getForSymbolInDevice(deviceOp, symbol);
  if (!runtimeSequenceOp) {
    if (!symbol.empty()) {
      deviceOp.emitError("No such runtime sequence: ") << symbol;
    } else {
      deviceOp.emitError("No runtime sequence in device");
    }
  }
  return runtimeSequenceOp;
}

LogicalResult RuntimeSequenceOp::verifyBeforeMaterialization() {
  // Check that all symbol references within the runtime sequence
  // are either to ShimDMAAllocationOp, DeviceOp or another RuntimeSequenceOp;
  // these are the only symbols that can be lowered with the NPU passes
  auto result = (*this)->walk([&](Operation *op) {
    for (NamedAttribute namedAttr : op->getAttrs()) {
      Attribute attr = namedAttr.getValue();
      auto walkResult = attr.walk([&](SymbolRefAttr symbolRef) {
        Operation *symbolDefOp =
            SymbolTable::lookupNearestSymbolFrom(*this, symbolRef);
        if (symbolDefOp) {
          if (!llvm::isa<ShimDMAAllocationOp>(symbolDefOp) &&
              !llvm::isa<DeviceOp>(symbolDefOp) &&
              !llvm::isa<RuntimeSequenceOp>(symbolDefOp) &&
              !llvm::isa<BufferOp>(symbolDefOp) &&
              !llvm::isa<ObjectFifoRearmBindingOp>(symbolDefOp) &&
              !llvm::isa<memref::GlobalOp>(symbolDefOp)) {
            op->emitOpError()
                << "references symbol '"
                << symbolRef.getRootReference().getValue()
                << "' which must be either a ShimDMAAllocationOp, DeviceOp, "
                   "RuntimeSequenceOp, BufferOp, ObjectFifoRearmBindingOp, or "
                   "GlobalOp, but got: "
                << symbolDefOp->getName().getStringRef();
            return WalkResult::interrupt();
          }
          if (BufferOp bufferOp = llvm::dyn_cast<BufferOp>(symbolDefOp)) {
            if (!bufferOp.getAddress()) {
              op->emitOpError()
                  << "Unallocated buffer; fixed addresses are required before "
                     "runtime sequence materialization.";
              return WalkResult::interrupt();
            }
          }
        }
        return WalkResult::advance();
      });
      if (walkResult.wasInterrupted()) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    return failure();
  }

  return success();
}

// Include implementations for custom attributes
#define GET_ATTRDEF_CLASSES
#include "aie/Dialect/AIE/IR/AIEAttrs.cpp.inc"
