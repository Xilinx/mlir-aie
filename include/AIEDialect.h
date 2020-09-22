// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
//===- AIEDialect.h - Dialect definition for the AIE IR ----------------===//
//
// Copyright 2019 Xilinx
//
//===---------------------------------------------------------------------===//

#ifndef MLIR_AIE_DIALECT_H
#define MLIR_AIE_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <map>

using namespace mlir;

#include "AIEEnums.h.inc"

namespace xilinx {
namespace AIE {

void registerAIETranslations();
void registerAIEFindFlowsPass();
void registerAIECreateFlowsPass();
void registerAIECreateCoresPass();
void registerAIECreateLocksPass();
void registerAIEBufferMergePass();
void registerAIEHerdRoutingPass();
void registerAIECreatePacketFlowsPass();


// FIXME: use this
//#include "AIEDialect.h.inc"

// The Dialect
class AIEDialect : public mlir::Dialect {
public:
  explicit AIEDialect(mlir::MLIRContext *ctx);
  static StringRef getDialectNamespace() { return "AIE"; }


  // /// Parse a type registered to this dialect. Overridding this method is
  // /// required for dialects that have custom types.
  // /// Technically this is only needed to be able to round-trip to textual IR.
  // mlir::Type parseType(DialectAsmParser &parser) const override;

  // /// Print a type registered to this dialect. Overridding this method is
  // /// only required for dialects that have custom types.
  // /// Technically this is only needed to be able to round-trip to textual IR.
  // void printType(mlir::Type type, DialectAsmPrinter &os) const override;
};

////////////////////////////////////////////////////////////////////////////////
/////////////////////// Custom Types for the Dialect ///////////////////////////
////////////////////////////////////////////////////////////////////////////////

// namespace detail {
// struct AIEListTypeStorage;
// }

// /// LLVM-style RTTI: one entry per subclass to allow dyn_cast/isa.
// enum AIETypeKind {
//   // The enum starts at the range reserved for this dialect.
//   AIE_TYPE = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
//   AIE_LIST,
// };

// /// Type for Toy arrays.
// /// In MLIR Types are reference to immutable and uniqued objects owned by the
// /// MLIRContext. As such `AIEListType` only wraps a pointer to an uniqued
// /// instance of `AIEListTypeStorage` (defined in our implementation file) and
// /// provides the public facade API to interact with the type.
// class AIEListType : public mlir::Type::TypeBase<AIEListType, mlir::Type,
//                                                  detail::AIEListTypeStorage> {
// public:
//   using Base::Base;

//   /// Return the type of individual elements in the array.
//   mlir::Type getElementType();

//   /// Get the unique instance of this Type from the context.
//   static AIEListType get(mlir::Type elementType);

//   /// Support method to enable LLVM-style RTTI type casting.
//   static bool kindof(unsigned kind) { return kind == AIETypeKind::AIE_LIST; }
// };


////////////////////////////////////////////////////////////////////////////////
//////////////////// Custom Operations for the Dialect /////////////////////////
////////////////////////////////////////////////////////////////////////////////

//#include "AIEOpInterfaces.h.inc"

typedef std::pair<WireBundle, int> Port;
typedef std::pair<Port, Port> Connect;
typedef std::pair<int, int> TileID;

static bool isValidTile(TileID src) {
  // FIXME: what about upper bound?
  return src.first >= 0 && src.second >= 0;
}
// Return the tile ID of the memory to the west of the given tile, if it exists.
static Optional<TileID> getMemWest(TileID src) {
  bool isEvenRow = ((src.first % 2) == 0);
  Optional<TileID> ret;
  if (isEvenRow)
    ret = src;
  else
    ret = std::make_pair(src.first - 1, src.second);
  if(!isValidTile(ret.getValue())) ret.reset();
  return ret;
}
// Return the tile ID of the memory to the west of the given tile, if it exists.
static Optional<TileID> getMemEast(TileID src) {
  bool isEvenRow = ((src.first % 2) == 0);
  Optional<TileID> ret;
  if (isEvenRow)
    ret = std::make_pair(src.first + 1, src.second);
  else
    ret = src;
  if(!isValidTile(ret.getValue())) ret.reset();
  return ret;
}
// Return the tile ID of the memory to the west of the given tile, if it exists.
static Optional<TileID> getMemNorth(TileID src) {
  Optional<TileID> ret = std::make_pair(src.first, src.second + 1);
  if(!isValidTile(ret.getValue())) ret.reset();
  return ret;
}
static Optional<TileID> getMemSouth(TileID src) {
  Optional<TileID> ret = std::make_pair(src.first, src.second - 1);
  if(!isValidTile(ret.getValue())) ret.reset();
  return ret;
}

static bool isInternal(int srcCol, int srcRow, int dstCol, int dstRow) {
  return ((srcCol == dstCol) && (srcRow == dstRow));
}

static bool isWest(int srcCol, int srcRow, int dstCol, int dstRow) {
  return ((srcCol == dstCol + 1) && (srcRow == dstRow));
}

static bool isMemWest(int srcCol, int srcRow, int dstCol, int dstRow) {
  bool IsEvenRow = ((srcRow % 2) == 0);
  return (IsEvenRow  && isInternal(srcCol, srcRow, dstCol, dstRow)) ||
         (!IsEvenRow && isWest(srcCol, srcRow, dstCol, dstRow));
}

static bool isEast(int srcCol, int srcRow, int dstCol, int dstRow) {
  return ((srcCol == dstCol - 1) && (srcRow == dstRow));
}

static bool isMemEast(int srcCol, int srcRow, int dstCol, int dstRow) {
  bool IsEvenRow = ((srcRow % 2) == 0);
  return (!IsEvenRow && isInternal(srcCol, srcRow, dstCol, dstRow)) ||
         (IsEvenRow  && isEast(srcCol, srcRow, dstCol, dstRow));
}

static bool isNorth(int srcCol, int srcRow, int dstCol, int dstRow) {
  return ((srcCol == dstCol) && (srcRow == dstRow - 1));
}

static bool isMemNorth(int srcCol, int srcRow, int dstCol, int dstRow) {
  return isNorth(srcCol, srcRow, dstCol, dstRow);
}

static bool isSouth(int srcCol, int srcRow, int dstCol, int dstRow) {
  return ((srcCol == dstCol) && (srcRow == dstRow + 1));
}

static bool isMemSouth(int srcCol, int srcRow, int dstCol, int dstRow) {
  return isSouth(srcCol, srcRow, dstCol, dstRow);
}

static bool isLegalMemAffinity(int coreCol, int coreRow, int memCol, int memRow) {
  bool IsEvenRow = ((coreRow % 2) == 0);

  bool IsMemWest = (isWest(coreCol, coreRow, memCol, memRow)   && !IsEvenRow) ||
                   (isInternal(coreCol, coreRow, memCol, memRow) &&  IsEvenRow);

  bool IsMemEast = (isEast(coreCol, coreRow, memCol, memRow)   &&  IsEvenRow) ||
                   (isInternal(coreCol, coreRow, memCol, memRow) && !IsEvenRow);

  bool IsMemNorth = isNorth(coreCol, coreRow, memCol, memRow);
  bool IsMemSouth = isSouth(coreCol, coreRow, memCol, memRow);

  return IsMemSouth || IsMemNorth || IsMemWest || IsMemEast;
}

// include TableGen generated Op definitions
#define GET_OP_CLASSES
#include "AIE.h.inc"

#define GEN_PASS_CLASSES
#include "AIEPasses.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createAIECoreToLLVMPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "AIEPasses.h.inc"

} // AIE
} // xilinx

#endif
