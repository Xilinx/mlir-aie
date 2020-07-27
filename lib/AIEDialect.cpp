// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#include "AIEDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
using namespace mlir;

namespace xilinx {
namespace AIE {

  //namespace detail {

// /// This class holds the implementation of the ATenListType.
// /// It is intended to be uniqued based on its content and owned by the context.
// struct ATenListTypeStorage : public mlir::TypeStorage {
//   ATenListTypeStorage(Type elementType) : elementType(elementType) {}

//   /// The hash key used for uniquing.
//   using KeyTy = mlir::Type;
//   bool operator==(const KeyTy &key) const { return key == getElementType(); }

//   /// This is a factory method to create our type storage. It is only
//   /// invoked after looking up the type in the context using the key and not
//   /// finding it.
//   static ATenListTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
//                                         const KeyTy &key) {

//     // Allocate the instance for the ATenListTypeStorage itself
//     auto *storage = allocator.allocate<ATenListTypeStorage>();
//     // Initialize the instance using placement new.
//     return new (storage) ATenListTypeStorage(key);
//   }

//   Type getElementType() const { return elementType; }

// private:
//   Type elementType;

// };
// } // namespace detail

// ATenListType ATenListType::get(mlir::Type elemType) {
//   return Base::get(elemType.getContext(), ATenTypeKind::ATEN_LIST, elemType);
// }

// mlir::Type ATenListType::getElementType() {
//   return getImpl()->getElementType();
// }

// mlir::Type ATenDialect::parseType(DialectAsmParser &parser) const {
//   Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());

//   // All types start with an identifier that we switch on.
//   StringRef typeNameSpelling;
//   if (failed(parser.parseKeyword(&typeNameSpelling)))
//     return nullptr;

//   if (typeNameSpelling == "list") {
//     if(failed(parser.parseLess()))
//       return nullptr;
//     Type t;
//     if(failed(parser.parseType(t)))
//       return nullptr;
//     if(failed(parser.parseGreater()))
//       return nullptr;
//     return ATenListType::get(t);
//   }

//   parser.emitError(parser.getCurrentLocation(), "Invalid ATen type '" + typeNameSpelling + "'");
//   return nullptr;
// }

// /// Print a ATenListType
// void ATenDialect::printType(mlir::Type type, DialectAsmPrinter &os) const {
//   auto ty = type.dyn_cast<ATenListType>();
//   if (!ty) {
//     os << "unknown aten type";
//     return;
//   }
//   os << "list<";
//   os.getStream() << ty.getElementType();
//   os << ">";
// }

AIEDialect::AIEDialect(mlir::MLIRContext *ctx) : mlir::Dialect("AIE", ctx) {
  //addTypes<AIEListType>();
  addOperations<
#define GET_OP_LIST
#include "AIE.cpp.inc"
    >();
}

} // namespace AIE
} // namespace xilinx

// static ParseResult parseArbiterOp(OpAsmParser &parser, OperationState &result) {
//   result.regions.reserve(1);
//   Region *mastersets = result.addRegion();

//   auto &builder = parser.getBuilder();
//   //  result.types.push_back(builder.getIndexType());

//   if (parser.parseLParen())
//     return failure();
//   if (parser.parseRParen())
//     return failure();

//   // Parse the mastersets.
//   if (parser.parseRegion(*mastersets, /*arguments=*/{}, /*argTypes=*/{}))
//     return failure();
//   // // Parse the optional attribute list.
//   // if (parser.parseOptionalAttrDict(result.attributes))
//   //   return failure();
//   xilinx::AIE::ArbiterOp::ensureTerminator(*mastersets, parser.getBuilder(), result.location);

//   return success();
// }


static ParseResult parsePacketRulesOp(OpAsmParser &parser, OperationState &result) {
  result.regions.reserve(1);
  Region *rules = result.addRegion();

  auto &builder = parser.getBuilder();
  //  result.types.push_back(builder.getIndexType());

  if (parser.parseLParen())
    return failure();
  {
    StringAttr attrVal;
    NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseAttribute(attrVal, parser.getBuilder().getNoneType(),
                              "sourceBundle", attrStorage))
      return failure();

    auto attrOptional = xilinx::AIE::symbolizeWireBundle(attrVal.getValue());
    if (!attrOptional)
      return parser.emitError(loc, "invalid ")
             << "sourceBundle attribute specification: " << attrVal;

    result.addAttribute("sourceBundle", parser.getBuilder().getI32IntegerAttr(static_cast<int32_t>(attrOptional.getValue())));
  }
  if (parser.parseColon())
    return failure();

  IntegerAttr sourceChannelAttr;
  if (parser.parseAttribute(sourceChannelAttr, parser.getBuilder().getIntegerType(32), "sourceChannel", result.attributes))
    return failure();

  if (parser.parseRParen())
    return failure();

  // Parse the rules.
  if (parser.parseRegion(*rules, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  // // Parse the optional attribute list.
  // if (parser.parseOptionalAttrDict(result.attributes))
  //   return failure();
  xilinx::AIE::PacketRulesOp::ensureTerminator(*rules, parser.getBuilder(), result.location);

  return success();
}

static ParseResult parseSwitchboxOp(OpAsmParser &parser, OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(1);
  Region *connections = result.addRegion();

  auto &builder = parser.getBuilder();
  result.types.push_back(builder.getIndexType());
  OpAsmParser::OperandType cond;
  Type iType = builder.getIndexType();
  SmallVector<Type, 4> types;
  types.push_back(iType);
  types.push_back(iType);


  if (parser.parseLParen())
    return failure();

  IntegerAttr colAttr;
  if (parser.parseAttribute(colAttr, parser.getBuilder().getIntegerType(32), "col", result.attributes))
    return failure();
  if (parser.parseComma())
    return failure();

  IntegerAttr rowAttr;
  if (parser.parseAttribute(rowAttr, parser.getBuilder().getIntegerType(32), "row", result.attributes))
    return failure();
  if (parser.parseRParen())
    return failure();

  // Parse the connections.
  if (parser.parseRegion(*connections, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  // // Parse the optional attribute list.
  // if (parser.parseOptionalAttrDict(result.attributes))
  //   return failure();
  xilinx::AIE::SwitchboxOp::ensureTerminator(*connections, parser.getBuilder(), result.location);

  return success();
}

// static void print(OpAsmPrinter &p, xilinx::AIE::ArbiterOp op) {
//   bool printBlockTerminators = false;

//   Region &body = op.region();
//   p << xilinx::AIE::ArbiterOp::getOperationName();
//   p << '(';
//   p << ')';

//   p.printRegion(body,
//                 /*printEntryBlockArgs=*/false,
//                 /*printBlockTerminators=*/false);
//   //  p.printOptionalAttrDict(op.getAttrs());

// }
static void print(OpAsmPrinter &p, xilinx::AIE::PacketRulesOp op) {
  bool printBlockTerminators = false;

  Region &body = op.rules();
  p << xilinx::AIE::PacketRulesOp::getOperationName();
  p << '(';
  p << "\"" << stringifyWireBundle(op.sourceBundle()) << "\"";
  p << " " << ":";
  p << " ";
  p.printAttributeWithoutType(op.sourceChannelAttr());
  p << ')';

  p.printRegion(body,
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  //  p.printOptionalAttrDict(op.getAttrs());

}
static void print(OpAsmPrinter &p, xilinx::AIE::SwitchboxOp op) {
  bool printBlockTerminators = false;

  Region &body = op.connections();
  p << xilinx::AIE::SwitchboxOp::getOperationName();
  p << '(';
  p << op.col() << ", " << op.row();
  p << ')';

  p.printRegion(body,
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  //  p.printOptionalAttrDict(op.getAttrs());

}


static LogicalResult verify(xilinx::AIE::SwitchboxOp op) {
  Region &body = op.connections();
  DenseSet<xilinx::AIE::Port> sourceset;
  DenseSet<xilinx::AIE::Port> destset;
  assert(op.getOperation()->getNumRegions());
  assert(!body.empty());
  for (auto &ops : body.front()) {
    if(auto connectOp = dyn_cast<xilinx::AIE::ConnectOp>(ops)) {
      xilinx::AIE::Port source = std::make_pair(connectOp.sourceBundle(),
                                                connectOp.sourceIndex());
      sourceset.insert(source);

      xilinx::AIE::Port dest = std::make_pair(connectOp.destBundle(),
                                              connectOp.destIndex());
      if(destset.count(dest)) {
        return connectOp.emitOpError("targets same destination ") <<
          stringifyWireBundle(dest.first) << dest.second <<
          " as another connect operation";
      } else {
        destset.insert(dest);
      }
      if(connectOp.sourceIndex() < 0) {
        connectOp.emitOpError("source index cannot be less than zero");
      }
      if(connectOp.sourceIndex() >=
         op.getNumSourceConnections(connectOp.sourceBundle())) {
        connectOp.emitOpError("source index for source bundle ") <<
          stringifyWireBundle(connectOp.sourceBundle()) <<
          " must be less than " <<
          op.getNumSourceConnections(connectOp.sourceBundle());
      }
      if(connectOp.destIndex() < 0) {
        connectOp.emitOpError("dest index cannot be less than zero");
      }
      if(connectOp.destIndex() >=
         op.getNumDestConnections(connectOp.destBundle())) {
        connectOp.emitOpError("dest index for dest bundle ") <<
          stringifyWireBundle(connectOp.destBundle()) <<
          " must be less than " <<
          op.getNumDestConnections(connectOp.destBundle());
      }
    } else if(auto connectOp = dyn_cast<xilinx::AIE::MasterSetOp>(ops)) {
      xilinx::AIE::Port dest = std::make_pair(connectOp.destBundle(),
                                              connectOp.destIndex());
      if(destset.count(dest)) {
        return connectOp.emitOpError("targets same destination ") <<
          stringifyWireBundle(dest.first) << dest.second <<
          " as another connect or masterset operation";
      } else {
        destset.insert(dest);
      }
      if(connectOp.destIndex() < 0) {
        connectOp.emitOpError("dest index cannot be less than zero");
      }
      if(connectOp.destIndex() >=
         op.getNumDestConnections(connectOp.destBundle())) {
        connectOp.emitOpError("dest index for dest bundle ") <<
          stringifyWireBundle(connectOp.destBundle()) <<
          " must be less than " <<
          op.getNumDestConnections(connectOp.destBundle());
      }
    } else if(auto connectOp = dyn_cast<xilinx::AIE::PacketRulesOp>(ops)) {
      xilinx::AIE::Port source = std::make_pair(connectOp.sourceBundle(),
                                                connectOp.sourceIndex());
      if(sourceset.count(source)) {
        return connectOp.emitOpError("packet switched source ") <<
          stringifyWireBundle(source.first) << source.second <<
          " cannot match another connect or masterset operation";
      } else {
        sourceset.insert(source);
      }
    } else if(auto endswitchOp = dyn_cast<xilinx::AIE::EndswitchOp>(ops)) {
    } else {
      return ops.emitOpError("cannot be contained in a Switchbox op");
    }
  }

  return success();
}

static ParseResult parseShimSwitchboxOp(OpAsmParser &parser, OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(1);
  Region *connections = result.addRegion();

  auto &builder = parser.getBuilder();
  result.types.push_back(builder.getIndexType());
  OpAsmParser::OperandType cond;
  Type iType = builder.getIndexType();
  SmallVector<Type, 4> types;
  types.push_back(iType);
  types.push_back(iType);


  if (parser.parseLParen())
    return failure();

  IntegerAttr colAttr;
  if (parser.parseAttribute(colAttr, parser.getBuilder().getIntegerType(32), "col", result.attributes))
    return failure();
  if (parser.parseRParen())
    return failure();

  // Parse the connections.
  if (parser.parseRegion(*connections, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  // // Parse the optional attribute list.
  // if (parser.parseOptionalAttrDict(result.attributes))
  //   return failure();
  xilinx::AIE::ShimSwitchboxOp::ensureTerminator(*connections, parser.getBuilder(), result.location);

  return success();
}

static void print(OpAsmPrinter &p, xilinx::AIE::ShimSwitchboxOp op) {
  bool printBlockTerminators = false;

  Region &body = op.connections();
  p << xilinx::AIE::ShimSwitchboxOp::getOperationName();
  p << '(';
  p << op.col();
  p << ')';

  p.printRegion(body,
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  //  p.printOptionalAttrDict(op.getAttrs());

}

static LogicalResult verify(xilinx::AIE::ShimSwitchboxOp op) {
  Region &body = op.connections();
  DenseSet<xilinx::AIE::Port> destset;
  assert(op.getOperation()->getNumRegions());
  assert(!body.empty());
  for (auto &ops : body.front()) {
    if(auto connectOp = dyn_cast<xilinx::AIE::ConnectOp>(ops)) {
      xilinx::AIE::Port dest = std::make_pair(connectOp.destBundle(),
                                              connectOp.destIndex());
      if(destset.count(dest)) {
        return connectOp.emitOpError("targets same destination ") <<
          stringifyWireBundle(dest.first) << dest.second << " as another connect operation";
      } else {
        destset.insert(dest);
      }
    } else if(auto endswitchOp = dyn_cast<xilinx::AIE::EndswitchOp>(ops)) {
    } else {
      return ops.emitOpError("cannot be contained in a Switchbox op");
    }
  }

  return success();
}

static ParseResult parsePacketFlowOp(OpAsmParser &parser, OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(1);
  Region *ports = result.addRegion();

  auto &builder = parser.getBuilder();
  //  result.types.push_back(builder.getIndexType());
  OpAsmParser::OperandType cond;

  if (parser.parseLParen())
    return failure();

  IntegerAttr IDAttr;
  if (parser.parseAttribute(IDAttr, parser.getBuilder().getIntegerType(8), "ID", result.attributes))
    return failure();

  if (parser.parseRParen())
    return failure();

  // Parse the ports.
  if (parser.parseRegion(*ports, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  // // Parse the optional attribute list.
  // if (parser.parseOptionalAttrDict(result.attributes))
  //   return failure();
  xilinx::AIE::PacketFlowOp::ensureTerminator(*ports, parser.getBuilder(), result.location);

  return success();
}

static void print(OpAsmPrinter &p, xilinx::AIE::PacketFlowOp op) {
  bool printBlockTerminators = false;

  Region &body = op.ports();
  p << xilinx::AIE::PacketFlowOp::getOperationName();
  p << '(' << op.ID() << ')';

  p.printRegion(body,
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  //  p.printOptionalAttrDict(op.getAttrs());

}

static LogicalResult verify(xilinx::AIE::PacketFlowOp op) {
  Region &body = op.ports();
  //DenseSet<xilinx::AIE::Port> destset;
  assert(op.getOperation()->getNumRegions());
  assert(!body.empty());
  for (auto &ops : body.front()) {
    if(auto Op = dyn_cast<xilinx::AIE::PacketSourceOp>(ops)) {
    } else if(auto Op = dyn_cast<xilinx::AIE::PacketDestOp>(ops)) {
    } else if(auto endswitchOp = dyn_cast<xilinx::AIE::EndswitchOp>(ops)) {
    } else {
      return ops.emitOpError("cannot be contained in a PacketFlow op");
    }
  }

  return success();
}

// MemOp
static ParseResult parseMemOp(OpAsmParser &parser, OperationState &result) {
  result.regions.reserve(1);
  Region *body = result.addRegion();

  auto &builder = parser.getBuilder();
  result.types.push_back(builder.getIndexType());

  if (parser.parseLParen())
    return failure();

  IntegerAttr colAttr;
  if (parser.parseAttribute(colAttr, parser.getBuilder().getIntegerType(32), "col", result.attributes))
    return failure();
  if (parser.parseComma())
    return failure();

  IntegerAttr rowAttr;
  if (parser.parseAttribute(rowAttr, parser.getBuilder().getIntegerType(32), "row", result.attributes))
    return failure();

  if (parser.parseRParen())
    return failure();

  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, xilinx::AIE::MemOp op) {
  p << xilinx::AIE::MemOp::getOperationName();

  p << "(";
  p.printAttributeWithoutType(op.colAttr());
  p << ",";
  p << " ";
  p.printAttributeWithoutType(op.rowAttr());
  p << ")";

  Region &body = op.body();

  p.printRegion(body,
                /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);
}

static LogicalResult verify(xilinx::AIE::MemOp op) {
  Region &body = op.body();
  assert(op.getOperation()->getNumRegions() == 1 && "MemOp has zero region!");
  assert(!body.empty() && "MemOp should have non-empty body");

  for (auto &bodyOp : body.getOps()) {
    if (auto allocOp = dyn_cast<AllocOp>(bodyOp)) {
      if (!allocOp.getAttr("id"))
        op.emitOpError() << "allocOp in MemOp region should have an id attribute\n";
    }
  }

  return success();
}

// CoreModuleOp
static ParseResult parseCoreModuleOp(OpAsmParser &parser, OperationState &result) {
  result.regions.reserve(1);
  Region *body = result.addRegion();

  auto &builder = parser.getBuilder();

  SmallVector<OpAsmParser::OperandType, 4> operandsOperands;
  llvm::SMLoc operandsOperandsLoc = parser.getCurrentLocation();
  (void)operandsOperandsLoc;
  SmallVector<Type, 4> operandsTypes;

  SmallVector<OpAsmParser::OperandType, 4> arguments;
  SmallVector<Type, 4> argTypes;

  if (parser.parseLess())
    return failure();

  if (parser.parseOperandList(operandsOperands))
    return failure();

  if (parser.parseGreater())
    return failure();

  if (parser.parseLParen())
    return failure();

  if (parser.parseRegionArgumentList(arguments))
    return failure();

  if (parser.parseRParen())
    return failure();

  for (unsigned i = 0; i < operandsOperands.size(); i++) {
    operandsTypes.push_back(builder.getIndexType());
    argTypes.push_back(builder.getIndexType());
  }

  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return failure();

//  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
  if (parser.parseRegion(*body, arguments, argTypes))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, xilinx::AIE::CoreModuleOp op) {
  p << xilinx::AIE::CoreModuleOp::getOperationName();

  p << "<";
  p << op.operands();
  p << ">";
  Region &body = op.body();

  p.printRegion(body,
                /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);
}

void xilinx::AIE::CoreModuleOp::build(OpBuilder &builder,
  OperationState &odsState, ValueRange operands) {
  odsState.addOperands(operands);
  int numOperands = operands.size();

  for (unsigned i = 0; i != 1; ++i) {
    Region *r = odsState.addRegion();
    r->push_back(new Block);
    Block &b = r->front();
    for (int i = 0; i < numOperands; i++) {
      b.addArgument(builder.getIndexType());
    }
  }
}

ArrayRef<Operation *> xilinx::AIE::CoreModuleOp::getCores() {
  SmallVector<Operation *, 4> cores;
  for (auto operand : operands()) {
    if (xilinx::AIE::CoreOp core =
            dyn_cast_or_null<xilinx::AIE::CoreOp>(operand.getDefiningOp())) {
      cores.push_back(core);
    }
  }

  return cores;
}

ArrayRef<Operation *> xilinx::AIE::CoreModuleOp::getMems() {
  SmallVector<Operation *, 4> mems;
  for (auto operand : operands()) {
    if (xilinx::AIE::MemOp mem =
                   dyn_cast_or_null<xilinx::AIE::MemOp>(operand.getDefiningOp())) {
      mems.push_back(mem);
    }
  }

  return mems;
}

ArrayRef<Operation *> xilinx::AIE::CoreModuleOp::getSwitchboxes() {
  SmallVector<Operation *, 4> switchboxes;
  for (auto operand : operands()) {
    if (xilinx::AIE::SwitchboxOp switchbox =
                   dyn_cast_or_null<xilinx::AIE::SwitchboxOp>(operand.getDefiningOp())) {
      switchboxes.push_back(switchbox);
    }
  }

  return switchboxes;
}

Operation *xilinx::AIE::CoreModuleOp::getMainCore() {
  ArrayRef<Operation *> cores(getCores());
  return cores.front();
}

Operation *xilinx::AIE::CoreModuleOp::getWestCore() {
  ArrayRef<Operation *> cores(getCores());
  xilinx::AIE::CoreOp mainCore = dyn_cast<xilinx::AIE::CoreOp>(getMainCore());
  int srcCol = mainCore.colIndex();
  int srcRow = mainCore.rowIndex();

  int westCol = srcCol - 1;
  int westRow = srcRow;

  for (auto Op : cores) {
    CoreOp westCore = dyn_cast<xilinx::AIE::CoreOp>(Op);
    if (westCol == westCore.colIndex() && westRow == westCore.rowIndex())
      return westCore;
  }

  return nullptr;
}

Operation *xilinx::AIE::CoreModuleOp::getEastCore() {
  ArrayRef<Operation *> cores(getCores());
  xilinx::AIE::CoreOp mainCore = dyn_cast<xilinx::AIE::CoreOp>(getMainCore());
  int srcCol = mainCore.colIndex();
  int srcRow = mainCore.rowIndex();

  int eastCol = srcCol + 1;
  int eastRow = srcRow;

  for (auto Op : cores) {
    CoreOp eastCore = dyn_cast<xilinx::AIE::CoreOp>(Op);
    if (eastCol == eastCore.colIndex() && eastRow == eastCore.rowIndex())
      return eastCore;
  }

  return nullptr;
}

Operation *xilinx::AIE::CoreModuleOp::getWestMem() {
  ArrayRef<Operation *> mems(getMems());

  xilinx::AIE::CoreOp mainCore = dyn_cast<xilinx::AIE::CoreOp>(getMainCore());
  int col = mainCore.colIndex();
  int row = mainCore.rowIndex();

  bool IsEvenRow = ((row % 2) == 0);

  int westCol = IsEvenRow ? (col) : (col - 1);
  int westRow = row;

  for (auto Op : mems) {
    xilinx::AIE::MemOp westMem = dyn_cast<xilinx::AIE::MemOp>(Op);
    if (westCol == westMem.colIndex() && westRow == westMem.rowIndex())
      return westMem;
  }

  return nullptr;
}

Operation *xilinx::AIE::CoreModuleOp::getEastMem() {
  ArrayRef<Operation *> mems(getMems());

  xilinx::AIE::CoreOp mainCore = dyn_cast<xilinx::AIE::CoreOp>(getMainCore());
  int col = mainCore.colIndex();
  int row = mainCore.rowIndex();

  bool IsEvenRow = ((row % 2) == 0);

  int eastCol = IsEvenRow ? (col + 1) : (col);
  int eastRow = row;

  for (auto Op : mems) {
    xilinx::AIE::MemOp eastMem = dyn_cast<xilinx::AIE::MemOp>(Op);
    if (eastCol == eastMem.colIndex() && eastRow == eastMem.rowIndex())
      return eastMem;
  }

  return nullptr;
}

Operation *xilinx::AIE::CoreModuleOp::getSouthMem() {
  ArrayRef<Operation *> mems(getMems());

  xilinx::AIE::CoreOp mainCore = dyn_cast<xilinx::AIE::CoreOp>(getMainCore());
  int col = mainCore.colIndex();
  int row = mainCore.rowIndex();

  int southCol = col;
  int southRow = row - 1;

  for (auto Op : mems) {
    xilinx::AIE::MemOp southMem = dyn_cast<xilinx::AIE::MemOp>(Op);
    if (southCol == southMem.colIndex() && southRow == southMem.rowIndex())
      return southMem;
  }

  return nullptr;
}

Operation *xilinx::AIE::CoreModuleOp::getNorthMem() {
  ArrayRef<Operation *> mems(getMems());

  xilinx::AIE::CoreOp mainCore = dyn_cast<xilinx::AIE::CoreOp>(getMainCore());
  int col = mainCore.colIndex();
  int row = mainCore.rowIndex();

  int northCol = col;
  int northRow = row + 1;

  for (auto Op : mems) {
    xilinx::AIE::MemOp northMem = dyn_cast<xilinx::AIE::MemOp>(Op);
    if (northCol == northMem.colIndex() && northRow == northMem.rowIndex())
      return northMem;
  }

  return nullptr;
}

Operation *xilinx::AIE::CoreModuleOp::getSwitchbox() {
  ArrayRef<Operation *> switchboxes(getSwitchboxes());

  return switchboxes.front();
}

bool xilinx::AIE::CoreModuleOp::isMainMemWest() {
  xilinx::AIE::CoreOp mainCore = dyn_cast<xilinx::AIE::CoreOp>(getMainCore());
  int col = mainCore.colIndex();
  int row = mainCore.rowIndex();

  bool IsEvenRow = ((row % 2) == 0);

  return IsEvenRow;
}

static LogicalResult verify(xilinx::AIE::CoreModuleOp op) {
  Region &body = op.body();
  assert(op.getOperation()->getNumRegions() == 1 && "CoreModule has zero region!");
  assert(!body.empty() && "CoreModule should have non-empty body");

  for (auto operand : op.operands()) {
    if (!(isa<xilinx::AIE::CoreOp>(operand.getDefiningOp()) ||
          isa<xilinx::AIE::MemOp>(operand.getDefiningOp()) ||
          isa<xilinx::AIE::SwitchboxOp>(operand.getDefiningOp()))) {
      op.emitOpError() << "Unsupported operand type!"
                          "An operand of a CoreModuleOp can only be"
                          "CoreOp, MemOp, or SwitchboxOp\n";
      return failure();
    }
  }

  if (op.getCores().size() < 1 || op.getCores().size() > 3) {
    op.emitOpError() << "A CoreModuleOp must have at least one CoreOp and at most three CoreOps\n";
    return failure();
  }

  if (op.getMems().size() > 4) {
    op.emitOpError() << "A CoreModuleOp can only have at most four memory modules\n";
    return failure();
  }

  if (op.getSwitchboxes().size() > 1) {
    op.emitOpError() << "A CoreModuleOp can only have at most one switchbox\n";
    return failure();
  }

  // the first CoreOp in the operands is the runnable core of the CoreModuleOp
  // (i.e., the code in the op region is executed on the core)
  xilinx::AIE::CoreOp mainCore = dyn_cast<xilinx::AIE::CoreOp>(op.getMainCore());
  int coreCol = mainCore.colIndex();
  int coreRow = mainCore.rowIndex();

  // Check switchbox
  for (auto Op : op.getSwitchboxes()) {
    xilinx::AIE::SwitchboxOp sb = dyn_cast<xilinx::AIE::SwitchboxOp>(Op);
    if ((coreCol != sb.colIndex()) || (coreRow != sb.rowIndex())) {
      op.emitOpError() << "core and switchbox (col, row) indices mismatched: "
                       << "switchbox(" << sb.colIndex() << "," << sb.rowIndex() << ")"
                       << "and "
                       << "core(" << coreCol << "," << coreRow << ")";
      return failure();
    }
  }

  // Check memory affinity
  for (auto Op : op.getMems()) {
    xilinx::AIE::MemOp mem = dyn_cast<xilinx::AIE::MemOp>(Op);
    int memCol = mem.colIndex();
    int memRow = mem.rowIndex();
    if (!xilinx::AIE::isLegalMemAffinity(coreCol, coreRow, memCol, memRow)) {
      op.emitOpError() << "Illegal memory affinity of a coreOp with a MemOp! "
                       << "mem(" << memCol << "," << memRow << ")"
                       << "and "
                       << "core(" << coreCol << "," << coreRow << ")";
      return failure();
    }
  }

  // Check core cascading
  for (unsigned i = 1; i < op.getCores().size(); i++) {
    xilinx::AIE::CoreOp nextCore = dyn_cast_or_null<xilinx::AIE::CoreOp>(op.getCores()[i]);
    bool IsCoreWest = xilinx::AIE::isWest(coreCol, coreRow, nextCore.colIndex(), nextCore.rowIndex());
    bool IsCoreEast = xilinx::AIE::isEast(coreCol, coreRow, nextCore.colIndex(), nextCore.rowIndex());
    if (!(IsCoreWest || IsCoreEast)) {
      op.emitOpError() << "Illegal core cascading!\n";
      return failure();
    }
  }

  return success();
}

static LogicalResult verify(xilinx::AIE::LockOp op) {
  Optional<Value> mem = op.mem();
  if (Value value = op.mem()) {
    if (BlockArgument arg = value.dyn_cast<BlockArgument>()) {
      int argNo = arg.getArgNumber();
      Operation *parentOp = arg.getOwner()->getParentOp();
      if (xilinx::AIE::CoreModuleOp coreModule = dyn_cast<xilinx::AIE::CoreModuleOp>(parentOp)) {
        value = coreModule.getOperands()[argNo];
      }
    }
    xilinx::AIE::MemOp memOp = dyn_cast_or_null<xilinx::AIE::MemOp>(value.getDefiningOp());
    if (!memOp) {
      op.emitOpError() << "Expected MemOp!\n";
      return failure();
    }
  }
  return success();
}

static LogicalResult verify(xilinx::AIE::UseLockOp op) {
  xilinx::AIE::LockOp lockOp = dyn_cast_or_null<xilinx::AIE::LockOp>(op.lock().getDefiningOp());
  if (!lockOp) {
    op.emitOpError() << "Expected LockOp!\n";
    return failure();
  }

  return success();
}

static LogicalResult verify(xilinx::AIE::BufferOp op) {
  Value value = op.mem();
  if (BlockArgument arg = value.dyn_cast<BlockArgument>()) {
    int argNo = arg.getArgNumber();
    Operation *parentOp = arg.getOwner()->getParentOp();
    if (xilinx::AIE::CoreModuleOp coreModule = dyn_cast<xilinx::AIE::CoreModuleOp>(parentOp)) {
      value = coreModule.getOperands()[argNo];
    }
  }

  xilinx::AIE::MemOp memOp = dyn_cast_or_null<xilinx::AIE::MemOp>(value.getDefiningOp());
  if (!memOp) {
    op.emitOpError() << "Expected MemOp!\n";
    return failure();
  }

  return success();
}

static LogicalResult verify(xilinx::AIE::GetStreamOp op) {
  Value value = op.switchbox();
  if (BlockArgument arg = value.dyn_cast<BlockArgument>()) {
    int argNo = arg.getArgNumber();
    Operation *parentOp = arg.getOwner()->getParentOp();
    if (xilinx::AIE::CoreModuleOp coreModule = dyn_cast<xilinx::AIE::CoreModuleOp>(parentOp)) {
      value = coreModule.getOperands()[argNo];
    }
  }

  xilinx::AIE::SwitchboxOp sbOp = dyn_cast_or_null<xilinx::AIE::SwitchboxOp>(value.getDefiningOp());
  if (!sbOp) {
    op.emitOpError() << "Expected SwitchboxOp!\n";
    return failure();
  }

  return success();
}

static LogicalResult verify(xilinx::AIE::PutStreamOp op) {
  Value value = op.switchbox();
  if (BlockArgument arg = value.dyn_cast<BlockArgument>()) {
    int argNo = arg.getArgNumber();
    Operation *parentOp = arg.getOwner()->getParentOp();
    if (xilinx::AIE::CoreModuleOp coreModule = dyn_cast<xilinx::AIE::CoreModuleOp>(parentOp)) {
      value = coreModule.getOperands()[argNo];
    }
  }

  xilinx::AIE::SwitchboxOp sbOp = dyn_cast_or_null<xilinx::AIE::SwitchboxOp>(value.getDefiningOp());
  if (!sbOp) {
    op.emitOpError() << "Expected SwitchboxOp!\n";
    return failure();
  }

  return success();
}

static LogicalResult verify(xilinx::AIE::GetCascadeOp op) {
  Value value = op.core();
  if (BlockArgument arg = value.dyn_cast<BlockArgument>()) {
    int argNo = arg.getArgNumber();
    Operation *parentOp = arg.getOwner()->getParentOp();
    if (xilinx::AIE::CoreModuleOp coreModule = dyn_cast<xilinx::AIE::CoreModuleOp>(parentOp)) {
      value = coreModule.getOperands()[argNo];
    }
  }

  xilinx::AIE::CoreOp coreOp = dyn_cast_or_null<xilinx::AIE::CoreOp>(value.getDefiningOp());
  if (!coreOp) {
    op.emitOpError() << "Expected CoreOp!\n";
    return failure();
  }

  return success();
}

static LogicalResult verify(xilinx::AIE::PutCascadeOp op) {
  Value value = op.core();
  if (BlockArgument arg = value.dyn_cast<BlockArgument>()) {
    int argNo = arg.getArgNumber();
    Operation *parentOp = arg.getOwner()->getParentOp();
    if (xilinx::AIE::CoreModuleOp coreModule = dyn_cast<xilinx::AIE::CoreModuleOp>(parentOp)) {
      value = coreModule.getOperands()[argNo];
    }
  }

  xilinx::AIE::CoreOp coreOp = dyn_cast_or_null<xilinx::AIE::CoreOp>(value.getDefiningOp());
  if (!coreOp) {
    op.emitOpError() << "Expected CoreOp!\n";
    return failure();
  }

  return success();
}

#include "AIEEnums.cpp.inc"

namespace xilinx {
  namespace AIE {
#define GET_OP_CLASSES
#include "AIE.cpp.inc"

    // void CoreOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, int col, int row) {
    //   odsState.addOperands(colValue);
    //   odsState.addOperands(rowValue);
    //   odsState.addTypes(resultType0);
    // }

  //#include "ATenOpInterfaces.cpp.inc"

    int SwitchboxOp::getNumSourceConnections(WireBundle bundle) {
      switch(bundle) {
      case WireBundle::ME: return 2;
      case WireBundle::DMA: return 2;
      case WireBundle::North: return 4;
      case WireBundle::West: return 4;
      case WireBundle::South: return 6;
      case WireBundle::East: return 4;
      default: return 0;
      }
    }
    int SwitchboxOp::getNumDestConnections(WireBundle bundle) {
      switch(bundle) {
      case WireBundle::ME: return 2;
      case WireBundle::DMA: return 2;
      case WireBundle::North: return 6;
      case WireBundle::West: return 4;
      case WireBundle::South: return 4;
      case WireBundle::East: return 4;
      default: return 0;
      }
    }
    int CoreOp::getNumSourceConnections(WireBundle bundle) {
      switch(bundle) {
      case WireBundle::ME: return 2;
      case WireBundle::DMA: return 2;
      default: return 0;
      }
    }
    int CoreOp::getNumDestConnections(WireBundle bundle) {
      switch(bundle) {
      case WireBundle::ME: return 2;
      case WireBundle::DMA: return 2;
      default: return 0;
      }
    }

  } // namespace AIE
} // namespace xilinx
