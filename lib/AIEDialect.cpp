// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#include "AIEDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;

namespace xilinx {
namespace aie {

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

AIEDialect::AIEDialect(mlir::MLIRContext *ctx) : mlir::Dialect("aie", ctx) {
  //addTypes<AIEListType>();
  addOperations<
#define GET_OP_LIST
#include "AIE.cpp.inc"
    >();
}

} // namespace aie
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
//   xilinx::aie::ArbiterOp::ensureTerminator(*mastersets, parser.getBuilder(), result.location);

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
    SmallVector<NamedAttribute, 1> attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseAttribute(attrVal, parser.getBuilder().getNoneType(),
                              "sourceBundle", attrStorage))
      return failure();

    auto attrOptional = xilinx::aie::symbolizeWireBundle(attrVal.getValue());
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
  xilinx::aie::PacketRulesOp::ensureTerminator(*rules, parser.getBuilder(), result.location);

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
  xilinx::aie::SwitchboxOp::ensureTerminator(*connections, parser.getBuilder(), result.location);

  return success();
}

// static void print(OpAsmPrinter &p, xilinx::aie::ArbiterOp op) {
//   bool printBlockTerminators = false;

//   Region &body = op.region();
//   p << xilinx::aie::ArbiterOp::getOperationName();
//   p << '(';
//   p << ')';

//   p.printRegion(body,
//                 /*printEntryBlockArgs=*/false,
//                 /*printBlockTerminators=*/false);
//   //  p.printOptionalAttrDict(op.getAttrs());

// }
static void print(OpAsmPrinter &p, xilinx::aie::PacketRulesOp op) {
  bool printBlockTerminators = false;

  Region &body = op.rules();
  p << xilinx::aie::PacketRulesOp::getOperationName();
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
static void print(OpAsmPrinter &p, xilinx::aie::SwitchboxOp op) {
  bool printBlockTerminators = false;

  Region &body = op.connections();
  p << xilinx::aie::SwitchboxOp::getOperationName();
  p << '(';
  p << op.col() << ", " << op.row();
  p << ')';

  p.printRegion(body,
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  //  p.printOptionalAttrDict(op.getAttrs());

}


static LogicalResult verify(xilinx::aie::SwitchboxOp op) {
  Region &body = op.connections();
  DenseSet<xilinx::aie::Port> sourceset;
  DenseSet<xilinx::aie::Port> destset;
  assert(op.getOperation()->getNumRegions());
  assert(!body.empty());
  for (auto &ops : body.front()) {
    if(auto connectOp = dyn_cast<xilinx::aie::ConnectOp>(ops)) {
      xilinx::aie::Port source = std::make_pair(connectOp.sourceBundle(),
                                                connectOp.sourceIndex());
      sourceset.insert(source);

      xilinx::aie::Port dest = std::make_pair(connectOp.destBundle(),
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
    } else if(auto connectOp = dyn_cast<xilinx::aie::MasterSetOp>(ops)) {
      xilinx::aie::Port dest = std::make_pair(connectOp.destBundle(),
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
    } else if(auto connectOp = dyn_cast<xilinx::aie::PacketRulesOp>(ops)) {
      xilinx::aie::Port source = std::make_pair(connectOp.sourceBundle(),
                                                connectOp.sourceIndex());
      if(sourceset.count(source)) {
        return connectOp.emitOpError("packet switched source ") <<
          stringifyWireBundle(source.first) << source.second <<
          " cannot match another connect or masterset operation";
      } else {
        sourceset.insert(source);
      }
    } else if(auto endswitchOp = dyn_cast<xilinx::aie::EndswitchOp>(ops)) {
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
  xilinx::aie::ShimSwitchboxOp::ensureTerminator(*connections, parser.getBuilder(), result.location);

  return success();
}

static void print(OpAsmPrinter &p, xilinx::aie::ShimSwitchboxOp op) {
  bool printBlockTerminators = false;

  Region &body = op.connections();
  p << xilinx::aie::ShimSwitchboxOp::getOperationName();
  p << '(';
  p << op.col();
  p << ')';

  p.printRegion(body,
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  //  p.printOptionalAttrDict(op.getAttrs());

}

static LogicalResult verify(xilinx::aie::ShimSwitchboxOp op) {
  Region &body = op.connections();
  DenseSet<xilinx::aie::Port> destset;
  assert(op.getOperation()->getNumRegions());
  assert(!body.empty());
  for (auto &ops : body.front()) {
    if(auto connectOp = dyn_cast<xilinx::aie::ConnectOp>(ops)) {
      xilinx::aie::Port dest = std::make_pair(connectOp.destBundle(),
                                              connectOp.destIndex());
      if(destset.count(dest)) {
        return connectOp.emitOpError("targets same destination ") <<
          stringifyWireBundle(dest.first) << dest.second << " as another connect operation";
      } else {
        destset.insert(dest);
      }
    } else if(auto endswitchOp = dyn_cast<xilinx::aie::EndswitchOp>(ops)) {
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
  xilinx::aie::PacketFlowOp::ensureTerminator(*ports, parser.getBuilder(), result.location);

  return success();
}

static void print(OpAsmPrinter &p, xilinx::aie::PacketFlowOp op) {
  bool printBlockTerminators = false;

  Region &body = op.ports();
  p << xilinx::aie::PacketFlowOp::getOperationName();
  p << '(' << op.ID() << ')';

  p.printRegion(body,
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  //  p.printOptionalAttrDict(op.getAttrs());

}

static LogicalResult verify(xilinx::aie::PacketFlowOp op) {
  Region &body = op.ports();
  //DenseSet<xilinx::aie::Port> destset;
  assert(op.getOperation()->getNumRegions());
  assert(!body.empty());
  for (auto &ops : body.front()) {
    if(auto Op = dyn_cast<xilinx::aie::PacketSourceOp>(ops)) {
    } else if(auto Op = dyn_cast<xilinx::aie::PacketDestOp>(ops)) {
    } else if(auto endswitchOp = dyn_cast<xilinx::aie::EndswitchOp>(ops)) {
    } else {
      return ops.emitOpError("cannot be contained in a PacketFlow op");
    }
  }

  return success();
}

#include "AIEEnums.cpp.inc"

namespace xilinx {
  namespace aie {
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

  } // namespace aie
} // namespace xilinx
