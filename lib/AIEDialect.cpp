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

  } // namespace aie
} // namespace xilinx
