//===- CIRToAIEpasses.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//===----------------------------------------------------------------------===//

#include <any>
#include <array>
#include <cassert>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <queue>

#include "aie/CIR/CIRToAIEPasses.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/InliningUtils.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/LowerToMLIR.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "cir-to-aie"

using namespace std::string_literals;

namespace {
// Erase a range of Operation* and its users recursively.
// Only consider the use-def chains and not the regions of blocks yet.
template <typename OpRange>
void eraseOpsAndUsers(OpRange &&opsToErase) {
  llvm::SetVector<mlir::Operation *> allOpsAndUsers;
  llvm::SmallVector<mlir::Operation *> newOps{
      std::forward<OpRange>(opsToErase)};
  //  While there are some operations to process
  while (!newOps.empty()) {
    auto *op = newOps.pop_back_val();
    LLVM_DEBUG(op->emitRemark("eraseOpsAndUsers: newOps.pop_back_val()"));
    // If the operation has not been visited yet, add it to the set and process
    // its users
    if (allOpsAndUsers.insert(op)) {
      LLVM_DEBUG(op->emitRemark("eraseOpsAndUsers: inserted!"));
      for (auto result : op->getResults())
        for (auto *user : result.getUsers()) {
          // Add each user to the visit queue
          newOps.push_back(user);
          LLVM_DEBUG(
              user->emitRemark("eraseOpsAndUsers: append to visit queue"));
        }
    }
  }
  // To avoid erasing operations with remaining users, topologically sort the
  // operations according to their use-def chains and erase them in reverse
  // order
  for (auto *op : allOpsAndUsers)
    LLVM_DEBUG(op->emitRemark("eraseOpsAndUsers: allOpsAndUsers"));
  // Does not work here
  // auto sorted = mlir::topologicalSort(allOpsAndUsers);
  llvm::SmallVector<mlir::Operation *> sorted{allOpsAndUsers.begin(),
                                              allOpsAndUsers.end()};
  mlir::computeTopologicalSorting(sorted);
  LLVM_DEBUG(for (auto *op
                  : sorted)
                 op->emitRemark("eraseOpsAndUsers: topologicalSort"));
  for (auto *op : llvm::reverse(sorted)) {
    LLVM_DEBUG(op->emitRemark("eraseOpsAndUsers: reverse"));
    op->erase();
  }
}

// Find in program order the first useful non cir.scope operation inside the
// root operation
mlir::Operation *findFirstNonCIRScopeOpInside(mlir::Operation *root) {
  mlir::Operation *firsttNonCIRScopeOp = nullptr;
  root->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (op == root || mlir::isa<cir::ScopeOp>(op))
      return mlir::WalkResult::advance();
    firsttNonCIRScopeOp = op;
    return mlir::WalkResult::interrupt();
  });
  return firsttNonCIRScopeOp;
}
} // namespace

namespace xilinx::AIE::CIR {

// Analyze all the C++ types used in a module and for aie++ types deconstruct
// them and keep track of the AIE dialect operations used to produce a value of
// its type. If some aie++ type values are produced by some AIE operations, keep
// track of these operations.
class CIRToAIETypesAnalysis {
public:
  // llvm::DenseMap<mlir::Type, std::optional<mlir::Type>> types;
  struct AIELikeTypesDeconstruction {
    // For example "aie::device<aie::npu1>"
    std::string fullName;
    // For example "aie::device"
    std::string base;
    // For example "npu1"
    std::vector<std::string> subMatches;
    // To attach something, like the aie.tile operation for example
    std::any data;
    // The AIE operation which is generated
    std::optional<mlir::Operation *> newAIEOperation;
    // The new operation producing the result (if any) instead for replacement,
    // typically an UnrealizedConversionCastOp fed by the newAIEOperation
    std::optional<mlir::Operation *> newProducer;

    // Display the content of AIELikeTypesDeconstruction
    void dump() {
      llvm::outs() << "Fullname = " + fullName + ", base = " + base +
                          ", subMatches = " + llvm::join(subMatches, ", ")
                   << '\n';
      if (newAIEOperation)
        (*newAIEOperation)->emitRemark("newAIEOperation = ");
      if (newProducer)
        (*newProducer)->emitRemark("newProducer = ");
    }
  };

private:
  // A map from a type to its aie:: deconstruction in the case it is a pointer
  // type to a well known aie:: struct
  llvm::DenseMap<mlir::Type, std::optional<AIELikeTypesDeconstruction>>
      moduleTypes;

  // Record whether an aie++ C++ type has been translated into some AIE
  // operation producing a value related to that type
  llvm::DenseSet<mlir::Type> isAIELoweredType;

public:
  void analyze() {
    // A struct with a name like "aie::device<aie::npu1>" (and the "npu1" is
    // used directly for the MLIR aie.device attribute) or aie::tile_t<8,50> for
    // example
    static const std::array typeNamePatterns{
        // A struct with a name like "aie::device<aie::npu1, aie::(lambda at
        // ./aie++.hpp:76:54)>.0". Drop the non-interesting unique-ing lambda
        // part of the type
        llvm::Regex{"^(aie::device)<aie::([^,]+).*$"},
        // A struct with a name like "aie::tile_t<8,50, aie::device<>>" for
        // example
        llvm::Regex{"^(aie::tile)<([[:digit:]]+), ([[:digit:]]+), .*$"},
        llvm::Regex{"^(aie::buffer)<([^,]+), ([^>]+)>$"}};

    for (auto &[type, value] : moduleTypes) {
      if (auto maybePointerType = mlir::dyn_cast<cir::PointerType>(type))
        if (auto maybeStructType =
                mlir::dyn_cast<cir::StructType>(maybePointerType.getPointee()))
          for (auto &tnp : typeNamePatterns)
            if (llvm::SmallVector<llvm::StringRef> matches;
                tnp.match(maybeStructType.getName(), &matches)) {
              value = {.fullName = matches[0].str(), .base = matches[1].str()};
              for (auto &e : llvm::ArrayRef(matches.begin() + 2, matches.end()))
                value->subMatches.emplace_back(e.str());
              // No need to look for a next match, go for the next type to
              // categorize
              break;
            }
    }
  }

  // Analysis called from pass getAnalysis()
  CIRToAIETypesAnalysis(mlir::ModuleOp module) {
    // First register all the types used in the module
    module->walk([this](mlir::Operation *op) {
      for (auto result : op->getResults()) {
        auto type = result.getType();
        moduleTypes.try_emplace(type, std::nullopt);
      }
    });
    // Deconstruct the aie++ C++ types found
    analyze();
    // If some AIE lowering has already be done in a previous pass, map an aie++
    // C++ type to the AIE operation generating such a value
    module->walk([this](mlir::UnrealizedConversionCastOp cast) {
      LLVM_DEBUG(cast.emitRemark("CIRToAIETypesAnalysis cast"));
      // Only a cast with 1 operand can have a potential AIE operation as
      // operand
      if (cast.getNumOperands() == 1) {
        auto type = cast.getType(0);
        // If this is an aie++ type
        if (auto &detail = getOptionalTypeDetail(type)) {
          auto *newOperation = cast.getOperand(0).getDefiningOp();
          auto dialectNamespace = newOperation->getDialect()->getNamespace();
          LLVM_DEBUG(cast.emitRemark("CIRToAIETypesAnalysis cast operand "
                                     "with  dialect namespace ")
                     << dialectNamespace);
          // If the operation producing the value is in AIE dialect, the aie++
          // type has already been translated, so record the translation for
          // this aie++ type
          if (dialectNamespace == "aie") {
            LLVM_DEBUG(
                newOperation->emitRemark(
                    "CIRToAIETypesAnalysis adding newAIEOperation");
                cast->emitRemark("CIRToAIETypesAnalysis adding newProducer"));
            detail.value().newAIEOperation = newOperation;
            detail.value().newProducer = cast;
          }
        }
      }
    });
  }

  // Get the deconstructed AIE type details behind the aie++ C++ type
  std::optional<AIELikeTypesDeconstruction> &
  getOptionalTypeDetail(mlir::Type t) {
    assert(moduleTypes.contains(t) && "This type should have been seen");
    return moduleTypes[t];
  }

  // Get the deconstructed AIE type details behind the aie++ C++ type
  AIELikeTypesDeconstruction &getTypeDetail(mlir::Type t) {
    auto &detail = getOptionalTypeDetail(t);
    assert(detail && "This type should have an analysis");
    return detail.value();
  }

  // Associate to a given aie++ C++ type the lowered AIE operation operation and
  // add the type to the operation as a "cir.type" type attribute for any later
  // introspection
  void setProducerOp(mlir::Type t, mlir::Operation *op, mlir::OpBuilder &b) {
    // Keep the original aie++ type for this AIE operation with a "cir.type"
    // attribute
    op->setAttr("cir.type", mlir::TypeAttr::get(t));
    auto &detail = getTypeDetail(t);
    detail.newAIEOperation = op;
    isAIELoweredType.insert(t);
  }

  // Associate to a given aie++ C++ type the operation producing the value for
  // this type
  void setProducerOpWithUCCast(mlir::Type t, mlir::Operation *op,
                               mlir::OpBuilder &b) {
    setProducerOp(t, op, b);
    auto &detail = getTypeDetail(t);
    detail.newAIEOperation = op;
    detail.newProducer = b.create<mlir::UnrealizedConversionCastOp>(
        op->getLoc(), t, mlir::ValueRange{op->getResult(0)});
  }

  // Get the optional operation producing the value for the given aie++ C++ type
  auto &getProducerOp(mlir::Type t) { return getTypeDetail(t).newProducer; }

  // Get the set of aie++ C++ types which have been lowered to an AIE operation
  // producing a value related to that type
  auto &getAIELoweredTypes() { return isAIELoweredType; }

  // Return true if the given type has a matching AIE operation to produce a
  // value related to that type
  bool isAIELowered(mlir::Type t) { return isAIELoweredType.contains(t); }

  // Visit recursively from a given root operation any operand with an
  // AIE-like C++ datatype
  template <typename FunctionRef>
  void visitAIEOperands(mlir::Operation *root, FunctionRef &&callBack) {
    root->walk([&](mlir::Operation *op) {
      for (auto &operand : op->getOpOperands()) {
        auto type = operand.get().getType();
        if (this->isAIELowered(type)) {
          LLVM_DEBUG(op->emitRemark("visitAIEOperands") << type);
          callBack(operand);
        }
      }
    });
  }

  // Display the analysis content
  void dump() {
    for (auto &[type, value] : moduleTypes) {
      llvm::outs() << "Type: " << type << " value: ";
      if (value)
        value->dump();
      else
        llvm::outs() << "None\n";
    }
  }
};

namespace {

// Return true if the call operation calls a function with any of the given
// string annotations
bool isCallingFunctionWithAnnotation(
    cir::CallOp op, llvm::ArrayRef<llvm::StringRef> anyAnnotations) {
  if (auto calledFunc = mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(
          op, op.getCalleeAttr())) {
    if (auto annnotations = calledFunc.getAnnotationsAttr())
      for (auto a : calledFunc.getAnnotationsAttr()) {
        for (auto one : anyAnnotations)
          if (mlir::cast<cir::AnnotationAttr>(a).getName() == one)
            return true;
      }
  }
  return false;
}

// Return true if the UnrealizedConversionCast operation has any of the given
// string annotations
bool isUnrealizedConversionCastWithAnnotation(
    mlir::UnrealizedConversionCastOp op,
    llvm::ArrayRef<llvm::StringRef> anyAnnotations) {
  for (auto attr : op->getAttrDictionary())
    for (auto needle : anyAnnotations)
      if (attr.getName() == needle)
        return true;
  return false;
}

// Generate the equivalent memref type of an aie::buffer
mlir::MemRefType bufferMemrefType(mlir::Type buffer) {
  static mlir::TypeConverter typeConverter = cir::prepareTypeConverter();
  LLVM_DEBUG(buffer.dump());
  if (auto p = mlir::dyn_cast<cir::PointerType>(buffer)) {
    if (auto bufferType = mlir::dyn_cast<cir::StructType>(p.getPointee())) {
      LLVM_DEBUG(bufferType.dump());
      // For now the aie::buffer is implemented as a std::array in the buffer
      // struct
      auto members = bufferType.getMembers();
      if (auto stdArrayType =
              mlir::dyn_cast<cir::StructType>(members.front())) {
        LLVM_DEBUG(stdArrayType.dump());
        // Access the array inside the std::array struct
        if (auto arrayType = mlir::dyn_cast<cir::ArrayType>(
                stdArrayType.getMembers().front())) {
          LLVM_DEBUG(arrayType.dump());
          auto memref = mlir::dyn_cast<mlir::MemRefType>(
              typeConverter.convertType(arrayType));
          LLVM_DEBUG(memref.dump());
          return memref;
        }
      }
    }
  }
  return {};
}

// Since an aie.device has its own symbol table, copy recursively all the
// symbols defined at the module level which are referenced by operations inside
// an aie.device into the aie.device.
void cloneReferencedSymbolsIntoDevice(xilinx::AIE::DeviceOp device) {
  // Speed-up symbol look-ups by defining some SymbolTable
  mlir::SymbolTable deviceSymbolTable{device};
  auto module = device->getParentOfType<mlir::ModuleOp>();
  mlir::SymbolTable moduleSymbolTable{module};
  mlir::OpBuilder builder{device};
  // Look recursively starting from the aie.device itself
  std::queue<mlir::Operation *> toVisit{{device}};
  while (!toVisit.empty()) {
    auto *opToVisit = toVisit.front();
    toVisit.pop();
    opToVisit->walk([&](mlir::Operation *op) {
      // Only look at the operations using some symbols
      if (auto user = mlir::dyn_cast<mlir::SymbolUserOpInterface>(op)) {
        LLVM_DEBUG(op->emitRemark(
            "importCalledFunctionsInSymbolTable: SymbolUserOpInterface!"));
        // Look for all the symbol references used by this operation
        op->getAttrDictionary().walk([&](mlir::SymbolRefAttr symbolRef) {
          LLVM_DEBUG(
              op->emitRemark("importCalledFunctionsInSymbolTable: symbolRef = ")
              << symbolRef);
          if (deviceSymbolTable.lookup(symbolRef.getRootReference())) {
            LLVM_DEBUG(llvm::outs() << "In Device!\n");
            // No need to visit it again if it is already in the device
            return;
          }
          // Get the referenced operation from the module symbol table
          auto *moduleSymbol =
              moduleSymbolTable.lookup(symbolRef.getRootReference());
          assert(moduleSymbol && "The symbol should be found in the module");
          LLVM_DEBUG(llvm::outs() << "In Module!\n"; moduleSymbol->emitRemark(
              "importCalledFunctionsInSymbolTable: cloning..."));
          // Newly discovered function not already in the device is used by
          // existing code and do not refer // TODO: o internal code, so add
          // it at the beginning inside the aie.device
          builder.setInsertionPointToStart(device.getBody());
          auto *clone = builder.clone(*moduleSymbol);
          deviceSymbolTable.insert(clone);
          LLVM_DEBUG(
              clone->emitRemark("importCalledFunctionsInSymbolTable: clone"));
          // Need to handle any missing symbols from the newly created
          // operation
          toVisit.push(clone);
        });
      }
    });
  }
}

// Lower C++ code like \code aie::device<aie::npu1> into an \code
// aie.device(npu1){} operation
struct PrepareDeviceLowering : public mlir::OpConversionPattern<cir::AllocaOp> {
  using mlir::OpConversionPattern<cir::AllocaOp>::OpConversionPattern;

  // \todo Find a less ugly way to access the analysis. How is it possible for a
  // pattern to access some contextual information?
  // It should be OK since it is a module pass, so no parallelism here.
  static inline CIRToAIETypesAnalysis *cat;

  mlir::LogicalResult
  matchAndRewrite(cir::AllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // The struct has a name like "aie::device<aie::npu1>" and the "npu1"
    // is used directly for the MLIR aie.device attribute
    if (auto aieLike = cat->getOptionalTypeDetail(op.getType());
        aieLike && aieLike->base == "aie::device") {
      auto deviceName = aieLike->subMatches[0];
      auto deviceId =
          xilinx::AIE::symbolizeEnum<xilinx::AIE::AIEDevice>(deviceName);
      if (!deviceId)
        // Actually this test cannot happens since the API of
        // xilinx::AIE::symbolizeEnum is strange: even if it returns a
        // std::optional it errors without returning
        op.emitError() << "aie::device incorrect for '" << deviceName << "'";
      // Replace the alloca of the aie::device by a temporary cast from
      // thin air and add a named attribute to the device name to make things
      // clearer
      rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
          op, op.getResult().getType(), mlir::ValueRange{},
          std::array{rewriter.getNamedAttr(
              aieLike->base, rewriter.getAttr<mlir::StringAttr>(deviceName))});
      return mlir::success();
    }
    return mlir::failure();
  }
};

// clang-format off
// Rewrite something like
//    %2 = cir.alloca !ty_aie3A3Atile3C12C_43E, !cir.ptr<!ty_aie3A3Atile3C12C_43E>, ["t", init] {alignment = 1 : i64} loc(#loc102)
//    %4 = cir.call @_ZN3aie6deviceILNS_3$_0E42EE4tileILi1ELi4EEENS_4tileIXT_EXT0_EEEv(%1) : (!cir.ptr<!ty_aie3A3Adevice3Caie3A3Anpu13E>) -> !ty_aie3A3Atile3C12C_43E loc(#loc70)
//    cir.store %4, %2 : !ty_aie3A3Atile3C12C_43E, !cir.ptr<!ty_aie3A3Atile3C12C_43E> loc(#loc70)
//
// Into
//
//    %2 = builtin.unrealized_conversion_cast %1 : !cir.ptr<!ty_aie3A3Adevice3Caie3A3Anpu13E> to !cir.ptr<!ty_aie3A3Atile3C12C_43E> {"aie::tile" = ["1", "4"]}
// clang-format on
struct PrepareTileBufferLowering
    : public mlir::OpConversionPattern<cir::CallOp> {
  using mlir::OpConversionPattern<cir::CallOp>::OpConversionPattern;

  // \todo Find a less ugly way to access the analysis. How is it possible for a
  // pattern to access some contextual information?
  // It should be OK since it is a module pass, so no parallelism here.
  static inline CIRToAIETypesAnalysis *cat;

  mlir::LogicalResult
  matchAndRewrite(cir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (isCallingFunctionWithAnnotation(
            op, {"aie.device.tile", "aie.tile.buffer"})) {
      auto device = op.getOperand(0);
      auto user = op.getResult().getUsers().begin();
      // Track the alloca where the tiled is stored
      auto store = mlir::dyn_cast<cir::StoreOp>(*user);
      auto alloca =
          mlir::dyn_cast<cir::AllocaOp>(store.getOperand(1).getDefiningOp());
      auto aieLike = cat->getTypeDetail(alloca.getResult().getType());
      // Replace the alloca by a conversion to be replaced later in
      // another pass.
      // Keep analyzed type information as named attribute to make things
      // clearer
      llvm::SmallVector<mlir::Attribute, 4> attrs;
      for (auto e : aieLike.subMatches)
        attrs.emplace_back(rewriter.getAttr<mlir::StringAttr>(e));
      rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
          alloca, alloca.getResult().getType(), device,
          std::array{rewriter.getNamedAttr(aieLike.base,
                                           rewriter.getArrayAttr(attrs))});
      // Remove the now useless original operations
      rewriter.eraseOp(store);
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return mlir::failure();
  }
};

/*
  Replace the call to

  cir.func internal private
 @_ZN3aie6tile_tILi1ELi4EE7programIZ4mainE3$_0EEvOT_(%arg0:
 !cir.ptr<!ty_aie3A3Atile_t3C12C_43E>, %arg1: !cir.ptr<!ty_anon2E0_>)
 [#cir.annotation<name = "aie.tile.program", args = []>] extra(#fn_attr)

 which ends up calling the lambda

 cir.call @_ZZ4mainENK3$_0clEv(%5) : (!cir.ptr<!ty_anon2E0_>) -> ()

 by just inlining the lambda body into the aie.core operation and replacing the
 capture by the direct def/use forwarding

*/
struct PrepareCoreLowering : public mlir::OpConversionPattern<cir::CallOp> {
  using mlir::OpConversionPattern<cir::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (isCallingFunctionWithAnnotation(op, {"aie.tile.program"})) {
      // Get tile::program() member function
      if (auto calledFunc =
              mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(
                  op, op.getCalleeAttr())) {
        // The last function instruction is cir.return and the one before
        // is the call to the lambda
        // calledFunc.getBlocks().front().back().dump();
        auto lambdaCall = mlir::dyn_cast<cir::CallOp>(
            *std::next(calledFunc.getBlocks().front().rbegin()));
        // lambdaCall.dump();
        if (auto lambdaFunc =
                mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(
                    lambdaCall, lambdaCall.getCalleeAttr())) {
          // lambdaFunc.dump();
          assert(lambdaFunc.getLambda());
          // auto scopeOp = op->getParentOfType<cir::ScopeOp>();
          // scopeOp.dump();
          //  The aie++ tile value
          rewriter.setInsertionPoint(op);
          rewriter.eraseOp(op);
          //        rewriter.insert(coreOp);
          // coreOp.dump();

          // auto bs = lambdaFunc.getBlocks().begin();
          //          rewriter.inlineBlockBefore(Block *source, Block *dest,
          //          Block::iterator before)
          return mlir::success();
        }
      }
    }

    return mlir::failure();
  }
};

struct CIRToAIEPrepare : CIRToAIEPrepareBase<CIRToAIEPrepare> {
  void runOnOperation() override {
    // Compute the analysis for the module since it is a module pass.
    // \todo Should this be a real pass?
    auto &cat = getAnalysis<CIRToAIETypesAnalysis>();
    // \todo Clean up this mess
    PrepareDeviceLowering::cat = &cat;
    PrepareTileBufferLowering::cat = &cat;
    // See mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp
    mlir::ConversionTarget target{getContext()};
    target.addLegalDialect<xilinx::AIE::AIEDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<cir::AllocaOp>([&](cir::AllocaOp op) {
      // If the struct has a name like "aie::device<aie::npu1>", mark
      // the operation illegal so it has to be rewritten
      auto aieLike = cat.getOptionalTypeDetail(op.getType());
      return !(aieLike && aieLike->base == "aie::device");
    });
    target.addDynamicallyLegalOp<cir::CallOp>([](cir::CallOp op) {
      return !isCallingFunctionWithAnnotation(
          op, {"aie.device.tile", "aie.tile.buffer"});
    });
    mlir::RewritePatternSet patterns{&getContext()};
    patterns.add<PrepareDeviceLowering>(&getContext());
    patterns.add<PrepareTileBufferLowering>(&getContext());
    //    patterns.add<PrepareCoreLowering>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

struct CIRToAIE : CIRToAIEBase<CIRToAIE> {
  // \todo Find a less ugly way to access the analysis. How is it possible for a
  // pattern to access some contextual information?
  // It should be OK since it is a module pass, so no parallelism here.
  static inline CIRToAIETypesAnalysis *cat;

  // Try to lower the operation as an aie.buffer and return true on success
  //
  // clang-format off
  // %3 = builtin.unrealized_conversion_cast %2 : !cir.ptr<!ty_aie3A3Atile3C12C_42C_aie3A3Adevice3C3E3E> to !cir.ptr<!ty_aie3A3Abuffer3Cint2C_8192UL3E> {"aie::buffer" = ["int", "8192UL"]}
  // is lowered to
  // %buffer_1_4 = aie.buffer(%tile_1_4) : memref<8192xi32>
  // %8 = builtin.unrealized_conversion_cast %buffer_1_4 : memref<8192xi32> to !cir.ptr<!ty_aie3A3Abuffer3Cint2C_8192UL3E>
  // clang-format on
  bool tryBufferLowering(mlir::Operation *op, mlir::OpBuilder &b) {
    if (auto bufCast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
      if (auto bufferDetail = cat->getTypeDetail(bufCast.getType(0));
          bufferDetail.base == "aie::buffer") {
        LLVM_DEBUG(bufCast.emitRemark("Buffer cast from tile"));
        auto mrt = bufferMemrefType(bufCast.getType(0));
        // \todo outline
        auto tileDetail = cat->getTypeDetail(bufCast.getOperand(0).getType());
        auto tileOp =
            mlir::dyn_cast<xilinx::AIE::TileOp>(*tileDetail.newAIEOperation);
        if (!tileOp)
          bufCast->emitError("No aie.device operation found for this tile");
        // Insert at the end of the aie.device to keep C++ program order
        auto deviceOp = tileOp->getParentOfType<xilinx::AIE::DeviceOp>();
        b.setInsertionPoint(deviceOp.getBody()->getTerminator());
        // The direct connection to tileOp is a peephole optimization but it
        // could be connected to the new tileOp UnrealizedConversionCastOp which
        // could be removed later by a cleaning phase
        auto bufferOp = b.create<xilinx::AIE::BufferOp>(bufCast.getLoc(), mrt,
                                                        tileOp.getResult());
        // Keep track of the buffer op behind the C++ type
        cat->setProducerOpWithUCCast(bufCast.getType(0), bufferOp, b);
        // Do not remap the old buffer users to the new one for now because the
        // new buffer is created inside the aie.device and the old users would
        // not be able to access it. Rely on the tile::program lowering for this
        // later.

        // The lowering is a success, no need to look further.
        return true;
      }
    }
    // Notify to try something else
    return false;
  }

  // During operation cloning, mlir::IRMapping is used to remap some leaf input
  // operands but cannot remap some internal ones. In some case, ClangIR lower
  // some lambda captures with aie::tile or with aie::device (). Having the
  // aie::device is problematic since it is remapped to the aie.device output
  // which leads to 2 issues for the verifyer:
  //
  // - this use is inside the device operation itself
  //
  // - the aie.device region is isolated from above.
  //
  // Since this aie::device is used only by an aie::tile, just remove the
  // aie::device part.
  void resolveSomeDeviceToTileAfterCloning(mlir::Operation *clone) {
    llvm::SmallVector<mlir::Operation *> oldCastsFromDevice;
    cat->visitAIEOperands(clone, [&](mlir::OpOperand &operand) {
      if (cat->getTypeDetail(operand.get().getType()).base == "aie::device") {
        auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(
            operand.getOwner());
        assert(cast && "There should be only an UnrealizedConversionCastOp "
                       "using the aie::device");
        // Connect directly the aie::tile user to the one produced by the
        // matching aie.tile
        cast.replaceAllUsesWith(cat->getProducerOp(cast.getType(0)).value());
        oldCastsFromDevice.push_back(cast);
      }
    });
    // Remove the problematic operations
    eraseOpsAndUsers(oldCastsFromDevice);
  }

  // Lower aie::tile::program(<tile code>) to aie.core
  bool tryTileProgramLowering(mlir::Operation *op, mlir::OpBuilder &b) {
    if (auto callOp = mlir::dyn_cast<cir::CallOp>(op)) {
      LLVM_DEBUG(
          callOp.emitRemark("tryTileProgramLowering: CallOp using a tile"));
      if (isCallingFunctionWithAnnotation(callOp, {"aie.tile.program"})) {
        LLVM_DEBUG(
            callOp.emitRemark("tryTileProgramLowering: CallOp using a tile"));
        if (auto calledFunc =
                mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(
                    callOp, callOp.getCalleeAttr())) {
          // The last function instruction is cir.return and the one before
          // is the call to the lambda
          if (auto lambdaCall = mlir::dyn_cast<cir::CallOp>(
                  *std::next(calledFunc.getBlocks().front().rbegin()))) {
            LLVM_DEBUG(lambdaCall.emitRemark("lambdaCall"));
            if (auto lambdaFunc =
                    mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(
                        lambdaCall, lambdaCall.getCalleeAttr())) {
              LLVM_DEBUG(lambdaFunc.emitRemark(
                  "tryTileProgramLowering: Tile core lambda"));
              assert(lambdaFunc.getLambda());
              auto scopeOp = callOp->getParentOfType<cir::ScopeOp>();
              LLVM_DEBUG(scopeOp.emitRemark("tryTileProgramLowering: Scope"));
              // \todo outline
              auto tileDetail =
                  cat->getTypeDetail(callOp.getOperand(0).getType());
              auto tileOp = mlir::dyn_cast<xilinx::AIE::TileOp>(
                  *tileDetail.newAIEOperation);
              if (!tileOp)
                LLVM_DEBUG(callOp->emitError(
                    "No aie.device operation found for this tile"));
              // Create the aie.core before the aie.end of the aie.device body
              // to keep the C++ order
              auto deviceOp = tileOp->getParentOfType<xilinx::AIE::DeviceOp>();
              b.setInsertionPoint(deviceOp.getBody()->getTerminator());

              auto coreOp =
                  b.create<xilinx::AIE::CoreOp>(callOp.getLoc(), tileOp);
              // Create the empty block of the aie.core op region
              coreOp.getRegion().emplaceBlock();
              // The aie.core requires a terminator
              b.setInsertionPointToEnd(&coreOp.getRegion().front());
              // Add right away an aie.end to have the verifyers happy even if
              // it makes the following more complicated
              b.create<xilinx::AIE::EndOp>(callOp.getLoc());
              LLVM_DEBUG(
                  coreOp.emitRemark("tryTileProgramLowering: Brand-new core"));
              // Get the cast connecting the aie.tile to the lambda call
              auto tileCastOp =
                  callOp.getArgOperand(0)
                      .getDefiningOp<mlir::UnrealizedConversionCastOp>();
              LLVM_DEBUG(
                  tileCastOp.emitRemark("tryTileProgramLowering: tileCastOp as "
                                        "callOp first argument"));
              // Values can be replaced while cloning, not operations
              mlir::IRMapping irm;
              // Compute the remapping to be done while cloning from the old
              // operands to the new one produced by the lowered AIE operations
              cat->visitAIEOperands(scopeOp, [&](auto &operand) {
                // Remap only if there is interesting result. Skip aie.device
                // for example
                if (auto producer = cat->getProducerOp(operand.get().getType()))
                  irm.map(operand.get(), producer.value()->getResult(0));
              });
              b.setInsertionPointToStart(&coreOp.getRegion().front());
              auto *clone = b.clone(*scopeOp.getOperation(), irm);
              // Since aie.device has a SymbolTable, all the called functions
              // need to be present in the aie.device
              cloneReferencedSymbolsIntoDevice(
                  clone->getParentOfType<xilinx::AIE::DeviceOp>());
              LLVM_DEBUG(clone->emitRemark("tryTileProgramLowering: Clone"));
              resolveSomeDeviceToTileAfterCloning(clone);
              LLVM_DEBUG(
                  coreOp.emitRemark("tryTileProgramLowering: Stuffed core");
                  coreOp->getParentOfType<cir::FuncOp>().emitRemark(
                      "tryTileProgramLowering: Top function"));
            }
          }
          // The bufCast should be removed as a dependent of tile cast
          // later. The lowering is a success, no need to look further.
          return true;
        }
      }
    }
    // Notify to try something else
    return false;
  }

  // Try to lower the operation as an aie.tile and return true on success
  //
  // clang-format off
  // %2 = builtin.unrealized_conversion_cast %1 : !cir.ptr<!ty_aie3A3Adevice3Caie3A3Anpu12C_aie3A3A28lambda_at_2E2Faie2B2B2Ehpp3A1453A56293E> to !cir.ptr<!ty_aie3A3Atile3C12C_42C_aie3A3Adevice3C3E3E> {"aie::tile" = ["1", "4"]}
  // is lowered to
  // %tile_1_4 = aie.tile(1, 4)
  // %7 = builtin.unrealized_conversion_cast %tile_1_4 : index to !cir.ptr<!ty_aie3A3Atile3C12C_42C_aie3A3Adevice3C3E3E>
  // clang-format on
  bool tryTileLowering(mlir::Operation *op, mlir::OpBuilder &b) {
    if (auto tileCast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
      auto tileCastOutputType = tileCast.getType(0);
      if (auto detail = cat->getTypeDetail(tileCastOutputType);
          detail.base == "aie::tile") {
        auto col = detail.subMatches[0];
        auto row = detail.subMatches[1];
        LLVM_DEBUG(tileCast.emitRemark("tryTileLowering: tileCast from device")
                   << ", col = " << col << ", row = " << row);
        auto deviceDetail =
            cat->getTypeDetail(tileCast.getOperand(0).getType());
        auto deviceOp = mlir::dyn_cast<xilinx::AIE::DeviceOp>(
            *deviceDetail.newAIEOperation);
        if (!deviceOp)
          tileCast->emitError("No aie.device operation found for this tile");
        // Create all the following code inside the device region. Add the tile
        // to the end to keep C++ program order.
        b.setInsertionPoint(deviceOp.getBody()->getTerminator());
        auto tileOp = b.create<xilinx::AIE::TileOp>(
            tileCast.getLoc(), std::stoi(col), std::stoi(row));
        cat->setProducerOpWithUCCast(tileCastOutputType, tileOp, b);
        // Do not remap the old tile users to the new one for now because the
        // new tile is created inside the aie.device and the old users would not
        // be able to access it. Rely on the tile::program lowering for this
        // later.

        // Tile lowering is done
        return true;
      }
    }
    // Try some other lowering
    return false;
  }

  // Try to lower the operation as an aie.device and return true on success
  //
  // clang-format off
  // %1 = builtin.unrealized_conversion_cast to !cir.ptr<!ty_aie3A3Adevice3Caie3A3Anpu12C_aie3A3A28lambda_at_2E2Faie2B2B2Ehpp3A1453A56293E> {"aie::device" = "npu1"}
  // is lowered to
  // %1 = aie.device(npu1) {
  // }
  // %2 = builtin.unrealized_conversion_cast %1 : index to !cir.ptr<!ty_aie3A3Adevice3Caie3A3Anpu12C_aie3A3A28lambda_at_2E2Faie2B2B2Ehpp3A1453A56293E>
  // clang-format on
  bool tryDeviceLowering(mlir::Operation *op, mlir::OpBuilder &b) {
    if (auto u = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
      if (!isUnrealizedConversionCastWithAnnotation(u, {"aie::device"}))
        // Try some other lowering
        return false;
      LLVM_DEBUG(u.emitRemark(
          "DeviceLowering found UnrealizedConversionCastOp inside the module"));
      auto aieLike = cat->getTypeDetail(u.getType(0));
      auto deviceName = aieLike.subMatches[0];
      auto deviceId =
          xilinx::AIE::symbolizeEnum<xilinx::AIE::AIEDevice>(deviceName);
      if (!deviceId)
        // Actually this test cannot happens since the API of
        // xilinx::AIE::symbolizeEnum is strange: even if it returns a
        // std::optional it errors without returning
        u.emitError("aie::device incorrect for '") << deviceName << "'";
      // Create an aie.device just before its equivalent
      // UnrealizedConversionCast. Since we visit in pre-order mode, this
      // should be fine.
      b.setInsertionPoint(u);
      auto deviceOp = b.create<xilinx::AIE::DeviceOp>(u.getLoc(), *deviceId);
      // The aie.device requires one block and a terminator
      b.setInsertionPointToEnd(&deviceOp.getRegion().emplaceBlock());
      b.create<xilinx::AIE::EndOp>(u.getLoc());
      // Keep for now the UnrealizedConversionCastOp for the aie.device since
      // aie.device do not returns value
      cat->setProducerOp(u.getType(0), deviceOp, b);
      // Note: aie.device does not require a terminator
      LLVM_DEBUG(deviceOp.emitRemark("DeviceLowering: end"));
      return true;
    }
    return false;
  }

  void runOnOperation() override {
    // Compute the analysis for the module since it is a module pass.
    cat = &getAnalysis<CIRToAIETypesAnalysis>();
    auto module = getOperation();
    mlir::OpBuilder b{module};
    // Use pre-order walk to keep the C++ ordered semantics while lowering the
    // AIE constructs
    module->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
      tryBufferLowering(op, b) || tryDeviceLowering(op, b) ||
          tryTileLowering(op, b) || tryTileProgramLowering(op, b);
    });
  }
};

// Inline the kernel lambda and its calling functions found in an aie.core
// operation
struct CIRToAIEInlineKernelLambda
    : CIRToAIEInlineKernelLambdaBase<CIRToAIEInlineKernelLambda> {
  // \todo Find a less ugly way to access the analysis. How is it possible for a
  // pattern to access some contextual information?
  // It should be OK since it is a module pass, so no parallelism here.
  static inline CIRToAIETypesAnalysis *cat;

  static void inlineAndEraseCall(cir::CallOp call, cir::FuncOp calledFunc) {
    LLVM_DEBUG(auto *entryBlock = &calledFunc.getCallableRegion()->front();
               calledFunc.emitRemark("CIRToAIEInlineKernelLambda calledFunc")
               << "call.getNumOperands()" << call.getNumOperands()
               << "entryBlock->getNumArguments()"
               << entryBlock->getNumArguments() << "call.getNumResults()"
               << call.getNumResults() << "calledFunc.getResultTypes().size()"
               << calledFunc.getResultTypes().size()
               << "calledFunc.getNumResults()" << calledFunc.getNumResults()
               << "calledFunc.getCallableResults().size()"
               << calledFunc.getCallableResults().size();
               calledFunc.getResultTypes()[0].dump());
    mlir::InlinerInterface interface{call.getContext()};
    if (mlir::inlineCall(interface, call, calledFunc,
                         calledFunc.getCallableRegion())
            .failed())
      call.emitError("CIRToAIEInlineKernelLambdaBase not able to "
                     "inline the lambda call");
    call.erase();
  }

  void runOnOperation() override {
    // Compute the analysis for the module since it is a module pass.
    cat = &getAnalysis<CIRToAIETypesAnalysis>();
    auto module = getOperation();
    mlir::OpBuilder b{module};
    // Use pre-order walk to keep the C++ ordered semantics while lowering the
    // AIE constructs
    module->walk<mlir::WalkOrder::PreOrder>([&](xilinx::AIE::CoreOp core) {
      LLVM_DEBUG(core.emitRemark("CIRToAIEInlineKernelLambda aie.core"));
      if (auto scope =
              mlir::dyn_cast<cir::ScopeOp>(core.getBody().front().front())) {
        LLVM_DEBUG(scope.emitRemark("CIRToAIEInlineKernelLambda cir.scope"));
        if (auto call = mlir::dyn_cast<cir::CallOp>(
                *std::next(scope.getScopeRegion().front().rbegin()))) {
          LLVM_DEBUG(call.emitRemark("CIRToAIEInlineKernelLambda call"));
          if (auto calledFunc =
                  mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(
                      call, call.getCalleeAttr())) {
            // The last function instruction is cir.return and the one before
            // is the call to the lambda
            if (auto lambdaCall = mlir::dyn_cast<cir::CallOp>(
                    *std::next(calledFunc.getBlocks().front().rbegin()))) {
              LLVM_DEBUG(lambdaCall.emitRemark("lambdaCall"));
              if (auto lambdaFunc =
                      mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(
                          lambdaCall, lambdaCall.getCalleeAttr())) {
                LLVM_DEBUG(lambdaFunc.emitRemark(
                    "CIRToAIEInlineKernelLambda: Tile core lambda"));
                if (lambdaFunc.getLambda()) {
                  inlineAndEraseCall(call, calledFunc);
                  LLVM_DEBUG(core.emitRemark(
                      "CIRToAIEInlineKernelLambda: core after first inlining"));
                  if (auto finalCall = mlir::dyn_cast<cir::CallOp>(*std::next(
                          scope.getScopeRegion().front().rbegin()))) {
                    inlineAndEraseCall(
                        finalCall,
                        mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(
                            finalCall, finalCall.getCalleeAttr()));
                  }
                }
              }
            }
          }
        }
      }
      // No need to dive further this aie.core operation since they cannot be
      // nested
      return mlir::WalkResult::skip();
    });
  }
};

struct CIRToAIEDecaptureKernel
    : CIRToAIEDecaptureKernelBase<CIRToAIEDecaptureKernel> {
  void runOnOperation() override {
    auto module = getOperation();
    mlir::OpBuilder b{module};
    // Use pre-order walk for early exit
    module->walk<mlir::WalkOrder::PreOrder>([&](xilinx::AIE::CoreOp core) {
      LLVM_DEBUG(core.emitRemark("CIRToAIEDecaptureKernel aie.core"));
      if (auto alloca = mlir::dyn_cast_if_present<cir::AllocaOp>(
              findFirstNonCIRScopeOpInside(core))) {
        LLVM_DEBUG(alloca.emitRemark("CIRToAIEInlineKernelLambda: alloca"));
        // Track the value loaded from or stored into each capture member
        llvm::DenseMap<llvm::StringRef, mlir::Value> loads, stores;
        for (auto *u : alloca.getResult().getUsers())
          if (auto gm = mlir::dyn_cast<cir::GetMemberOp>(u)) {
            auto memberName = gm.getName();
            llvm::TypeSwitch<mlir::Operation *>(
                *gm.getResult().getUsers().begin())
                .Case(
                    [&](cir::StoreOp s) { stores[memberName] = s.getValue(); })
                .Case([&](cir::LoadOp l) { loads[memberName] = l.getResult(); })
                .Default([&](auto op) {
                  op->emitError(
                      "CIRToAIEInlineKernelLambda unknown user for member ")
                      << memberName;
                });
          } else
            u->emitError("CIRToAIEInlineKernelLambda unknown use for alloca");
        // Connect directly all the capture read users to the capture stored
        // values
        for (auto &&[memberName, value] : loads)
          value.replaceAllUsesWith(stores[memberName]);
        // Remove all the lambda capture leftover
        eraseOpsAndUsers(alloca);
      }
      // No need to dive further this aie.core operation since they cannot be
      // nested
      return mlir::WalkResult::skip();
    });
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCIRToAIEPreparePass() {
  return std::make_unique<CIRToAIEPrepare>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createCIRToAIEPass() {
  return std::make_unique<CIRToAIE>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCIRToAIEInlineKernelLambdaPass() {
  return std::make_unique<CIRToAIEInlineKernelLambda>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCIRToAIEDecaptureKernelPass() {
  return std::make_unique<CIRToAIEDecaptureKernel>();
}

} // namespace xilinx::AIE::CIR
