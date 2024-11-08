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
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/LowerToMLIR.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

using namespace std::string_literals;

namespace xilinx::AIE::CIR {

// Analysis all the C++ types used in a module and for aie++ types deconstruct
// them and keep track of the AIE dialect operations used to produce a value of
// its type.
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
    // The new operation producing the result instead for replacement
    std::optional<mlir::Operation *> newProducer;

    std::string str() {
      return "Fullname = " + fullName + ", base = " + base +
             ", subMatches = " + llvm::join(subMatches, ", ");
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
      if (auto maybePointerType = mlir::dyn_cast<mlir::cir::PointerType>(type))
        if (auto maybeStructType = mlir::dyn_cast<mlir::cir::StructType>(
                maybePointerType.getPointee()))
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
    return *detail;
  }

  // Associate to a given aie++ C++ type the operation producing the value for
  // this type
  void setProducerOpWithUCCast(mlir::Type t, mlir::Operation *op,
                               mlir::OpBuilder &b) {
    auto cast = b.create<mlir::UnrealizedConversionCastOp>(
        op->getLoc(), t, mlir::ValueRange{op->getResult(0)});
    getTypeDetail(t).newProducer = cast;
    isAIELoweredType.insert(t);
  }

  // Associate to a given aie++ C++ type the operation producing the value for
  // this type
  mlir::Operation *getProducerOp(mlir::Type t) {
    auto &detail = getTypeDetail(t);
    assert(detail.newProducer &&
           "This type should have an operation registered "
           "with a previous setProducerOp()");
    return *detail.newProducer;
  }

  // Get the set of aie++ C++ types which have been lowered to an AIE operation
  // producing a value related to that type
  auto &getAIELoweredTypes() { return isAIELoweredType; }

  // Return true if the given type has a matching AIE operation to produce a
  // value related to that type
  bool isAIELowered(mlir::Type t) { return isAIELoweredType.contains(t); }

  CIRToAIETypesAnalysis(mlir::ModuleOp module) {
    module->walk([this](mlir::Operation *op) {
      for (auto result : op->getResults()) {
        auto type = result.getType();
        moduleTypes.try_emplace(type, std::nullopt);
      }
    });
    analyze();
  }

  void dump() {
    for (auto &[type, value] : moduleTypes) {
      llvm::outs() << "Type: " << type << " value: ";
      if (value) {
        llvm::outs() << value->str() << '\n';
      } else
        llvm::outs() << "None\n";
    }
  }
};

namespace {

// Return true if the call operation calls a function with any of the given
// string annotations
bool isCallingFunctionWithAnnotation(
    mlir::cir::CallOp op, llvm::ArrayRef<llvm::StringRef> anyAnnotations) {
  if (auto calledFunc =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::cir::FuncOp>(
              op, op.getCalleeAttr())) {
    if (auto annnotations = calledFunc.getAnnotationsAttr())
      for (auto a : calledFunc.getAnnotationsAttr()) {
        for (auto one : anyAnnotations)
          if (mlir::cast<mlir::cir::AnnotationAttr>(a).getName() == one)
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
  buffer.dump();
  if (auto p = mlir::dyn_cast<mlir::cir::PointerType>(buffer)) {
    if (auto bufferType =
            mlir::dyn_cast<mlir::cir::StructType>(p.getPointee())) {
      bufferType.dump();
      // For now the aie::buffer is implemented as a std::array in the buffer
      // struct
      auto members = bufferType.getMembers();
      if (auto stdArrayType =
              mlir::dyn_cast<mlir::cir::StructType>(members.front())) {
        stdArrayType.dump();
        // Access the array inside the std::array struct
        if (auto arrayType = mlir::dyn_cast<mlir::cir::ArrayType>(
                stdArrayType.getMembers().front())) {
          arrayType.dump();
          auto memref = mlir::dyn_cast<mlir::MemRefType>(
              typeConverter.convertType(arrayType));
          memref.dump();
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
        op->emitRemark(
            "importCalledFunctionsInSymbolTable: SymbolUserOpInterface!");
        // Look for all the symbol references used by this operation
        op->getAttrDictionary().walk([&](mlir::SymbolRefAttr symbolRef) {
          op->emitRemark("importCalledFunctionsInSymbolTable: symbolRef = ")
              << symbolRef;
          if (deviceSymbolTable.lookup(symbolRef.getRootReference())) {
            llvm::errs() << "In Device!\n";
            // No need to visit it again if it is already in the device
            return;
          }
          // Get the referenced operation from the module symbol table
          auto *moduleSymbol =
              moduleSymbolTable.lookup(symbolRef.getRootReference());
          assert(moduleSymbol && "The symbol should be found in the module");
          llvm::errs() << "In Module!\n";
          moduleSymbol->emitRemark(
              "importCalledFunctionsInSymbolTable: cloning...");
          // Newly discovered function not already in the device is used by
          // existing code and do not refer // TODO: o internal code, so add
          // it at the beginning inside the aie.device
          builder.setInsertionPointToStart(device.getBody());
          auto *clone = builder.clone(*moduleSymbol);
          deviceSymbolTable.insert(clone);
          clone->emitRemark("importCalledFunctionsInSymbolTable: clone");
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
struct PrepareDeviceLowering
    : public mlir::OpConversionPattern<mlir::cir::AllocaOp> {
  using mlir::OpConversionPattern<mlir::cir::AllocaOp>::OpConversionPattern;

  // \todo Find a less ugly way to access the analysis. How is it possible for a
  // pattern to access some contextual information?
  // It should be OK since it is a module pass, so no parallelism here.
  static inline CIRToAIETypesAnalysis *cat;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::AllocaOp op, OpAdaptor adaptor,
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
    : public mlir::OpConversionPattern<mlir::cir::CallOp> {
  using mlir::OpConversionPattern<mlir::cir::CallOp>::OpConversionPattern;

  // \todo Find a less ugly way to access the analysis. How is it possible for a
  // pattern to access some contextual information?
  // It should be OK since it is a module pass, so no parallelism here.
  static inline CIRToAIETypesAnalysis *cat;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (isCallingFunctionWithAnnotation(
            op, {"aie.device.tile", "aie.tile.buffer"})) {
      auto device = op.getOperand(0);
      auto user = op.getResult().getUsers().begin();
      // Track the alloca where the tiled is stored
      auto store = mlir::dyn_cast<mlir::cir::StoreOp>(*user);
      auto alloca = mlir::dyn_cast<mlir::cir::AllocaOp>(
          store.getOperand(1).getDefiningOp());
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
struct PrepareCoreLowering
    : public mlir::OpConversionPattern<mlir::cir::CallOp> {
  using mlir::OpConversionPattern<mlir::cir::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (isCallingFunctionWithAnnotation(op, {"aie.tile.program"})) {
      // Get tile::program() member function
      if (auto calledFunc =
              mlir::SymbolTable::lookupNearestSymbolFrom<mlir::cir::FuncOp>(
                  op, op.getCalleeAttr())) {
        // The last function instruction is cir.return and the one before
        // is the call to the lambda
        // calledFunc.getBlocks().front().back().dump();
        auto lambdaCall = mlir::dyn_cast<mlir::cir::CallOp>(
            *std::next(calledFunc.getBlocks().front().rbegin()));
        // lambdaCall.dump();
        if (auto lambdaFunc =
                mlir::SymbolTable::lookupNearestSymbolFrom<mlir::cir::FuncOp>(
                    lambdaCall, lambdaCall.getCalleeAttr())) {
          // lambdaFunc.dump();
          assert(lambdaFunc.getLambda());
          // auto scopeOp = op->getParentOfType<mlir::cir::ScopeOp>();
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
    target.addDynamicallyLegalOp<mlir::cir::AllocaOp>(
        [&](mlir::cir::AllocaOp op) {
          // If the struct has a name like "aie::device<aie::npu1>", mark
          // the operation illegal so it has to be rewritten
          auto aieLike = cat.getOptionalTypeDetail(op.getType());
          return !(aieLike && aieLike->base == "aie::device");
        });
    target.addDynamicallyLegalOp<mlir::cir::CallOp>([](mlir::cir::CallOp op) {
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
    op->emitRemark("eraseOpsAndUsers: newOps.pop_back_val()");
    // If the operation has not been visited yet, add it to the set and process
    // its users
    if (allOpsAndUsers.insert(op)) {
      op->emitRemark("eraseOpsAndUsers: inserted!");
      for (auto result : op->getResults())
        for (auto *user : result.getUsers()) {
          // Add each user to the visit queue
          newOps.push_back(user);
          user->emitRemark("eraseOpsAndUsers: append to visit queue");
        }
    }
  }
  // To avoid erasing operations with remaining users, topologically sort the
  // operations according to their use-def chains and erase them in reverse
  // order
  for (auto *op : allOpsAndUsers)
    op->emitRemark("eraseOpsAndUsers: allOpsAndUsers");
  // Does not work here
  // auto sorted = mlir::topologicalSort(allOpsAndUsers);
  llvm::SmallVector<mlir::Operation *> sorted{allOpsAndUsers.begin(),
                                              allOpsAndUsers.end()};
  mlir::computeTopologicalSorting(sorted);
  for (auto *op : sorted)
    op->emitRemark("eraseOpsAndUsers: topologicalSort");
  for (auto *op : llvm::reverse(sorted)) {
    op->emitRemark("eraseOpsAndUsers: reverse");
    op->erase();
  }
}

struct CIRToAIE : CIRToAIEBase<CIRToAIE> {

  static inline CIRToAIETypesAnalysis *cat;

  bool tryBufferLowering(mlir::Operation *op, xilinx::AIE::TileOp tileOp,
                         mlir::OpBuilder &b,
                         llvm::SmallVector<mlir::Operation *> &opsToErase) {
    if (auto bufCast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
      bufCast.emitRemark("Buffer cast from tile");
      auto mrt = bufferMemrefType(bufCast.getType(0));
      // The direct connection to tileOp is a peephole optimization but it could
      // be connected to the new tileOp UnrealizedConversionCastOp which could
      // be removed later by a cleaning phase
      auto bufferOp = b.create<xilinx::AIE::BufferOp>(bufCast.getLoc(), mrt,
                                                      tileOp.getResult());
      // Keep track of the buffer op behind the C++ type
      cat->setProducerOpWithUCCast(bufCast.getType(0), bufferOp, b);
      // The bufCast should be removed as a dependent of its tile cast later,
      // but be redundant here for symmetry and to exercise the final erasing
      // and its topological sort
      opsToErase.push_back(bufCast);
      // The lowering is a success, no need to look further.
      return true;
    }
    // Notify to try something else
    return false;
  }

  // Lower aie::tile::program(<tile code>) to aie.core
  bool
  tryTileProgramLowering(mlir::Operation *op, xilinx::AIE::TileOp tileOp,
                         mlir::OpBuilder &b,
                         llvm::SmallVector<mlir::Operation *> &opsToErase) {
    if (auto callOp = mlir::dyn_cast<mlir::cir::CallOp>(op)) {
      callOp.emitRemark("tryTileProgramLowering: CallOp using a tile");
      if (isCallingFunctionWithAnnotation(callOp, {"aie.tile.program"})) {
        if (auto calledFunc =
                mlir::SymbolTable::lookupNearestSymbolFrom<mlir::cir::FuncOp>(
                    callOp, callOp.getCalleeAttr())) {
          // The last function instruction is cir.return and the one before
          // is the call to the lambda
          // calledFunc.getBlocks().front().back().dump();
          auto lambdaCall = mlir::dyn_cast<mlir::cir::CallOp>(
              *std::next(calledFunc.getBlocks().front().rbegin()));
          lambdaCall.emitRemark("lambdaCall");
          if (auto lambdaFunc =
                  mlir::SymbolTable::lookupNearestSymbolFrom<mlir::cir::FuncOp>(
                      lambdaCall, lambdaCall.getCalleeAttr())) {
            lambdaFunc.emitRemark("tryTileProgramLowering: Tile core lambda");
            assert(lambdaFunc.getLambda());
            auto scopeOp = callOp->getParentOfType<mlir::cir::ScopeOp>();
            scopeOp.emitRemark("tryTileProgramLowering: Scope");
            auto coreOp =
                b.create<xilinx::AIE::CoreOp>(callOp.getLoc(), tileOp);
            // Create the empty block of the aie.core op region
            coreOp.getRegion().emplaceBlock();
            // Do not mess up with current insertion point inside aie.tile
            mlir::OpBuilder::InsertionGuard _{b};
            // The aie.core requires a terminator
            b.setInsertionPointToEnd(&coreOp.getRegion().back());
            // Add right away an aie.end to have the verifyers happy even if it
            // makes the following more complicated
            b.create<xilinx::AIE::EndOp>(callOp.getLoc());
            coreOp.emitRemark("tryTileProgramLowering: Brand-new core");
            // Get the cast connecting the aie.tile to the lambda call
            auto tileCastOp =
                callOp.getArgOperand(0)
                    .getDefiningOp<mlir::UnrealizedConversionCastOp>();
            tileCastOp.emitRemark(
                "tryTileProgramLowering: tileCastOp as callOp first argument");
            // Values can be replaced while cloning, not operations
            mlir::IRMapping irm;
            // Compute the remapping to be done while cloning from the old
            // operands to the new one produced by the lowered AIE operations
            scopeOp.walk([&](mlir::Operation *op) {
              for (auto &operand : op->getOpOperands()) {
                auto type = operand.get().getType();
                if (cat->isAIELowered(type)) {
                  op->emitRemark("Mapping ")
                      << type << " to "
                      << cat->getProducerOp(type)->getResult(0);
                  if (cat->getTypeDetail(type).base == "aie::device")
                    // Do not bring in any device-dependent code since it would
                    // be C++ left-over which cannot use the aie.device from
                    // inside the aie.device anyway
                    continue;
                  // Remap the current potential value use to its new producer
                  irm.map(operand.get(),
                          cat->getProducerOp(type)->getResult(0));
                }
              }
            });
            b.setInsertionPointToStart(&coreOp.getRegion().front());
            auto *clone = b.clone(*scopeOp.getOperation(), irm);
            // Since aie.device has a SymbolTable, all the called function need
            // to be present in the aie.device
            cloneReferencedSymbolsIntoDevice(
                clone->getParentOfType<xilinx::AIE::DeviceOp>());
            clone->emitRemark("tryTileProgramLowering: Clone");
            coreOp.emitRemark("tryTileProgramLowering: Stuffed core");
            scopeOp.emitRemark("tryTileProgramLowering: Scope after cloning");
            coreOp->getParentOfType<mlir::cir::FuncOp>().emitRemark(
                "tryTileProgramLowering: Top function");
          }
        }
        // The bufCast should be removed as a dependent of its tile cast later.
        // The lowering is a success, no need to look further.
        return true;
      }
    }
    // Notify to try something else
    return false;
  }

  bool tryTileLowering(mlir::Operation *op, mlir::OpBuilder &b,
                       llvm::SmallVector<mlir::Operation *> &opsToErase) {
    if (auto tileCast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
      tileCast.emitRemark("tryTileLowering: tileCast from device");
      auto aieLike = cat->getTypeDetail(
          tileCast.getType(0)); // llvm::errs() << aieLike.str() << " value \n";
      auto col = aieLike.subMatches[0];
      // llvm::errs() << "col" << col << " \n";
      auto row = aieLike.subMatches[1];
      // llvm::errs() << " row " << row << " \n";
      auto tileOp = b.create<xilinx::AIE::TileOp>(
          tileCast.getLoc(), std::stoi(col), std::stoi(row));
      cat->setProducerOpWithUCCast(tileCast.getType(0), tileOp, b);
      for (mlir::Operation *user : tileCast.getResult(0).getUsers())
        if (!(tryBufferLowering(user, tileOp, b, opsToErase) ||
              tryTileProgramLowering(user, tileOp, b, opsToErase)))
          user->emitError("User of tile cast not handled");
      // Tile lowering is done
      return true;
    }
    // Try some other lowering
    return false;
  }

  void deviceLowering(mlir::Operation *op) {
    llvm::SmallVector<mlir::Operation *> opsToErase;
    // Use pre-order walk to enable rewriting the code before the visited
    // operation
    op->walk<mlir::WalkOrder::PreOrder>([&](mlir::UnrealizedConversionCastOp
                                                u) {
      u.emitRemark(
          "DeviceLowering found UnrealizedConversionCastOp inside the module");
      if (!isUnrealizedConversionCastWithAnnotation(u, {"aie::device"}))
        return;
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
      mlir::OpBuilder b{u};
      auto deviceOp = b.create<xilinx::AIE::DeviceOp>(u.getLoc(), *deviceId);
      // The aie.device requires one block
      deviceOp.getRegion().emplaceBlock();
      cat->setProducerOpWithUCCast(u.getType(0), deviceOp, b);
      // Create all the following code inside the device region
      b.setInsertionPointToStart(deviceOp.getBody());
      // Lazily move all the code depending on the device to the trash
      opsToErase.push_back(u);
      for (mlir::Operation *user : u.getResult(0).getUsers())
        if (!tryTileLowering(user, b, opsToErase))
          user->emitRemark("User of device cast not handled");
      // Note: aie.device does not require a terminator
      deviceOp.emitRemark("DeviceLowering: end");
    });
    // Remove the useless operations
    eraseOpsAndUsers(opsToErase);
  }

  void runOnOperation() override {
    // Compute the analysis for the module since it is a module pass.
    cat = &getAnalysis<CIRToAIETypesAnalysis>();
    deviceLowering(getOperation());
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

} // namespace xilinx::AIE::CIR
