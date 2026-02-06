//===- AIEExternalMangle.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Demangle/Demangle.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

struct AIEExternalManglePass
    : public AIEExternalMangleBase<AIEExternalManglePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Helper to mangle a function based on an object file
    auto mangleFunction = [&](func::FuncOp func,
                              StringRef objectFileName) -> func::FuncOp {
      StringRef functionName = func.getName();
      StringRef symbolName = functionName;

      if (auto linkSymbolAttr =
              func->getAttrOfType<StringAttr>("link_symbol")) {
        symbolName = linkSymbolAttr.getValue();
      }

      // Attempt to open the object file
      auto objectFileOrError =
          llvm::object::ObjectFile::createObjectFile(objectFileName);

      if (!objectFileOrError) {
        llvm::consumeError(objectFileOrError.takeError());
        func.emitWarning() << "Could not open object file: " << objectFileName;
        return func;
      }

      auto objectFile = std::move(objectFileOrError.get());

      // Iterate over symbols in the object file
      for (const auto &symbol : objectFile.getBinary()->symbols()) {
        auto nameOrError = symbol.getName();
        if (!nameOrError) {
          llvm::consumeError(nameOrError.takeError());
          continue;
        }
        StringRef mangledName = nameOrError.get();

        // Check if the symbol is a function
        auto typeOrError = symbol.getType();
        if (!typeOrError ||
            *typeOrError != llvm::object::SymbolRef::ST_Function)
          continue;

        // Check if the name matches directly (C linkage)
        bool match = (mangledName == symbolName);

        if (!match) {
          // Demangle the symbol name
          std::string demangled = llvm::demangle(mangledName.str());
          if (demangled == symbolName)
            match = true;
        }

        if (match) {
          if (functionName == mangledName) {
            return func;
          }

          SymbolTable parentSymbolTable(func->getParentOp());

          // Found a match! Check if we need to rename/clone.
          std::string newName = mangledName.str();
          int suffix = 0;
          while (parentSymbolTable.lookup(newName)) {
            // If the existing symbol is the function itself, we are good.
            auto existingOp = parentSymbolTable.lookup(newName);
            if (existingOp == func)
              break;

            // If existing function links to the same object file, reuse it.
            if (auto existingFunc = dyn_cast<func::FuncOp>(existingOp)) {
              if (auto existingLinkWith =
                      existingFunc->getAttrOfType<StringAttr>("link_with")) {
                if (existingLinkWith.getValue() == objectFileName) {
                  return existingFunc;
                }
              }
            }

            newName = mangledName.str() + "_" + std::to_string(++suffix);
          }

          if (newName != functionName) {
            // If 'func' already has the correct name and link_with, return it.
            if (auto existingOp = parentSymbolTable.lookup(newName)) {
              return cast<func::FuncOp>(existingOp);
            }

            // Create a new function declaration
            OpBuilder builder(func);
            auto newFunc = func::FuncOp::create(builder, func.getLoc(), newName,
                                                func.getFunctionType());
            newFunc.setPrivate();
            if (auto linkName = func->getAttr("link_name"))
              newFunc->setAttr("link_name", linkName);
            if (auto linkSymbol = func->getAttr("link_symbol"))
              newFunc->setAttr("link_symbol", linkSymbol);

            newFunc->setAttr("link_with", StringAttr::get(func.getContext(),
                                                          objectFileName));

            if (newName != mangledName) {
              newFunc->setAttr("link_name",
                               StringAttr::get(func.getContext(), mangledName));
            }

            // Insert the new function in the same symbol table as the original
            // function.
            parentSymbolTable.insert(newFunc, func->getIterator());
            return newFunc;
          }
          return func;
        }
      }

      func.emitWarning() << "Could not find symbol for " << symbolName << " in "
                         << objectFileName;
      return func;
    };

    // Process AIE cores
    module.walk([&](CoreOp core) {
      if (auto linkWithAttr = core->getAttrOfType<StringAttr>("link_with")) {
        StringRef objectFileName = linkWithAttr.getValue();

        core.walk([&](func::CallOp call) {
          auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
              call, call.getCalleeAttr());
          if (callee && callee.isExternal()) {
            // Mangle/Clone the callee for this core
            auto newCallee = mangleFunction(callee, objectFileName);
            if (newCallee != callee) {
              call.setCallee(newCallee.getName());
            }
          }
        });
      }
    });

    // Process func.func with link_with (legacy/explicit mode)
    module.walk([&](func::FuncOp func) {
      if (auto linkWithAttr = func->getAttrOfType<StringAttr>("link_with")) {
        StringRef objectFileName = linkWithAttr.getValue();
        auto newFunc = mangleFunction(func, objectFileName);
        if (newFunc != func) {
          // If newFunc is a different op, replace uses and erase original.
          if (failed(SymbolTable::replaceAllSymbolUses(
                  func, newFunc.getNameAttr(), func->getParentOp()))) {
            func.emitError("failed to replace symbol uses");
            return;
          }

          func.erase();
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIEExternalManglePass() {
  return std::make_unique<AIEExternalManglePass>();
}
