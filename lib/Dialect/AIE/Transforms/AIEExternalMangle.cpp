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
    SymbolTable symbolTable(module);

    module.walk([&](func::FuncOp func) {
      if (auto linkWithAttr = func->getAttrOfType<StringAttr>("link_with")) {
        StringRef objectFileName = linkWithAttr.getValue();
        StringRef functionName = func.getName();

        // Attempt to open the object file
        auto objectFileOrError =
            llvm::object::ObjectFile::createObjectFile(objectFileName);

        if (!objectFileOrError) {
          llvm::consumeError(objectFileOrError.takeError());
          func.emitWarning()
              << "Could not open object file: " << objectFileName;
          return;
        }

        auto objectFile = std::move(objectFileOrError.get());
        bool found = false;

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
          if (mangledName == functionName) {
            // Already matches, nothing to do
            found = true;
            break;
          }

          // Demangle the symbol name
          std::string demangled = llvm::demangle(mangledName.str());

          if (demangled == functionName) {
            // Found a match! Rename the function to the mangled name by
            // updating the symbol table.

            std::string newName = mangledName.str();
            int suffix = 0;
            while (symbolTable.lookup(newName)) {
              newName = mangledName.str() + "_" + std::to_string(++suffix);
            }

            if (newName != mangledName) {
              // Collision detected, uniquified the name.
              // Store the original mangled name so the backend can handle it
              // (e.g. by renaming symbols in object file).
              func->setAttr("link_name",
                            StringAttr::get(func.getContext(), mangledName));
            }

            SymbolTable::setSymbolName(func, newName);
            found = true;
            break;
          }
        }

        if (!found) {
          func.emitWarning() << "Could not find symbol for " << functionName
                             << " in " << objectFileName;
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
