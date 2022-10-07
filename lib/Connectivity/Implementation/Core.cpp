//===- Core.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/Implementation/Core.h"

#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

#include <set>
#include <utility>

using namespace mlir;
using namespace xilinx::phy;
using namespace xilinx::phy::connectivity;

Operation *CoreImplementation::createOperation() {
  assert(node && "a core must be associated with a node");

  auto builder = OpBuilder::atBlockEnd(context.module.getBody());
  return builder.create<physical::CoreOp>(
      builder.getUnknownLoc(), physical::CoreType::get(builder.getContext()),
      translateFunction(), translateOperands());
}

llvm::SmallVector<std::pair<std::weak_ptr<Implementation>, Value>>
CoreImplementation::getOperandImpls(Value operand) {
  llvm::SmallVector<std::pair<std::weak_ptr<Implementation>, Value>> impls;

  auto queue = dyn_cast<spatial::QueueOp>(operand.getDefiningOp());
  assert(queue && "operand is a defined queue");

  for (auto impl : queue_impls[queue]) {
    auto *impl_op = impl.lock()->getOperation();
    assert(impl_op->getNumResults() == 1 && "returns one value");
    auto value = impl_op->getResult(0);
    impls.emplace_back(impl, value);
  }

  return impls;
}

llvm::SmallVector<Value> CoreImplementation::translateOperands() {
  llvm::SmallVector<Value> translated;

  for (auto operand : node.operands())
    for (auto impl : getOperandImpls(operand))
      translated.push_back(impl.second);

  return translated;
}

StringRef CoreImplementation::translateFunction() {

  auto original_op = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      node, StringAttr::get(node.getContext(), node.getCallee()));
  assert(original_op && "function must be defined");
  auto original_fn_type = original_op.getFunctionType();
  int original_num_inputs = original_fn_type.getNumInputs();

  // Clone function
  OpBuilder builder(original_op);
  auto translated_op =
      dyn_cast<func::FuncOp>(builder.clone(*(original_op.getOperation())));
  assert(translated_op && "function cloned");

  // Set name to translated name
  auto tranlated_name = context.getUniqueSymbol(node.getCallee(), node);
  translated_op.setName(tranlated_name);

  // Clone function type
  llvm::SmallVector<Type> translated_inputs;
  for (int i = 0; i < original_num_inputs; i++)
    translated_inputs.push_back(original_fn_type.getInput(i));

  // Build new function arguments and translate the usages
  std::set<Operation *> users_to_be_erased;
  for (int i = 0; i < original_num_inputs; i++)
    for (auto impl : getOperandImpls(node.getOperand(i))) {
      translated_inputs.push_back(impl.second.getType());
      auto arg = translated_op.getCallableRegion()->addArgument(
          impl.second.getType(), translated_op.getLoc());
      for (auto *user : translated_op.getArgument(i).getUsers()) {
        impl.first.lock()->translateUserOperation(arg, user);
        users_to_be_erased.insert(user);
      }
    }
  translated_op.setType(builder.getFunctionType(translated_inputs, {}));

  // Erase original arguments' users
  for (auto *user : users_to_be_erased) {
#ifndef NDEBUG
    for (auto result : user->getResults())
      assert(result.getUsers().empty() && "all results must be replaced");
#endif
    user->erase();
  }

  // Erase original arguments
  for (int idx = original_num_inputs - 1; idx >= 0; idx--) {
    if (!translated_op.getArgument(idx).use_empty()) {
      node->emitError() << " argument #" << idx << " is not translated";
      continue;
    }
    translated_op.eraseArgument(idx);
  }

  return tranlated_name;
}

void CoreImplementation::addPredecessor(std::weak_ptr<Implementation> pred,
                                        Operation *src, Operation *dest) {
  if (auto queue = dyn_cast<spatial::QueueOp>(src))
    addQueueImpl(queue, pred);
}

void CoreImplementation::addSuccessor(std::weak_ptr<Implementation> succ,
                                      Operation *src, Operation *dest) {
  if (auto queue = dyn_cast<spatial::QueueOp>(dest))
    addQueueImpl(queue, succ);
}

void CoreImplementation::addSpatialOperation(Operation *spatial) {
  if (auto node_op = dyn_cast<spatial::NodeOp>(spatial)) {
    assert(!node && "a core can only hold a node");
    node = node_op;
  } else {
    assert(false && "a core can only implement a node");
  }
}

void CoreImplementation::addQueueImpl(spatial::QueueOp queue,
                                      std::weak_ptr<Implementation> impl) {
  queue_impls[queue].push_back(impl);
}
