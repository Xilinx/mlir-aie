//===- AIEStructuralRank.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEStructuralRank.h"

#include "mlir/IR/SymbolTable.h"

#include <map>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

static bool isSymbolReference(Attribute value) {
  auto array = dyn_cast<ArrayAttr>(value);
  return isa<SymbolRefAttr>(value) ||
         (array && llvm::all_of(array, llvm::IsaPred<SymbolRefAttr>));
}

/// An op's name and attributes. Symbol names and references are left out:
/// what a reference names enters through a link instead, and a fifo name
/// only says whether there is one. A consumed_by proof is left out too, so
/// whether verification ran never moves a channel.
static void describe(Operation *op, llvm::raw_ostream &os) {
  os << op->getName();
  for (NamedAttribute attr : op->getAttrs()) {
    if (isSymbolReference(attr.getValue()) ||
        attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == "consumed_by")
      continue;
    os << ' ' << attr.getName();
    if (attr.getName() != "fifoName")
      os << '=' << attr.getValue();
  }
}

std::string &StructuralGraph::add(Operation *op) {
  nodes.push_back(op);
  llvm::raw_string_ostream os(features[op]);
  describe(op, os);
  return features[op];
}

void StructuralGraph::link(Operation *from, Operation *to, StringRef label) {
  links[from].push_back({label.str(), to});
  links[to].push_back({(label + "~").str(), from});
}

DenseMap<Operation *, unsigned> StructuralGraph::rank() const {
  DenseMap<Operation *, unsigned> rank;
  unsigned classes = 0;
  auto rerank = [&](auto &signature) {
    SmallVector<Operation *> order(nodes);
    llvm::stable_sort(order, [&](Operation *a, Operation *b) {
      return signature.at(a) < signature.at(b);
    });
    classes = 0;
    for (auto [index, op] : llvm::enumerate(order)) {
      if (index && signature.at(order[index - 1]) < signature.at(op))
        ++classes;
      rank[op] = classes;
    }
    ++classes;
  };
  std::map<Operation *, std::string> initial;
  for (Operation *op : nodes)
    initial[op] = features.lookup(op);
  rerank(initial);
  for (unsigned previous = 0; classes != previous;) {
    previous = classes;
    std::map<Operation *,
             std::pair<unsigned, SmallVector<std::pair<std::string, unsigned>>>>
        signature;
    for (Operation *op : nodes) {
      auto &[own, around] = signature[op];
      own = rank[op];
      for (const auto &[label, other] : links.lookup(op))
        around.push_back({label, rank[other]});
      llvm::sort(around);
    }
    rerank(signature);
  }
  return rank;
}

StructuralGraph StructuralGraph::of(DeviceOp device) {
  StructuralGraph graph;
  // Where each nested op sits in its top-level op, for links into it.
  DenseMap<Operation *, std::pair<Operation *, unsigned>> position;
  for (Operation &op : device.getBody()->without_terminator()) {
    if (isa<RuntimeSequenceOp>(op))
      continue;
    llvm::raw_string_ostream os(graph.add(&op));
    unsigned index = 0;
    op.walk<WalkOrder::PreOrder>([&](Operation *nested) {
      position[nested] = {&op, index++};
      if (nested != &op) {
        os << " {";
        describe(nested, os);
        os << '}';
      }
    });
  }
  for (Operation *node : SmallVector<Operation *>(graph.nodes)) {
    node->walk<WalkOrder::PreOrder>([&](Operation *nested) {
      std::string at = std::to_string(position.lookup(nested).second);
      for (auto [index, operand] : llvm::enumerate(nested->getOperands())) {
        Operation *def = operand.getDefiningOp();
        if (def && def->getParentOp() == device && graph.features.count(def))
          graph.link(node, def, at + ":" + std::to_string(index));
      }
      for (NamedAttribute attr : nested->getAttrs()) {
        if (!isSymbolReference(attr.getValue()))
          continue;
        SmallVector<SymbolRefAttr> refs;
        if (auto ref = dyn_cast<SymbolRefAttr>(attr.getValue()))
          refs.push_back(ref);
        else
          llvm::append_range(
              refs,
              cast<ArrayAttr>(attr.getValue()).getAsRange<SymbolRefAttr>());
        for (auto [index, ref] : llvm::enumerate(refs)) {
          auto it =
              position.find(SymbolTable::lookupNearestSymbolFrom(nested, ref));
          if (it == position.end())
            continue;
          auto [target, within] = it->second;
          graph.link(node, target,
                     at + ":" + attr.getName().str() + ":" +
                         std::to_string(index) + "@" + std::to_string(within));
        }
      }
    });
  }
  return graph;
}
