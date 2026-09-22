// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Actions.h"

#include "mlir/IR/Verifier.h"

#include <cassert>
#include <string>
#include <vector>

using namespace xilinx::aiecc;
using mlir::ModuleOp;
using ModRef = mlir::OwningOpRef<ModuleOp>;
using Focus = OpInModule<ModuleOp>;

static void checkItems(const std::vector<Item<Focus>> &items) {
  assert(items.size() == 6);
  ModuleOp shared = items.front().get().module.get();
  assert(mlir::succeeded(mlir::verify(shared)));
  size_t count = 0;
  shared.walk([&](ModuleOp) { ++count; });
  assert(count == items.size());
  for (const auto &item : items) {
    ModuleOp focus = item.get().op;
    assert(item.get().module.get() == shared);
    assert(focus.getSymName() == item.key);
    assert(opWalkIndex(shared, focus) >= 0);
  }
}

int main(int argc, char **argv) {
  assert(argc == 2);
  mlir::MLIRContext context;
  // Nested symbol tables model device/sequence splits without AIE libraries.
  Item<ModRef> input;
  input.value = mlir::parseSourceString<ModuleOp>(R"mlir(
    module @design {
      module @device_a {
        module @seq_a attributes {test.target = @device_b::@seq_c} {}
        module @seq_b {}
      }
      module @device_b {
        module @seq_c {}
      }
    }
  )mlir",
                                                  &context);
  assert(input.get());
  SplitIRAction<ModuleOp> split(
      [](ModuleOp op) { return op.getSymName()->str(); });
  auto result = split(input);
  assert(mlir::succeeded(result));
  std::vector<Item<Focus>> items;
  for (auto &[key, focus] : *result) {
    Item<Focus> item;
    item.key = std::move(key);
    item.value = std::move(focus);
    items.push_back(std::move(item));
  }
  result->clear();
  checkItems(items);
  assert(items.front().get().module.get() != input.get().get());

  // Transforming a consumer's clone must not mutate any sibling or the input.
  auto consumer = asModule(items.front(), &context);
  consumer->getBody()->clear();
  checkItems(items);
  assert(!input.get().get().getBody()->empty());
  input.value.reset();
  checkItems(items);

  assert(!llvm::sys::fs::create_directories(argv[1]));
  auto desc = NodeSerializer<Focus>::write(items, argv[1]);
  std::vector<std::string> keys;
  for (const auto &item : items)
    keys.push_back(item.key);
  items.clear();
  auto restored = NodeDeserializer<Focus>::read(desc, argv[1],
                                                DeserializeContext{&context});
  assert(mlir::succeeded(restored));
  checkItems(*restored);
  for (size_t i = 0; i < keys.size(); ++i)
    assert((*restored)[i].key == keys[i]);

  // A surviving focus keeps the shared module alive after siblings go away.
  Focus last = restored->back().get();
  restored->clear();
  assert(last.op.getSymName() == keys.back());
  assert(mlir::succeeded(mlir::verify(last.module.get())));
  llvm::outs() << "split sharing: all checks passed\n";
}
