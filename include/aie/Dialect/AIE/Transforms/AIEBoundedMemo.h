//===- AIEBoundedMemo.h -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_TRANSFORMS_AIEBOUNDEDMEMO_H
#define AIE_DIALECT_AIE_TRANSFORMS_AIEBOUNDEDMEMO_H

#include "llvm/ADT/Hashing.h"

#include <list>
#include <unordered_map>
#include <vector>

namespace xilinx::AIE {

/// Hashes a vector key by its elements.
struct VectorKeyHash {
  template <typename T>
  size_t operator()(const std::vector<T> &key) const {
    return llvm::hash_combine_range(key.begin(), key.end());
  }
};

/// A memo that keeps its `capacity` most recently used results, each key
/// stored once.
template <typename Key, typename Value, typename Hash = VectorKeyHash>
class BoundedMemo {
public:
  explicit BoundedMemo(size_t capacity) : capacity(capacity) {}

  /// The result recorded for `key`, now the most recently used, or null.
  /// Valid until the next record().
  const Value *find(const Key &key) {
    auto it = entries.find(key);
    if (it == entries.end())
      return nullptr;
    order.splice(order.begin(), order, it->second.used);
    return &it->second.value;
  }

  /// Records `value` for `key`, replacing any result it had, and forgets the
  /// least recently used result past `capacity`.
  void record(Key key, Value value) {
    auto [it, added] = entries.try_emplace(std::move(key));
    it->second.value = std::move(value);
    if (!added) {
      order.splice(order.begin(), order, it->second.used);
      return;
    }
    order.push_front(&it->first);
    it->second.used = order.begin();
    if (entries.size() > capacity) {
      auto oldest = entries.find(*order.back());
      order.pop_back();
      entries.erase(oldest);
    }
  }

  void clear() {
    order.clear();
    entries.clear();
  }

private:
  // Keys live in `entries`' nodes, which never move; `order` points at them,
  // most recently used first.
  struct Entry {
    Value value;
    typename std::list<const Key *>::iterator used;
  };
  size_t capacity;
  std::unordered_map<Key, Entry, Hash> entries;
  std::list<const Key *> order;
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIEBOUNDEDMEMO_H
