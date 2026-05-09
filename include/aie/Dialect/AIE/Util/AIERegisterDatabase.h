//===- AIERegisterDatabase.h ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Register and event database for AIE trace configuration
//===----------------------------------------------------------------------===//

#ifndef AIE_REGISTER_DATABASE_H
#define AIE_REGISTER_DATABASE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace xilinx {
namespace AIE {

/// Bit field information for a register
struct BitFieldInfo {
  std::string name;
  uint32_t bit_start; // LSB position
  uint32_t bit_end;   // MSB position
  std::string type;
  std::string reset;
  std::string description;

  uint32_t getWidth() const { return bit_end - bit_start + 1; }
};

/// Register information
struct RegisterInfo {
  std::string name;
  uint32_t offset;
  std::string module;
  uint32_t width;
  std::string type;
  std::string reset;
  std::string description;
  std::vector<BitFieldInfo> bit_fields;

  const BitFieldInfo *getField(llvm::StringRef fieldName) const;
};

/// Event information
struct EventInfo {
  std::string name;
  uint32_t number;
  std::string module; // core, memory, pl, mem_tile
};

/// Register and event database for a specific architecture
class RegisterDatabase {
public:
  /// Load database for AIE2 architecture
  static std::unique_ptr<RegisterDatabase> loadAIE2();

  /// Lookup register by name and module
  const RegisterInfo *lookupRegister(llvm::StringRef name,
                                     llvm::StringRef module) const;

  /// Lookup register by tile-relative offset and module name. Used by the
  /// transaction-binary locmap to decorate each transaction word with its
  /// semantic register name. Module names are the JSON keys in
  /// aie_registers_aie2.json (e.g. "core_module", "memory_module",
  /// "mem_tile_module", "pl_module"). Returns nullptr if no register matches.
  const RegisterInfo *lookupRegisterByOffset(uint32_t offset,
                                             llvm::StringRef module) const;

  /// Lookup event by name and module
  std::optional<uint32_t> lookupEvent(llvm::StringRef name,
                                      llvm::StringRef module) const;

  /// Encode a value for a specific bitfield
  uint32_t encodeFieldValue(const BitFieldInfo &field, uint32_t value) const;

private:
  RegisterDatabase() = default;

  bool loadFromJSON(llvm::StringRef registerPath, llvm::StringRef eventPath);

  // Build the offset-keyed reverse index from `registers_`. Called once at
  // load time after `loadFromJSON` finishes populating `registers_`.
  void buildOffsetIndex();

  llvm::StringMap<RegisterInfo> registers_;
  llvm::StringMap<EventInfo> events_;
  // (lowercased module name) -> (offset -> *RegisterInfo). Pointers into
  // entries owned by `registers_`; safe because `registers_` is never
  // mutated after construction.
  llvm::StringMap<llvm::DenseMap<uint32_t, const RegisterInfo *>>
      registersByOffset_;
};

} // namespace AIE
} // namespace xilinx

#endif // AIE_REGISTER_DATABASE_H
