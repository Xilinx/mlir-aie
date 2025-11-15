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

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <optional>
#include <string>
#include <vector>

namespace mlir {
class Operation;
}

namespace xilinx {
namespace AIE {

class TileOp;

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

  /// Lookup register by name, determining module from tile
  const RegisterInfo *lookupRegister(llvm::StringRef name, TileOp tile,
                                     bool isMem = false) const;

  /// Lookup event by name and module
  std::optional<uint32_t> lookupEvent(llvm::StringRef name,
                                      llvm::StringRef module) const;

  /// Lookup event by name, determining module from tile
  std::optional<uint32_t> lookupEvent(llvm::StringRef name, TileOp tile,
                                      bool isMem = false) const;

  /// Encode a value for a specific bitfield
  uint32_t encodeFieldValue(const BitFieldInfo &field, uint32_t value) const;

  /// Resolve "PORT:CHANNEL" value to stream switch port index
  /// @param value String in format "PORT:CHANNEL" (e.g., "NORTH:1", "DMA:0")
  /// @param tile Tile operation (for context)
  /// @param master True for master port, false for slave
  /// @return Port index, or nullopt if invalid/unsupported
  std::optional<uint32_t> resolvePortValue(llvm::StringRef value, TileOp tile,
                                           bool master) const;

  /// Get register module name for a tile (CORE_MODULE, PL_MODULE, etc.)
  static llvm::StringRef getRegisterModuleForTile(TileOp tile,
                                                  bool isMem = false);

  /// Get event module name for a tile (core, pl, memory, etc.)
  static llvm::StringRef getEventModuleForTile(TileOp tile, bool isMem = false);

private:
  RegisterDatabase() = default;

  bool loadFromJSON(llvm::StringRef registerPath, llvm::StringRef eventPath);

  llvm::StringMap<RegisterInfo> registers_;
  llvm::StringMap<EventInfo> events_;
};

} // namespace AIE
} // namespace xilinx

#endif // AIE_REGISTER_DATABASE_H
