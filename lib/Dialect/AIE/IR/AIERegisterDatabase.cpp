//===- AIERegisterDatabase.cpp ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Register and event database implementation
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIERegisterDatabase.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <fstream>

using namespace xilinx::AIE;
using namespace llvm;

const BitFieldInfo *RegisterInfo::getField(StringRef fieldName) const {
  for (const auto &field : bit_fields) {
    if (field.name == fieldName) {
      return &field;
    }
  }
  return nullptr;
}

std::unique_ptr<RegisterDatabase> RegisterDatabase::loadAIE2() {
  auto db = std::unique_ptr<RegisterDatabase>(new RegisterDatabase());

  // Get paths relative to source directory
  // In a real build, these would be found via CMake-configured paths
  // FIXME: update paths to look in the installed location
  const char *registerPath = "/work/acdc/aie/utils/aie_registers_aie2.json";
  const char *eventPath = "/work/acdc/aie/utils/events_database.json";

  if (!db->loadFromJSON(registerPath, eventPath)) {
    return nullptr;
  }

  return db;
}

bool RegisterDatabase::loadFromJSON(StringRef registerPath,
                                    StringRef eventPath) {
  // Load register database
  auto registerBuf = MemoryBuffer::getFile(registerPath);
  if (!registerBuf) {
    llvm::errs() << "Failed to load register database: " << registerPath
                 << "\n";
    return false;
  }

  auto registerJSON = json::parse(registerBuf.get()->getBuffer());
  if (!registerJSON) {
    llvm::errs() << "Failed to parse register JSON\n";
    return false;
  }

  // Parse register database
  auto *root = registerJSON->getAsObject();
  if (!root)
    return false;

  auto *modules = root->getObject("modules");
  if (!modules)
    return false;

  // Iterate through modules (CORE_MODULE, MEMORY_MODULE, etc.)
  for (const auto &modulePair : *modules) {
    StringRef moduleName = modulePair.first;
    auto *moduleObj = modulePair.second.getAsObject();
    if (!moduleObj)
      continue;

    auto *registers = moduleObj->getArray("registers");
    if (!registers)
      continue;

    // Parse each register
    for (const auto &regVal : *registers) {
      auto *regObj = regVal.getAsObject();
      if (!regObj)
        continue;

      RegisterInfo regInfo;
      regInfo.module = moduleName.str();
      regInfo.offset = 0; // Initialize to 0

      if (auto name = regObj->getString("name"))
        regInfo.name = name->str();
      if (auto offset = regObj->getString("offset")) {
        // Parse hex offset (string like "0x00000340D0")
        StringRef offsetStr = *offset;
        // Remove "0x" prefix if present
        if (offsetStr.starts_with("0x") || offsetStr.starts_with("0X"))
          offsetStr = offsetStr.drop_front(2);

        uint64_t offsetVal;
        if (offsetStr.getAsInteger(16, offsetVal)) {
          llvm::errs() << "Warning: Failed to parse offset '" << *offset
                       << "' for register " << regInfo.name << "\n";
        } else {
          regInfo.offset = static_cast<uint32_t>(offsetVal);
        }
      }
      if (auto width = regObj->getInteger("width"))
        regInfo.width = static_cast<uint32_t>(*width);
      if (auto type = regObj->getString("type"))
        regInfo.type = type->str();
      if (auto reset = regObj->getString("reset"))
        regInfo.reset = reset->str();
      if (auto desc = regObj->getString("description"))
        regInfo.description = desc->str();

      // Parse bit fields
      if (auto *fields = regObj->getArray("bit_fields")) {
        for (const auto &fieldVal : *fields) {
          auto *fieldObj = fieldVal.getAsObject();
          if (!fieldObj)
            continue;

          BitFieldInfo fieldInfo;
          if (auto name = fieldObj->getString("name"))
            fieldInfo.name = name->str();
          if (auto *bitRange = fieldObj->getArray("bit_range")) {
            if (bitRange->size() == 2) {
              if (auto start = (*bitRange)[0].getAsInteger())
                fieldInfo.bit_start = static_cast<uint32_t>(*start);
              if (auto end = (*bitRange)[1].getAsInteger())
                fieldInfo.bit_end = static_cast<uint32_t>(*end);
            }
          }
          if (auto type = fieldObj->getString("type"))
            fieldInfo.type = type->str();
          if (auto reset = fieldObj->getString("reset"))
            fieldInfo.reset = reset->str();
          if (auto desc = fieldObj->getString("description"))
            fieldInfo.description = desc->str();

          regInfo.bit_fields.push_back(fieldInfo);
        }
      }

      // Store with module::name as key for uniqueness (lowercase for
      // case-insensitive lookup)
      std::string key = moduleName.lower() + "::" + regInfo.name;
      std::transform(key.begin(), key.end(), key.begin(), ::tolower);
      registers_[key] = regInfo;
    }
  }

  // Load event database
  auto eventBuf = MemoryBuffer::getFile(eventPath);
  if (!eventBuf) {
    llvm::errs() << "Failed to load event database: " << eventPath << "\n";
    return false;
  }

  auto eventJSON = json::parse(eventBuf.get()->getBuffer());
  if (!eventJSON) {
    llvm::errs() << "Failed to parse event JSON\n";
    return false;
  }

  // Parse event database for aieml (AIE2)
  auto *eventRoot = eventJSON->getAsObject();
  if (!eventRoot)
    return false;

  // Use "aieml" for AIE2 architecture
  auto *aieml = eventRoot->getObject("aieml");
  if (!aieml) {
    llvm::errs() << "Failed to find 'aieml' architecture in event database\n";
    return false;
  }

  auto *eventModules = aieml->getObject("modules");
  if (!eventModules)
    return false;

  // Iterate through modules (core, memory, pl, mem_tile)
  for (const auto &modulePair : *eventModules) {
    StringRef moduleName = modulePair.first;
    auto *events = modulePair.second.getAsArray();
    if (!events)
      continue;

    // Parse each event
    for (const auto &eventVal : *events) {
      auto *eventObj = eventVal.getAsObject();
      if (!eventObj)
        continue;

      EventInfo eventInfo;
      eventInfo.module = moduleName.str();

      if (auto name = eventObj->getString("name"))
        eventInfo.name = name->str();
      if (auto number = eventObj->getInteger("number"))
        eventInfo.number = static_cast<uint32_t>(*number);

      // Store with module::name as key (lowercase for case-insensitive lookup)
      std::string key = moduleName.str() + "::" + eventInfo.name;
      std::transform(key.begin(), key.end(), key.begin(), ::tolower);
      events_[key] = eventInfo;
    }
  }

  return true;
}

const RegisterInfo *RegisterDatabase::lookupRegister(StringRef name,
                                                     StringRef module) const {
  std::string key = module.str() + "::" + name.str();
  std::transform(key.begin(), key.end(), key.begin(), ::tolower);
  auto it = registers_.find(key);
  return it != registers_.end() ? &it->second : nullptr;
}

std::optional<uint32_t> RegisterDatabase::lookupEvent(StringRef name,
                                                      StringRef module) const {
  std::string key = module.str() + "::" + name.str();
  std::transform(key.begin(), key.end(), key.begin(), ::tolower);
  auto it = events_.find(key);
  if (it != events_.end()) {
    return it->second.number;
  }
  return std::nullopt;
}

uint32_t RegisterDatabase::encodeFieldValue(const BitFieldInfo &field,
                                            uint32_t value) const {
  // Validate value fits in field width
  uint32_t maxValue = (1u << field.getWidth()) - 1;
  if (value > maxValue) {
    llvm::errs() << "Warning: value " << value << " exceeds field width "
                 << field.getWidth() << "\n";
    value = value & maxValue; // Truncate
  }

  // Shift to correct bit position
  return value << field.bit_start;
}

StringRef RegisterDatabase::getRegisterModuleForTile(TileOp tile, bool isMem) {
  if (tile.isShimTile()) {
    return "PL_MODULE";
  } else if (tile.isMemTile()) {
    return "MEMORY_TILE_MODULE";
  } else {
    // Core tile - has both CORE_MODULE and MEMORY_MODULE
    if (isMem) {
      return "MEMORY_MODULE";
    } else {
      return "CORE_MODULE";
    }
  }
}

StringRef RegisterDatabase::getEventModuleForTile(TileOp tile, bool isMem) {
  if (tile.isShimTile()) {
    return "pl";
  } else if (tile.isMemTile()) {
    return "mem_tile";
  } else {
    // Core tile - has both core and memory events
    if (isMem) {
      return "memory";
    } else {
      return "core";
    }
  }
}

const RegisterInfo *RegisterDatabase::lookupRegister(StringRef name,
                                                     TileOp tile,
                                                     bool isMem) const {
  return lookupRegister(name, getRegisterModuleForTile(tile, isMem));
}

std::optional<uint32_t>
RegisterDatabase::lookupEvent(StringRef name, TileOp tile, bool isMem) const {
  return lookupEvent(name, getEventModuleForTile(tile, isMem));
}

std::optional<uint32_t> RegisterDatabase::resolvePortValue(StringRef value,
                                                           TileOp tile,
                                                           bool master) const {
  // Parse "PORT:CHANNEL" format
  auto colonPos = value.find(':');
  if (colonPos == StringRef::npos) {
    return std::nullopt; // Not a port value
  }

  StringRef portName = value.substr(0, colonPos);
  StringRef channelStr = value.substr(colonPos + 1);

  // Parse channel number
  int channel;
  if (channelStr.getAsInteger(10, channel)) {
    return std::nullopt; // Invalid channel number
  }

  // Convert port name to WireBundle enum
  WireBundle bundle;
  if (portName == "North" || portName == "NORTH") {
    bundle = WireBundle::North;
  } else if (portName == "South" || portName == "SOUTH") {
    bundle = WireBundle::South;
  } else if (portName == "East" || portName == "EAST") {
    bundle = WireBundle::East;
  } else if (portName == "West" || portName == "WEST") {
    bundle = WireBundle::West;
  } else if (portName == "DMA") {
    bundle = WireBundle::DMA;
  } else if (portName == "FIFO") {
    bundle = WireBundle::FIFO;
  } else if (portName == "Core" || portName == "CORE") {
    bundle = WireBundle::Core;
  } else if (portName == "CTRL") {
    bundle = WireBundle::Core; // or TileControl?
  } else {
    return std::nullopt; // Unknown port type
  }

  // Get device and target model
  auto device = tile->getParentOfType<DeviceOp>();
  if (!device) {
    return std::nullopt;
  }

  const auto &targetModel = device.getTargetModel();

  // Look up port index from target model
  return targetModel.getStreamSwitchPortIndex(tile.getCol(), tile.getRow(),
                                              bundle, channel, master);
}
