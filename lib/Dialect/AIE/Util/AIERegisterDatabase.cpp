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

#include "aie/Dialect/AIE/Util/AIERegisterDatabase.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>
#include <limits>

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

namespace {
std::optional<std::string> findRegDBFile(llvm::StringRef fileName) {
  auto checkDirectory = [&](llvm::StringRef dir) -> std::optional<std::string> {
    if (dir.empty())
      return std::nullopt;
    llvm::SmallString<256> candidate(dir);
    llvm::sys::path::append(candidate, fileName);
    if (llvm::sys::fs::exists(candidate))
      return std::string(candidate.str());
    return std::nullopt;
  };

  if (const char *installDir = std::getenv("MLIR_AIE_INSTALL_DIR")) {
    llvm::SmallString<256> dir(installDir);
    llvm::sys::path::append(dir, "lib", "regdb");
    if (auto path = checkDirectory(dir))
      return path;
  }

  std::string mainExecutable = llvm::sys::fs::getMainExecutable(
      nullptr, reinterpret_cast<void *>(&findRegDBFile));
  if (!mainExecutable.empty()) {
    llvm::SmallString<256> dir(mainExecutable);
    llvm::sys::path::remove_filename(dir);
    llvm::sys::path::append(dir, "..", "lib", "regdb");
    llvm::sys::path::remove_dots(dir, true);
    if (auto path = checkDirectory(dir))
      return path;
  }

  return std::nullopt;
}
} // namespace

std::unique_ptr<RegisterDatabase> RegisterDatabase::loadAIE2() {
  auto registerPath = findRegDBFile("aie_registers_aie2.json");
  auto eventPath = findRegDBFile("events_database.json");

  if (!registerPath || !eventPath) {
    llvm::errs() << "Failed to locate AIE register database resources. "
                 << "Set MLIR_AIE_INSTALL_DIR to the "
                 << "MLIR-AIE installation path.\n";
    return nullptr;
  }

  auto db = std::unique_ptr<RegisterDatabase>(new RegisterDatabase());
  if (!db->loadFromJSON(*registerPath, *eventPath))
    return nullptr;

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

  // Parse event database for aie2
  auto *eventRoot = eventJSON->getAsObject();
  if (!eventRoot)
    return false;

  auto *aie2 = eventRoot->getObject("aie2");
  if (!aie2) {
    llvm::errs() << "Failed to find 'aie2' architecture in event database\n";
    return false;
  }

  auto *eventModules = aie2->getObject("modules");
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
  uint32_t width = field.getWidth();
  if (width == 0)
    return 0;

  uint64_t maxValue = width >= 32 ? std::numeric_limits<uint32_t>::max()
                                  : ((1ULL << width) - 1ULL);

  if (value > maxValue) {
    llvm::errs() << "Warning: value " << value << " exceeds field width "
                 << width << "\n";
    value = static_cast<uint32_t>(value & maxValue); // Truncate
  } else if (width < 32) {
    value &= static_cast<uint32_t>(maxValue);
  }

  // Shift to correct bit position
  uint64_t encoded = (static_cast<uint64_t>(value) << field.bit_start);
  return static_cast<uint32_t>(encoded);
}
