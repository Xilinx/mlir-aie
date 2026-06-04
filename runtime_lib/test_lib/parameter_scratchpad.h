//===- parameter_scratchpad.h - Host-side parameter runtime ------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// ParameterScratchpad: Host-side runtime class for writing named parameters
// to AIE cores via the scratchpad mechanism.
//
// Usage:
//   auto params = test_utils::ParameterScratchpad(run, "params.json");
//   params.write("foo", 42u);
//   params.write("bar", std::bfloat16_t(3.14f));
//   params.sync();
//
//===----------------------------------------------------------------------===//

#ifndef AIE_RUNTIME_TEST_LIB_PARAMETER_SCRATCHPAD_H
#define AIE_RUNTIME_TEST_LIB_PARAMETER_SCRATCHPAD_H

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef TEST_UTILS_USE_XRT
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#endif

namespace test_utils {

class ParameterScratchpad {
public:
#ifdef TEST_UTILS_USE_XRT
  /// Construct from an XRT run handle (C++ usage).
  ParameterScratchpad(xrt::run &run, const std::string &paramsPath) {
    parseParams(paramsPath);
    scratchpadBo = run.get_ctrl_scratchpad_bo();
    boMap = scratchpadBo.map<uint32_t *>();
    init();
  }
#endif

  /// Construct from a raw buffer pointer (Python bindings / testing).
  ParameterScratchpad(uint32_t *buffer, const std::string &paramsPath)
      : boMap(buffer) {
    parseParams(paramsPath);
    init();
  }

  /// Write raw bytes (up to 4) by name, interpreted as a little-endian
  /// uint32.  The bits are left-shifted by 2 (firmware requirement) and
  /// delta-encoded against the previous write.
  /// For addr-kind parameters, the value is written raw (no shift, no delta).
  void writeBytes(const std::string &name, const void *data, size_t len) {
    uint32_t bits = 0;
    std::memcpy(&bits, data, std::min(len, size_t(4)));
    writeBits(name, bits);
  }

  /// Write a raw 32-bit value by name.  For core-kind parameters, the bits
  /// are left-shifted by 2 (firmware requirement) and delta-encoded against
  /// the previous write.  For addr-kind parameters, the value is written
  /// directly (no shift-2, no delta encoding).
  void writeBits(const std::string &name, uint32_t bits) {
    auto it = paramMap.find(name);
    if (it == paramMap.end()) {
      throw std::runtime_error("ParameterScratchpad: unknown parameter '" +
                               name + "'");
    }
    uint8_t idx = it->second;
    if (addrParams.count(name)) {
      // addr-kind: raw absolute write, no shift-2, no delta encoding.
      // The firmware multiplies by element_size and adds to BD address.
      boMap[idx] = bits;
    } else {
      // core-kind: shift-2 + delta encoding.
      uint32_t encoded = bits << 2;
      boMap[idx] = encoded - prevEncoded[idx];
      prevEncoded[idx] = encoded;
    }
  }

  /// Write a typed parameter value. The raw bits of the value are
  /// left-shifted by 2 as required by the firmware's UPDATE_REG Incr mode.
  /// Supports any type up to 32 bits (uint32_t, int16_t, std::bfloat16_t,
  /// float, etc.).
  template <typename T>
  void write(const std::string &name, T value) {
    static_assert(sizeof(T) <= 4, "Parameter values must be at most 32 bits");
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(T));
    writeBits(name, bits);
  }

#ifdef TEST_UTILS_USE_XRT
  /// Sync the scratchpad buffer to device. Call after all writes for this run.
  void sync() { scratchpadBo.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
#endif

  /// Read back a parameter's current encoded value (for debugging).
  uint32_t read(const std::string &name) const {
    auto it = paramMap.find(name);
    if (it == paramMap.end()) {
      throw std::runtime_error("ParameterScratchpad: unknown parameter '" +
                               name + "'");
    }
    if (addrParams.count(name)) {
      return boMap[it->second]; // addr-kind: raw value
    }
    return prevEncoded[it->second] >> 2;
  }

private:
#ifdef TEST_UTILS_USE_XRT
  xrt::bo scratchpadBo;
#endif
  uint32_t *boMap = nullptr;
  size_t scratchpadSizeBytes = 0;
  std::unordered_map<std::string, uint8_t> paramMap;
  std::unordered_set<std::string> addrParams; // params with kind="addr"
  std::vector<uint32_t> prevEncoded;

  void init() {
    prevEncoded.resize(scratchpadSizeBytes / 4, 0);
    for (size_t i = 0; i < scratchpadSizeBytes / 4; i++) {
      boMap[i] = 0;
    }
  }

  void parseParams(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("ParameterScratchpad: cannot open '" + path +
                               "'");
    }

    // Format:
    //   <num_parameters>
    //   <name> <state_table_idx> <type> <kind>
    //   ...
    // where kind is "core" or "addr".
    unsigned numParams = 0;
    file >> numParams;
    scratchpadSizeBytes = numParams * 4;

    for (unsigned i = 0; i < numParams; i++) {
      std::string name, type, kind;
      unsigned idx;
      file >> name >> idx >> type;
      // kind column is optional for backward compatibility
      if (file.peek() != '\n' && file.peek() != EOF) {
        file >> kind;
      } else {
        kind = "core";
      }
      paramMap[name] = static_cast<uint8_t>(idx);
      if (kind == "addr")
        addrParams.insert(name);
    }
  }
};

} // namespace test_utils

#endif // AIE_RUNTIME_TEST_LIB_PARAMETER_SCRATCHPAD_H
