//===- parameter_scratchpad.h - Host-side parameter runtime ------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// ParameterScratchpad: Host-side runtime class for writing named parameters
// to AIE cores via the scratchpad mechanism.
//
// Usage:
//   auto params = test_utils::ParameterScratchpad(run, "params.txt");
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

#if !defined(TEST_UTILS_USE_XRT) && defined(__has_include)
#if __has_include(<xrt/xrt_bo.h>) && __has_include(<xrt/xrt_kernel.h>)
#define TEST_UTILS_USE_XRT 1
#endif
#endif

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
    if (scratchpadBo.size() < scratchpadSizeBytes)
      throw std::runtime_error("ParameterScratchpad: BO size (" +
                               std::to_string(scratchpadBo.size()) +
                               ") < required scratchpad size (" +
                               std::to_string(scratchpadSizeBytes) + ")");
    clear();
  }
#endif

  /// Construct from a raw buffer pointer (Python bindings / testing).
  ParameterScratchpad(uint32_t *buffer, const std::string &paramsPath)
      : boMap(buffer) {
    parseParams(paramsPath);
    clear();
  }

  /// Write raw bytes (up to 4) by name, interpreted as a little-endian
  /// uint32.  For core-kind parameters, the bits are left-shifted by 2
  /// (firmware requirement).
  /// For addr-kind parameters, the value is written raw (no shift).
  void writeBytes(const std::string &name, const void *data, size_t len) {
    uint32_t bits = 0;
    std::memcpy(&bits, data, std::min(len, sizeof(bits)));
    writeBits(name, bits);
  }

  /// Write a raw 32-bit value by name.  For core-kind parameters, the bits
  /// are left-shifted by 2 (firmware requirement).  For addr-kind parameters,
  /// the value is written directly (no shift).
  void writeBits(const std::string &name, uint32_t bits) {
    auto it = paramMap.find(name);
    if (it == paramMap.end()) {
      throw std::runtime_error("ParameterScratchpad: unknown parameter '" +
                               name + "'");
    }
    uint8_t idx = it->second;
    uint32_t encoded = bits;
    if (coreParams.count(name)) {
      // core parameters require shift-2 to survive masking of lowest bits by
      // firmware op
      encoded = bits << 2;
    }
    boMap[idx] = encoded;
  }

  /// Write a typed parameter value. For core-kind parameters, the raw bits
  /// are left-shifted by 2 as required by the firmware's UPDATE_REG Incr mode,
  /// and the core right-shifts (unsigned) by 2 after reading. The round trip
  /// therefore zeroes the top 2 bits of the value (the bottom 2 bits are
  /// preserved). Types that fit in 30 bits (e.g. uint16_t, int16_t,
  /// std::bfloat16_t, and uint32_t values < 2^30) round-trip losslessly.
  /// `float` is not supported as a core-kind parameter (the verifier in
  /// `--aie-lower-scratchpad-parameters` rejects it).
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
    return boMap[it->second];
  }

private:
#ifdef TEST_UTILS_USE_XRT
  xrt::bo scratchpadBo;
#endif
  uint32_t *boMap = nullptr;
  size_t scratchpadSizeBytes = 0;
  std::unordered_map<std::string, uint8_t> paramMap;
  std::unordered_set<std::string> coreParams; // params with kind="core"

  void clear() {
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
      file >> name >> idx >> type >> kind;
      if (idx > 255)
        throw std::runtime_error("ParameterScratchpad: state_table_idx " +
                                 std::to_string(idx) + " for '" + name +
                                 "' exceeds uint8_t range");
      if (paramMap.count(name))
        throw std::runtime_error(
            "ParameterScratchpad: duplicate parameter name '" + name + "'");
      paramMap[name] = static_cast<uint8_t>(idx);
      if (kind != "core" && kind != "addr") {
        throw std::runtime_error("ParameterScratchpad: invalid kind '" + kind +
                                 "' for parameter '" + name + "'");
      } else if (kind == "core") {
        coreParams.insert(name);
      }
    }
  }
};

} // namespace test_utils

#endif // AIE_RUNTIME_TEST_LIB_PARAMETER_SCRATCHPAD_H
