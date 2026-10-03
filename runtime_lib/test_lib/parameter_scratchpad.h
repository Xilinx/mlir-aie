//===- parameter_scratchpad.h - Host-side parameter runtime ------*- C++-*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include <utility>
#include <vector>

#if !defined(TEST_UTILS_USE_XRT)
#if defined(__has_include) && __has_include(<xrt/xrt_bo.h>) && \
    __has_include(<xrt/xrt_kernel.h>)
#define TEST_UTILS_USE_XRT 1
#else
#define TEST_UTILS_USE_XRT 0
#endif
#endif

#if TEST_UTILS_USE_XRT
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#endif

namespace test_utils {

class ParameterScratchpad {
public:
#if TEST_UTILS_USE_XRT
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

  /// Construct from a raw buffer and its size in bytes (Python bindings /
  /// testing). The size is required: clear() and every write index off
  /// scratchpadSizeBytes, which parseParams derives from the params file, not
  /// from the buffer.
  ParameterScratchpad(uint32_t *buffer, size_t bufferSizeBytes,
                      const std::string &paramsPath)
      : boMap(buffer) {
    parseParams(paramsPath);
    if (bufferSizeBytes < scratchpadSizeBytes)
      throw std::runtime_error("ParameterScratchpad: buffer size (" +
                               std::to_string(bufferSizeBytes) +
                               ") < required scratchpad size (" +
                               std::to_string(scratchpadSizeBytes) + ")");
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
  /// the value is written directly (no shift). A DMA offset or length
  /// parameter's value, read as an int32, must lie in the range the compiler
  /// gave it, which keeps its transfers within their buffers.
  void writeBits(const std::string &name, uint32_t bits) {
    auto it = paramMap.find(name);
    if (it == paramMap.end()) {
      throw std::runtime_error("ParameterScratchpad: unknown parameter '" +
                               name + "'");
    }
    if (auto range = ranges.find(name); range != ranges.end()) {
      int32_t value = static_cast<int32_t>(bits);
      if (value < range->second.first || value > range->second.second)
        throw std::invalid_argument(
            "ParameterScratchpad: value " + std::to_string(value) + " of '" +
            name + "' is outside [" + std::to_string(range->second.first) +
            ", " + std::to_string(range->second.second) +
            "], which keeps its DMA transfers within their buffers");
      values[name] = value;
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

  /// Check the written values of each offset and length parameter pair that
  /// one DMA transfer uses. Throws std::invalid_argument if they take the
  /// transfer past the end of its buffer.
  void validate() const {
    for (const JointBound &bound : jointBounds) {
      int64_t offset = values.at(bound.offset);
      int64_t length = values.at(bound.length);
      int64_t end = offset + bound.lengthStep * length;
      if (end > bound.max)
        throw std::invalid_argument(
            "ParameterScratchpad: '" + bound.offset + "' = " +
            std::to_string(offset) + " plus " +
            std::to_string(bound.lengthStep) + " * '" + bound.length +
            "' = " + std::to_string(length) + " is " + std::to_string(end) +
            ", above " + std::to_string(bound.max) +
            ", so a DMA transfer using both runs past its buffer");
    }
  }

#if TEST_UTILS_USE_XRT
  /// Validate, then sync the scratchpad buffer to device. Call after all writes
  /// for this run.
  void sync() {
    validate();
    scratchpadBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
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
#if TEST_UTILS_USE_XRT
  xrt::bo scratchpadBo;
#endif
  uint32_t *boMap = nullptr;
  size_t scratchpadSizeBytes = 0;
  std::unordered_map<std::string, uint8_t> paramMap;
  std::unordered_set<std::string> coreParams; // params with kind="core"
  std::unordered_map<std::string, std::pair<int32_t, int32_t>> ranges;
  std::unordered_map<std::string, int32_t> values; // of the ranged params
  struct JointBound {
    std::string length, offset;
    int64_t lengthStep, max;
  };
  std::vector<JointBound> jointBounds;

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
    //   <name> <state_table_idx> <type> <kind> <min> <max>
    //   ...
    //   <num_joint_bounds>
    //   <length> <offset> <length_step> <max>
    //   ...
    // where kind is "core" or "addr", and <min> <max> is "- -" for a
    // parameter with no range. A joint bound requires
    // offset + length_step * length <= max.
    unsigned numParams = 0;
    file >> numParams;
    scratchpadSizeBytes = numParams * 4;

    for (unsigned i = 0; i < numParams; i++) {
      std::string name, type, kind, min, max;
      unsigned idx;
      file >> name >> idx >> type >> kind >> min >> max;
      if (!file)
        throw std::runtime_error("ParameterScratchpad: malformed entry " +
                                 std::to_string(i) + " in '" + path + "'");
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
      if (min != "-") {
        try {
          ranges[name] = {std::stoi(min), std::stoi(max)};
        } catch (const std::logic_error &) {
          throw std::runtime_error("ParameterScratchpad: malformed range '" +
                                   min + " " + max + "' for parameter '" +
                                   name + "' in '" + path + "'");
        }
        values[name] = 0;
      }
    }

    unsigned numJointBounds = 0;
    file >> numJointBounds;
    if (!file)
      throw std::runtime_error(
          "ParameterScratchpad: missing joint bound count in '" + path + "'");
    for (unsigned i = 0; i < numJointBounds; i++) {
      JointBound bound;
      file >> bound.length >> bound.offset >> bound.lengthStep >> bound.max;
      if (!file)
        throw std::runtime_error("ParameterScratchpad: malformed joint bound " +
                                 std::to_string(i) + " in '" + path + "'");
      for (const std::string &name : {bound.length, bound.offset})
        if (!ranges.count(name))
          throw std::runtime_error("ParameterScratchpad: joint bound " +
                                   std::to_string(i) + " names '" + name +
                                   "', which has no range, in '" + path + "'");
      jointBounds.push_back(bound);
    }
  }
};

} // namespace test_utils

#endif // AIE_RUNTIME_TEST_LIB_PARAMETER_SCRATCHPAD_H
