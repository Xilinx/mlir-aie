//===- npu_instructions.h - NPU Instruction Encoding -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Standalone header-only library for NPU transaction instruction encoding.
// This has NO dependencies on LLVM or MLIR and can be used by:
// - MLIR compiler (AIETargetNPU.cpp)
// - Generated runtime code (.so files)
// - Standalone host applications
//
//===----------------------------------------------------------------------===//

#ifndef NPU_INSTRUCTIONS_H
#define NPU_INSTRUCTIONS_H

#include <cstdint>
#include <vector>

namespace aie {
namespace npu {

// Transaction opcodes from XAie_TxnOpcode
enum class Opcode : uint8_t {
  WRITE = 0,
  BLOCKWRITE = 1,
  BLOCKSET = 2,
  MASKWRITE = 3,
  MASKPOLL = 4,
  NOOP = 5,
  PREEMPT = 6,
  MASKPOLL_BUSY = 7,
  LOADPDI = 8,
  LOAD_PM_START = 9,
  CREATE_SCRATCHPAD = 10,
  UPDATE_STATE_TABLE = 11,
  UPDATE_REG = 12,
  UPDATE_SCRATCH = 13,
  CONFIG_SHIMDMA_BD = 14,
  CONFIG_SHIMDMA_DMABUF_BD = 15,
  // Custom opcodes
  CUSTOM_OP_BEGIN = 128,
  CUSTOM_OP_TCT = 128,        // Task Complete Token
  CUSTOM_OP_DDR_PATCH = 129,  // DDR address patch
  CUSTOM_OP_READ_REGS = 130,
  CUSTOM_OP_RECORD_TIMER = 131,
  CUSTOM_OP_MERGE_SYNC = 132,
};

/// Helper to reserve space and get mutable tail of instruction vector
inline uint32_t* reserveAndGetTail(std::vector<uint32_t> &instructions,
                                   size_t tailSize) {
  size_t oldSize = instructions.size();
  instructions.resize(oldSize + tailSize, 0);
  return instructions.data() + oldSize;
}

/// Append NPU write32 instruction
/// Writes a 32-bit value to a 32-bit address
inline void appendWrite32(std::vector<uint32_t> &instructions,
                          uint32_t address, uint32_t value) {
  uint32_t *words = reserveAndGetTail(instructions, 6);
  words[0] = static_cast<uint32_t>(Opcode::WRITE);
  words[1] = 0; // Reserved
  words[2] = address;
  words[3] = 0; // Extra bits for register offset
  words[4] = value;
  words[5] = 6 * sizeof(uint32_t); // Operation size
}

/// Append NPU maskwrite32 instruction
/// Writes a 32-bit value with mask to a 32-bit address
inline void appendMaskWrite32(std::vector<uint32_t> &instructions,
                              uint32_t address, uint32_t value, uint32_t mask) {
  uint32_t *words = reserveAndGetTail(instructions, 7);
  words[0] = static_cast<uint32_t>(Opcode::MASKWRITE);
  words[1] = 0;
  words[2] = address;
  words[3] = 0;
  words[4] = value;
  words[5] = mask;
  words[6] = 7 * sizeof(uint32_t);
}

/// Append NPU sync instruction (Task Complete Token)
/// Waits for a task completion token from specified tile/channel
inline void appendSync(std::vector<uint32_t> &instructions,
                       uint32_t column, uint32_t row,
                       uint32_t direction, uint32_t channel,
                       uint32_t columnNum, uint32_t rowNum) {
  uint32_t *words = reserveAndGetTail(instructions, 4);
  words[0] = static_cast<uint32_t>(Opcode::CUSTOM_OP_TCT);
  words[1] = 4 * sizeof(uint32_t); // Operation size
  words[2] = (direction & 0xff) | ((row & 0xff) << 8) | ((column & 0xff) << 16);
  words[3] = ((rowNum & 0xff) << 8) | ((columnNum & 0xff) << 16) | ((channel & 0xff) << 24);
}

/// Append NPU block write instruction
/// Writes a block of data to specified address
inline void appendBlockWrite(std::vector<uint32_t> &instructions,
                             uint32_t address,
                             const uint32_t *data, size_t dataSize) {
  uint32_t *words = reserveAndGetTail(instructions, 4 + dataSize);
  words[0] = static_cast<uint32_t>(Opcode::BLOCKWRITE);
  words[1] = 0; // Optional tile coordinates
  words[2] = address;
  words[3] = (4 + dataSize) * sizeof(uint32_t); // Operation size
  // Copy data
  for (size_t i = 0; i < dataSize; ++i) {
    words[4 + i] = data[i];
  }
}

/// Append NPU load PDI instruction
inline void appendLoadPdi(std::vector<uint32_t> &instructions,
                          uint32_t id, uint32_t size, uint64_t address) {
  uint32_t *words = reserveAndGetTail(instructions, 4);
  words[0] = static_cast<uint32_t>(Opcode::LOADPDI) | (id << 16);
  words[1] = size;
  words[2] = static_cast<uint32_t>(address);        // Low 32 bits
  words[3] = static_cast<uint32_t>(address >> 32);  // High 32 bits
}

/// Append NPU address patch instruction
inline void appendAddressPatch(std::vector<uint32_t> &instructions,
                               uint32_t patchAddr, uint32_t argIdx,
                               uint32_t argPlusOffset) {
  uint32_t *words = reserveAndGetTail(instructions, 12);
  words[0] = static_cast<uint32_t>(Opcode::CUSTOM_OP_DDR_PATCH);
  words[1] = 12 * sizeof(uint32_t); // Operation size
  words[2] = 0;
  words[3] = 0;
  words[4] = 0;
  words[5] = 0; // Action
  words[6] = patchAddr;
  words[7] = 0;
  words[8] = argIdx;
  words[9] = 0;
  words[10] = argPlusOffset;
  words[11] = 0;
}

/// Append NPU preempt instruction
inline void appendPreempt(std::vector<uint32_t> &instructions, uint32_t level) {
  uint32_t *words = reserveAndGetTail(instructions, 1);
  words[0] = static_cast<uint32_t>(Opcode::PREEMPT) | (level << 8);
}

/// Prepend transaction header to instruction sequence
/// This should be called after all instructions are appended
inline void prependHeader(std::vector<uint32_t> &instructions,
                          uint32_t numRows = 6, uint32_t numCols = 5,
                          uint32_t devGen = 4, uint32_t numMemTileRows = 1) {
  // Estimate operation count (each op is variable size, this is approximate)
  uint32_t totalSize = instructions.size() * sizeof(uint32_t);
  uint32_t count = instructions.size() / 4; // Rough estimate

  std::vector<uint32_t> header = {
    (numRows << 24) | (devGen << 16) | (0 << 8) | 1,  // Version info
    (numMemTileRows << 8) | numCols,                  // Geometry
    count,                                             // Operation count
    totalSize                                          // Total size in bytes
  };

  instructions.insert(instructions.begin(), header.begin(), header.end());
}

} // namespace npu
} // namespace aie

#endif // NPU_INSTRUCTIONS_H
