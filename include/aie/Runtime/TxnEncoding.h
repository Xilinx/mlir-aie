//===- TxnEncoding.h - Standalone TXN instruction encoding -------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Header-only library for encoding AI Engine TXN (transaction) instructions.
// This has ZERO dependencies on MLIR or LLVM and can be used standalone in
// host applications to generate TXN binaries at runtime.
//
// The encoding logic is extracted from AIETargetNPU.cpp and is the single
// source of truth for instruction format, used by both the compiler and
// generated host code.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_RUNTIME_TXNENCODING_H
#define AIE_RUNTIME_TXNENCODING_H

#include <cstdint>
#include <cstring>
#include <vector>

namespace aie_runtime {

// Transaction opcodes - mirroring xaie_txn.h from aie-rt.
// See aie-rt commit a6196eb, xaiengine/xaie_txn.h.
enum TxnOpcode : uint32_t {
  TXN_OPC_WRITE = 0,
  TXN_OPC_BLOCKWRITE = 1,
  TXN_OPC_BLOCKSET = 2,
  TXN_OPC_MASKWRITE = 3,
  TXN_OPC_MASKPOLL = 4,
  TXN_OPC_NOOP = 5,
  TXN_OPC_PREEMPT = 6,
  TXN_OPC_MASKPOLL_BUSY = 7,
  TXN_OPC_LOADPDI = 8,
  TXN_OPC_LOAD_PM_START = 9,
  TXN_OPC_CREATE_SCRATCHPAD = 10,
  TXN_OPC_UPDATE_STATE_TABLE = 11,
  TXN_OPC_UPDATE_REG = 12,
  TXN_OPC_UPDATE_SCRATCH = 13,
  TXN_OPC_CONFIG_SHIMDMA_BD = 14,
  TXN_OPC_CONFIG_SHIMDMA_DMABUF_BD = 15,
  TXN_OPC_CUSTOM_OP_BEGIN = 1U << 7U,
  TXN_OPC_TCT = TXN_OPC_CUSTOM_OP_BEGIN,
  TXN_OPC_DDR_PATCH = TXN_OPC_CUSTOM_OP_BEGIN + 1,
  TXN_OPC_READ_REGS = TXN_OPC_CUSTOM_OP_BEGIN + 2,
  TXN_OPC_RECORD_TIMER = TXN_OPC_CUSTOM_OP_BEGIN + 3,
  TXN_OPC_MERGE_SYNC = TXN_OPC_CUSTOM_OP_BEGIN + 4,
  TXN_OPC_CUSTOM_OP_NEXT = TXN_OPC_CUSTOM_OP_BEGIN + 5,
  TXN_OPC_LOAD_PM_END_INTERNAL = 200,
  TXN_OPC_CUSTOM_OP_MAX = 255,
};

// Device information for the TXN header.
struct TxnDeviceInfo {
  uint8_t major = 0;
  uint8_t minor = 1;
  uint8_t devGen = 3; // 3 = NPU (PHX/HWK), 4 = NPU2 (STX/KRK)
  uint8_t numRows = 6;
  uint8_t numCols = 5;
  uint8_t numMemTileRows = 1;
};

// Append a 6-word write32 instruction.
inline void txn_append_write32(std::vector<uint32_t> &txn, uint32_t addr,
                               uint32_t val) {
  size_t pos = txn.size();
  txn.resize(pos + 6, 0);
  txn[pos + 0] = TXN_OPC_WRITE;
  // txn[pos + 1] is reserved (0)
  txn[pos + 2] = addr;
  txn[pos + 3] = 0; // extra bits for reg offset
  txn[pos + 4] = val;
  txn[pos + 5] = 6 * sizeof(uint32_t); // operation size
}

// Append a 7-word maskwrite32 instruction.
inline void txn_append_maskwrite32(std::vector<uint32_t> &txn, uint32_t addr,
                                   uint32_t val, uint32_t mask) {
  size_t pos = txn.size();
  txn.resize(pos + 7, 0);
  txn[pos + 0] = TXN_OPC_MASKWRITE;
  // txn[pos + 1] is reserved (0)
  txn[pos + 2] = addr;
  txn[pos + 3] = 0;
  txn[pos + 4] = val;
  txn[pos + 5] = mask;
  txn[pos + 6] = 7 * sizeof(uint32_t); // operation size
}

// Append a 4-word sync (TCT) instruction.
inline void txn_append_sync(std::vector<uint32_t> &txn, uint32_t col,
                            uint32_t row, uint32_t dir, uint32_t chan,
                            uint32_t ncol, uint32_t nrow) {
  size_t pos = txn.size();
  txn.resize(pos + 4, 0);
  txn[pos + 0] = TXN_OPC_TCT;
  txn[pos + 1] = 4 * sizeof(uint32_t); // operation size
  txn[pos + 2] = (dir & 0xff) | ((row & 0xff) << 8) | ((col & 0xff) << 16);
  txn[pos + 3] =
      ((nrow & 0xff) << 8) | ((ncol & 0xff) << 16) | ((chan & 0xff) << 24);
}

// Append a variable-length blockwrite instruction.
// `data` points to `count` uint32_t words of payload.
inline void txn_append_blockwrite(std::vector<uint32_t> &txn, uint32_t addr,
                                  const uint32_t *data, size_t count) {
  const unsigned headerSize = 4;
  size_t pos = txn.size();
  txn.resize(pos + headerSize + count, 0);
  txn[pos + 0] = TXN_OPC_BLOCKWRITE;
  // txn[pos + 1] is col/row (set to 0; caller can set if needed)
  txn[pos + 2] = addr;
  txn[pos + 3] = static_cast<uint32_t>((headerSize + count) * sizeof(uint32_t));
  for (size_t i = 0; i < count; ++i)
    txn[pos + headerSize + i] = data[i];
}

// Append a 12-word address_patch (DDR_PATCH) instruction.
inline void txn_append_address_patch(std::vector<uint32_t> &txn, uint32_t addr,
                                     int32_t arg_idx, int32_t arg_plus) {
  size_t pos = txn.size();
  txn.resize(pos + 12, 0);
  txn[pos + 0] = TXN_OPC_DDR_PATCH;
  txn[pos + 1] = 12 * sizeof(uint32_t); // operation size
  txn[pos + 5] = 0;                     // action
  txn[pos + 6] = addr;
  txn[pos + 8] = static_cast<uint32_t>(arg_idx);
  txn[pos + 10] = static_cast<uint32_t>(arg_plus);
}

// Append a 4-word loadpdi instruction.
inline void txn_append_loadpdi(std::vector<uint32_t> &txn, uint32_t id,
                               uint32_t size, uint64_t addr) {
  size_t pos = txn.size();
  txn.resize(pos + 4, 0);
  txn[pos + 0] = TXN_OPC_LOADPDI | (id << 16);
  txn[pos + 1] = size;
  txn[pos + 2] = static_cast<uint32_t>(addr);
  txn[pos + 3] = static_cast<uint32_t>(addr >> 32);
}

// Append a 1-word preempt instruction.
inline void txn_append_preempt(std::vector<uint32_t> &txn, uint32_t level) {
  txn.push_back(TXN_OPC_PREEMPT | (level << 8));
}

// Prepend a 4-word TXN header. Call this AFTER all instructions are appended.
// `op_count` is the number of operations appended.
inline void txn_prepend_header(std::vector<uint32_t> &txn, uint32_t op_count,
                               TxnDeviceInfo info = {}) {
  uint32_t header[4];
  header[0] = (static_cast<uint32_t>(info.numRows) << 24) |
              (static_cast<uint32_t>(info.devGen) << 16) |
              (static_cast<uint32_t>(info.minor) << 8) |
              static_cast<uint32_t>(info.major);
  header[1] = (static_cast<uint32_t>(info.numMemTileRows) << 8) |
              static_cast<uint32_t>(info.numCols);
  header[2] = op_count;
  header[3] = static_cast<uint32_t>((txn.size() + 4) * sizeof(uint32_t));
  txn.insert(txn.begin(), header, header + 4);
}

} // namespace aie_runtime

#endif // AIE_RUNTIME_TXNENCODING_H
