//===- TxnEncoding.h - Standalone TXN instruction encoding ------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Header-only library for encoding AI Engine TXN (transaction) instructions.
// This has ZERO dependencies on MLIR or LLVM and can be used standalone in
// host applications to generate TXN binaries at runtime. It is the single
// in-tree source of truth for the instruction format, used by both the
// compiler (AIETargetNPU.cpp) and generated host code.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_RUNTIME_TXNENCODING_H
#define AIE_RUNTIME_TXNENCODING_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aie_runtime {

// Transaction opcodes for the firmware TXN format the compiler currently
// targets.
//
// These DO NOT match the third_party/aie-rt xaie_txn.h enum, which is an older
// layout (CONFIG_SHIMDMA_BD=5, no NOOP/PREEMPT/LOADPDI block). They match the
// newer firmware opcodes the compiler emits (e.g. CREATE_SCRATCHPAD=0x0A,
// UPDATE_REG=0x0C).
//
// DRIFT WARNING: when the firmware TXN format changes, update these values
// here. Do not "correct" them against the submodule's xaie_txn.h — it tracks a
// different (older) numbering and reconciling to it would break the emitted
// transactions.
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

// Upper bound on a tile's buffer-descriptor count across all tile types, sizing
// the pool's fixed storage. This is a standalone header (no MLIR / target model
// here), so the ACTUAL per-tile count is supplied at runtime by the compiler via
// bd_pool_init(getNumBDs(col,row)) -- this constant only bounds the array. AIE2
// memtiles have the most BDs (48); shim/core have 16. Keep this >= the largest
// getNumBDs any target model returns.
constexpr uint32_t kMaxBDsPerTile = 48;

// Runtime buffer-descriptor free-list pool, for dynamic runtime sequences whose
// BD IDs are chosen at runtime (a rolled scf.for the compiler cannot unroll).
// The static allocator assigns BD IDs at compile time; this is its runtime
// counterpart. `head` is the number of currently-free IDs at the front of
// `free_ids`; pop takes from the top, push returns to it.
struct BdPool {
  uint32_t free_ids[kMaxBDsPerTile];
  int head;
};

// Initialize a pool with `n` free IDs. `n` is the tile's BD count, which the
// compiler reads from the target model (getNumBDs) and passes in. IDs are
// stacked so that pop() hands out the LOWEST free id first (id 0, then 1, ...),
// matching the static allocator's lowest-free-first order -- so a first pop from
// a fresh pool equals a pinned bd_id = 0.
inline BdPool bd_pool_init(uint32_t n) {
  BdPool p;
  p.head = 0;
  uint32_t count = n < kMaxBDsPerTile ? n : kMaxBDsPerTile;
  for (uint32_t i = 0; i < count; ++i)
    p.free_ids[p.head++] = count - 1 - i; // top of stack is id 0
  return p;
}

// Pop a free BD ID into `out`. Returns false if the pool is empty -- the
// generated builder turns that into a `return std::nullopt`, so a runtime
// working set that exceeds the tile's BD count yields no stream rather than a
// silently-corrupt one.
inline bool bd_pool_pop(BdPool &p, uint32_t &out) {
  if (p.head == 0)
    return false;
  out = p.free_ids[--p.head];
  return true;
}

// Return a BD ID to the pool for reuse.
inline void bd_pool_push(BdPool &p, uint32_t id) {
  if (p.head < static_cast<int>(kMaxBDsPerTile))
    p.free_ids[p.head++] = id;
}

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
//
// The destination tile is encoded twice in the firmware blockwrite format: once
// in the upper bits of the absolute `addr` (column/row folded in by the
// compiler before this point) and once in the dedicated col/row field at
// word[1]. Callers that build an absolute address from a tile pass that same
// (col, row) here so both representations agree; callers with a purely flat
// address pass 0/0.
inline void txn_append_blockwrite(std::vector<uint32_t> &txn, uint32_t addr,
                                  const uint32_t *data, size_t count,
                                  uint32_t col = 0, uint32_t row = 0) {
  const unsigned headerSize = 4;
  size_t pos = txn.size();
  txn.resize(pos + headerSize + count, 0);
  txn[pos + 0] = TXN_OPC_BLOCKWRITE;
  txn[pos + 1] = (col & 0xff) | ((row & 0xff) << 8);
  txn[pos + 2] = addr;
  txn[pos + 3] = static_cast<uint32_t>((headerSize + count) * sizeof(uint32_t));
  for (size_t i = 0; i < count; ++i)
    txn[pos + headerSize + i] = data[i];
}

// Append a 12-word address_patch (DDR_PATCH) instruction.
inline void txn_append_address_patch(std::vector<uint32_t> &txn, uint32_t addr,
                                     int32_t arg_idx, uint32_t arg_plus) {
  size_t pos = txn.size();
  txn.resize(pos + 12, 0);
  txn[pos + 0] = TXN_OPC_DDR_PATCH;     // opcode
  txn[pos + 1] = 12 * sizeof(uint32_t); // operation size
  // pos+2..4 are reserved (zero)
  txn[pos + 5] = 0;    // action (0 = patch)
  txn[pos + 6] = addr; // register address to patch
  // pos+7 is reserved (zero)
  txn[pos + 8] = static_cast<uint32_t>(arg_idx); // buffer argument index
  // pos+9 is reserved (zero)
  txn[pos + 10] = arg_plus; // byte offset into buffer
  // pos+11 is reserved (zero)
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

// Reserve 4 placeholder words for the TXN header. Call this BEFORE appending
// any instructions, so that the header space is already allocated.
inline void txn_init(std::vector<uint32_t> &txn) { txn.resize(4, 0); }

// Finalize the 4-word TXN header in-place. Call this AFTER all instructions
// are appended. The first 4 words must have been reserved by txn_init().
// `op_count` is the number of operations appended.
inline void txn_prepend_header(std::vector<uint32_t> &txn, uint32_t op_count,
                               TxnDeviceInfo info = {}) {
  if (txn.size() < 4)
    txn.resize(4, 0);
  txn[0] = (static_cast<uint32_t>(info.numRows) << 24) |
           (static_cast<uint32_t>(info.devGen) << 16) |
           (static_cast<uint32_t>(info.minor) << 8) |
           static_cast<uint32_t>(info.major);
  txn[1] = (static_cast<uint32_t>(info.numMemTileRows) << 8) |
           static_cast<uint32_t>(info.numCols);
  txn[2] = op_count;
  txn[3] = static_cast<uint32_t>(txn.size() * sizeof(uint32_t));
}

} // namespace aie_runtime

#endif // AIE_RUNTIME_TXNENCODING_H
