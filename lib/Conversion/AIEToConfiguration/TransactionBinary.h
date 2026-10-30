//===- TransactionBinary.h --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private transaction parser shared by configuration conversion and
// core-issued DMA lowering. The latter records aie-rt calls as transactions
// and reuses this parser to extract the register writes executed by the core.

#ifndef AIE_CONVERSION_AIETOCONFIGURATION_TRANSACTIONBINARY_H
#define AIE_CONVERSION_AIETOCONFIGURATION_TRANSACTIONBINARY_H

#include "mlir/IR/Location.h"

#include <cstdint>
#include <optional>
#include <vector>

extern "C" {
#include "xaiengine/xaiegbl.h"
// xaie_txn.h needs the types and declarations from xaiegbl.h first.
#include "xaiengine/xaie_txn.h"
}

namespace xilinx::AIE {

// A TransactionBinaryOperation encapsulates an aie-rt XAie_TxnCmd struct and
// any additional metadata needed for custom operations that do not map cleanly
// onto the core command fields.
struct TransactionBinaryOperation {
  struct XAie_TxnCmd cmd = {};

  struct SyncPayload {
    int32_t column;
    int32_t row;
    int32_t direction;
    int32_t channel;
    int32_t columnCount;
    int32_t rowCount;
  };

  struct LoadPdiPayload {
    uint32_t id;
    uint32_t size;
    uint64_t address;
  };

  struct AddressPatchPayload {
    uint32_t action;
    uint32_t addr;
    int32_t argIdx;
    // Words 10-11 of the op; see txn_append_address_patch.
    uint64_t argPlus;
  };

  std::optional<SyncPayload> sync;
  std::optional<LoadPdiPayload> loadPdi;
  std::optional<AddressPatchPayload> addressPatch;
  std::optional<mlir::Location> sourceLoc;

  TransactionBinaryOperation() = default;

  TransactionBinaryOperation(XAie_TxnOpcode opc, uint32_t mask, uint64_t addr,
                             uint32_t value, const uint8_t *data,
                             uint32_t size) {
    cmd.Opcode = opc;
    cmd.Mask = mask;
    cmd.RegOff = addr;
    cmd.Value = value;
    cmd.DataPtr = reinterpret_cast<uint64_t>(data);
    cmd.Size = size;
  }
};

/// Parse an exported transaction. Returns the header's column count and fills
/// `ops`, or std::nullopt when the blob does not parse.
std::optional<int>
parseTransactionBinary(const std::vector<uint8_t> &data,
                       std::vector<TransactionBinaryOperation> &ops);

} // namespace xilinx::AIE

#endif // AIE_CONVERSION_AIETOCONFIGURATION_TRANSACTIONBINARY_H
