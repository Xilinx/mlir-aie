//===- AIEPoolLocks.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEPoolLocks.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

PoolLocks::PoolLocks(DeviceOp device) {
  for (auto endpoint : device.getOps<ObjectFifoCoreEndpointOp>())
    if (!endpoint.drains())
      filled.insert(endpoint.getPoolOp());
  for (auto endpoint : device.getOps<ObjectFifoDmaEndpointOp>())
    if (!endpoint.drains())
      filled.insert(endpoint.getPoolOp());
  device.walk([&](AIEX::DmaChannelResetForOp reset) {
    // Split may already have pointed it at the fifo's shim endpoint.
    StringRef name = reset.getObjfifo();
    if (auto endpoint = dyn_cast_or_null<RouteEndpoint>(
            SymbolTable::lookupNearestSymbolFrom(device,
                                                 reset.getObjfifoAttr())))
      if (std::optional<StringRef> fifoName = endpoint.getFifoName())
        name = *fifoName;
    rearmsByFifo[name].push_back(reset);
  });
}

bool PoolLocks::needsLocks(ObjectFifoPoolOp pool) const {
  if (pool.getDisableSynchronization())
    return false;
  std::optional<ArrayAttr> initValues = pool.getInitValues();
  int64_t full = initValues ? initValues->size() : 0;
  // Locks would let a pool of constants be read only as many times as they
  // count, so a second launch would hang, unless a reset re-arms them.
  std::optional<StringRef> fifo = pool.getFifoName();
  return full != pool.getDepth() || full <= 0 || filled.contains(pool) ||
         (fifo && rearmsByFifo.contains(*fifo));
}
