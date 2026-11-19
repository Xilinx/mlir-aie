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
  std::optional<ArrayAttr> initValues = pool.getInitValues();
  std::optional<StringRef> fifo = pool.getFifoName();
  return needsLocks(pool.getDisableSynchronization(),
                    initValues ? initValues->size() : 0, pool.getDepth(),
                    filled.contains(pool),
                    fifo && rearmsByFifo.contains(*fifo));
}
