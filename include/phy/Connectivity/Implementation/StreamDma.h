//===- StreamDma.h ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_STREAM_DMA_H
#define MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_STREAM_DMA_H

#include "phy/Connectivity/Implementation.h"
#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Dialect/Spatial/SpatialDialect.h"

#include <map>
#include <memory>
#include <utility>

namespace xilinx {
namespace phy {
namespace connectivity {

class StreamDmaImplementation : public Implementation {

  // Overrides
protected:
  mlir::Operation *createOperation() override;

public:
  using Implementation::Implementation;
  ~StreamDmaImplementation() override {}

  void addPredecessor(std::weak_ptr<Implementation> pred, mlir::Operation *src,
                      mlir::Operation *dest) override;
  void addSuccessor(std::weak_ptr<Implementation> succ, mlir::Operation *src,
                    mlir::Operation *dest) override;

protected:
  // a stream dma can only connect to one stream, either istream or ostream
  std::weak_ptr<Implementation> istream;
  std::weak_ptr<Implementation> ostream;

  // but it can be connected to multiple buffers and locks for sequential dmas
  // buffers/locks[{src, dest}] == buffer/lock
  std::map<std::pair<mlir::Operation *, mlir::Operation *>,
           std::weak_ptr<Implementation>>
      buffers;
  std::map<std::pair<mlir::Operation *, mlir::Operation *>,
           std::weak_ptr<Implementation>>
      locks;

  void addStorage(std::weak_ptr<Implementation> storage, mlir::Operation *src,
                  mlir::Operation *dest);
};

} // namespace connectivity
} // namespace phy
} // namespace xilinx

#endif // MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_STREAM_DMA_H
