//===- AIEGenerateColumnControlOverlay.h ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_ASSIGN_BUFFER_DESCRIPTOR_IDS_H
#define AIE_ASSIGN_BUFFER_DESCRIPTOR_IDS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"

using namespace mlir;
using namespace xilinx::AIE;

// Populate column control streaming interconnect overlay
void populateAIEColumnControlOverlay(DeviceOp &device);

// AIE arch-specific row id to shim dma mm2s channel mapping. All shim mm2s
// channels were assumed to be available for control packet flow routing (i.e.
// not reserved by any aie.flow circuit-switched routing).
DenseMap<int, int> getRowToShimChanMap(const AIETargetModel &targetModel,
                                       WireBundle bundle);

// AIE arch-specific tile id to controller id mapping. Users can use those
// packet ids for design but run into risk of deadlocking control packet flows.
DenseMap<TileID, int>
getTileToControllerIdMap(bool clColumnWiseUniqueIDs,
                         const AIETargetModel &targetModel);

#endif
