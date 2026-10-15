//===- AIEGenerateColumnControlOverlay.h ------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_ASSIGN_BUFFER_DESCRIPTOR_IDS_H
#define AIE_ASSIGN_BUFFER_DESCRIPTOR_IDS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include <optional>
#include <string>

using namespace mlir;
using namespace xilinx::AIE;

// Populate column control streaming interconnect overlay
void populateAIEColumnControlOverlay(DeviceOp &device);

// Name of the shim DMA allocation the runtime sequence issues or receives
// control packets through.
std::string getCtrlPktAllocName(int col, bool isShimMM2S, int chan);

// Found by its is_ctrl_pkt_overlay mark, not by name; null if none.
ShimDMAAllocationOp getCtrlPktShimAlloc(DeviceOp device, int col);

// AIE arch-specific tile id to controller id mapping. Users can use those
// packet ids for design but run into risk of deadlocking control packet flows.
DenseMap<TileID, int>
getTileToControllerIdMap(bool clColumnWiseUniqueIDs,
                         const AIETargetModel &targetModel);

#endif
