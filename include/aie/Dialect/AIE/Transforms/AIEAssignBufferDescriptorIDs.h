//===- AIEAssignBufferDescriptorIDs.h ---------------------------*- C++ -*-===//
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
#include "aie/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "aie-assign-bd-ids"
#define EVEN_BD_ID_START 0
#define ODD_BD_ID_START 24

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct BdIdGenerator {
  BdIdGenerator(int col, int row, const AIETargetModel &targetModel);

  int32_t nextBdId(int channelIndex);

  void assignBdId(int32_t bdId);

  bool bdIdAlreadyAssigned(int32_t bdId);

  int col;
  int row;
  int oddBdId = ODD_BD_ID_START;
  int evenBdId = EVEN_BD_ID_START;
  bool isMemTile;
  std::set<int32_t> alreadyAssigned;
};

#endif