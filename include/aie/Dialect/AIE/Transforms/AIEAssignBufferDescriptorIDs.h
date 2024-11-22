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

#include "aie/Dialect/AIE/IR/AIETargetModel.h"

using namespace xilinx::AIE;

struct BdIdGenerator {
  const int col;
  const int row;
  const AIETargetModel &targetModel;
  std::set<uint32_t> alreadyAssigned;

  BdIdGenerator(int col, int row, const AIETargetModel &targetModel);

  std::optional<uint32_t> nextBdId(int channelIndex);

  void assignBdId(uint32_t bdId);

  bool bdIdAlreadyAssigned(uint32_t bdId);

  void freeBdId(uint32_t bdId);
};

#endif
