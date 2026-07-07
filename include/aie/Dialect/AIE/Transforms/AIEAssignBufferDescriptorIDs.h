//===- AIEAssignBufferDescriptorIDs.h ---------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_ASSIGN_BUFFER_DESCRIPTOR_IDS_H
#define AIE_ASSIGN_BUFFER_DESCRIPTOR_IDS_H

#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include <optional>
#include <set>

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

  // Save / restore / merge the set of assigned ids, for exploring mutually
  // exclusive allocation paths (e.g. scf.if arms) without them interfering.
  // AssignedState is opaque; callers pass it back verbatim.
  using AssignedState = std::set<uint32_t>;
  AssignedState saveAssigned() const;
  void restoreAssigned(AssignedState state);
  void mergeAssigned(const AssignedState &state);
};

#endif
