//===- ResourceListTests.cpp - resource list unit tests -------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/ResourceList.h"
#include "gtest/gtest.h"

using namespace xilinx::phy::connectivity;

namespace {

TEST(ResourceList, Parses) {
  ResourceList i("tile/1.1/bank/0/buffer,tile/1.1/lock");
  EXPECT_EQ(i.phys.front().key, "buffer");
  EXPECT_EQ(i.phys.front().metadata["tile"], "1.1");
  EXPECT_EQ(i.phys.front().metadata["bank"], "0");
  EXPECT_EQ(i.phys.back().key, "lock");
  EXPECT_EQ(i.phys.back().metadata["tile"], "1.1");
}

TEST(ResourceList, Serializes) {
  ResourceList i;
  i.phys.push_back(PhysicalResource("buffer", {{"tile", "1.1"}}));
  i.phys.push_back(PhysicalResource("lock", {{"tile", "1.1"}}));
  EXPECT_EQ(i.toString(), "tile/1.1/buffer,tile/1.1/lock");
}

} // namespace
