//===- ResourceTests.cpp - physical resource unit tests -------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/Resource.h"
#include "gtest/gtest.h"

using namespace xilinx::phy::connectivity;

namespace {

TEST(Resource, Parses) {
  Resource r("tile/1.1/bank/0/buffer");
  EXPECT_EQ(r.key, "buffer");
  EXPECT_EQ(r.metadata["tile"], "1.1");
  EXPECT_EQ(r.metadata["bank"], "0");
}

TEST(Resource, Serializes) {
  Resource r("buffer", {{"tile", "1.1"}});
  EXPECT_EQ(r.toString(), "tile/1.1/buffer");
}

} // namespace
