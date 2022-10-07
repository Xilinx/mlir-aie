//===- PhysicalOps.td --------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_OPS_TD
#define MLIR_PHY_DIALECT_PHYSICAL_OPS_TD

include "phy/Dialect/Physical/PhysicalBase.td"

include "phy/Dialect/Physical/Operations/AsyncOps.td"
include "phy/Dialect/Physical/Operations/BufferOps.td"
include "phy/Dialect/Physical/Operations/BusCacheOps.td"
include "phy/Dialect/Physical/Operations/BusOps.td"
include "phy/Dialect/Physical/Operations/CoreOps.td"
include "phy/Dialect/Physical/Operations/EndOps.td"
include "phy/Dialect/Physical/Operations/LockOps.td"
include "phy/Dialect/Physical/Operations/StreamDmaOps.td"
include "phy/Dialect/Physical/Operations/StreamHubOps.td"
include "phy/Dialect/Physical/Operations/StreamOps.td"

#endif  // MLIR_PHY_DIALECT_PHYSICAL_OPS_TD
