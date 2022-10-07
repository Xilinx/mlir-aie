//===- SpatialOps.td ---------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_SPATIAL_OPS_TD
#define MLIR_PHY_DIALECT_SPATIAL_OPS_TD

include "phy/Dialect/Spatial/SpatialBase.td"

include "phy/Dialect/Spatial/Operations/FlowAccessOps.td"
include "phy/Dialect/Spatial/Operations/FlowControlOps.td"
include "phy/Dialect/Spatial/Operations/GraphOps.td"

#endif  // MLIR_PHY_DIALECT_SPATIAL_OPS_TD
