//===- Constraint.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_CONNECTIVITY_CONSTRAINT_H
#define MLIR_PHY_CONNECTIVITY_CONSTRAINT_H

#include <map>
#include <string>

namespace xilinx {
namespace phy {
namespace connectivity {

/**
 * A resource utilization is the occupation of a phyisical resource.
 * For example, a vertex can be marked {"count": 1, "bytes": 1024} utilization.
 */
using Utilization = std::map<std::string, int>;

/**
 * A capacity is the resource constraints of a phyisical resource.  When a
 * vertex occupies a phyisical resource, the resource utilization of the vertex
 * occupies the capacity of all its phyisical resources.  For example, if a
 * {"count": 1} vertex occupies a buffer and a lock, both the buffer and the
 * lock's count is reduced by 1..
 *
 * A capacity constraint is effective in the selection of virtual resource for
 * a spatial operation.
 */
using Capacity = std::map<std::string, int>;

/**
 * A target support is the implementation constraints of a phyisical resource.
 * For example, if a {"states": 2} lock is implemented, only lock_acquire<0>
 * and lock_acquire<1> will be used.  Another example, a {"width_bytes": 32}
 * stream limits the physical implementation to use this width.
 *
 * A target support constraint is effective in the selection of physical
 * resource for a virtual resource.
 */
using TargetSupport = std::map<std::string, int>;

} // namespace connectivity
} // namespace phy
} // namespace xilinx

#endif // MLIR_PHY_CONNECTIVITY_CONSTRAINT_H
