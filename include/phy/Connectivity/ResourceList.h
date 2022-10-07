//===- ResourceList.h -------------------------------------------*- C++ -*-===//
//
// This file defines a list of resources.
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_CONNECTIVITY_RESOURCE_LIST_H
#define MLIR_PHY_CONNECTIVITY_RESOURCE_LIST_H

#include "phy/Connectivity/Resource.h"

#include <list>
#include <map>
#include <string>

namespace xilinx {
namespace phy {
namespace connectivity {

class ResourceList {
  /**
   * A list of Physical Resource.
   *
   * To serialize the list into a string, the resources are concatenated
   * seperated by a comma. For example, "tile/7.0/buffer,tile/7.0/lock".
   */
  const std::string delim = ",";

public:
  std::list<PhysicalResource> phys;

  ResourceList() {}
  ResourceList(std::string serialized);
  ResourceList(std::list<PhysicalResource> phys) : phys(phys) {}
  std::string toString();
};

} // namespace connectivity
} // namespace phy
} // namespace xilinx

#endif // MLIR_PHY_CONNECTIVITY_RESOURCE_LIST_H
