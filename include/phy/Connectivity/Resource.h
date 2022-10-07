//===- Resource.h -----------------------------------------------*- C++ -*-===//
//
// This file defines the virtual resources, and the physical resources to be
// lowered from a virtual resource on a connectivity graph.
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_CONNECTIVITY_RESOURCE_H
#define MLIR_PHY_CONNECTIVITY_RESOURCE_H

#include <list>
#include <map>
#include <string>

namespace xilinx {
namespace phy {
namespace connectivity {

class Resource {
  /**
   * A resource contains two pieces of information: implementation method (key)
   * and target information (metadata).  The implementation method is target-
   * independent, for example, "buffer".  The target information is target-
   * dependent and is directly handled by the lowering passes.  It is
   * represented as a string map.  For example, "tile" = "7.0".
   *
   * To serialize the resource into a string, the target information is
   * concatenated first seperated by a slash.  The implementation method
   * follows with a slash seperated.  For example, "tile/7.0/buffer".
   */

  const std::string delim = "/";

public:
  std::string key;
  std::map<std::string, std::string> metadata;

  Resource() {}
  Resource(std::string serialized);
  Resource(std::string key, std::map<std::string, std::string> metadata)
      : key(key), metadata(metadata) {}
  std::string toString();
};

class PhysicalResource : public Resource {
  /**
   * A phyisical resource directly maps to the operations in the physical
   * dialect.  The key is the operation name and the metadata is directly
   * handled by the lowering passes as attribute strings.
   */
public:
  using Resource::Resource;
};

class VirtualResource : public Resource {
  /**
   * A virtual resource is the vertices on the connectivity graph, which is
   * afterwards lowered into physical resources.
   */
public:
  using Resource::Resource;
};

} // namespace connectivity
} // namespace phy
} // namespace xilinx

#endif // MLIR_PHY_CONNECTIVITY_RESOURCE_H
