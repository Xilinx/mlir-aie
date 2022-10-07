//===- LiteralVector.h ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Support/LexicalCast.h"

#include <string>
#include <vector>

#ifndef MLIR_PHY_CONNECTIVITY_SERIALIZATION_LITERAL_VECTOR_H
#define MLIR_PHY_CONNECTIVITY_SERIALIZATION_LITERAL_VECTOR_H

namespace xilinx {
namespace phy {
namespace connectivity {

template <typename T, char delim = '.'> class LiteralVector {
  std::vector<T> data;

public:
  inline LiteralVector(std::vector<T> &data) : data(data) {}
  inline LiteralVector(std::string serialized) {
    size_t pos = 0;
    std::string token;
    while ((pos = serialized.find(delim)) != std::string::npos) {
      token = serialized.substr(0, pos);
      serialized.erase(0, pos + 1);
      data.push_back(lexicalCast<T>(token));
    }
    data.push_back(lexicalCast<T>(serialized));
  }

  inline std::vector<T> vec() { return data; }

  inline std::string str() {
    std::ostringstream result;
    bool first = true;

    for (auto d : data) {
      if (!first) {
        result << delim;
      }
      result << d;
      first = false;
    }

    return result.str();
  }
};

} // namespace connectivity
} // namespace phy
} // namespace xilinx

#endif // MLIR_PHY_CONNECTIVITY_SERIALIZATION_LITERAL_VECTOR_H
