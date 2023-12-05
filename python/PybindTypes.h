//===- PybindTypes.h --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_PYBINDTYPES_H
#define AIE_PYBINDTYPES_H

#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace xilinx::AIE {
using Flow = struct Flow {
  PathEndPoint src;
  std::vector<PathEndPoint> dsts;

  friend std::ostream &operator<<(std::ostream &os, const Flow &s) {
    os << "Flow(" << s.src << ": {"
       << join(map_range(llvm::ArrayRef(s.dsts),
                         [](const PathEndPoint &pe) { return to_string(pe); }),
               ", ")
       << "})";
    return os;
  }

  GENERATE_TO_STRING(Flow)

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Flow &s) {
    os << to_string(s);
    return os;
  }

};

void bindTypes(py::module_ &m);

} // namespace xilinx::AIE

#endif // AIE_PYBINDTYPES_H
