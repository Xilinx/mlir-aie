//===- AIERTModule.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "aie-c/TargetModel.h"
#include "aie-c/Translation.h"

#include "aie/Bindings/PyTypes.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <algorithm>

namespace py = pybind11;
using namespace py::literals;

class PyAIERTControl {
public:
  PyAIERTControl(AieTargetModel targetModel)
      : ctl(getAieRtControl(targetModel)) {}

  ~PyAIERTControl() { freeAieRtControl(ctl); }

  AieRtControl ctl;
};

PYBIND11_MODULE(_aiert, m) {

  py::class_<PyAIERTControl>(m, "AIERTControl", py::module_local())
      .def(py::init<PyAieTargetModel>(), "target_model"_a)
      .def("start_transaction",
           [](PyAIERTControl &self) { aieRtStartTransaction(self.ctl); })
      .def("export_serialized_transaction",
           [](PyAIERTControl &self) {
             aieRtExportSerializedTransaction(self.ctl);
           })
      .def(
          "dma_update_bd_addr",
          [](PyAIERTControl &self, int col, int row, size_t addr, size_t bdId) {
            aieRtDmaUpdateBdAddr(self.ctl, col, row, addr, bdId);
          },
          "col"_a, "row"_a, "addr"_a, "bd_id"_a);
}
