//===- XRTModule.cpp -------------------------------------------000---*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie-c/Translation.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <algorithm>

namespace py = pybind11;
using namespace py::literals;

class PyAIERTXControl {
public:
  PyAIERTXControl(size_t partitionStartCol, size_t partitionNumCols)
      : ctl(getAieRtxControl(partitionStartCol, partitionNumCols)) {}

  ~PyAIERTXControl() { freeAieRtxControl(ctl); }

  AieRtxControl ctl;
};

PYBIND11_MODULE(_aiertx, m) {

  py::class_<PyAIERTXControl>(m, "AIERTXControl", py::module_local())
      .def(py::init<size_t, size_t>(), "partition_start_col"_a,
           "partition_num_cols"_a)
      .def("start_transaction",
           [](PyAIERTXControl &self) { aieRtxStartTransaction(self.ctl); })
      .def("export_serialized_transaction",
           [](PyAIERTXControl &self) {
             aieRtxExportSerializedTransaction(self.ctl);
           })
      .def(
          "dma_update_bd_addr",
          [](PyAIERTXControl &self, int col, int row, size_t addr,
             size_t bdId) {
            aieRtxDmaUpdateBdAddr(self.ctl, col, row, addr, bdId);
          },
          "col"_a, "row"_a, "addr"_a, "bd_id"_a);
}
