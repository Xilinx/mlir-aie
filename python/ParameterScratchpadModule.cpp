//===- ParameterScratchpadModule.cpp - Python bindings ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "parameter_scratchpad.h"

namespace py = pybind11;

PYBIND11_MODULE(_parameter_scratchpad, m) {
  m.doc() = "Python bindings for test_utils::ParameterScratchpad";

  py::class_<test_utils::ParameterScratchpad>(m, "ParameterScratchpad")
      .def(py::init([](py::buffer buf, const std::string &paramsPath) {
             py::buffer_info info = buf.request(/*writable=*/true);
             auto *ptr = static_cast<uint32_t *>(info.ptr);
             return new test_utils::ParameterScratchpad(ptr, paramsPath);
           }),
           py::arg("buffer"), py::arg("params_path"),
           py::keep_alive<1, 2>()) // prevent GC of buffer while alive
      .def("write_bytes",
           [](test_utils::ParameterScratchpad &self, const std::string &name,
              py::bytes data) {
             std::string s = data;
             self.writeBytes(name, s.data(), s.size());
           },
           py::arg("name"), py::arg("data"))
      .def("read", &test_utils::ParameterScratchpad::read, py::arg("name"));
}
