//===- ParameterScratchpadModule.cpp - Python bindings ----------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "parameter_scratchpad.h"

namespace nb = nanobind;
using namespace nb::literals;

using test_utils::ParameterScratchpad;

// Any writable, C-contiguous host buffer: a numpy table, or a mapped buffer
// object's memoryview. The scratchpad keeps its pointer, so a buffer is taken
// as it is (noconvert): a copy made to meet these would be written instead.
using HostBuffer = nb::ndarray<nb::c_contig, nb::device::cpu>;

NB_MODULE(_parameter_scratchpad, m) {
  m.doc() = "Python bindings for test_utils::ParameterScratchpad";

  nb::class_<ParameterScratchpad::Slot>(m, "Slot")
      .def_ro("index", &ParameterScratchpad::Slot::index)
      .def_ro("shift", &ParameterScratchpad::Slot::shift)
      .def_ro("kind", &ParameterScratchpad::Slot::kind)
      .def_ro("type", &ParameterScratchpad::Slot::type);

  nb::class_<ParameterScratchpad>(m, "ParameterScratchpad")
      .def(
          "__init__",
          [](ParameterScratchpad *self, HostBuffer buf,
             const std::string &paramsPath, const std::string &command) {
            if (reinterpret_cast<uintptr_t>(buf.data()) % alignof(uint32_t))
              throw nb::value_error("buffer must be 4-byte aligned");
            new (self) ParameterScratchpad(static_cast<uint32_t *>(buf.data()),
                                           buf.nbytes(), paramsPath, command);
          },
          "buffer"_a.noconvert(), "params_path"_a, "command"_a = "",
          nb::keep_alive<1, 2>()) // prevent GC of buffer while alive
      .def(
          "write_bytes",
          [](ParameterScratchpad &self, const std::string &name,
             nb::bytes data) {
            self.writeBytes(name, data.c_str(), data.size());
          },
          "name"_a, "data"_a)
      .def("validate", &ParameterScratchpad::validate)
      .def("names", &ParameterScratchpad::names)
      .def("read", &ParameterScratchpad::read, "name"_a)
      .def("slot", &ParameterScratchpad::slot, "name"_a)
      .def_static("slot_count", &ParameterScratchpad::slotCount,
                  "params_path"_a);
}
